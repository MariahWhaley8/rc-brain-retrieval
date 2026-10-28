"""
Hybrid retrieval helpers for POST /retrieve.

Vector search and Postgres full-text search each produce a ranking. Reciprocal
rank fusion merges those rankings, identical and near-identical passages are
collapsed (including Drive copies of the same file), the Voyage reranker orders
what remains, and a per-document cap keeps one transcript from filling the list.
When that leaves the list short, later distinct passages fill the open slots.

The full-text configuration is `simple` (lowercase, no stemming) so scanner
codes, acronyms, and names stay intact. Alphabetic terms are prefix-matched
so "RGU" still hits "RGUs".
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass

log = logging.getLogger(__name__)

# Titles that contain this text are answer keys for the retrieval eval.
# They are always excluded, including when the caller passes their own list.
BUILTIN_TITLE_EXCLUSION = "RC Brain Agent Evaluation Q&A"

RRF_K = 60

# Planner settings that force a sequential scan. An IVFFlat or HNSW index
# built outside this repo can return fewer rows than LIMIT (including zero)
# when probes / ef_search are low. About 9,000 chunks is small enough that
# an exact scan is the reliable choice.
EXACT_SCAN_SETTINGS = (
    "SET LOCAL enable_indexscan = off",
    "SET LOCAL enable_bitmapscan = off",
)
# Applied even when the planner still picks an ANN index. Unknown GUCs
# (older pgvector builds) are skipped per session.
BROAD_ANN_SETTINGS = (
    "SET LOCAL ivfflat.probes = 1000",
    "SET LOCAL hnsw.ef_search = 1000",
    "SET LOCAL hnsw.iterative_scan = relaxed_order",
    "SET LOCAL ivfflat.iterative_scan = relaxed_order",
    "SET LOCAL hnsw.max_scan_tuples = 20000",
    "SET LOCAL ivfflat.max_probes = 1000",
)

HIT_COLUMNS = """
    document_id,
    title,
    category,
    topic,
    source,
    source_url,
    source_date,
    is_authoritative,
    chunk_text,
    chunk_index
"""

# Exact text identity used in SQL so Drive copies cannot fill the LIMIT
# before a different passage is ever read. Punctuation and case are ignored.
TEXT_DEDUP_KEY = (
    "md5(trim(regexp_replace(lower(COALESCE(chunk_text, '')), "
    "'[^a-z0-9]+', ' ', 'g')))"
)
# 0 disables the per-document pool cap. A huge window cap is the same thing
# and keeps the placeholder count stable.
DOC_RANK_CAP_DISABLED = 1_000_000

# Across documents, this much shared wording is the same passage.
NEAR_DUP_JACCARD = 0.82
# Inside one document, only an almost-verbatim repeat is collapsed. Adjacent
# chunks share a short overlap and must both survive.
SAME_DOC_JACCARD = 0.97
_SHINGLE_SIZE = 8

DEFAULT_NEIGHBOR_DOCS = 2
DEFAULT_NEIGHBOR_RADIUS = 1

_WS = re.compile(r"\s+")
_NON_WORD = re.compile(r"[^a-z0-9\s]+")
_COPY_SUFFIX = re.compile(r"\s*\(\d+\)\s*$")

_STOPWORDS = frozenset(
    """
    a an the of and or to for in on at by with from is are was were be been being
    as that this it its what which who whom how when where why do does did can
    could should would about into your our their not but if then than so such
    there here also we you they he she his her them my me i am have has had will
    just over under after before between out up down only other any all each more
    most some no nor too very per via within without during both few these those
    same own
    """.split()
)

_HYPHENATED = re.compile(r"\b[A-Za-z0-9]+(?:-[A-Za-z0-9]+)+\b")
_ACRONYM = re.compile(r"\b[A-Z]{2,}s?\b")
_CODE = re.compile(r"\b\d{2,}\b")
_TOKEN = re.compile(r"[A-Za-z0-9]+")
_SAFE_LEXEME = re.compile(r"^[a-z0-9]+$")

_skipped_settings: set[str] = set()


@dataclass
class ChunkHit:
    document_id: str
    title: str | None
    category: str | None
    topic: str | None
    source: str | None
    source_url: str | None
    source_date: str | None
    is_authoritative: bool
    chunk_text: str
    chunk_index: int | None
    score: float | None = None

    def identity(self) -> tuple:
        return (self.document_id, self.chunk_index, self.chunk_text)


@dataclass(frozen=True)
class KeywordPlan:
    """OR full-text query plus exact patterns for codes, acronyms, and ids."""

    tsquery: str | None
    exact_patterns: tuple[str, ...]
    lexemes: tuple[str, ...]


def hit_from_row(row) -> ChunkHit:
    score = None
    if len(row) > 10 and row[10] is not None:
        score = float(row[10])
    chunk_index = row[9]
    if chunk_index is not None:
        chunk_index = int(chunk_index)
    return ChunkHit(
        document_id=str(row[0]),
        title=row[1],
        category=row[2],
        topic=row[3],
        source=row[4],
        source_url=row[5],
        source_date=None if row[6] is None else str(row[6]),
        is_authoritative=bool(row[7]),
        chunk_text=row[8] or "",
        chunk_index=chunk_index,
        score=score,
    )


def title_exclusion_substrings(extra: list[str] | None = None) -> list[str]:
    """Builtin eval-key exclusion, plus env and per-request substrings."""
    raw: list[str] = [BUILTIN_TITLE_EXCLUSION]
    configured = os.environ.get("RETRIEVAL_EXCLUDE_TITLES", "")
    if configured.strip():
        raw.extend(part.strip() for part in configured.split("||"))
    if extra:
        raw.extend(part.strip() for part in extra if part and str(part).strip())
    found: list[str] = []
    seen: set[str] = set()
    for item in raw:
        text = str(item).strip()
        if not text:
            continue
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        found.append(text)
    return found


def ilike_contains(substring: str) -> str:
    """Case-insensitive substring pattern. `%` and `_` in the text are literal."""
    escaped = (
        substring.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    )
    return f"%{escaped}%"


def title_is_excluded(title: str | None, substrings: list[str]) -> bool:
    if not title or not substrings:
        return False
    folded = title.casefold()
    return any(item.casefold() in folded for item in substrings)


def category_is_excluded(category: str | None, categories: list[str]) -> bool:
    if category is None or not categories:
        return False
    return category in set(categories)


def clean_strings(values: list[str] | None) -> list[str]:
    if not values:
        return []
    found: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        found.append(text)
    return found


def _add_unique(items: list[str], seen: set[str], token: str) -> None:
    key = token.casefold()
    if not token or key in seen:
        return
    seen.add(key)
    items.append(token)


def exact_tokens(question: str, limit: int = 12) -> list[str]:
    """Surface forms worth an exact bonus: MGB-020, RGU, RGUs, 02, 07."""
    found: list[str] = []
    seen: set[str] = set()
    for match in _HYPHENATED.finditer(question):
        _add_unique(found, seen, match.group(0))
    for match in _ACRONYM.finditer(question):
        token = match.group(0)
        _add_unique(found, seen, token)
        # "RGUs" should also hit the uninflected acronym.
        if token.endswith("s") and token[:-1].isupper() and len(token) > 3:
            _add_unique(found, seen, token[:-1])
    for match in _CODE.finditer(question):
        _add_unique(found, seen, match.group(0))
    return found[:limit]


def lexemes_for(question: str, limit: int = 24) -> list[str]:
    acronyms = {match.group(0).casefold() for match in _ACRONYM.finditer(question)}
    found: list[str] = []
    seen: set[str] = set()

    def add(token: str) -> None:
        token = token.casefold()
        if not _SAFE_LEXEME.fullmatch(token or ""):
            return
        if token in seen:
            return
        if token in _STOPWORDS and token not in acronyms:
            return
        if len(token) < 2:
            return
        if len(token) == 2 and not any(char.isdigit() for char in token) and token not in acronyms:
            return
        seen.add(token)
        found.append(token)

    for token in _TOKEN.findall(question):
        add(token)
        folded = token.casefold()
        if len(folded) >= 4 and folded.endswith("s") and not folded.endswith("ss"):
            add(folded[:-1])
    return found[:limit]


def plan_keyword_query(question: str) -> KeywordPlan:
    lexemes = lexemes_for(question)
    parts: list[str] = []
    for lexeme in lexemes:
        if len(lexeme) >= 3 and lexeme.isalpha():
            parts.append(f"{lexeme}:*")
        else:
            parts.append(lexeme)
    patterns = tuple(_boundary_pattern(token) for token in exact_tokens(question))
    return KeywordPlan(
        tsquery=" | ".join(parts) if parts else None,
        exact_patterns=patterns,
        lexemes=tuple(lexemes),
    )


def _boundary_pattern(token: str) -> str:
    """POSIX regex: the token as its own word, so '02' does not match '2024'."""
    return rf"(^|[^[:alnum:]]){re.escape(token)}([^[:alnum:]]|$)"


def _text_expr() -> str:
    return "COALESCE(title, '') || ' ' || COALESCE(chunk_text, '')"


def retrieval_where(
    *,
    category: str | None,
    source: str | None,
    exclude_titles: list[str] | None,
    exclude_categories: list[str] | None,
) -> tuple[str, list]:
    parts: list[str] = []
    params: list = []
    if category:
        parts.append("category = %s")
        params.append(category)
    if source:
        parts.append("source = %s")
        params.append(source)
    patterns = [ilike_contains(item) for item in title_exclusion_substrings(exclude_titles)]
    if patterns:
        parts.append("COALESCE(title, '') NOT ILIKE ALL(%s)")
        params.append(patterns)
    categories = clean_strings(exclude_categories)
    if categories:
        parts.append("(category IS NULL OR category <> ALL(%s))")
        params.append(categories)
    return " AND ".join(parts), params


def doc_cap_param(per_document: int) -> int:
    """SQL window cap. Zero means 'no cap' without dropping the placeholder."""
    if per_document <= 0:
        return DOC_RANK_CAP_DISABLED
    return per_document


def _diversify(inner_sql: str) -> str:
    """
    Keep the best row per document, then one row per exact passage text, then LIMIT.

    The inner query selects HIT_COLUMNS and a numeric `score` (higher is better).
    Two extra placeholders follow the inner ones: the per-document cap, then LIMIT.
    """
    return f"""
        SELECT
            {HIT_COLUMNS},
            score
        FROM (
            SELECT
                {HIT_COLUMNS},
                score,
                ROW_NUMBER() OVER (
                    PARTITION BY {TEXT_DEDUP_KEY}
                    ORDER BY score DESC, document_id, chunk_index
                ) AS text_rank
            FROM (
                SELECT
                    {HIT_COLUMNS},
                    score,
                    ROW_NUMBER() OVER (
                        PARTITION BY document_id
                        ORDER BY score DESC, chunk_index
                    ) AS doc_rank
                FROM (
                    {inner_sql}
                ) raw_hits
            ) per_document
            WHERE doc_rank <= %s
        ) per_text
        WHERE text_rank = 1
        ORDER BY score DESC, document_id, chunk_index
        LIMIT %s
    """


def vector_search_sql(where_sql: str) -> str:
    where = f"WHERE {where_sql}" if where_sql else ""
    inner = f"""
        SELECT
            {HIT_COLUMNS},
            1 - (embedding <=> %s::vector) AS score
        FROM knowledge_chunks
        {where}
    """
    return _diversify(inner)


def keyword_search_sql(
    where_sql: str,
    plan: KeywordPlan,
    filter_params: list,
    limit: int,
    per_document: int = 4,
) -> tuple[str | None, list]:
    """Full-text OR exact-token match. Params follow the `%s` order in the SQL."""
    if not plan.tsquery and not plan.exact_patterns:
        return None, []

    rank_parts: list[str] = []
    rank_params: list = []
    match_parts: list[str] = []
    match_params: list = []
    text = _text_expr()

    if plan.tsquery:
        rank_parts.append("COALESCE(ts_rank_cd(search_tsv, to_tsquery('simple', %s)), 0)")
        rank_params.append(plan.tsquery)
        match_parts.append("search_tsv @@ to_tsquery('simple', %s)")
        match_params.append(plan.tsquery)
    for pattern in plan.exact_patterns:
        rank_parts.append(f"(CASE WHEN {text} ~* %s THEN 10 ELSE 0 END)")
        rank_params.append(pattern)
        match_parts.append(f"{text} ~* %s")
        match_params.append(pattern)

    where_parts = ["(" + " OR ".join(match_parts) + ")"]
    params = rank_params + match_params
    if where_sql:
        where_parts.append(f"({where_sql})")
        params.extend(filter_params)
    params.append(doc_cap_param(per_document))
    params.append(limit)
    inner = f"""
        SELECT
            {HIT_COLUMNS},
            ({" + ".join(rank_parts)}) AS score
        FROM knowledge_chunks
        WHERE {" AND ".join(where_parts)}
    """
    return _diversify(inner), params


def rrf_merge(rankings: list[list[ChunkHit]], k: int = RRF_K) -> list[ChunkHit]:
    """Merge rankings with reciprocal rank fusion. Ties keep the earlier list."""
    scores: dict[tuple, float] = {}
    order: dict[tuple, int] = {}
    chosen: dict[tuple, ChunkHit] = {}
    seen = 0
    for ranking in rankings:
        for rank, hit in enumerate(ranking, start=1):
            key = hit.identity()
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            if key not in chosen:
                chosen[key] = hit
                order[key] = seen
                seen += 1
    keys = sorted(scores, key=lambda key: (-scores[key], order[key]))
    return [chosen[key] for key in keys]


def select_with_document_cap(
    items: list[ChunkHit],
    limit: int,
    per_document: int,
    floor: int | None = None,
    group_key=None,
) -> list[ChunkHit]:
    """
    Keep at most `per_document` chunks from each document, up to `limit`.

    `group_key` lets Drive copies that share a base title ("Notes" and
    "Notes (1)") count as one document. When the capped list is shorter than
    `floor`, further distinct chunks from those documents fill the gap. A
    question that only matches one transcript still returns a full set; a
    question that matches several does not get five parts of the same
    transcript at the expense of the others.

    Callers collapse identical text before this function. The floor backfill
    must not put that text back.
    """
    if limit <= 0:
        return []
    if floor is None:
        floor = limit
    floor = max(0, min(floor, limit))

    def key_of(item: ChunkHit) -> str:
        if group_key is None:
            return item.document_id
        return group_key(item)

    chosen: list[ChunkHit] = []
    overflow: list[ChunkHit] = []
    counts: dict[str, int] = {}
    seen_text: set[str] = set()
    for item in items:
        if len(chosen) >= limit:
            break
        norm = normalize_passage_text(item.chunk_text)
        if norm and norm in seen_text:
            continue
        doc_key = key_of(item)
        seen = counts.get(doc_key, 0)
        if per_document <= 0 or seen < per_document:
            chosen.append(item)
            counts[doc_key] = seen + 1
            if norm:
                seen_text.add(norm)
        else:
            overflow.append(item)
    if len(chosen) < floor:
        for item in overflow:
            if len(chosen) >= floor:
                break
            norm = normalize_passage_text(item.chunk_text)
            if norm and norm in seen_text:
                continue
            chosen.append(item)
            if norm:
                seen_text.add(norm)
    return chosen


def normalize_passage_text(text: str | None) -> str:
    """Case, punctuation, and whitespace folded so two copies compare equal."""
    folded = (text or "").casefold().replace("\u00a0", " ")
    folded = _NON_WORD.sub(" ", folded)
    return _WS.sub(" ", folded).strip()


def canonical_title(title: str | None) -> str:
    """Title without a trailing Drive copy suffix such as ' (1)'."""
    text = (title or "").strip()
    while True:
        updated = _COPY_SUFFIX.sub("", text).strip()
        if updated == text:
            return text.casefold()
        text = updated


def is_copy_title(title: str | None) -> bool:
    raw = (title or "").strip()
    base = canonical_title(title)
    return bool(base) and raw.casefold() != base


def _copy_sort_key(title: str | None) -> tuple[int, int]:
    """Lower sorts first. The file without a copy suffix beats ' (1)', then ' (2)'."""
    if not is_copy_title(title):
        return (0, 0)
    nums = [int(value) for value in re.findall(r"\((\d+)\)", (title or ""))]
    return (1, nums[-1] if nums else 999)


def document_cap_keys(items: list[ChunkHit]) -> dict[str, str]:
    """
    Map document_id to a shared cap key.

    Only titles that actually look like Drive copies are grouped, so two
    unrelated documents that happen to share a title keep separate caps.
    """
    members: dict[str, list[str]] = {}
    has_copy: dict[str, bool] = {}
    for item in items:
        base = canonical_title(item.title)
        if not base:
            continue
        bucket = members.setdefault(base, [])
        if item.document_id not in bucket:
            bucket.append(item.document_id)
        if is_copy_title(item.title):
            has_copy[base] = True
    keys: dict[str, str] = {}
    for base, doc_ids in members.items():
        if len(doc_ids) < 2 or not has_copy.get(base):
            continue
        group = "title:" + base
        for doc_id in doc_ids:
            keys[doc_id] = group
    return keys


def make_group_key(items: list[ChunkHit]):
    groups = document_cap_keys(items)

    def key(item: ChunkHit) -> str:
        return groups.get(item.document_id, item.document_id)

    return key


def _shingles(norm: str) -> frozenset:
    words = norm.split()
    if not words:
        return frozenset()
    if len(words) > 1500:
        words = words[:1500]
    if len(words) <= _SHINGLE_SIZE:
        return frozenset((tuple(words),))
    step = 1 if len(words) < 80 else 2
    return frozenset(
        tuple(words[index : index + _SHINGLE_SIZE])
        for index in range(0, len(words) - _SHINGLE_SIZE + 1, step)
    )


def _jaccard(left: frozenset, right: frozenset) -> float:
    if not left or not right:
        return 0.0
    overlap = len(left & right)
    if overlap == 0:
        return 0.0
    return overlap / (len(left) + len(right) - overlap)


def texts_are_near_duplicate(
    left: str,
    right: str,
    left_shingles: frozenset | None = None,
    right_shingles: frozenset | None = None,
    *,
    same_document: bool = False,
) -> bool:
    """True when two normalized passages are the same text for retrieval."""
    if not left or not right:
        return False
    if left == right:
        return True
    shorter, longer = (left, right) if len(left) <= len(right) else (right, left)
    length_ratio = len(shorter) / len(longer)
    if length_ratio < 0.85:
        return False
    if length_ratio >= 0.9 and shorter in longer:
        return not same_document or length_ratio >= 0.98
    threshold = SAME_DOC_JACCARD if same_document else NEAR_DUP_JACCARD
    if left_shingles is None:
        left_shingles = _shingles(left)
    if right_shingles is None:
        right_shingles = _shingles(right)
    return _jaccard(left_shingles, right_shingles) >= threshold


def collapse_similar_passages(items: list[ChunkHit]) -> list[ChunkHit]:
    """
    Drop identical and near-identical passages, including copies in other docs.

    The earlier hit wins (it ranked higher). A title without a Drive copy
    suffix replaces "Notes (2)" so later neighbor lookup reads the original file.
    """
    kept: list[ChunkHit] = []
    norms: list[str] = []
    shingles: list[frozenset] = []
    exact: dict[str, int] = {}

    for item in items:
        norm = normalize_passage_text(item.chunk_text)
        if not norm:
            kept.append(item)
            norms.append("")
            shingles.append(frozenset())
            continue
        def adopt(index: int) -> None:
            previous_norm = norms[index]
            if exact.get(previous_norm) == index:
                del exact[previous_norm]
            item.score = kept[index].score
            kept[index] = item
            norms[index] = norm
            shingles[index] = item_shingles
            exact[norm] = index

        existing = exact.get(norm)
        if existing is not None:
            if _copy_sort_key(item.title) < _copy_sort_key(kept[existing].title):
                item_shingles = shingles[existing]
                adopt(existing)
            continue
        item_shingles = _shingles(norm)
        match_at = None
        for index, previous in enumerate(norms):
            if not previous:
                continue
            same_document = kept[index].document_id == item.document_id
            if texts_are_near_duplicate(
                norm,
                previous,
                item_shingles,
                shingles[index],
                same_document=same_document,
            ):
                match_at = index
                break
        if match_at is not None:
            if _copy_sort_key(item.title) < _copy_sort_key(kept[match_at].title):
                adopt(match_at)
            continue
        exact[norm] = len(kept)
        kept.append(item)
        norms.append(norm)
        shingles.append(item_shingles)
    return kept


def neighbor_budget(limit: int) -> int:
    """How many adjacent chunks may displace lower-ranked passages."""
    if limit < 4:
        return 0
    return min(4, limit // 3)


def neighbor_pairs(
    hits: list[ChunkHit],
    doc_ids: list[str],
    radius: int = DEFAULT_NEIGHBOR_RADIUS,
) -> list[tuple[str, int]]:
    """(document_id, chunk_index) pairs sitting beside an already chosen chunk."""
    wanted: list[tuple[str, int]] = []
    seen: set[tuple[str, int]] = set()
    selected = {
        (hit.document_id, hit.chunk_index)
        for hit in hits
        if hit.chunk_index is not None
    }
    allowed = set(doc_ids)
    for hit in hits:
        if hit.document_id not in allowed or hit.chunk_index is None:
            continue
        for offset in range(-radius, radius + 1):
            if offset == 0:
                continue
            pair = (hit.document_id, hit.chunk_index + offset)
            if pair[1] < 0 or pair in seen or pair in selected:
                continue
            seen.add(pair)
            wanted.append(pair)
    return wanted


def top_document_ids(hits: list[ChunkHit], count: int) -> list[str]:
    found: list[str] = []
    for hit in hits:
        if hit.document_id in found:
            continue
        found.append(hit.document_id)
        if len(found) >= count:
            break
    return found


def neighbor_sql() -> str:
    return f"""
        /* neighbor_chunks */
        SELECT
            {HIT_COLUMNS},
            NULL::float8 AS score
        FROM knowledge_chunks
        WHERE (document_id, chunk_index) IN %s
    """


def splice_neighbors(
    primary: list[ChunkHit],
    neighbors: list[ChunkHit],
    *,
    limit: int,
    expand_doc_ids: list[str],
    max_neighbors: int,
) -> list[ChunkHit]:
    """
    Place adjacent chunks next to the passage they continue.

    Numbered lists are split across chunk boundaries. The neighbor inherits a
    score just under its parent. The list stays at `limit`; lower-ranked
    passages from other documents make room first.
    """
    by_pair = {
        (hit.document_id, hit.chunk_index): hit
        for hit in neighbors
        if hit.chunk_index is not None
    }
    allowed = set(expand_doc_ids)
    emitted_norm: set[str] = set()
    emitted_id: set[tuple] = set()
    output: list[ChunkHit] = []
    budget = max(0, max_neighbors)

    def add(hit: ChunkHit, *, is_neighbor: bool) -> None:
        nonlocal budget
        norm = normalize_passage_text(hit.chunk_text)
        ident = (hit.document_id, hit.chunk_index, norm)
        if ident in emitted_id:
            return
        if norm and norm in emitted_norm:
            return
        if is_neighbor:
            if budget <= 0:
                return
            budget -= 1
        emitted_id.add(ident)
        if norm:
            emitted_norm.add(norm)
        output.append(hit)

    def add_adjacent(hit: ChunkHit, offset: int) -> None:
        adjacent = by_pair.get((hit.document_id, hit.chunk_index + offset))
        if adjacent is None:
            return
        if adjacent.score is None and hit.score is not None:
            adjacent.score = hit.score * 0.99
        add(adjacent, is_neighbor=True)

    for hit in primary:
        if hit.document_id in allowed and hit.chunk_index is not None:
            add_adjacent(hit, -1)
            add(hit, is_neighbor=False)
            add_adjacent(hit, 1)
        else:
            add(hit, is_neighbor=False)

    if len(output) <= limit:
        return output
    protected = set(expand_doc_ids)
    while len(output) > limit:
        removed = False
        for index in range(len(output) - 1, -1, -1):
            if output[index].document_id not in protected:
                del output[index]
                removed = True
                break
        if not removed:
            return output[:limit]
    return output


# Exact duplicate rows: same document, same chunk slot, same text.
IDENTICAL_CHUNK_SQL = """
    /* duplicate_identical */
    SELECT
        document_id,
        MIN(title) AS title,
        chunk_index,
        COUNT(*) AS copies,
        LEFT(MIN(chunk_text), 180) AS preview
    FROM knowledge_chunks
    GROUP BY document_id, chunk_index, md5(COALESCE(chunk_text, ''))
    HAVING COUNT(*) > 1
    ORDER BY copies DESC, document_id, chunk_index
"""

CROSS_DOCUMENT_SQL = """
    /* duplicate_cross_doc */
    SELECT
        COUNT(*) AS rows,
        COUNT(DISTINCT document_id) AS documents,
        LEFT(MIN(chunk_text), 180) AS preview,
        (ARRAY_AGG(DISTINCT title))[1:8] AS titles
    FROM knowledge_chunks
    GROUP BY md5(COALESCE(chunk_text, ''))
    HAVING COUNT(DISTINCT document_id) > 1
    ORDER BY documents DESC, rows DESC
"""

REPEATED_WITHIN_DOCUMENT_SQL = """
    /* duplicate_repeated_within */
    SELECT
        document_id,
        MIN(title) AS title,
        COUNT(*) AS copies,
        COUNT(DISTINCT chunk_index) AS chunk_slots,
        LEFT(MIN(chunk_text), 180) AS preview
    FROM knowledge_chunks
    GROUP BY document_id, md5(COALESCE(chunk_text, ''))
    HAVING COUNT(*) > 1 AND COUNT(DISTINCT chunk_index) > 1
    ORDER BY copies DESC, document_id
"""

DELETE_IDENTICAL_SQL = """
    /* duplicate_delete */
    DELETE FROM knowledge_chunks
    WHERE ctid IN (
        SELECT ctid FROM (
            SELECT
                ctid,
                ROW_NUMBER() OVER (
                    PARTITION BY document_id, chunk_index, md5(COALESCE(chunk_text, ''))
                    ORDER BY ctid
                ) AS rn
            FROM knowledge_chunks
        ) numbered
        WHERE rn > 1
    )
"""


def _group_row(row, kind: str) -> dict:
    if kind == "identical":
        return {
            "document_id": row[0],
            "title": row[1],
            "chunk_index": None if row[2] is None else int(row[2]),
            "copies": int(row[3]),
            "extra_rows": int(row[3]) - 1,
            "preview": row[4] or "",
        }
    if kind == "cross":
        titles = row[3] or []
        return {
            "rows": int(row[0]),
            "documents": int(row[1]),
            "preview": row[2] or "",
            "titles": [title for title in titles if title],
        }
    return {
        "document_id": row[0],
        "title": row[1],
        "copies": int(row[2]),
        "chunk_slots": int(row[3]),
        "preview": row[4] or "",
    }


def collect_duplicate_chunks(cur, apply: bool = False, preview_limit: int = 50) -> dict:
    """
    Report exact duplicate chunk rows. Delete them only when `apply` is true.

    A duplicate row is the same document_id, chunk_index, and chunk_text stored
    more than once. The same text under two document ids (a Drive copy indexed
    as its own file) is reported and left in place; retrieval collapses it,
    and the next reindex would insert it again. Repeated paragraphs at two
    chunk indexes of one document are also report-only.
    """
    cur.execute(IDENTICAL_CHUNK_SQL)
    identical_rows = cur.fetchall()
    cur.execute(CROSS_DOCUMENT_SQL)
    cross_rows = cur.fetchall()
    cur.execute(REPEATED_WITHIN_DOCUMENT_SQL)
    repeated_rows = cur.fetchall()

    extra_rows = sum(int(row[3]) - 1 for row in identical_rows)
    deleted = 0
    if apply and extra_rows:
        cur.execute(DELETE_IDENTICAL_SQL)
        deleted = int(cur.rowcount or 0)

    preview_limit = max(0, preview_limit)
    return {
        "apply": bool(apply),
        "mode": "deleted" if apply else "report",
        "deleted": deleted,
        "identical_row_groups": len(identical_rows),
        "extra_identical_rows": extra_rows,
        "identical_groups": [
            _group_row(row, "identical") for row in identical_rows[:preview_limit]
        ],
        "identical_groups_truncated": len(identical_rows) > preview_limit,
        "cross_document_text_groups": len(cross_rows),
        "cross_document_preview": [
            _group_row(row, "cross") for row in cross_rows[:preview_limit]
        ],
        "cross_document_truncated": len(cross_rows) > preview_limit,
        "repeated_within_document_groups": len(repeated_rows),
        "repeated_within_document_preview": [
            _group_row(row, "within") for row in repeated_rows[:preview_limit]
        ],
        "repeated_within_document_truncated": len(repeated_rows) > preview_limit,
        "note": (
            "Deletion removes extra physical rows that share document_id, "
            "chunk_index, and chunk_text, and keeps one. Cross-document copies "
            "and repeated paragraphs at different chunk indexes are reported only."
        ),
    }


def _set_local(cur, statement: str) -> None:
    cur.execute("SAVEPOINT retrieval_guc")
    try:
        cur.execute(statement)
    except Exception as exc:
        cur.execute("ROLLBACK TO SAVEPOINT retrieval_guc")
        cur.execute("RELEASE SAVEPOINT retrieval_guc")
        if statement not in _skipped_settings:
            _skipped_settings.add(statement)
            log.info("Vector setting skipped (%s): %s", statement, exc)
        return
    cur.execute("RELEASE SAVEPOINT retrieval_guc")


def apply_vector_settings(cur, exact: bool = True) -> None:
    """
    Make the following vector query return a full exact neighbor list.

    Keyword search should run before this. Disabling index scans also keeps
    the full-text GIN index from being used, which is fine at this table size
    but pointless for a query that has already finished.
    """
    if exact:
        for statement in EXACT_SCAN_SETTINGS:
            _set_local(cur, statement)
    for statement in BROAD_ANN_SETTINGS:
        _set_local(cur, statement)
