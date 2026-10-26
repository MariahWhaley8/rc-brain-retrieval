"""
Hybrid retrieval helpers for POST /retrieve.

Vector search and Postgres full-text search each produce a ranking. Reciprocal
rank fusion merges those rankings, the Voyage reranker orders the merged
candidates, and a per-document cap keeps one transcript from filling the list.

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


def vector_search_sql(where_sql: str) -> str:
    where = f"WHERE {where_sql}" if where_sql else ""
    return f"""
        SELECT
            {HIT_COLUMNS},
            1 - (embedding <=> %s::vector) AS similarity
        FROM knowledge_chunks
        {where}
        ORDER BY embedding <=> %s::vector
        LIMIT %s
    """


def keyword_search_sql(
    where_sql: str,
    plan: KeywordPlan,
    filter_params: list,
    limit: int,
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
    params.append(limit)
    sql = f"""
        SELECT
            {HIT_COLUMNS},
            ({" + ".join(rank_parts)}) AS rank
        FROM knowledge_chunks
        WHERE {" AND ".join(where_parts)}
        ORDER BY rank DESC
        LIMIT %s
    """
    return sql, params


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
) -> list[ChunkHit]:
    """
    Keep at most `per_document` chunks from each document, up to `limit`.

    When the capped list is shorter than `floor`, further chunks from those
    documents fill the gap. A question that only matches one transcript still
    returns a full set; a question that matches several does not get five
    parts of the same transcript at the expense of the others.
    """
    if limit <= 0:
        return []
    if floor is None:
        floor = limit
    floor = max(0, min(floor, limit))
    chosen: list[ChunkHit] = []
    overflow: list[ChunkHit] = []
    counts: dict[str, int] = {}
    for item in items:
        if len(chosen) >= limit:
            break
        doc_id = item.document_id
        seen = counts.get(doc_id, 0)
        if per_document <= 0 or seen < per_document:
            chosen.append(item)
            counts[doc_id] = seen + 1
        else:
            overflow.append(item)
    if len(chosen) < floor:
        for item in overflow:
            if len(chosen) >= floor:
                break
            chosen.append(item)
    return chosen


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
