#!/usr/bin/env python3
"""
RC Brain — Knowledge Indexer
=============================
Pulls KnowledgeDocument records from Base44, splits each one into passages,
embeds those passages with Voyage AI, and upserts them into PostgreSQL + pgvector.

The walk is incremental. A document is re-embedded only when its content hash
changes, when its stored embed version is older than the current title-prefixed
embedding, or when --force is set. Chunks whose Base44 document has been deleted
are removed. The existing index is never truncated. A tsvector column is added
in place for keyword search; embeddings already in the table are left as they are
until a document is re-embedded.

The Base44 list call uses cursor pagination
(GET /api/apps/{app_id}/entities/KnowledgeDocument/v2/list on
https://app.base44.com). The previous indexer called https://api.base44.com
with skip/limit. That host now returns an HTML 404, and the current list API
rejects `skip` with HTTP 400, so the one-shot Railway deployment crashed.

Usage:
    python indexer.py            # skip unchanged documents
    python indexer.py --force    # re-embed every document

Environment variables (set in Railway or a .env file):
    DATABASE_URL          - PostgreSQL connection string
    VOYAGE_API_KEY        - Voyage AI API key
    BASE44_API_KEY        - Base44 personal access token (or a legacy account key)
    BASE44_APP_ID         - RC Brain app ID (default is the existing app id)
    BASE44_API_BASE       - API origin (default: https://app.base44.com)
    BASE44_RPM            - Max Base44 requests per minute (default: 60)
    BASE44_PAGE_SIZE      - Cursor page size, 1–500 (default: 100)
    VOYAGE_EMBED_MODEL    - Must match the model used for the live index
                            (default: voyage-large-2)
    VOYAGE_EMBED_BATCH_SIZE
    VOYAGE_MAX_TOKENS_PER_REQUEST
    VOYAGE_MIN_INTERVAL_SECONDS
    CHUNK_SIZE            - Max characters per chunk (default: 800)
    CHUNK_OVERLAP         - Overlap between chunks in characters (default: 150)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import sys
import time
from datetime import datetime, timedelta, timezone
from typing import Callable
from urllib.parse import quote

import psycopg2
import requests
import voyageai
from dotenv import load_dotenv

import search

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

DEFAULT_BASE44_API_BASE = "https://app.base44.com"
DEFAULT_APP_ID = "69ef93740d8ea9e5021345b0"
ENTITY_NAME = "KnowledgeDocument"
USER_AGENT = "rc-brain-retrieval/1.1"
# Arbitrary constant. Session advisory lock so two Railway replicas cannot
# index at the same time. Released when the connection closes.
ADVISORY_LOCK_KEY = 7482011267
# voyage-large-2 accepts 16k tokens per string. Stay far under that, and under
# the ~120k token request cap, using a chars/4 estimate.
MAX_CHUNK_CHARS = 20_000
DEFAULT_EMBED_BATCH_SIZE = 64
DEFAULT_EMBED_MAX_TOKENS = 100_000
DEFAULT_EMBED_MIN_INTERVAL = 0.5
DEFAULT_BASE44_RPM = 60
MAX_ERROR_DETAILS = 20
# List endpoint base limit is 70 rpm; record GETs are 150 rpm. 60 leaves room.
LIST_FIELD_ATTEMPTS = (
    "title,updated_date,category,topic,subtopic,source,source_url,source_date,is_authoritative",
    "title,updated_date",
    None,
)
TRANSIENT_VOYAGE = (
    voyageai.error.RateLimitError,
    voyageai.error.ServiceUnavailableError,
    voyageai.error.Timeout,
    voyageai.error.APIConnectionError,
    voyageai.error.TryAgain,
    voyageai.error.ServerError,
    voyageai.error.APIError,
)
RETRYABLE_REQUESTS = (
    requests.Timeout,
    requests.ConnectionError,
    requests.exceptions.ChunkedEncodingError,
)
SUCCESS_RUN_STATUSES = {"succeeded", "completed_with_errors"}

_voyage_client = None


class Base44Error(RuntimeError):
    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


def coerce_bool(value, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def content_hash(text: str) -> str:
    """MD5 of the stripped document body. Existing rows were hashed this way."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()


# Bump when the string sent to Voyage changes. Rows stored before this version
# are re-embedded on the next incremental reindex. The body hash stays the
# same, so a title-only embedding change is not confused with an edit.
EMBED_TEXT_VERSION = 2


def passage_for_model(title: str | None, chunk: str) -> str:
    """
    Text embedded and reranked for a passage.

    The stored chunk_text stays the passage itself. The title is prepended so
    sibling documents (Vendor Leads vs Sponsorships) are distinguishable when
    the body never repeats the title.
    """
    body = chunk or ""
    heading = (title or "").strip()
    if not heading or heading.casefold() == "untitled":
        return body
    prefix = f"Title: {heading}\n\n"
    if body.startswith(prefix):
        return body
    return prefix + body


def search_vector_expr(title_sql: str, topic_sql: str, body_sql: str) -> str:
    """
    Weighted tsvector. Title and topic are weight A; the passage is weight B.

    `simple` keeps codes and names unstemmed. The same expression is used for
    the backfill, writes, and the trigger.
    """
    return (
        "setweight(to_tsvector('simple', "
        f"coalesce({title_sql}, '') || ' ' || coalesce({topic_sql}, '')), 'A') || "
        "setweight(to_tsvector('simple', "
        f"coalesce({body_sql}, '')), 'B')"
    )


def normalize_timestamp(value) -> str | None:
    """Compare Base44 timestamps that sometimes include a trailing Z."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1]
    if text.endswith("+00:00"):
        text = text[:-6]
    return text


def as_text(value) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        return text or None
    return str(value)


def document_content(doc: dict) -> str:
    raw = doc.get("content")
    if raw is None:
        return ""
    if not isinstance(raw, str):
        raw = str(raw)
    return raw.replace("\x00", "").strip()


def choose_document_id(doc: dict, local_ids: set[str]) -> str | None:
    """Prefer an id already stored, so a rename from `_id` to `id` does not duplicate rows."""
    candidates: list[str] = []
    for key in ("id", "_id", "document_id"):
        value = doc.get(key)
        if value:
            candidates.append(str(value))
    if not candidates:
        return None
    for candidate in candidates:
        if candidate in local_ids:
            return candidate
    return candidates[0]


def base44_entity_id(doc: dict) -> str | None:
    value = doc.get("id") or doc.get("_id") or doc.get("document_id")
    return str(value) if value else None


def chunk_text(text: str, size: int | None = None, overlap: int | None = None) -> list[str]:
    """
    Split text into overlapping chunks, breaking on a sentence boundary when
    one falls in the last 200 characters of the window.

    The window always advances. A chunk size smaller than the overlap, or a
    boundary that would not move the cursor, cannot loop forever — including
    on documents around 100k characters.
    """
    if size is None:
        size = int(os.environ.get("CHUNK_SIZE", "800"))
    if overlap is None:
        overlap = int(os.environ.get("CHUNK_OVERLAP", "150"))
    if size < 1:
        raise ValueError("chunk size must be positive")
    if size > MAX_CHUNK_CHARS:
        log.warning(
            "CHUNK_SIZE %s exceeds %s characters; clamping so a chunk stays under the Voyage context limit",
            size,
            MAX_CHUNK_CHARS,
        )
        size = MAX_CHUNK_CHARS
    overlap = max(0, min(overlap, size - 1))

    chunks: list[str] = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + size, length)
        if end < length:
            window = max(start + 1, end - min(200, size - 1))
            boundary = text.rfind(". ", window, end)
            if boundary != -1 and boundary + 1 > start:
                end = boundary + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= length:
            break
        next_start = end - overlap
        if next_start <= start:
            next_start = end
        start = next_start
    return chunks


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def iter_embed_batches(texts: list[str], max_items: int, max_tokens: int):
    """Yield batches that stay within Voyage's per-request item and token caps."""
    if max_items < 1:
        max_items = 1
    if max_tokens < 1:
        max_tokens = 1
    batch: list[str] = []
    tokens = 0
    for text in texts:
        est = estimate_tokens(text)
        if batch and (len(batch) >= max_items or tokens + est > max_tokens):
            yield batch
            batch = []
            tokens = 0
        batch.append(text)
        tokens += est
    if batch:
        yield batch


def retry_delay(attempt: int, headers, rand: Callable[[], float] = random.random) -> float:
    raw = None
    if headers:
        raw = headers.get("retry-after")
        if raw is None:
            raw = headers.get("Retry-After")
    if raw is not None:
        try:
            return max(0.0, float(raw))
        except (TypeError, ValueError):
            pass
    # attempt is 1-based after a failure
    return min(60.0, 2 ** max(0, attempt - 1)) + rand() * 0.25


def _is_token_limit(exc: Exception) -> bool:
    message = str(exc).lower()
    return "token" in message or "context length" in message


def get_voyage_client():
    global _voyage_client
    if _voyage_client is None:
        key = os.environ.get("VOYAGE_API_KEY")
        if not key:
            raise RuntimeError("VOYAGE_API_KEY is not set")
        _voyage_client = voyageai.Client(api_key=key)
    return _voyage_client


def embed_texts(
    texts: list[str],
    input_type: str = "document",
    client=None,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
    rand: Callable[[], float] = random.random,
    max_attempts: int = 5,
) -> list[list[float]]:
    """Embed texts in token-aware batches, backing off on 429s and 5xxs."""
    if not texts:
        return []
    vo = client or get_voyage_client()
    model = os.environ.get("VOYAGE_EMBED_MODEL", "voyage-large-2")
    batch_size = int(os.environ.get("VOYAGE_EMBED_BATCH_SIZE", str(DEFAULT_EMBED_BATCH_SIZE)))
    max_tokens = int(os.environ.get("VOYAGE_MAX_TOKENS_PER_REQUEST", str(DEFAULT_EMBED_MAX_TOKENS)))
    min_interval = float(os.environ.get("VOYAGE_MIN_INTERVAL_SECONDS", str(DEFAULT_EMBED_MIN_INTERVAL)))
    embeddings: list[list[float]] = []
    next_allowed = 0.0

    def pace():
        nonlocal next_allowed
        if min_interval <= 0:
            return
        delay = next_allowed - clock()
        if delay > 0:
            sleep(delay)
        next_allowed = clock() + min_interval

    def embed_batch(batch: list[str]) -> list[list[float]]:
        attempt = 0
        while True:
            pace()
            try:
                result = vo.embed(batch, model=model, input_type=input_type)
                vectors = list(result.embeddings)
                if len(vectors) != len(batch):
                    raise RuntimeError(
                        f"Voyage returned {len(vectors)} embeddings for {len(batch)} inputs"
                    )
                return vectors
            except voyageai.error.InvalidRequestError as exc:
                if _is_token_limit(exc) and len(batch) > 1:
                    mid = len(batch) // 2
                    log.warning(
                        "Voyage rejected a batch of %s texts (%s); splitting",
                        len(batch),
                        exc,
                    )
                    return embed_batch(batch[:mid]) + embed_batch(batch[mid:])
                raise
            except TRANSIENT_VOYAGE as exc:
                attempt += 1
                if attempt >= max_attempts:
                    raise
                headers = getattr(exc, "headers", None) or {}
                delay = retry_delay(attempt, headers, rand=rand)
                log.warning(
                    "Voyage embed failed (%s); retry %s/%s in %.1fs",
                    exc,
                    attempt,
                    max_attempts - 1,
                    delay,
                )
                sleep(delay)

    for batch in iter_embed_batches(texts, batch_size, max_tokens):
        embeddings.extend(embed_batch(batch))
    return embeddings


def _snippet(resp: requests.Response) -> str:
    try:
        text = resp.text or ""
    except Exception:
        return ""
    return " ".join(text.split())[:300]


class Base44Client:
    """Cursor-paged Base44 client with auth fallback, pacing, and retries."""

    AUTH_MODES = ("bearer", "api_key", "x-api-key")

    def __init__(
        self,
        api_key: str,
        app_id: str,
        base_url: str = DEFAULT_BASE44_API_BASE,
        session: requests.Session | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        rand: Callable[[], float] = random.random,
        rpm: float = DEFAULT_BASE44_RPM,
        page_size: int = 100,
        max_attempts: int = 6,
        timeout: float = 60,
    ):
        if not api_key:
            raise RuntimeError("BASE44_API_KEY is not set")
        if not app_id:
            raise RuntimeError("BASE44_APP_ID is not set")
        self.api_key = api_key
        self.app_id = app_id
        self.base_url = base_url.rstrip("/")
        self.session = session or requests.Session()
        self.sleep = sleep
        self.clock = clock
        self.rand = rand
        self.rpm = rpm
        self.page_size = max(1, min(int(page_size), 500))
        self.max_attempts = max_attempts
        self.timeout = timeout
        self._auth_index = 0
        self._next_allowed = 0.0

    @classmethod
    def from_env(cls) -> "Base44Client":
        api_key = os.environ.get("BASE44_API_KEY", "")
        app_id = os.environ.get("BASE44_APP_ID", DEFAULT_APP_ID)
        base_url = os.environ.get("BASE44_API_BASE", DEFAULT_BASE44_API_BASE)
        rpm = float(os.environ.get("BASE44_RPM", str(DEFAULT_BASE44_RPM)))
        page_size = int(os.environ.get("BASE44_PAGE_SIZE", "100"))
        return cls(api_key=api_key, app_id=app_id, base_url=base_url, rpm=rpm, page_size=page_size)

    def _headers(self) -> dict[str, str]:
        mode = self.AUTH_MODES[self._auth_index]
        headers = {"Accept": "application/json", "User-Agent": USER_AGENT}
        if mode == "bearer":
            headers["Authorization"] = f"Bearer {self.api_key}"
        elif mode == "api_key":
            headers["api_key"] = self.api_key
        else:
            headers["x-api-key"] = self.api_key
        return headers

    def _pace(self) -> None:
        if self.rpm <= 0:
            return
        interval = 60.0 / self.rpm
        delay = self._next_allowed - self.clock()
        if delay > 0:
            self.sleep(delay)
        self._next_allowed = self.clock() + interval

    def request(self, method: str, url: str, params: dict | None = None, timeout: float | None = None):
        timeout = self.timeout if timeout is None else timeout
        last_error: Exception | None = None
        attempt = 0
        while attempt < self.max_attempts:
            self._pace()
            try:
                resp = self.session.request(
                    method,
                    url,
                    headers=self._headers(),
                    params=params,
                    timeout=timeout,
                )
            except RETRYABLE_REQUESTS as exc:
                attempt += 1
                last_error = exc
                if attempt >= self.max_attempts:
                    break
                delay = retry_delay(attempt, None, rand=self.rand)
                log.warning(
                    "Base44 %s %s failed (%s); retrying in %.1fs",
                    method,
                    url,
                    type(exc).__name__,
                    delay,
                )
                self.sleep(delay)
                continue

            if resp.status_code == 401 and self._auth_index < len(self.AUTH_MODES) - 1:
                previous = self.AUTH_MODES[self._auth_index]
                self._auth_index += 1
                log.warning(
                    "Base44 returned 401 for %s auth; trying %s",
                    previous,
                    self.AUTH_MODES[self._auth_index],
                )
                continue

            if resp.status_code == 401:
                raise Base44Error(
                    "Base44 rejected BASE44_API_KEY (401). Set it to a personal access "
                    "token from Workspace Settings → Secrets → Personal access tokens, "
                    "with read access to this app. Account API keys are being shut off, "
                    "and https://api.base44.com no longer serves the entity API.",
                    status_code=401,
                )

            if resp.status_code == 429 or resp.status_code >= 500:
                attempt += 1
                last_error = Base44Error(
                    f"{resp.status_code} from Base44: {_snippet(resp)}",
                    status_code=resp.status_code,
                )
                if attempt >= self.max_attempts:
                    break
                delay = retry_delay(attempt, resp.headers, rand=self.rand)
                log.warning("Base44 returned %s; retrying in %.1fs", resp.status_code, delay)
                self.sleep(delay)
                continue

            if resp.status_code >= 400:
                raise Base44Error(
                    f"Base44 returned {resp.status_code}: {_snippet(resp)}",
                    status_code=resp.status_code,
                )
            return resp

        raise Base44Error(
            f"Base44 request failed after {self.max_attempts} attempts: {last_error}",
            status_code=getattr(last_error, "status_code", None),
        )

    def _list_url(self) -> str:
        app_id = quote(self.app_id, safe="")
        return f"{self.base_url}/api/apps/{app_id}/entities/{ENTITY_NAME}/v2/list"

    def _record_url(self, entity_id: str) -> str:
        app_id = quote(self.app_id, safe="")
        entity_id = quote(entity_id, safe="")
        return f"{self.base_url}/api/apps/{app_id}/entities/{ENTITY_NAME}/{entity_id}"

    def iter_records(self):
        """
        Yield every KnowledgeDocument metadata row.

        Content is not requested here. Bodies of 100k characters are fetched
        one document at a time, and only when the document is new or changed.
        A full page without `has_more` is an error: treating it as the catalog
        would delete the rest of the index.
        """
        url = self._list_url()
        cursor = None
        seen_cursors: set[str] = set()
        field_mode = 0
        limit = self.page_size
        pages = 0
        while True:
            pages += 1
            if pages > 10_000:
                raise Base44Error("Base44 pagination exceeded 10000 pages")
            params: dict = {"limit": limit}
            timeout = self.timeout
            if cursor:
                params["cursor"] = cursor
            else:
                fields = LIST_FIELD_ATTEMPTS[field_mode]
                if fields:
                    params["sort"] = "created_date"
                    params["fields"] = fields
                else:
                    # Full records. Keep the page small so a 100k-character body
                    # cannot blow the response or the timeout.
                    limit = min(limit, 5)
                    params["limit"] = limit
                    timeout = max(timeout, 180)
            if "skip" in params:
                raise Base44Error("refusing to send skip; the list API returns 400")
            try:
                resp = self.request("GET", url, params=params, timeout=timeout)
            except Base44Error as exc:
                if cursor is None and exc.status_code == 400 and field_mode < len(LIST_FIELD_ATTEMPTS) - 1:
                    field_mode += 1
                    log.warning(
                        "Base44 rejected list parameters (%s); retrying with field mode %s",
                        exc,
                        field_mode,
                    )
                    continue
                raise
            try:
                payload = resp.json()
            except ValueError as exc:
                raise Base44Error(f"Base44 list returned non-JSON: {_snippet(resp)}") from exc
            docs, next_cursor, has_more = parse_page(payload, limit)
            log.info(
                "Base44 page %s: %s document(s), has_more=%s",
                pages,
                len(docs),
                has_more,
            )
            for doc in docs:
                yield doc
            if not has_more:
                return
            if not next_cursor:
                raise Base44Error("Base44 reported more records but did not return a cursor")
            if next_cursor in seen_cursors:
                raise Base44Error("Base44 pagination cursor repeated; refusing to loop")
            seen_cursors.add(next_cursor)
            cursor = next_cursor

    def get_document(self, entity_id: str) -> dict:
        resp = self.request("GET", self._record_url(entity_id), timeout=max(self.timeout, 120))
        try:
            payload = resp.json()
        except ValueError as exc:
            raise Base44Error(f"Base44 record {entity_id} returned non-JSON") from exc
        if isinstance(payload, dict):
            if "content" not in payload:
                for key in ("item", "document", "data", "record"):
                    nested = payload.get(key)
                    if isinstance(nested, dict):
                        return nested
            return payload
        raise Base44Error(f"Base44 record {entity_id} was not an object")


def parse_page(payload, page_size: int) -> tuple[list[dict], str | None, bool]:
    if isinstance(payload, list):
        raise Base44Error(
            "Base44 returned a bare list instead of a cursor page. "
            "Refusing to treat a partial list as the full catalog."
        )
    if not isinstance(payload, dict):
        raise Base44Error(f"Unexpected Base44 response type: {type(payload).__name__}")
    items = None
    for key in ("items", "documents", "data", "results", "records"):
        candidate = payload.get(key)
        if isinstance(candidate, list):
            items = candidate
            break
    if items is None:
        raise Base44Error("Base44 page did not include a list of records")
    docs = [item for item in items if isinstance(item, dict)]
    next_cursor = payload.get("next_cursor")
    if next_cursor is not None:
        next_cursor = str(next_cursor)
    if "has_more" not in payload:
        if len(docs) >= page_size:
            raise Base44Error(
                "Base44 page omitted has_more on a full page. "
                "Refusing to treat it as the end of the catalog."
            )
        return docs, next_cursor, False
    return docs, next_cursor, bool(payload.get("has_more"))


def is_reindex_due(
    last_finished_at: datetime | None,
    last_status: str | None,
    now: datetime,
    max_age: timedelta = timedelta(days=8),
) -> bool:
    """True when startup should catch up a missed or never-finished reindex."""
    if last_finished_at is None or last_status not in SUCCESS_RUN_STATUSES:
        return True
    finished = last_finished_at
    if finished.tzinfo is None:
        finished = finished.replace(tzinfo=timezone.utc)
    current = now if now.tzinfo else now.replace(tzinfo=timezone.utc)
    return current - finished > max_age


def default_connect():
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL is not set")
    return psycopg2.connect(url)


def safe_rollback(conn) -> None:
    try:
        conn.rollback()
    except Exception:
        log.warning("Database rollback failed", exc_info=True)


def ensure_embed_version_column(conn) -> bool:
    """Additive. Existing rows stay NULL until they are re-embedded."""
    try:
        cur = conn.cursor()
        cur.execute(
            "ALTER TABLE indexed_documents ADD COLUMN IF NOT EXISTS embed_version INTEGER"
        )
        conn.commit()
        return True
    except Exception:
        safe_rollback(conn)
        log.warning(
            "Could not add indexed_documents.embed_version; title embeddings will not refresh on their own",
            exc_info=True,
        )
        return False


def _ensure_search_trigger(conn) -> None:
    expr_title = "NEW.title"
    expr_topic = "NEW.topic"
    expr_body = "NEW.chunk_text"
    cur = conn.cursor()
    cur.execute(
        f"""
        CREATE OR REPLACE FUNCTION knowledge_chunks_search_tsv_update() RETURNS trigger
        LANGUAGE plpgsql AS $fn$
        BEGIN
            NEW.search_tsv := {search_vector_expr(expr_title, expr_topic, expr_body)};
            RETURN NEW;
        END
        $fn$
        """
    )
    cur.execute(
        "DROP TRIGGER IF EXISTS knowledge_chunks_search_tsv_update ON knowledge_chunks"
    )
    for keyword in ("FUNCTION", "PROCEDURE"):
        cur.execute("SAVEPOINT search_trigger")
        try:
            cur.execute(
                f"""
                CREATE TRIGGER knowledge_chunks_search_tsv_update
                BEFORE INSERT OR UPDATE OF title, topic, chunk_text
                ON knowledge_chunks
                FOR EACH ROW
                EXECUTE {keyword} knowledge_chunks_search_tsv_update()
                """
            )
        except Exception:
            cur.execute("ROLLBACK TO SAVEPOINT search_trigger")
            cur.execute("RELEASE SAVEPOINT search_trigger")
            continue
        cur.execute("RELEASE SAVEPOINT search_trigger")
        conn.commit()
        return
    log.warning(
        "Could not create the search_tsv trigger; inserts and updates still set the column"
    )
    conn.commit()


def ensure_search_schema(conn) -> bool:
    """
    Add full-text search without dropping or rewriting embeddings.

    Adds knowledge_chunks.search_tsv, a GIN index, a trigger, and a backfill
    of rows whose vector is still NULL. Safe to run on every boot.
    """
    try:
        cur = conn.cursor()
        cur.execute(
            "ALTER TABLE knowledge_chunks ADD COLUMN IF NOT EXISTS search_tsv tsvector"
        )
        cur.execute(
            f"""
            UPDATE knowledge_chunks
            SET search_tsv = {search_vector_expr("title", "topic", "chunk_text")}
            WHERE search_tsv IS NULL
            """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS knowledge_chunks_search_tsv_idx
            ON knowledge_chunks USING GIN (search_tsv)
            """
        )
        conn.commit()
    except Exception:
        safe_rollback(conn)
        log.warning(
            "Full-text column is not available; retrieval will use vectors only",
            exc_info=True,
        )
        return False
    try:
        cur = conn.cursor()
        cur.execute("ANALYZE knowledge_chunks")
        conn.commit()
    except Exception:
        safe_rollback(conn)
        log.info("ANALYZE knowledge_chunks skipped", exc_info=True)
    try:
        _ensure_search_trigger(conn)
    except Exception:
        safe_rollback(conn)
        log.warning("Search trigger was not created", exc_info=True)
    return True


def ensure_updated_date_column(conn) -> bool:
    try:
        cur = conn.cursor()
        cur.execute(
            "ALTER TABLE indexed_documents ADD COLUMN IF NOT EXISTS updated_date TEXT"
        )
        conn.commit()
        return True
    except Exception:
        safe_rollback(conn)
        log.warning(
            "Could not add indexed_documents.updated_date; change detection will use content hashes only",
            exc_info=True,
        )
        return False


def ensure_state_table(conn) -> bool:
    try:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS indexer_state (
                id TEXT PRIMARY KEY,
                last_started_at TIMESTAMPTZ,
                last_finished_at TIMESTAMPTZ,
                last_status TEXT,
                last_error TEXT
            )
            """
        )
        conn.commit()
        return True
    except Exception:
        safe_rollback(conn)
        log.warning("Could not ensure indexer_state table", exc_info=True)
        return False


def mark_run_started(conn) -> None:
    try:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO indexer_state (id, last_started_at, last_status, last_error)
            VALUES ('reindex', NOW(), 'running', NULL)
            ON CONFLICT (id) DO UPDATE
            SET last_started_at = NOW(),
                last_status = 'running',
                last_error = NULL
            """
        )
        conn.commit()
    except Exception:
        safe_rollback(conn)
        log.warning("Could not record reindex start", exc_info=True)


def mark_run_finished(conn, status: str, error: str | None) -> None:
    safe_rollback(conn)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            UPDATE indexer_state
            SET last_finished_at = NOW(),
                last_status = %s,
                last_error = %s
            WHERE id = 'reindex'
            """,
            (status, (error or None)),
        )
        conn.commit()
    except Exception:
        safe_rollback(conn)
        log.warning("Could not record reindex result", exc_info=True)


_STATE_SELECT = """
    SELECT last_finished_at, last_status
    FROM indexer_state
    WHERE id = %s
"""


def read_reindex_state(connect: Callable = default_connect) -> tuple[datetime | None, str | None]:
    """Return the last finished run. Creates the state table only if it is missing."""
    conn = connect()
    try:
        cur = conn.cursor()
        try:
            cur.execute(_STATE_SELECT, ("reindex",))
        except psycopg2.Error as exc:
            safe_rollback(conn)
            # 42P01 = undefined_table. Connection and permission errors must
            # propagate so a database outage is not treated as "never indexed".
            if getattr(exc, "pgcode", None) != "42P01":
                raise
            if not ensure_state_table(conn):
                return None, None
            cur = conn.cursor()
            cur.execute(_STATE_SELECT, ("reindex",))
        row = cur.fetchone()
        if not row:
            return None, None
        return row[0], row[1]
    finally:
        conn.close()


def load_local_index(conn, has_updated: bool, has_embed_version: bool) -> dict[str, dict]:
    columns = ["document_id", "content_hash"]
    if has_updated:
        columns.append("updated_date")
    if has_embed_version:
        columns.append("embed_version")
    # Title is part of the embedded text, so a rename has to be visible here.
    columns.append("title")
    cur = conn.cursor()
    cur.execute(f"SELECT {', '.join(columns)} FROM indexed_documents")
    local: dict[str, dict] = {}
    for row in cur.fetchall():
        record = {
            "content_hash": row[1],
            "updated_date": None,
            "embed_version": None,
            "title": None,
        }
        index = 2
        if has_updated:
            record["updated_date"] = normalize_timestamp(row[index])
            index += 1
        if has_embed_version:
            version = row[index]
            if version is not None:
                try:
                    version = int(version)
                except (TypeError, ValueError):
                    version = None
            record["embed_version"] = version
            index += 1
        record["title"] = row[index]
        local[str(row[0])] = record
    return local


def _note_error(stats: dict, doc_id, title, exc) -> None:
    message = str(exc).replace("\n", " ")[:500]
    if isinstance(exc, BaseException):
        log.error("Failed to index %s (%s): %s", title, doc_id, message, exc_info=exc)
    else:
        log.error("Failed to index %s (%s): %s", title, doc_id, message)
    if len(stats["error_details"]) < MAX_ERROR_DETAILS:
        stats["error_details"].append(
            {
                "document_id": doc_id,
                "title": (str(title)[:120] if title else None),
                "error": message,
            }
        )


def document_has_chunks(conn, doc_id: str) -> bool:
    cur = conn.cursor()
    cur.execute(
        "SELECT 1 FROM knowledge_chunks WHERE document_id = %s LIMIT 1",
        (doc_id,),
    )
    return cur.fetchone() is not None


def chunks_exist(conn, doc_id: str) -> bool:
    """Read-only check that does not leave a transaction open across HTTP calls."""
    try:
        return document_has_chunks(conn, doc_id)
    finally:
        safe_rollback(conn)


def delete_document(conn, doc_id: str) -> None:
    cur = conn.cursor()
    cur.execute("DELETE FROM knowledge_chunks WHERE document_id = %s", (doc_id,))
    cur.execute("DELETE FROM indexed_documents WHERE document_id = %s", (doc_id,))
    conn.commit()


def update_unchanged_document(
    conn,
    doc_id: str,
    merged: dict,
    remote_updated: str | None,
    chash: str,
    has_updated: bool,
    has_search: bool,
) -> None:
    """Refresh metadata on chunks whose body did not change, without re-embedding."""
    title = as_text(merged.get("title")) or "Untitled"
    topic = as_text(merged.get("topic"))
    assignments = """
            title = %s,
            category = %s,
            topic = %s,
            subtopic = %s,
            source = %s,
            source_url = %s,
            source_date = %s,
            is_authoritative = %s
    """
    params: list = [
        title,
        as_text(merged.get("category")),
        topic,
        as_text(merged.get("subtopic")),
        as_text(merged.get("source")),
        as_text(merged.get("source_url")),
        as_text(merged.get("source_date")),
        coerce_bool(merged.get("is_authoritative"), default=False),
    ]
    if has_search:
        assignments += f""",
            search_tsv = {search_vector_expr("%s", "%s", "chunk_text")}
        """
        params.extend([title, topic])
    params.append(doc_id)
    cur = conn.cursor()
    cur.execute(
        f"""
        UPDATE knowledge_chunks
        SET {assignments}
        WHERE document_id = %s
        """,
        params,
    )
    if has_updated:
        cur.execute(
            """
            UPDATE indexed_documents
            SET title = %s, updated_date = %s
            WHERE document_id = %s AND content_hash = %s
            """,
            (as_text(merged.get("title")) or "Untitled", remote_updated, doc_id, chash),
        )
    else:
        cur.execute(
            """
            UPDATE indexed_documents
            SET title = %s
            WHERE document_id = %s AND content_hash = %s
            """,
            (as_text(merged.get("title")) or "Untitled", doc_id, chash),
        )
    conn.commit()


def replace_document(
    conn,
    doc_id: str,
    merged: dict,
    chunks: list[str],
    embeddings: list[list[float]],
    chash: str,
    remote_updated: str | None,
    has_updated: bool,
    has_embed_version: bool,
    has_search: bool,
) -> None:
    title = as_text(merged.get("title")) or "Untitled"
    topic = as_text(merged.get("topic"))
    cur = conn.cursor()
    cur.execute("DELETE FROM knowledge_chunks WHERE document_id = %s", (doc_id,))
    cur.execute("DELETE FROM indexed_documents WHERE document_id = %s", (doc_id,))
    search_column = ""
    search_value = ""
    for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        params: list = [
            doc_id,
            title,
            as_text(merged.get("category")),
            topic,
            as_text(merged.get("subtopic")),
            as_text(merged.get("source")),
            as_text(merged.get("source_url")),
            as_text(merged.get("source_date")),
            coerce_bool(merged.get("is_authoritative"), default=False),
            idx,
            chunk,
            json.dumps(embedding),
        ]
        if has_search:
            search_column = ", search_tsv"
            search_value = ", " + search_vector_expr("%s", "%s", "%s")
            params.extend([title, topic, chunk])
        cur.execute(
            f"""
            INSERT INTO knowledge_chunks
                (document_id, title, category, topic, subtopic,
                 source, source_url, source_date, is_authoritative,
                 chunk_index, chunk_text, embedding{search_column})
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::vector{search_value})
            """,
            params,
        )
    columns = ["document_id", "title", "chunk_count", "content_hash"]
    values: list = [doc_id, title, len(chunks), chash]
    if has_updated:
        columns.append("updated_date")
        values.append(remote_updated)
    if has_embed_version:
        columns.append("embed_version")
        values.append(EMBED_TEXT_VERSION)
    placeholders = ", ".join(["%s"] * len(values))
    cur.execute(
        f"INSERT INTO indexed_documents ({', '.join(columns)}) VALUES ({placeholders})",
        values,
    )
    conn.commit()


def delete_absent(conn, remote_ids: list[str]) -> tuple[int, int]:
    cur = conn.cursor()
    cur.execute(
        "DELETE FROM knowledge_chunks WHERE NOT (document_id = ANY(%s))",
        (remote_ids,),
    )
    chunks_deleted = cur.rowcount
    cur.execute(
        "DELETE FROM indexed_documents WHERE NOT (document_id = ANY(%s))",
        (remote_ids,),
    )
    docs_deleted = cur.rowcount
    conn.commit()
    return docs_deleted, chunks_deleted


def _embed_version_current(local_row: dict | None, has_embed_version: bool) -> bool:
    if not has_embed_version:
        # The column could not be added, so there is nowhere to record progress.
        return True
    if not local_row:
        return False
    return local_row.get("embed_version") == EMBED_TEXT_VERSION


def sync_one(
    conn,
    record: dict,
    storage_id: str,
    local_row: dict | None,
    force: bool,
    client: Base44Client,
    embed,
    has_updated: bool,
    has_embed_version: bool,
    has_search: bool,
    stats: dict,
) -> str:
    remote_updated = normalize_timestamp(record.get("updated_date"))
    version_ok = _embed_version_current(local_row, has_embed_version)
    if (
        not force
        and has_updated
        and local_row
        and local_row.get("content_hash")
        and remote_updated
        and local_row.get("updated_date") == remote_updated
        and version_ok
        and chunks_exist(conn, storage_id)
    ):
        return "skipped"

    entity_id = base44_entity_id(record)
    if not entity_id:
        raise Base44Error(f"Document {storage_id} has no Base44 id")
    full = client.get_document(entity_id)
    merged = {**record, **full}
    remote_updated = normalize_timestamp(merged.get("updated_date")) or remote_updated
    content = document_content(merged)
    title = as_text(merged.get("title")) or "Untitled"
    if not content:
        delete_document(conn, storage_id)
        log.info("Removed empty document '%s' (%s)", title, storage_id)
        return "empty"

    chash = content_hash(content)
    title_changed = local_row is not None and (local_row.get("title") or "") != title
    if (
        not force
        and local_row
        and local_row.get("content_hash") == chash
        and version_ok
        and not title_changed
        and chunks_exist(conn, storage_id)
    ):
        update_unchanged_document(
            conn, storage_id, merged, remote_updated, chash, has_updated, has_search
        )
        return "skipped"

    if local_row and local_row.get("content_hash") == chash and (not version_ok or title_changed):
        stats["title_reembeds"] += 1
        log.info(
            "Re-embedding '%s' (%s) so the vector includes the document title",
            title,
            storage_id,
        )

    chunks = chunk_text(content)
    if not chunks:
        delete_document(conn, storage_id)
        return "empty"
    embeddings = embed(
        [passage_for_model(title, chunk) for chunk in chunks],
        input_type="document",
    )
    if len(embeddings) != len(chunks):
        raise RuntimeError(
            f"embedding count {len(embeddings)} != chunk count {len(chunks)} for {storage_id}"
        )
    replace_document(
        conn,
        storage_id,
        merged,
        chunks,
        embeddings,
        chash,
        remote_updated,
        has_updated,
        has_embed_version,
        has_search,
    )
    stats["chunks_written"] += len(chunks)
    log.info("Indexed '%s' (%s) → %s chunk(s)", title, storage_id, len(chunks))
    return "indexed"


def _empty_stats(force: bool) -> dict:
    return {
        "ok": True,
        "force": force,
        "fetched": 0,
        "indexed": 0,
        "skipped": 0,
        "deleted": 0,
        "chunks_deleted": 0,
        "chunks_written": 0,
        "empty": 0,
        "errors": 0,
        "error_details": [],
        "title_reembeds": 0,
    }


def _acquire_lock(connect) -> tuple[object | None, object | None, bool]:
    lock_conn = connect()
    lock_conn.autocommit = True
    cur = lock_conn.cursor()
    cur.execute("SELECT pg_try_advisory_lock(%s)", (ADVISORY_LOCK_KEY,))
    locked = bool(cur.fetchone()[0])
    if not locked:
        lock_conn.close()
        return None, None, False
    return lock_conn, cur, True


def _release_lock(lock_conn, lock_cur) -> None:
    try:
        lock_cur.execute("SELECT pg_advisory_unlock(%s)", (ADVISORY_LOCK_KEY,))
    except Exception:
        log.warning("Could not release the indexer advisory lock", exc_info=True)
    finally:
        try:
            lock_conn.close()
        except Exception:
            pass


def index_documents(
    force_reindex: bool = False,
    *,
    client: Base44Client | None = None,
    connect: Callable | None = None,
    embed: Callable | None = None,
    use_advisory_lock: bool = True,
) -> dict:
    """
    Reconcile the local index with Base44.

    Returns a stats dict. Raises only when the Base44 catalog walk itself
    fails. Per-document embed/database errors are counted and the rest of the
    catalog continues. Documents are deleted locally only after a complete walk
    that returned at least one id. An empty catalog does not wipe the index.
    """
    client = client or Base44Client.from_env()
    connect = connect or default_connect
    embed = embed or embed_texts
    stats = _empty_stats(force_reindex)

    lock_conn = None
    lock_cur = None
    if use_advisory_lock:
        lock_conn, lock_cur, locked = _acquire_lock(connect)
        if not locked:
            stats["ok"] = False
            stats["reason"] = "already_running"
            log.info("Reindex skipped; another session holds the advisory lock")
            return stats

    conn = None
    remote_ids: list[str] = []
    try:
        conn = connect()
        has_updated = ensure_updated_date_column(conn)
        has_embed_version = ensure_embed_version_column(conn)
        has_search = ensure_search_schema(conn)
        if ensure_state_table(conn):
            mark_run_started(conn)
        try:
            local = load_local_index(conn, has_updated, has_embed_version)
            safe_rollback(conn)
            local_ids = set(local)
            seen_remote: set[str] = set()
            for record in client.iter_records():
                stats["fetched"] += 1
                storage_id = choose_document_id(record, local_ids)
                title = record.get("title")
                if not storage_id:
                    stats["errors"] += 1
                    _note_error(stats, None, title, "missing document id")
                    continue
                if storage_id in seen_remote:
                    continue
                seen_remote.add(storage_id)
                remote_ids.append(storage_id)
                try:
                    outcome = sync_one(
                        conn,
                        record,
                        storage_id,
                        local.get(storage_id),
                        force_reindex,
                        client,
                        embed,
                        has_updated,
                        has_embed_version,
                        has_search,
                        stats,
                    )
                    stats[outcome] += 1
                except Exception as exc:
                    safe_rollback(conn)
                    stats["errors"] += 1
                    _note_error(stats, storage_id, title, exc)
                if stats["fetched"] % 100 == 0:
                    log.info(
                        "Progress: fetched=%s indexed=%s skipped=%s empty=%s errors=%s",
                        stats["fetched"],
                        stats["indexed"],
                        stats["skipped"],
                        stats["empty"],
                        stats["errors"],
                    )

            if not remote_ids:
                stats["ok"] = False
                stats["reason"] = "empty_catalog"
                if stats["fetched"] == 0:
                    stats["warning"] = (
                        "Base44 returned no KnowledgeDocument records; the existing index was left unchanged"
                    )
                else:
                    stats["warning"] = (
                        "Base44 returned records but none had an id; the existing index was left unchanged"
                    )
                log.error(stats["warning"])
                mark_run_finished(conn, "refused_empty_catalog", stats["warning"])
                return stats

            deleted, chunks_deleted = delete_absent(conn, remote_ids)
            stats["deleted"] = deleted
            stats["chunks_deleted"] = chunks_deleted
            status = "succeeded" if stats["errors"] == 0 else "completed_with_errors"
            error = None
            if stats["error_details"]:
                error = stats["error_details"][0]["error"]
            mark_run_finished(conn, status, error)
            log.info(
                "Done. Indexed: %s | Skipped: %s | Deleted: %s | Empty: %s | Errors: %s",
                stats["indexed"],
                stats["skipped"],
                stats["deleted"],
                stats["empty"],
                stats["errors"],
            )
            return stats
        except Exception as exc:
            stats["ok"] = False
            stats["error"] = str(exc)[:500]
            log.error("Reindex failed before the catalog walk finished: %s", stats["error"])
            if conn is not None:
                mark_run_finished(conn, "failed", stats["error"])
            raise
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
        if lock_conn is not None and lock_cur is not None:
            _release_lock(lock_conn, lock_cur)


def report_duplicate_chunks(apply: bool = False, connect: Callable | None = None) -> dict:
    """
    Report exact duplicate chunk rows. Delete them only when apply is true.

    Same rules as POST /maintenance/duplicate-chunks. The default is report-only.
    """
    connect = connect or default_connect
    conn = connect()
    try:
        cur = conn.cursor()
        result = search.collect_duplicate_chunks(cur, apply=apply)
        if apply:
            conn.commit()
        else:
            conn.rollback()
        return result
    finally:
        conn.close()


if __name__ == "__main__":
    if "--report-duplicate-chunks" in sys.argv or "--delete-duplicate-chunks" in sys.argv:
        deleting = "--delete-duplicate-chunks" in sys.argv
        try:
            report = report_duplicate_chunks(apply=deleting)
        except Exception:
            log.exception("Duplicate-chunk maintenance failed")
            sys.exit(1)
        print(json.dumps(report, default=str, indent=2))
        sys.exit(0)
    force = "--force" in sys.argv
    if force:
        log.info("Force mode: re-embedding every document.")
    try:
        result = index_documents(force_reindex=force)
    except Exception:
        log.exception("Indexer failed")
        sys.exit(1)
    if not result.get("ok", False):
        sys.exit(1)
    if result.get("errors"):
        sys.exit(2)
