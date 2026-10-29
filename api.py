#!/usr/bin/env python3
"""
RC Brain — Retrieval API
=========================
FastAPI service that answers "what are the most relevant passages for this question?"

Base44 calls POST /retrieve with the user's question.
The API returns the top passages from the knowledge store, ready to pass to Claude.

Search is hybrid. A full-text leg (Postgres `simple` tsvector) and a vector
leg are merged with reciprocal rank fusion. Identical and near-identical
passages are collapsed, including duplicate Drive copies, then Voyage reranks
what remains. A per-document cap runs after that collapse. Titles containing
"RC Brain Agent Evaluation Q&A" are always left out.

The vector leg uses a cosine HNSW index once that index is valid. Until the
background build finishes, and whenever a request sets exact_search, the leg
is the original exact scan. Query embeddings are cached in memory. Database
connections come from a pool.

Endpoints:
    GET  /health           — liveness check; returns chunk count
    POST /retrieve         — hybrid search + rerank; returns top passages
    POST /reindex          — start an incremental reindex in the background
    GET  /reindex/status   — progress of the latest reindex
    GET  /maintenance/duplicate-chunks
                           — report exact duplicate chunk rows (no deletes)
    POST /maintenance/duplicate-chunks
                           — same report; {"apply": true} deletes those rows

Environment variables (set in Railway):
    DATABASE_URL        - PostgreSQL connection string (from Railway)
    VOYAGE_API_KEY      - Voyage AI API key
    VOYAGE_EMBED_MODEL  - Model used at index time (default: voyage-large-2)
                          MUST match the model used by the indexer.
    VOYAGE_RERANK_MODEL - Reranker model (default: rerank-2)
    TOP_K               - Candidates kept after fusion, before reranking (default: 40)
    RERANK_TOP_N        - Final passages when the request omits top_k (default: 8)
    MAX_CHUNKS_PER_DOCUMENT
                        - Final cap per document (default: 2). 0 disables it.
                          Drive copies that share a base title count as one document.
    CANDIDATE_CHUNKS_PER_DOCUMENT
                        - Cap per document in the rerank candidate set (default: 4)
    EXPAND_NEIGHBORS    - "false" disables adjacent-chunk expansion (default: on)
    VECTOR_EXACT_SEARCH - "true" forces the exact vector scan on every request
                          (rollback). Default is the HNSW path once that index
                          is valid; until then retrieval stays on the exact scan.
    ANN_CANDIDATES      - Nearest chunks read before the per-document window
                          (default: 1000). hnsw.ef_search is at least this.
    ANN_INDEX_ON_START  - "false" skips the background HNSW build
    DB_POOL_MAX         - Pooled Postgres connections (default: 8)
    DB_POOL_MIN         - Connections opened when the pool is created (default: 1)
    QUERY_EMBED_CACHE_SIZE
                        - Cached query embeddings (default: 2048)
    QUERY_EMBED_CACHE_TTL_SECONDS
                        - How long a cached query embedding is kept (default: 3600).
                          0 disables the cache.
    HNSW_MAINTENANCE_WORK_MEM
                        - Optional maintenance_work_mem for the index build
                          session only (for example 512MB). Unset keeps the
                          server default. Ignored unless it is a plain size.
    RETRIEVAL_EXCLUDE_TITLES
                        - Extra title substrings to drop, separated by "||"
    PORT                - Port to listen on (Railway sets this automatically)
    RETRIEVAL_API_KEY   - Secret key callers must send as X-API-Key header
    BASE44_API_KEY      - Base44 personal access token used by the indexer
    BASE44_APP_ID       - Base44 app id (indexer default is the RC Brain app)
    REINDEX_SCHEDULER   - "false" disables the Monday 19:00 America/Chicago job
    REINDEX_ON_STARTUP  - "false" disables the catch-up reindex when the last
                          successful run is missing or older than 8 days
    RETRIEVAL_SCHEMA_ON_START
                        - "false" skips the additive full-text migration at boot
"""

import os
import json
import logging
import threading
import time
import psycopg2
import psycopg2.pool
import voyageai
from collections import OrderedDict
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime, timezone
from typing import Any, Optional
from zoneinfo import ZoneInfo

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from fastapi import Body, FastAPI, HTTPException, Security, Depends
from fastapi.responses import JSONResponse
from fastapi.security.api_key import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
from dotenv import load_dotenv

import search

load_dotenv()

# ── Logging ────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

from indexer import (  # noqa: E402
    default_connect,
    ensure_search_schema,
    index_documents,
    is_reindex_due,
    passage_for_model,
    read_reindex_state,
)

# ── Config ───────────────────────────────────────────────
VOYAGE_API_KEY      = os.environ["VOYAGE_API_KEY"]
DATABASE_URL        = os.environ["DATABASE_URL"]
VOYAGE_EMBED_MODEL  = os.environ.get("VOYAGE_EMBED_MODEL",  "voyage-large-2")
VOYAGE_RERANK_MODEL = os.environ.get("VOYAGE_RERANK_MODEL", "rerank-2")
TOP_K               = int(os.environ.get("TOP_K",         "40"))
RERANK_TOP_N        = int(os.environ.get("RERANK_TOP_N",  "8"))
MAX_CHUNKS_PER_DOCUMENT = int(os.environ.get("MAX_CHUNKS_PER_DOCUMENT", "2"))
CANDIDATE_CHUNKS_PER_DOCUMENT = int(os.environ.get("CANDIDATE_CHUNKS_PER_DOCUMENT", "4"))
# A caller can still ask for a long list. These caps keep one request from
# sending hundreds of passages to the reranker.
MAX_FINAL_N = 100
MAX_CANDIDATES = 200
RETRIEVAL_API_KEY   = os.environ.get("RETRIEVAL_API_KEY", "")

_search_ready = False
_search_lock = threading.Lock()
_ann_index_ready = False
_ann_build_started = False
_ann_index_lock = threading.Lock()

vo = voyageai.Client(api_key=VOYAGE_API_KEY)

REINDEX_SCHEDULE_DESCRIPTION = "Mondays at 19:00 America/Chicago"
# A few hours, so a deploy or a busy event loop around 19:00 still runs the job.
REINDEX_MISFIRE_GRACE_SECONDS = 6 * 60 * 60

_state_lock = threading.Lock()
_reindex_state = {
    "state": "idle",
    "force": False,
    "trigger": None,
    "started_at": None,
    "finished_at": None,
    "stats": None,
    "error": None,
}


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        log.warning("Ignoring %s=%r", name, raw)
        return default


def _float_env(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        log.warning("Ignoring %s=%r", name, raw)
        return default


def _scheduler_enabled() -> bool:
    return _env_flag("REINDEX_SCHEDULER", True)


def _startup_reindex_enabled() -> bool:
    return _env_flag("REINDEX_ON_STARTUP", True)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _chicago() -> ZoneInfo:
    return ZoneInfo("America/Chicago")


def weekly_reindex_trigger() -> CronTrigger:
    return CronTrigger(day_of_week="mon", hour=19, minute=0, timezone=_chicago())


def _snapshot() -> dict:
    with _state_lock:
        body = dict(_reindex_state)
        if body.get("stats") is not None:
            body["stats"] = dict(body["stats"])
    body["schedule"] = REINDEX_SCHEDULE_DESCRIPTION
    body["scheduler_enabled"] = _scheduler_enabled()
    return body


def reindex_status() -> dict:
    body = _snapshot()
    if body["state"] == "idle":
        try:
            finished, status = read_reindex_state(default_connect)
        except Exception as exc:
            log.warning("Could not read indexer state: %s", exc)
            finished, status = None, None
        body["last_finished_at"] = finished.isoformat() if finished else None
        body["last_status"] = status
    return body


def start_reindex(force: bool, trigger: str) -> tuple[int, dict]:
    with _state_lock:
        if _reindex_state["state"] == "running":
            body = dict(_reindex_state)
            body["status"] = "already_running"
            body["schedule"] = REINDEX_SCHEDULE_DESCRIPTION
            return 409, body
        _reindex_state.update(
            {
                "state": "running",
                "force": force,
                "trigger": trigger,
                "started_at": _now(),
                "finished_at": None,
                "stats": None,
                "error": None,
            }
        )

    def run() -> None:
        try:
            stats = index_documents(force_reindex=force)
            with _state_lock:
                _reindex_state["stats"] = stats
                _reindex_state["finished_at"] = _now()
                if stats.get("reason") == "already_running":
                    _reindex_state["state"] = "skipped"
                    _reindex_state["error"] = "Another indexer holds the database lock"
                elif not stats.get("ok", False):
                    _reindex_state["state"] = "failed"
                    _reindex_state["error"] = (
                        stats.get("warning") or stats.get("error") or stats.get("reason")
                    )
                elif stats.get("errors"):
                    _reindex_state["state"] = "completed_with_errors"
                else:
                    _reindex_state["state"] = "succeeded"
        except Exception as exc:
            log.exception("Reindex failed")
            with _state_lock:
                _reindex_state["state"] = "failed"
                _reindex_state["error"] = str(exc)[:500]
                _reindex_state["finished_at"] = _now()

    inline = os.environ.get("REINDEX_INLINE", "").strip().lower() in {"1", "true", "yes", "on"}
    if inline:
        run()
    else:
        threading.Thread(target=run, name="reindex", daemon=True).start()
    code = 409 if _snapshot()["state"] == "skipped" and trigger == "manual" else 202
    # A manual call that lost the advisory lock still accepted the request;
    # report the terminal state so the caller can tell it did not start.
    body = _snapshot()
    if body["state"] == "running":
        body["status"] = "started"
        return 202, body
    body["status"] = body["state"]
    return code, body


def ensure_search_ready(conn) -> bool:
    """Add the tsvector column if this process has not done so yet."""
    global _search_ready
    if _search_ready:
        return True
    with _search_lock:
        if _search_ready:
            return True
        ready = ensure_search_schema(conn)
        if ready:
            _search_ready = True
        return ready


def _prepare_search_on_startup() -> None:
    if not _env_flag("RETRIEVAL_SCHEMA_ON_START", True):
        return
    try:
        with get_db() as conn:
            ensure_search_ready(conn)
    except Exception:
        log.exception("Full-text setup failed at startup; /retrieve will retry")


def passage_limits(requested: Optional[int]) -> tuple[int, int, int, int, int]:
    """Return final_n, candidate_n, pool_n, final_cap, candidate_cap."""
    if requested is not None and requested > 0:
        final_n = min(int(requested), MAX_FINAL_N)
    else:
        final_n = max(1, RERANK_TOP_N)
    candidates_n = min(max(final_n * 4, TOP_K, final_n), MAX_CANDIDATES)
    pool_n = min(max(candidates_n * 3, 80), 400)
    final_cap = MAX_CHUNKS_PER_DOCUMENT
    candidate_cap = CANDIDATE_CHUNKS_PER_DOCUMENT
    if final_cap <= 0 or candidate_cap <= 0:
        candidate_cap = 0 if final_cap <= 0 else candidate_cap
    else:
        candidate_cap = max(candidate_cap, final_cap)
    return final_n, candidates_n, pool_n, final_cap, candidate_cap


def _maybe_reindex_on_startup() -> None:
    if not _startup_reindex_enabled():
        return
    if not os.environ.get("BASE44_API_KEY"):
        log.warning(
            "Skipping startup reindex: BASE44_API_KEY is not set on this service"
        )
        return
    try:
        finished, status = read_reindex_state(default_connect)
    except Exception:
        log.exception("Could not read indexer state; skipping startup reindex")
        return
    if is_reindex_due(finished, status, datetime.now(timezone.utc)):
        log.info(
            "Reindex is due (last_status=%s, last_finished_at=%s); starting in the background",
            status,
            finished,
        )
        start_reindex(False, "startup")
    else:
        log.info("Startup reindex not due (last finished %s, status %s)", finished, status)


@asynccontextmanager
async def lifespan(app: FastAPI):
    scheduler = None
    if _scheduler_enabled():
        try:
            scheduler = BackgroundScheduler(timezone=_chicago())
            scheduler.add_job(
                lambda: start_reindex(False, "schedule"),
                weekly_reindex_trigger(),
                id="weekly_reindex",
                replace_existing=True,
                max_instances=1,
                coalesce=True,
                misfire_grace_time=REINDEX_MISFIRE_GRACE_SECONDS,
            )
            scheduler.start()
            log.info("Weekly reindex scheduled for %s", REINDEX_SCHEDULE_DESCRIPTION)
        except Exception:
            log.exception(
                "Weekly scheduler did not start; POST /reindex still works"
            )
            scheduler = None
    _prepare_search_on_startup()
    if _ann_index_on_start():
        start_ann_index_build()
    _maybe_reindex_on_startup()
    try:
        yield
    finally:
        if scheduler is not None:
            scheduler.shutdown(wait=False)
        close_db_pool()


# ── Auth ──────────────────────────────────────────────────────────
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def require_api_key(key: str = Security(api_key_header)):
    if not RETRIEVAL_API_KEY:
        return   # key not configured — open access (dev mode)
    if key != RETRIEVAL_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")

# ── App ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="RC Brain Retrieval API",
    description="Hybrid search over RC Brain's KnowledgeDocument library.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ── Query-embedding cache ─────────────────────────────────────────────
# Absolute TTL from insert. A repeat of the same question skips Voyage.
# Failures are not stored. The key includes the model so a model change
# cannot reuse a vector from the previous one.

class QueryEmbeddingCache:
    def __init__(self, maxsize: int, ttl_seconds: float):
        self.maxsize = int(maxsize)
        self.ttl = float(ttl_seconds)
        self._lock = threading.Lock()
        self._items: OrderedDict = OrderedDict()

    def get(self, key):
        if self.ttl <= 0 or self.maxsize <= 0:
            return None
        now = time.monotonic()
        with self._lock:
            item = self._items.get(key)
            if item is None:
                return None
            expires, value = item
            if expires <= now:
                self._items.pop(key, None)
                return None
            self._items.move_to_end(key)
            return value

    def put(self, key, value) -> None:
        if self.ttl <= 0 or self.maxsize <= 0:
            return
        expires = time.monotonic() + self.ttl
        with self._lock:
            self._items[key] = (expires, value)
            self._items.move_to_end(key)
            while len(self._items) > self.maxsize:
                self._items.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._items.clear()


_embed_cache = QueryEmbeddingCache(
    maxsize=_int_env("QUERY_EMBED_CACHE_SIZE", 2048),
    ttl_seconds=_float_env("QUERY_EMBED_CACHE_TTL_SECONDS", 3600),
)


def embed_query(text: str) -> list[float]:
    """Query embedding, from the in-process cache when the entry is still fresh."""
    key = (VOYAGE_EMBED_MODEL, text)
    cached = _embed_cache.get(key)
    if cached is not None:
        return list(cached)
    result = vo.embed(
        [text],
        model=VOYAGE_EMBED_MODEL,
        input_type="query",
    )
    vector = tuple(float(value) for value in result.embeddings[0])
    _embed_cache.put(key, vector)
    return list(vector)


# ── DB pool ───────────────────────────────────────────────────────────
# One pool per process. SET LOCAL and an open transaction are cleared with
# rollback before the connection is reused. The HNSW build does not use this
# pool: CREATE INDEX CONCURRENTLY holds its connection for the whole build.

_pool_lock = threading.Lock()
_POOL = None


def _pool():
    global _POOL
    with _pool_lock:
        if _POOL is None:
            maxconn = max(1, _int_env("DB_POOL_MAX", 8))
            minconn = _int_env("DB_POOL_MIN", 1)
            minconn = min(max(1, minconn), maxconn)
            _POOL = psycopg2.pool.ThreadedConnectionPool(
                minconn,
                maxconn,
                DATABASE_URL,
            )
        return _POOL


def close_db_pool() -> None:
    global _POOL
    with _pool_lock:
        pool = _POOL
        _POOL = None
    if pool is not None:
        pool.closeall()


def _release_conn(pool, conn) -> None:
    discard = bool(getattr(conn, "closed", 1))
    if not discard:
        try:
            conn.rollback()
        except Exception:
            log.warning(
                "Discarding a database connection that could not be reset",
                exc_info=True,
            )
            discard = True
    try:
        pool.putconn(conn, close=discard)
    except Exception:
        log.warning(
            "Could not return a database connection to the pool",
            exc_info=True,
        )


@contextmanager
def get_db():
    pool = _pool()
    conn = pool.getconn()
    try:
        yield conn
    finally:
        _release_conn(pool, conn)


# ── HNSW index ────────────────────────────────────────────────────────
# Built in the background so a long CREATE INDEX cannot fail the healthcheck
# or hold up startup. Requests use the exact scan until the index is valid.

def ann_index_ready() -> bool:
    with _ann_index_lock:
        return _ann_index_ready


def set_ann_index_ready(ready: bool) -> None:
    global _ann_index_ready
    with _ann_index_lock:
        _ann_index_ready = bool(ready)


def _ann_index_on_start() -> bool:
    return _env_flag("ANN_INDEX_ON_START", True)


def _use_exact_vector_search(requested: Optional[bool]) -> bool:
    """Exact scan when asked, when the env rollback is on, or when HNSW is not ready."""
    if requested is True:
        return True
    if _env_flag("VECTOR_EXACT_SEARCH", False):
        return True
    return not ann_index_ready()


def _connect_for_index():
    return psycopg2.connect(DATABASE_URL, connect_timeout=10)


def _maintenance_work_mem() -> Optional[str]:
    """A plain Postgres memory size, or None. Anything else is ignored."""
    raw = os.environ.get("HNSW_MAINTENANCE_WORK_MEM", "").strip().replace(" ", "")
    if not raw:
        return None
    if len(raw) < 3 or not raw[:-2].isdigit() or int(raw[:-2]) < 1:
        log.warning("Ignoring HNSW_MAINTENANCE_WORK_MEM=%r", raw)
        return None
    canonical = {"kb": "kB", "mb": "MB", "gb": "GB"}.get(raw[-2:].lower())
    if canonical is None:
        log.warning("Ignoring HNSW_MAINTENANCE_WORK_MEM=%r", raw)
        return None
    return raw[:-2] + canonical


def _set_hnsw_maintenance_work_mem(cur) -> None:
    value = _maintenance_work_mem()
    if not value:
        return
    # SET does not take a bound parameter. `value` is digits plus a fixed unit.
    cur.execute(f"SET maintenance_work_mem = '{value}'")


def _cursor_index_ready(cur) -> bool:
    """True when this database has a valid cosine HNSW index on embedding.

    The detailed catalog query confirms the opclass. If that query cannot run
    on this server, the index name plus indisvalid/indisready is enough: the
    build creates that name with vector_cosine_ops and does not drop it.
    """
    try:
        cur.execute(search.HNSW_READY_SQL)
        row = cur.fetchone()
        return bool(row and row[0])
    except Exception:
        log.warning(
            "Detailed HNSW check failed; falling back to the index name",
            exc_info=True,
        )
        cur.execute(search.HNSW_STATE_SQL)
        row = cur.fetchone()
        return bool(row and row[0] and row[1])


def build_hnsw_index(connect=None) -> bool:
    """
    Create the cosine HNSW index if this database does not already have a valid one.

    Uses a dedicated autocommit connection. CREATE INDEX CONCURRENTLY cannot
    run inside a transaction and must not occupy a pooled connection for the
    whole build. An invalid index is left in place; this function does not
    drop anything.
    """
    connect = connect or _connect_for_index
    conn = connect()
    try:
        conn.autocommit = True
        if not conn.autocommit:
            raise RuntimeError(
                "CREATE INDEX CONCURRENTLY requires autocommit and cannot run in a transaction"
            )
        cur = conn.cursor()
        try:
            if _cursor_index_ready(cur):
                log.info("HNSW index %s is already valid", search.HNSW_INDEX_NAME)
                return True
        except Exception:
            log.warning(
                "Could not confirm an existing HNSW index; attempting CREATE INDEX",
                exc_info=True,
            )
        try:
            cur.execute(search.HNSW_STATE_SQL)
            state = cur.fetchone()
        except Exception:
            log.info("Could not inspect %s", search.HNSW_INDEX_NAME, exc_info=True)
            state = None
        if state is not None and not state[0]:
            log.warning(
                "HNSW index %s exists but is not valid. It will not be dropped. "
                "Retrieval stays on the exact scan until that index is valid.",
                search.HNSW_INDEX_NAME,
            )
        _set_hnsw_maintenance_work_mem(cur)
        log.info(
            "Creating HNSW index %s with CREATE INDEX CONCURRENTLY",
            search.HNSW_INDEX_NAME,
        )
        try:
            cur.execute(search.HNSW_INDEX_SQL)
        except Exception:
            # A second process may have created the same index first. Recheck
            # below instead of treating that race as a failed build.
            log.warning(
                "CREATE INDEX for %s did not finish; checking whether it is valid anyway",
                search.HNSW_INDEX_NAME,
                exc_info=True,
            )
        try:
            cur.execute("ANALYZE knowledge_chunks")
        except Exception:
            log.info("ANALYZE knowledge_chunks after HNSW build skipped", exc_info=True)
        try:
            ready = _cursor_index_ready(cur)
        except Exception:
            log.warning(
                "Could not confirm the HNSW index after building it",
                exc_info=True,
            )
            return False
        if not ready:
            log.warning(
                "HNSW index %s is not valid after CREATE INDEX; staying on the exact scan",
                search.HNSW_INDEX_NAME,
            )
        return ready
    finally:
        conn.close()


def _ann_index_thread() -> None:
    started = time.perf_counter()
    ready = False
    try:
        ready = bool(build_hnsw_index())
    except Exception:
        log.exception("HNSW index build failed; retrieval stays on the exact scan")
        ready = False
    set_ann_index_ready(ready)
    log.info(
        "HNSW index background task finished in %.1fs (ready=%s)",
        time.perf_counter() - started,
        ready,
    )


def start_ann_index_build() -> Optional[threading.Thread]:
    """Start the HNSW build and return immediately. A second call does nothing."""
    global _ann_build_started
    with _ann_index_lock:
        if _ann_build_started:
            return None
        _ann_build_started = True
    thread = threading.Thread(target=_ann_index_thread, name="hnsw-index", daemon=True)
    thread.start()
    log.info("HNSW index build started in the background")
    return thread


_TIMING_KEYS = ("embed", "keyword", "vector_search", "rerank", "neighbors")


def _timing_payload(started: float, marks: dict[str, float]) -> dict[str, int]:
    payload = {key: int(round(marks.get(key, 0.0) * 1000.0)) for key in _TIMING_KEYS}
    payload["total"] = int(round((time.perf_counter() - started) * 1000.0))
    return payload


# ── Request / Response models ───────────────────────────────────────────────────

class RetrievalRequest(BaseModel):
    question:   str
    top_k:      Optional[int] = None
    category:   Optional[str] = None
    source:     Optional[str] = None
    # Substrings matched against the document title, in addition to the
    # built-in eval-key exclusion. A single string is accepted too.
    exclude_titles: Optional[list[str]] = None
    # Exact category values to leave out of the result.
    exclude_categories: Optional[list[str]] = None
    # Adjacent chunks of the top documents. None follows EXPAND_NEIGHBORS,
    # which defaults to on, so a numbered list split across chunks stays together.
    expand_neighbors: Optional[bool] = None
    # True forces the exact vector scan for this request, so a deployed ANN
    # path can be compared with the old results. False and null follow the
    # default: HNSW when that index is valid, otherwise the exact scan.
    exact_search: Optional[bool] = None

    @field_validator("exclude_titles", "exclude_categories", mode="before")
    @classmethod
    def _string_or_list(cls, value):
        if value is None or value == "":
            return None
        if isinstance(value, str):
            return [value]
        return value


class Passage(BaseModel):
    document_id:      str
    title:            str
    category:         Optional[str]
    topic:            Optional[str]
    source:           Optional[str]
    source_url:       Optional[str]
    source_date:      Optional[str]
    is_authoritative: bool
    chunk_text:       str
    relevance_score:  float


class RetrievalResponse(BaseModel):
    question: str
    passages: list[Passage]
    # Milliseconds for embed, keyword, vector_search, rerank, neighbors, and total.
    timings_ms: dict[str, int]


# ── Endpoints ───────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """Liveness check. Returns the total number of indexed chunks.

    vector_index is "ready" once the background HNSW build has been confirmed
    valid, and "pending" until then. The check does not wait on that build.
    """
    try:
        with get_db() as conn:
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM knowledge_chunks")
            chunk_count = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM indexed_documents")
            doc_count = cur.fetchone()[0]
        return {
            "status":       "ok",
            "chunk_count":  chunk_count,
            "doc_count":    doc_count,
            "embed_model":  VOYAGE_EMBED_MODEL,
            "vector_index": "ready" if ann_index_ready() else "pending",
        }
    except Exception as exc:
        log.error(f"Health check failed: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


def _expand_enabled(requested: Optional[bool]) -> bool:
    if requested is not None:
        return bool(requested)
    return _env_flag("EXPAND_NEIGHBORS", True)


def _apply_flag(payload: Optional[dict]) -> bool:
    """True only for an explicit apply. Anything else stays report-only."""
    if not isinstance(payload, dict) or "apply" not in payload:
        return False
    value = payload.get("apply")
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return value is True


def _duplicate_chunk_body(apply: bool) -> dict:
    with get_db() as conn:
        cur = conn.cursor()
        body = search.collect_duplicate_chunks(cur, apply=apply)
        if apply:
            conn.commit()
        else:
            conn.rollback()
        return body


def _keep_hit(hit: search.ChunkHit, titles: list[str], categories: list[str]) -> bool:
    if search.title_is_excluded(hit.title, titles):
        return False
    if search.category_is_excluded(hit.category, categories):
        return False
    return True


def _passages_from_hits(hits: list[search.ChunkHit]) -> list[Passage]:
    passages = []
    for hit in hits:
        passages.append(
            Passage(
                document_id=hit.document_id,
                title=hit.title or "",
                category=hit.category,
                topic=hit.topic,
                source=hit.source,
                source_url=hit.source_url,
                source_date=hit.source_date,
                is_authoritative=hit.is_authoritative,
                chunk_text=hit.chunk_text,
                relevance_score=float(hit.score or 0.0),
            )
        )
    return passages


@app.post("/retrieve", response_model=RetrievalResponse, dependencies=[Depends(require_api_key)])
def retrieve(req: RetrievalRequest):
    """
    Hybrid search + rerank.

    1. Embed the question with Voyage AI. A repeated question reuses the cached vector.
    2. Full-text search (codes, acronyms, names, titles) and vector search.
       The vector leg uses HNSW when that index is valid. exact_search, the
       VECTOR_EXACT_SEARCH rollback, or a missing index keeps the exact scan.
       Each leg keeps a few chunks per document and one row per exact passage.
    3. Fuse the rankings, collapse near-duplicate text, cap chunks per document,
       then rerank. The cap counts Drive copies of one title as one document.
    4. Backfill open slots with the next distinct passages. When expand_neighbors
       is on, splice in adjacent chunks of the top documents.
    5. Return the top passages plus timings_ms for this request.
    """
    started = time.perf_counter()
    marks = {key: 0.0 for key in _TIMING_KEYS}
    final_n, candidates_n, pool_n, final_cap, candidate_cap = passage_limits(req.top_k)
    title_filters = search.title_exclusion_substrings(req.exclude_titles)
    category_filters = search.clean_strings(req.exclude_categories)
    exact = _use_exact_vector_search(req.exact_search)
    mode = "exact" if exact else "ann"
    ann_limit = None if exact else search.ann_candidate_limit(pool_n)

    t_embed = time.perf_counter()
    try:
        q_vector = embed_query(req.question)
    except Exception as exc:
        log.error(f"Voyage embed failed: {exc}")
        raise HTTPException(status_code=502, detail=f"Embedding failed: {exc}")
    finally:
        marks["embed"] = time.perf_counter() - t_embed

    where_sql, filter_params = search.retrieval_where(
        category=req.category,
        source=req.source,
        exclude_titles=req.exclude_titles,
        exclude_categories=req.exclude_categories,
    )
    vector_json = json.dumps(q_vector)
    vector_sql = search.vector_search_sql(where_sql, ann_limit=ann_limit)
    vector_params = search.vector_search_params(
        vector_json,
        filter_params,
        candidate_cap,
        pool_n,
        ann_limit=ann_limit,
    )
    plan = search.plan_keyword_query(req.question)
    keyword_sql, keyword_params = search.keyword_search_sql(
        where_sql,
        plan,
        filter_params,
        pool_n,
        per_document=candidate_cap,
    )

    try:
        with get_db() as conn:
            cur = conn.cursor()
            keyword_ready = ensure_search_ready(conn)
            keyword_rows = []
            t_keyword = time.perf_counter()
            if keyword_ready and keyword_sql:
                # Run this before exact-scan settings. Those settings also
                # disable the GIN index, and the rows are already in memory
                # by the time the vector query needs a sequential scan.
                try:
                    cur.execute("SAVEPOINT keyword_search")
                    cur.execute(keyword_sql, keyword_params)
                    keyword_rows = cur.fetchall()
                    cur.execute("RELEASE SAVEPOINT keyword_search")
                except Exception as exc:
                    log.warning("Keyword search failed (%s); continuing with vectors", exc)
                    try:
                        cur.execute("ROLLBACK TO SAVEPOINT keyword_search")
                        cur.execute("RELEASE SAVEPOINT keyword_search")
                    except Exception:
                        log.warning("Could not recover the keyword-search savepoint", exc_info=True)
            marks["keyword"] = time.perf_counter() - t_keyword
            t_vector = time.perf_counter()
            search.apply_vector_settings(cur, exact=exact, ef_search=ann_limit)
            cur.execute(vector_sql, vector_params)
            vector_rows = cur.fetchall()
            marks["vector_search"] = time.perf_counter() - t_vector
    except Exception as exc:
        log.error(f"Postgres query failed: {exc}")
        raise HTTPException(status_code=500, detail=f"Database error: {exc}")

    vector_hits = [
        hit
        for hit in (search.hit_from_row(row) for row in vector_rows)
        if _keep_hit(hit, title_filters, category_filters)
    ]
    keyword_hits = [
        hit
        for hit in (search.hit_from_row(row) for row in keyword_rows)
        if _keep_hit(hit, title_filters, category_filters)
    ]
    merged = search.rrf_merge([vector_hits, keyword_hits])
    collapsed = search.collapse_similar_passages(merged)
    group_key = search.make_group_key(collapsed)
    candidates = search.select_with_document_cap(
        collapsed,
        limit=candidates_n,
        per_document=candidate_cap,
        floor=final_n,
        group_key=group_key,
    )
    if not candidates:
        timings = _timing_payload(started, marks)
        log.info(
            "No results for: %r (vector=%s keyword=%s mode=%s) timings_ms=%s",
            req.question[:80],
            len(vector_hits),
            len(keyword_hits),
            mode,
            timings,
        )
        return RetrievalResponse(question=req.question, passages=[], timings_ms=timings)

    rerank_inputs = [passage_for_model(hit.title, hit.chunk_text) for hit in candidates]
    t_rerank = time.perf_counter()
    try:
        rerank_result = vo.rerank(
            req.question,
            rerank_inputs,
            model=VOYAGE_RERANK_MODEL,
            top_k=len(candidates),
        )
        ordered: list[search.ChunkHit] = []
        for result in rerank_result.results:
            hit = candidates[result.index]
            hit.score = float(result.relevance_score)
            ordered.append(hit)
    except Exception as exc:
        log.warning(f"Reranking failed ({exc}), falling back to fused order")
        ordered = candidates
    finally:
        marks["rerank"] = time.perf_counter() - t_rerank

    ordered = search.collapse_similar_passages(ordered)
    group_key = search.make_group_key(ordered)
    final_hits = search.select_with_document_cap(
        ordered,
        limit=final_n,
        per_document=final_cap,
        floor=final_n,
        group_key=group_key,
    )
    expand = _expand_enabled(req.expand_neighbors)
    budget = search.neighbor_budget(final_n)
    if expand and budget and final_hits:
        doc_ids = search.top_document_ids(final_hits, search.DEFAULT_NEIGHBOR_DOCS)
        pairs = search.neighbor_pairs(final_hits, doc_ids)
        if pairs:
            # Checked out after the rerank returns, so a pooled connection is
            # not held during the Voyage call.
            t_neighbors = time.perf_counter()
            try:
                with get_db() as conn:
                    cur = conn.cursor()
                    cur.execute(search.neighbor_sql(), (tuple(pairs),))
                    neighbor_rows = cur.fetchall()
            except Exception as exc:
                log.warning("Neighbor lookup failed (%s); returning the capped list", exc)
                neighbor_rows = []
            finally:
                marks["neighbors"] = time.perf_counter() - t_neighbors
            neighbors = [
                hit
                for hit in (search.hit_from_row(row) for row in neighbor_rows)
                if _keep_hit(hit, title_filters, category_filters)
            ]
            if neighbors:
                final_hits = search.splice_neighbors(
                    final_hits,
                    neighbors,
                    limit=final_n,
                    expand_doc_ids=doc_ids,
                    max_neighbors=budget,
                )
    passages = _passages_from_hits(final_hits)
    timings = _timing_payload(started, marks)
    log.info(
        "Retrieved %s passage(s) for %r (vector=%s keyword=%s merged=%s unique=%s mode=%s) timings_ms=%s",
        len(passages),
        req.question[:80],
        len(vector_hits),
        len(keyword_hits),
        len(merged),
        len(collapsed),
        mode,
        timings,
    )
    return RetrievalResponse(question=req.question, passages=passages, timings_ms=timings)


@app.post("/reindex", dependencies=[Depends(require_api_key)])
def reindex(payload: Optional[dict[str, Any]] = Body(default=None)):
    """
    Start an incremental reindex in the background.

    Unchanged documents (same updated_date, or the same content hash) are not
    re-embedded. Chunks for documents Base44 no longer returns are deleted.
    The index is not dropped. Send {"force": true} to re-embed every document.
    """
    force = False
    if isinstance(payload, dict):
        force = payload.get("force", False)
        if isinstance(force, str):
            force = force.strip().lower() in {"1", "true", "yes", "y", "on"}
        else:
            force = bool(force)
    code, body = start_reindex(force=bool(force), trigger="manual")
    return JSONResponse(status_code=code, content=body)


@app.get("/reindex/status", dependencies=[Depends(require_api_key)])
def reindex_status_endpoint():
    """In-memory progress of this process, plus the last finished run when idle."""
    return reindex_status()


@app.get("/maintenance/duplicate-chunks", dependencies=[Depends(require_api_key)])
def duplicate_chunks_report():
    """
    Report exact duplicate chunk rows. Does not delete anything.

    Same API key as /retrieve. Cross-document Drive copies are listed and left
    in the table; retrieval already collapses them.
    """
    try:
        return _duplicate_chunk_body(False)
    except Exception as exc:
        log.error("Duplicate-chunk report failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Database error: {exc}")


@app.post("/maintenance/duplicate-chunks", dependencies=[Depends(require_api_key)])
def duplicate_chunks_apply(payload: Optional[dict[str, Any]] = Body(default=None)):
    """
    Report exact duplicate chunk rows. Send {"apply": true} to delete the extras.

    An empty body, or any payload that does not set apply to true, is report-only.
    One row is kept for each (document_id, chunk_index, chunk_text).
    """
    apply = _apply_flag(payload)
    try:
        return _duplicate_chunk_body(apply)
    except Exception as exc:
        log.error("Duplicate-chunk maintenance failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Database error: {exc}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "8000")),
        reload=False,
    )
