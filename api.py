#!/usr/bin/env python3
"""
RC Brain — Retrieval API
=========================
FastAPI service that answers "what are the most relevant passages for this question?"

Base44 calls POST /retrieve with the user's question.
The API returns the top passages from the knowledge store, ready to pass to Claude.

Search is hybrid. A full-text leg (Postgres `simple` tsvector) and an exact
vector leg are merged with reciprocal rank fusion. Identical and near-identical
passages are collapsed, including duplicate Drive copies, then Voyage reranks
what remains. A per-document cap runs after that collapse. Titles containing
"RC Brain Agent Evaluation Q&A" are always left out.

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
    VECTOR_EXACT_SEARCH - "false" leaves ANN index scans on (default: exact scan)
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
import psycopg2
import voyageai
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
    _maybe_reindex_on_startup()
    try:
        yield
    finally:
        if scheduler is not None:
            scheduler.shutdown(wait=False)


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


# ── DB connection ─────────────────────────────────────────────────────

@contextmanager
def get_db():
    conn = psycopg2.connect(DATABASE_URL)
    try:
        yield conn
    finally:
        conn.close()


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


# ── Endpoints ───────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """Liveness check. Returns the total number of indexed chunks."""
    try:
        with get_db() as conn:
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM knowledge_chunks")
            chunk_count = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM indexed_documents")
            doc_count = cur.fetchone()[0]
        return {
            "status":      "ok",
            "chunk_count": chunk_count,
            "doc_count":   doc_count,
            "embed_model": VOYAGE_EMBED_MODEL,
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

    1. Embed the question with Voyage AI.
    2. Full-text search (codes, acronyms, names, titles) and exact vector search.
       Each leg keeps a few chunks per document and one row per exact passage.
    3. Fuse the rankings, collapse near-duplicate text, cap chunks per document,
       then rerank. The cap counts Drive copies of one title as one document.
    4. Backfill open slots with the next distinct passages. When expand_neighbors
       is on, splice in adjacent chunks of the top documents.
    5. Return the top passages. The response fields are unchanged.
    """
    final_n, candidates_n, pool_n, final_cap, candidate_cap = passage_limits(req.top_k)
    title_filters = search.title_exclusion_substrings(req.exclude_titles)
    category_filters = search.clean_strings(req.exclude_categories)

    try:
        embed_result = vo.embed(
            [req.question],
            model=VOYAGE_EMBED_MODEL,
            input_type="query",
        )
        q_vector = embed_result.embeddings[0]
    except Exception as exc:
        log.error(f"Voyage embed failed: {exc}")
        raise HTTPException(status_code=502, detail=f"Embedding failed: {exc}")

    where_sql, filter_params = search.retrieval_where(
        category=req.category,
        source=req.source,
        exclude_titles=req.exclude_titles,
        exclude_categories=req.exclude_categories,
    )
    vector_json = json.dumps(q_vector)
    vector_sql = search.vector_search_sql(where_sql)
    vector_params = [
        vector_json,
        *filter_params,
        search.doc_cap_param(candidate_cap),
        pool_n,
    ]
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
            if _env_flag("VECTOR_EXACT_SEARCH", True):
                search.apply_vector_settings(cur, exact=True)
            else:
                search.apply_vector_settings(cur, exact=False)
            cur.execute(vector_sql, vector_params)
            vector_rows = cur.fetchall()
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
        log.info(
            "No results for: %r (vector=%s keyword=%s)",
            req.question[:80],
            len(vector_hits),
            len(keyword_hits),
        )
        return RetrievalResponse(question=req.question, passages=[])

    rerank_inputs = [passage_for_model(hit.title, hit.chunk_text) for hit in candidates]
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
            try:
                with get_db() as conn:
                    cur = conn.cursor()
                    cur.execute(search.neighbor_sql(), (tuple(pairs),))
                    neighbor_rows = cur.fetchall()
            except Exception as exc:
                log.warning("Neighbor lookup failed (%s); returning the capped list", exc)
                neighbor_rows = []
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
    log.info(
        "Retrieved %s passage(s) for %r (vector=%s keyword=%s merged=%s unique=%s)",
        len(passages),
        req.question[:80],
        len(vector_hits),
        len(keyword_hits),
        len(merged),
        len(collapsed),
    )
    return RetrievalResponse(question=req.question, passages=passages)


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
