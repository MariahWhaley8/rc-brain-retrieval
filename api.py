#!/usr/bin/env python3
"""
RC Brain — Retrieval API
=========================
FastAPI service that answers "what are the most relevant passages for this question?"

Base44 calls POST /retrieve with the user's question.
The API returns the top passages from the knowledge store, ready to pass to Claude.

Endpoints:
    GET  /health           — liveness check; returns chunk count
    POST /retrieve         — semantic search + rerank; returns top passages
    POST /reindex          — start an incremental reindex in the background
    GET  /reindex/status   — progress of the latest reindex

Environment variables (set in Railway):
    DATABASE_URL        - PostgreSQL connection string (from Railway)
    VOYAGE_API_KEY      - Voyage AI API key
    VOYAGE_EMBED_MODEL  - Model used at index time (default: voyage-large-2)
                          MUST match the model used by the indexer.
    VOYAGE_RERANK_MODEL - Reranker model (default: rerank-2)
    TOP_K               - Candidates to fetch from Postgres before reranking (default: 20)
    RERANK_TOP_N        - Final passages returned after reranking (default: 5)
    PORT                - Port to listen on (Railway sets this automatically)
    RETRIEVAL_API_KEY   - Secret key callers must send as X-API-Key header
    BASE44_API_KEY      - Base44 personal access token used by the indexer
    BASE44_APP_ID       - Base44 app id (indexer default is the RC Brain app)
    REINDEX_SCHEDULER   - "false" disables the Monday 19:00 America/Chicago job
    REINDEX_ON_STARTUP  - "false" disables the catch-up reindex when the last
                          successful run is missing or older than 8 days
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
from pydantic import BaseModel
from dotenv import load_dotenv

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
    index_documents,
    is_reindex_due,
    read_reindex_state,
)

# ── Config ───────────────────────────────────────────────
VOYAGE_API_KEY      = os.environ["VOYAGE_API_KEY"]
DATABASE_URL        = os.environ["DATABASE_URL"]
VOYAGE_EMBED_MODEL  = os.environ.get("VOYAGE_EMBED_MODEL",  "voyage-large-2")
VOYAGE_RERANK_MODEL = os.environ.get("VOYAGE_RERANK_MODEL", "rerank-2")
TOP_K               = int(os.environ.get("TOP_K",         "20"))
RERANK_TOP_N        = int(os.environ.get("RERANK_TOP_N",  "5"))
RETRIEVAL_API_KEY   = os.environ.get("RETRIEVAL_API_KEY", "")

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
    description="Semantic search over RC Brain's KnowledgeDocument library.",
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


@app.post("/retrieve", response_model=RetrievalResponse, dependencies=[Depends(require_api_key)])
def retrieve(req: RetrievalRequest):
    """
    Semantic search + rerank.

    1. Embed the question with Voyage AI.
    2. Fetch the TOP_K nearest chunks from Postgres using cosine distance.
    3. Rerank those candidates with Voyage's reranker.
    4. Return the top RERANK_TOP_N passages with metadata.
    """
    final_n = req.top_k or RERANK_TOP_N
    candidates_n = max(final_n * 4, TOP_K)

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

    where_parts  = []
    filter_params: list = []

    if req.category:
        where_parts.append("category = %s")
        filter_params.append(req.category)
    if req.source:
        where_parts.append("source = %s")
        filter_params.append(req.source)

    where_clause = ("WHERE " + " AND ".join(where_parts)) if where_parts else ""

    sql = f"""
        SELECT
            document_id,
            title,
            category,
            topic,
            source,
            source_url,
            source_date,
            is_authoritative,
            chunk_text,
            1 - (embedding <=> %s::vector) AS similarity
        FROM knowledge_chunks
        {where_clause}
        ORDER BY embedding <=> %s::vector
        LIMIT %s
    """
    params = [json.dumps(q_vector)] + filter_params + [json.dumps(q_vector), candidates_n]

    try:
        with get_db() as conn:
            cur = conn.cursor()
            cur.execute(sql, params)
            rows = cur.fetchall()
    except Exception as exc:
        log.error(f"Postgres query failed: {exc}")
        raise HTTPException(status_code=500, detail=f"Database error: {exc}")

    if not rows:
        log.info(f"No results for: {req.question!r}")
        return RetrievalResponse(question=req.question, passages=[])

    chunk_texts = [row[8] for row in rows]

    try:
        rerank_result = vo.rerank(
            req.question,
            chunk_texts,
            model=VOYAGE_RERANK_MODEL,
            top_k=final_n,
        )
        top_results = rerank_result.results
    except Exception as exc:
        log.warning(f"Reranking failed ({exc}), falling back to similarity order")
        top_results = [
            type("R", (), {"index": i, "relevance_score": rows[i][9]})()
            for i in range(min(final_n, len(rows)))
        ]

    passages = []
    for result in top_results:
        row = rows[result.index]
        passages.append(
            Passage(
                document_id=      row[0],
                title=            row[1],
                category=         row[2],
                topic=            row[3],
                source=           row[4],
                source_url=       row[5],
                source_date=      row[6],
                is_authoritative= bool(row[7]),
                chunk_text=       row[8],
                relevance_score=  float(result.relevance_score),
            )
        )

    log.info(
        f"Retrieved {len(passages)} passage(s) for: {req.question[:80]!r}"
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "8000")),
        reload=False,
    )
