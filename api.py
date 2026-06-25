#!/usr/bin/env python3
"""
RC Brain — Retrieval API
=========================
FastAPI service that answers "what are the most relevant passages for this question?"

Base44 calls POST /retrieve with the user's question.
The API returns the top passages from the knowledge store, ready to pass to Claude.

Endpoints:
    GET  /health    — liveness check; returns chunk count
    POST /retrieve  — semantic search + rerank; returns top passages

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
"""

import os
import json
import logging
import psycopg2
import voyageai
from contextlib import contextmanager
from typing import Optional
from fastapi import FastAPI, HTTPException, Security, Depends
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

# ── Config ───────────────────────────────────────────────
VOYAGE_API_KEY      = os.environ["VOYAGE_API_KEY"]
DATABASE_URL        = os.environ["DATABASE_URL"]
VOYAGE_EMBED_MODEL  = os.environ.get("VOYAGE_EMBED_MODEL",  "voyage-large-2")
VOYAGE_RERANK_MODEL = os.environ.get("VOYAGE_RERANK_MODEL", "rerank-2")
TOP_K               = int(os.environ.get("TOP_K",         "20"))
RERANK_TOP_N        = int(os.environ.get("RERANK_TOP_N",  "5"))
RETRIEVAL_API_KEY   = os.environ.get("RETRIEVAL_API_KEY", "")

vo = voyageai.Client(api_key=VOYAGE_API_KEY)

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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "8000")),
        reload=False,
    )
