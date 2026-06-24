#!/usr/bin/env python3
"""
RC Brain — Knowledge Indexer
=============================
Pulls every KnowledgeDocument from Base44, splits each one into short passages
(chunks), creates a Voyage AI meaning fingerprint (embedding) for every chunk,
and stores the result in PostgreSQL + pgvector.

Run once to seed the database, then on a schedule whenever new content is added.

Usage:
    python indexer.py            # skip unchanged documents
    python indexer.py --force    # re-index everything regardless of changes

Environment variables (set in Railway or a .env file):
    DATABASE_URL       - PostgreSQL connection string (from Railway)
    VOYAGE_API_KEY     - Voyage AI API key
    BASE44_API_KEY     - Base44 backend API key (Settings → API Keys in Base44)
    BASE44_APP_ID      - RC Brain app ID in Base44 (default set below)
    VOYAGE_EMBED_MODEL - Voyage model to use (default: voyage-large-2)
    CHUNK_SIZE         - Max characters per chunk (default: 800)
    CHUNK_OVERLAP      - Overlap between chunks in characters (default: 150)
"""

import os
import sys
import json
import time
import hashlib
import logging
import requests
import psycopg2
import voyageai
from dotenv import load_dotenv

load_dotenv()

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Config ───────────────────────────────────────────────────────────────────
VOYAGE_API_KEY     = os.environ["VOYAGE_API_KEY"]
DATABASE_URL       = os.environ["DATABASE_URL"]
BASE44_API_KEY     = os.environ["BASE44_API_KEY"]
BASE44_APP_ID      = os.environ.get("BASE44_APP_ID", "69ef93740d8ea9e5021345b0")
VOYAGE_EMBED_MODEL = os.environ.get("VOYAGE_EMBED_MODEL", "voyage-large-2")
CHUNK_SIZE         = int(os.environ.get("CHUNK_SIZE", "800"))
CHUNK_OVERLAP      = int(os.environ.get("CHUNK_OVERLAP", "150"))
EMBED_BATCH_SIZE   = 32   # Voyage API limit per call

vo = voyageai.Client(api_key=VOYAGE_API_KEY)

# ── Base44 helpers ────────────────────────────────────────────────────────────

def fetch_all_knowledge_documents() -> list[dict]:
    """Page through the Base44 KnowledgeDocument entity and return all records."""
    base_url = f"https://api.base44.com/api/apps/{BASE44_APP_ID}/entities/KnowledgeDocument"
    headers  = {
        "x-api-key":     BASE44_API_KEY,
        "Content-Type":  "application/json",
    }

    all_docs = []
    skip = 0
    limit = 100

    while True:
        resp = requests.get(
            base_url,
            headers=headers,
            params={"limit": limit, "skip": skip},
            timeout=30,
        )
        resp.raise_for_status()
        batch = resp.json()

        # Base44 returns either a list directly or {"documents": [...]}
        if isinstance(batch, dict):
            batch = batch.get("documents", [])

        if not batch:
            break

        all_docs.extend(batch)
        log.info(f"Fetched {len(all_docs)} documents so far …")

        if len(batch) < limit:
            break
        skip += limit

    log.info(f"Total documents fetched from Base44: {len(all_docs)}")
    return all_docs


# ── Text chunking ─────────────────────────────────────────────────────────────

def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Split text into overlapping chunks.
    Tries to break on sentence boundaries ('. ') when possible.
    """
    chunks = []
    start = 0

    while start < len(text):
        end = start + size

        if end < len(text):
            # Try to break on a sentence boundary within the last 200 chars
            boundary = text.rfind(". ", start + size - 200, end)
            if boundary != -1:
                end = boundary + 1  # include the period

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break
        start = end - overlap

    return chunks


# ── Embedding ─────────────────────────────────────────────────────────────────

def embed_texts(texts: list[str], input_type: str = "document") -> list[list[float]]:
    """Embed a list of texts using Voyage AI, batched to respect API limits."""
    all_embeddings = []

    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[i : i + EMBED_BATCH_SIZE]
        result = vo.embed(batch, model=VOYAGE_EMBED_MODEL, input_type=input_type)
        all_embeddings.extend(result.embeddings)
        if i + EMBED_BATCH_SIZE < len(texts):
            time.sleep(0.05)  # gentle rate-limit buffer

    return all_embeddings


# ── Hashing for change detection ──────────────────────────────────────────────

def content_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


# ── Main indexing logic ───────────────────────────────────────────────────────

def index_documents(force_reindex: bool = False) -> None:
    docs = fetch_all_knowledge_documents()

    conn = psycopg2.connect(DATABASE_URL)
    cur  = conn.cursor()

    indexed_new  = 0
    skipped_same = 0
    errors       = 0

    for doc in docs:
        doc_id  = doc.get("_id") or doc.get("id") or doc.get("document_id")
        title   = doc.get("title", "Untitled")
        content = (doc.get("content") or "").strip()

        if not content:
            log.warning(f"Skipping '{title}' ({doc_id}) — no content")
            continue
        if not doc_id:
            log.warning(f"Skipping '{title}' — no document ID")
            continue

        chash = content_hash(content)

        # Skip unchanged documents unless --force
        if not force_reindex:
            cur.execute(
                "SELECT content_hash FROM indexed_documents WHERE document_id = %s",
                (doc_id,),
            )
            row = cur.fetchone()
            if row and row[0] == chash:
                skipped_same += 1
                continue

        try:
            # Remove stale chunks and registry entry for this doc
            cur.execute("DELETE FROM knowledge_chunks   WHERE document_id = %s", (doc_id,))
            cur.execute("DELETE FROM indexed_documents  WHERE document_id = %s", (doc_id,))

            # Chunk
            chunks = chunk_text(content)
            if not chunks:
                log.warning(f"Skipping '{title}' — chunking produced no output")
                continue

            # Embed all chunks for this document
            embeddings = embed_texts(chunks, input_type="document")

            # Insert chunks
            for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                cur.execute(
                    """
                    INSERT INTO knowledge_chunks
                        (document_id, title, category, topic, subtopic,
                         source, source_url, is_authoritative,
                         chunk_index, chunk_text, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::vector)
                    """,
                    (
                        doc_id,
                        title,
                        doc.get("category"),
                        doc.get("topic"),
                        doc.get("subtopic"),
                        doc.get("source"),
                        doc.get("source_url"),
                        bool(doc.get("is_authoritative", False)),
                        idx,
                        chunk,
                        json.dumps(embedding),
                    ),
                )

            # Record in registry
            cur.execute(
                """
                INSERT INTO indexed_documents (document_id, title, chunk_count, content_hash)
                VALUES (%s, %s, %s, %s)
                """,
                (doc_id, title, len(chunks), chash),
            )

            conn.commit()
            indexed_new += 1
            log.info(f"✓ Indexed '{title}' → {len(chunks)} chunk(s)")

        except Exception as exc:
            conn.rollback()
            errors += 1
            log.error(f"✗ Failed to index '{title}' ({doc_id}): {exc}")

    cur.close()
    conn.close()

    log.info(
        f"\nDone.  Indexed: {indexed_new}  |  Skipped (unchanged): {skipped_same}  |  Errors: {errors}"
    )


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    force = "--force" in sys.argv
    if force:
        log.info("Force mode: re-indexing all documents regardless of changes.")
    index_documents(force_reindex=force)
