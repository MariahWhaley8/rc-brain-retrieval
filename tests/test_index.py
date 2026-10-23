from datetime import datetime, timedelta, timezone

import pytest

import indexer
from indexer import content_hash, is_reindex_due


def norm(sql: str) -> str:
    return " ".join(sql.split()).lower()


class DB:
    def __init__(self):
        self.docs = {}
        self.chunks = {}
        self.locked = False
        self.run_status = None
        self.run_error = None


class Conn:
    def __init__(self, db):
        self.db = db
        self.autocommit = False

    def cursor(self):
        return Cur(self)

    def commit(self):
        return None

    def rollback(self):
        return None

    def close(self):
        return None


class Cur:
    def __init__(self, conn):
        self.conn = conn
        self.rowcount = 0
        self._one = None
        self._all = []

    def execute(self, sql, params=None):
        db = self.conn.db
        statement = norm(sql)
        params = params or ()
        self.rowcount = 0
        self._one = None
        self._all = []

        if "pg_try_advisory_lock" in statement:
            if db.locked:
                self._one = (False,)
            else:
                db.locked = True
                self._one = (True,)
            return
        if "pg_advisory_unlock" in statement:
            db.locked = False
            self._one = (True,)
            return
        if statement.startswith("alter table indexed_documents"):
            return
        if "create table if not exists indexer_state" in statement:
            return
        if "insert into indexer_state" in statement:
            db.run_status = "running"
            return
        if statement.startswith("update indexer_state"):
            db.run_status = params[0]
            db.run_error = params[1]
            return
        if "from indexer_state" in statement:
            self._one = (None, db.run_status)
            return
        if statement.startswith("select document_id"):
            self._all = [
                (doc_id, row["content_hash"], row.get("updated_date"))
                for doc_id, row in db.docs.items()
            ]
            return
        if statement.startswith("select 1 from knowledge_chunks"):
            chunks = db.chunks.get(params[0]) or []
            self._one = (1,) if chunks else None
            return
        if "any(" in statement and "knowledge_chunks" in statement:
            keep = set(params[0])
            removed = 0
            for doc_id in list(db.chunks):
                if doc_id not in keep:
                    removed += len(db.chunks.pop(doc_id))
            self.rowcount = removed
            return
        if "any(" in statement and "indexed_documents" in statement:
            keep = set(params[0])
            removed = 0
            for doc_id in list(db.docs):
                if doc_id not in keep:
                    db.docs.pop(doc_id)
                    removed += 1
            self.rowcount = removed
            return
        if statement.startswith("delete from knowledge_chunks"):
            db.chunks.pop(params[0], None)
            return
        if statement.startswith("delete from indexed_documents"):
            db.docs.pop(params[0], None)
            return
        if statement.startswith("update knowledge_chunks"):
            for chunk in db.chunks.get(params[8], []):
                chunk["title"] = params[0]
                chunk["source_date"] = params[6]
                chunk["source"] = params[4]
            return
        if statement.startswith("update indexed_documents"):
            doc_id = params[2]
            row = db.docs.get(doc_id)
            if row and row["content_hash"] == params[-1]:
                row["title"] = params[0]
                if "updated_date" in statement:
                    row["updated_date"] = params[1]
            return
        if statement.startswith("insert into knowledge_chunks"):
            db.chunks.setdefault(params[0], []).append(
                {
                    "text": params[10],
                    "title": params[1],
                    "source": params[5],
                    "source_url": params[6],
                    "source_date": params[7],
                    "category": params[2],
                    "topic": params[3],
                }
            )
            return
        if statement.startswith("insert into indexed_documents"):
            db.docs[params[0]] = {
                "title": params[1],
                "chunk_count": params[2],
                "content_hash": params[3],
                "updated_date": params[4] if len(params) > 4 else None,
            }
            return
        raise AssertionError(f"unexpected SQL: {statement}")

    def fetchone(self):
        return self._one

    def fetchall(self):
        return list(self._all)


class Catalog:
    def __init__(self, records, bodies=None, boom_after=None):
        self.records = records
        self.bodies = bodies or {}
        self.gets = []
        self.boom_after = boom_after

    def iter_records(self):
        for index, record in enumerate(self.records):
            if self.boom_after is not None and index >= self.boom_after:
                raise RuntimeError("catalog walk dropped")
            yield record

    def get_document(self, entity_id):
        self.gets.append(entity_id)
        if entity_id not in self.bodies:
            raise RuntimeError(f"missing {entity_id}")
        return self.bodies[entity_id]


def connect_to(db):
    def connect():
        return Conn(db)

    return connect


def embedder(store):
    def embed(texts, input_type="document"):
        store.append(list(texts))
        return [[0.1, 0.2] for _ in texts]

    return embed


def seed(db, doc_id, body, updated="2026-06-25T00:00:00", title="Stored"):
    db.docs[doc_id] = {
        "title": title,
        "chunk_count": 1,
        "content_hash": content_hash(body),
        "updated_date": updated,
    }
    db.chunks[doc_id] = [
        {"text": body, "title": title, "source": "drive", "source_url": None, "source_date": None, "category": None, "topic": None}
    ]


def run(db, catalog, force=False, embedded=None):
    embedded = [] if embedded is None else embedded
    return indexer.index_documents(
        force_reindex=force,
        client=catalog,
        connect=connect_to(db),
        embed=embedder(embedded),
    )


def test_unchanged_updated_date_does_not_fetch_or_embed():
    db = DB()
    body = "The policy is unchanged."
    seed(db, "doc-1", body)
    embedded = []
    catalog = Catalog(
        [{"id": "doc-1", "title": "Stored", "updated_date": "2026-06-25T00:00:00Z"}],
        bodies={"doc-1": {"content": "should not be read"}},
    )
    stats = run(db, catalog, embedded=embedded)
    assert stats["ok"] is True
    assert stats["skipped"] == 1
    assert stats["indexed"] == 0
    assert stats["deleted"] == 0
    assert catalog.gets == []
    assert embedded == []
    assert db.chunks["doc-1"][0]["text"] == body


def test_changed_body_is_reembedded_and_deleted_docs_are_removed():
    db = DB()
    seed(db, "keep", "old body")
    seed(db, "gone", "delete me")
    embedded = []
    catalog = Catalog(
        [
            {"id": "keep", "title": "Keep", "updated_date": "2026-10-23T00:00:00"},
            {"id": "fresh", "title": "Fresh", "updated_date": "2026-10-23T00:00:00", "category": "hr"},
        ],
        bodies={
            "keep": {
                "id": "keep",
                "title": "Keep",
                "content": "new body about benefits.",
                "updated_date": "2026-10-23T00:00:00",
                "source": "Drive",
                "source_url": "https://example.com/keep",
                "source_date": "2026-10-01",
                "category": "hr",
                "topic": "benefits",
            },
            "fresh": {
                "id": "fresh",
                "title": "Fresh",
                "content": "A brand new document. " * 50,
                "source": "Drive",
                "source_url": "https://example.com/fresh",
                "source_date": "2026-10-20",
                "category": "hr",
                "topic": "new",
            },
        },
    )
    stats = run(db, catalog, embedded=embedded)
    assert stats["indexed"] == 2
    assert stats["deleted"] == 1
    assert "gone" not in db.docs
    assert "gone" not in db.chunks
    assert db.chunks["keep"][0]["text"].startswith("new body")
    assert db.chunks["keep"][0]["source_date"] == "2026-10-01"
    assert db.chunks["keep"][0]["source_url"] == "https://example.com/keep"
    assert db.docs["fresh"]["content_hash"] == content_hash(catalog.bodies["fresh"]["content"].strip())
    assert db.docs["keep"]["updated_date"] == "2026-10-23T00:00:00"
    assert len(embedded) == 2


def test_same_hash_updates_metadata_without_embedding():
    db = DB()
    body = "Same words as June."
    seed(db, "doc-1", body, updated=None, title="Old title")
    embedded = []
    catalog = Catalog(
        [{"id": "doc-1", "title": "Old title", "updated_date": "2026-10-02T00:00:00"}],
        bodies={
            "doc-1": {
                "title": "New title",
                "content": body,
                "updated_date": "2026-10-02T00:00:00",
                "source_date": "2026-09-30",
                "source": "Drive",
            }
        },
    )
    stats = run(db, catalog, embedded=embedded)
    assert stats["skipped"] == 1
    assert stats["indexed"] == 0
    assert embedded == []
    assert db.chunks["doc-1"][0]["title"] == "New title"
    assert db.chunks["doc-1"][0]["source_date"] == "2026-09-30"
    assert db.chunks["doc-1"][0]["text"] == body
    assert db.docs["doc-1"]["updated_date"] == "2026-10-02T00:00:00"


def test_stored_legacy_id_is_kept_when_base44_also_sends_id():
    db = DB()
    body = "Legacy id body."
    seed(db, "stored-id", body, updated=None)
    embedded = []
    catalog = Catalog(
        [{"id": "new-id", "_id": "stored-id", "title": "Legacy", "updated_date": "2026-10-01T00:00:00"}],
        bodies={"new-id": {"id": "new-id", "title": "Legacy", "content": body, "updated_date": "2026-10-01T00:00:00"}},
    )
    stats = run(db, catalog, embedded=embedded)
    assert catalog.gets == ["new-id"]
    assert stats["indexed"] == 0
    assert "stored-id" in db.docs
    assert "new-id" not in db.docs
    assert embedded == []


def test_empty_catalog_does_not_wipe_the_index():
    db = DB()
    seed(db, "doc-1", "keep me")
    stats = run(db, Catalog([]))
    assert stats["ok"] is False
    assert stats["reason"] == "empty_catalog"
    assert stats["deleted"] == 0
    assert "doc-1" in db.docs
    assert db.chunks["doc-1"]
    assert db.run_status == "refused_empty_catalog"


def test_failed_walk_does_not_delete_documents_not_yet_seen():
    db = DB()
    seed(db, "keep", "keep me", updated=None)
    seed(db, "gone", "still here")
    catalog = Catalog(
        [
            {"id": "keep", "title": "Keep", "updated_date": "2026-10-01T00:00:00"},
            {"id": "gone", "title": "Gone"},
        ],
        bodies={"keep": {"content": "keep me", "title": "Keep", "updated_date": "2026-10-01T00:00:00"}},
        boom_after=1,
    )
    with pytest.raises(RuntimeError, match="catalog walk dropped"):
        run(db, catalog)
    assert "gone" in db.docs
    assert "gone" in db.chunks
    assert db.run_status == "failed"


def test_document_error_keeps_old_chunks_and_continues():
    db = DB()
    seed(db, "bad", "old bad")
    seed(db, "good", "old good", updated="2026-01-01T00:00:00")
    embedded = []
    catalog = Catalog(
        [
            {"id": "bad", "title": "Bad", "updated_date": "2026-10-01T00:00:00"},
            {"id": "good", "title": "Good", "updated_date": "2026-10-01T00:00:00"},
        ],
        bodies={
            "good": {
                "content": "fresh good text",
                "title": "Good",
                "updated_date": "2026-10-01T00:00:00",
                "source_date": "2026-10-01",
            }
        },
    )
    stats = run(db, catalog, embedded=embedded)
    assert stats["ok"] is True
    assert stats["errors"] == 1
    assert stats["indexed"] == 1
    assert db.chunks["bad"][0]["text"] == "old bad"
    assert db.chunks["good"][0]["text"] == "fresh good text"
    assert db.run_status == "completed_with_errors"


def test_empty_content_removes_stale_chunks():
    db = DB()
    seed(db, "doc-1", "obsolete")
    catalog = Catalog(
        [{"id": "doc-1", "title": "Blank", "updated_date": "2026-10-01T00:00:00"}],
        bodies={"doc-1": {"title": "Blank", "content": "   ", "updated_date": "2026-10-01T00:00:00"}},
    )
    stats = run(db, catalog)
    assert stats["empty"] == 1
    assert "doc-1" not in db.chunks
    assert "doc-1" not in db.docs


def test_force_reembeds_even_when_the_date_matches():
    db = DB()
    body = "Same body."
    seed(db, "doc-1", body)
    embedded = []
    catalog = Catalog(
        [{"id": "doc-1", "title": "Stored", "updated_date": "2026-06-25T00:00:00"}],
        bodies={"doc-1": {"content": body, "title": "Stored", "updated_date": "2026-06-25T00:00:00"}},
    )
    stats = run(db, catalog, force=True, embedded=embedded)
    assert stats["indexed"] == 1
    assert len(embedded) == 1


def test_second_indexer_does_not_run_while_the_lock_is_held():
    db = DB()
    seed(db, "doc-1", "body")
    db.locked = True
    catalog = Catalog([{"id": "doc-1", "updated_date": "2026-10-01T00:00:00"}])
    stats = run(db, catalog)
    assert stats["reason"] == "already_running"
    assert catalog.gets == []
    assert db.chunks["doc-1"][0]["text"] == "body"


def test_reindex_is_due_only_when_the_last_success_is_old_or_missing():
    now = datetime(2026, 10, 23, tzinfo=timezone.utc)
    recent = now - timedelta(days=2)
    stale = now - timedelta(days=8, seconds=1)
    assert is_reindex_due(None, None, now) is True
    assert is_reindex_due(recent, "running", now) is True
    assert is_reindex_due(recent, "succeeded", now) is False
    assert is_reindex_due(recent, "completed_with_errors", now) is False
    assert is_reindex_due(stale, "succeeded", now) is True
    assert is_reindex_due(recent.replace(tzinfo=None), "succeeded", now) is False
