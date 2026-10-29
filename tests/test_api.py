import logging
import threading
import time
from contextlib import contextmanager
from datetime import datetime
from zoneinfo import ZoneInfo

import psycopg2
import pytest
from fastapi.testclient import TestClient

import api

AUTH = {"X-API-Key": "test-retrieval-key"}
PASSAGE_KEYS = {
    "document_id",
    "title",
    "category",
    "topic",
    "source",
    "source_url",
    "source_date",
    "is_authoritative",
    "chunk_text",
    "relevance_score",
}


def _join_hnsw_threads():
    for thread in threading.enumerate():
        if thread.name == "hnsw-index" and thread is not threading.current_thread():
            thread.join(2)


@pytest.fixture(autouse=True)
def reset_state():
    def clear():
        with api._state_lock:
            api._reindex_state.update(
                {
                    "state": "idle",
                    "force": False,
                    "trigger": None,
                    "started_at": None,
                    "finished_at": None,
                    "stats": None,
                    "error": None,
                }
            )

    def clear_runtime():
        api.set_ann_index_ready(False)
        api._ann_build_started = False
        api._embed_cache.clear()

    clear()
    clear_runtime()
    api._search_ready = True
    yield
    _join_hnsw_threads()
    clear()
    clear_runtime()
    api._search_ready = True


@pytest.fixture
def client():
    with TestClient(api.app) as test_client:
        yield test_client


def test_weekly_trigger_is_monday_1900_chicago():
    now = datetime(2026, 10, 23, 20, 0, tzinfo=ZoneInfo("America/Chicago"))
    nxt = api.weekly_reindex_trigger().get_next_fire_time(None, now)
    local = nxt.astimezone(ZoneInfo("America/Chicago"))
    assert local.weekday() == 0
    assert local.hour == 19
    assert local.minute == 0
    assert local.date().isoformat() == "2026-10-26"


def test_reindex_requires_the_same_api_key(client):
    assert client.post("/reindex", json={"force": False}).status_code == 401
    assert client.get("/reindex/status").status_code == 401
    denied = client.post("/reindex", headers={"X-API-Key": "wrong"})
    assert denied.status_code == 401


def test_manual_reindex_runs_and_reports_status(client, monkeypatch):
    calls = []

    def fake_index(force_reindex=False):
        calls.append(force_reindex)
        return {"ok": True, "errors": 0, "indexed": 3, "skipped": 9, "deleted": 1, "fetched": 12}

    monkeypatch.setattr(api, "index_documents", fake_index)
    started = client.post("/reindex", headers=AUTH, json={"force": False})
    assert started.status_code == 202
    body = started.json()
    assert body["status"] == "succeeded"
    assert body["force"] is False
    assert body["stats"]["indexed"] == 3
    assert "Monday" in body["schedule"]
    assert calls == [False]

    forced = client.post("/reindex", headers=AUTH, json={"force": True})
    assert forced.status_code == 202
    assert forced.json()["force"] is True
    assert calls == [False, True]


def test_reindex_accepts_an_empty_body(client, monkeypatch):
    monkeypatch.setattr(
        api,
        "index_documents",
        lambda force_reindex=False: {"ok": True, "errors": 0},
    )
    response = client.post("/reindex", headers=AUTH)
    assert response.status_code == 202
    assert response.json()["force"] is False


def test_reindex_reports_partial_and_fatal_failures(client, monkeypatch):
    monkeypatch.setattr(
        api,
        "index_documents",
        lambda force_reindex=False: {
            "ok": True,
            "errors": 2,
            "error_details": [{"document_id": "x", "title": "X", "error": "voyage 429"}],
        },
    )
    partial = client.post("/reindex", headers=AUTH).json()
    assert partial["status"] == "completed_with_errors"
    assert partial["stats"]["errors"] == 2

    def explode(force_reindex=False):
        raise RuntimeError("catalog walk dropped")

    monkeypatch.setattr(api, "index_documents", explode)
    failed = client.post("/reindex", headers=AUTH)
    assert failed.status_code == 202
    assert failed.json()["status"] == "failed"
    assert "catalog walk dropped" in failed.json()["error"]


def test_reindex_conflict_while_running(client):
    with api._state_lock:
        api._reindex_state["state"] = "running"
    response = client.post("/reindex", headers=AUTH, json={"force": False})
    assert response.status_code == 409
    assert response.json()["status"] == "already_running"


def test_background_reindex_returns_before_the_job_finishes(client, monkeypatch):
    monkeypatch.setenv("REINDEX_INLINE", "false")
    started = threading.Event()
    release = threading.Event()

    def fake_index(force_reindex=False):
        started.set()
        assert release.wait(2)
        return {"ok": True, "errors": 0, "indexed": 1}

    monkeypatch.setattr(api, "index_documents", fake_index)
    response = client.post("/reindex", headers=AUTH, json={})
    assert response.status_code == 202
    assert response.json()["status"] == "started"
    assert started.wait(2)
    release.set()
    for _ in range(50):
        status = client.get("/reindex/status", headers=AUTH).json()
        if status["state"] == "succeeded":
            break
    else:
        raise AssertionError(status)
    assert status["stats"]["indexed"] == 1


def test_status_when_idle_includes_the_schedule(client, monkeypatch):
    monkeypatch.setattr(api, "read_reindex_state", lambda connect=None: (None, None))
    response = client.get("/reindex/status", headers=AUTH)
    assert response.status_code == 200
    body = response.json()
    assert body["state"] == "idle"
    assert body["schedule"] == "Mondays at 19:00 America/Chicago"
    assert body["last_status"] is None


def _row(doc_id, title, text, chunk_index=0, category="hr", topic="benefits", score=0.2):
    return (
        doc_id,
        title,
        category,
        topic,
        "Drive",
        "https://example.com/handbook",
        "2026-06-01",
        True,
        text,
        chunk_index,
        score,
    )


class RecordingDB:
    """Fake connection. Keyword and vector queries can return different rows."""

    def __init__(self, rows_for):
        self.rows_for = rows_for
        self.statements = []
        self.params = []
        self.rerank_inputs = []

    def install(self, monkeypatch):
        db = self

        class Embed:
            embeddings = [[0.1, 0.2, 0.3]]

        class Cur:
            def execute(self, sql, params=None):
                db.statements.append(sql)
                db.params.append(params)

            def fetchall(self):
                sql = db.statements[-1]
                if "neighbor_chunks" in sql:
                    return db.rows_for("neighbor")
                if "search_tsv" in sql:
                    return db.rows_for("keyword")
                return db.rows_for("vector")

        class Conn:
            def cursor(self):
                return Cur()

            def close(self):
                return None

            def commit(self):
                return None

            def rollback(self):
                return None

        @contextmanager
        def fake_db():
            yield Conn()

        def fake_rerank(query, docs, model, top_k):
            db.rerank_inputs = list(docs)

            class Result:
                pass

            result = Result()
            items = []
            for index in range(min(top_k, len(docs))):
                item = Result()
                item.index = index
                item.relevance_score = 1 - (index * 0.01)
                items.append(item)
            result.results = items
            return result

        monkeypatch.setattr(api.vo, "embed", lambda *args, **kwargs: Embed())
        monkeypatch.setattr(api.vo, "rerank", fake_rerank)
        monkeypatch.setattr(api, "get_db", fake_db)
        return db


def test_retrieve_response_shape_is_unchanged(client, monkeypatch):
    handbook = _row("doc-1", "Employee handbook", "PTO accrues each pay period.", score=0.2)
    db = RecordingDB(lambda kind: [handbook]).install(monkeypatch)

    denied = client.post("/retrieve", json={"question": "How does PTO work?"})
    assert denied.status_code == 401

    response = client.post(
        "/retrieve",
        headers=AUTH,
        json={"question": "How does PTO work?", "category": "hr"},
    )
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"question", "passages", "timings_ms"}
    assert set(body["timings_ms"]) == {
        "embed",
        "keyword",
        "vector_search",
        "rerank",
        "neighbors",
        "total",
    }
    assert all(isinstance(value, int) and value >= 0 for value in body["timings_ms"].values())
    assert body["question"] == "How does PTO work?"
    assert len(body["passages"]) == 1
    passage = body["passages"][0]
    assert set(passage) == PASSAGE_KEYS
    assert passage["document_id"] == "doc-1"
    assert passage["title"] == "Employee handbook"
    assert passage["category"] == "hr"
    assert passage["topic"] == "benefits"
    assert passage["source"] == "Drive"
    assert passage["source_url"] == "https://example.com/handbook"
    assert passage["source_date"] == "2026-06-01"
    assert passage["is_authoritative"] is True
    assert passage["chunk_text"] == "PTO accrues each pay period."
    assert passage["relevance_score"] == 1
    sqls = [" ".join(sql.split()) for sql in db.statements]
    vector_sql = next(sql for sql in sqls if "embedding <=>" in sql)
    assert "document_id, title, category, topic, source, source_url, source_date, is_authoritative, chunk_text" in vector_sql
    assert "category = %s" in vector_sql
    assert "NOT ILIKE ALL" in vector_sql
    assert "vector_exact" in vector_sql
    assert "ORDER BY embedding" not in vector_sql
    assert "enable_indexscan = off" in " ".join(sqls)
    assert api.ann_index_ready() is False
    assert any("RC Brain Agent Evaluation Q&A" in str(params) for params in db.params)
    assert db.rerank_inputs == ["Title: Employee handbook\n\nPTO accrues each pay period."]
    vector_params = next(
        params
        for sql, params in zip(db.statements, db.params)
        if params and "embedding <=>" in sql
    )
    assert vector_params[-1] >= 80


def test_default_passage_count_is_eight(client, monkeypatch):
    rows = [
        _row(f"doc-{i}", f"Doc {i}", f"Passage {i}.", chunk_index=0)
        for i in range(12)
    ]
    RecordingDB(lambda kind: rows if kind == "vector" else []).install(monkeypatch)
    response = client.post(
        "/retrieve",
        headers=AUTH,
        json={"question": "How does PTO work?"},
    )
    assert response.status_code == 200
    assert [item["document_id"] for item in response.json()["passages"]] == [
        f"doc-{i}" for i in range(8)
    ]


def test_keyword_hits_survive_an_empty_vector_search(client, monkeypatch):
    hans = _row(
        "kickstarter-3",
        "EMS Kickstarter Part 3 of 9",
        "Hans walked through revenue examples for the model years.",
        topic="ems",
        category="transcripts",
    )

    def rows_for(kind):
        if kind == "keyword":
            return [hans]
        return []

    RecordingDB(rows_for).install(monkeypatch)
    response = client.post(
        "/retrieve",
        headers=AUTH,
        json={"question": "Hans revenue examples and model years"},
    )
    assert response.status_code == 200
    passages = response.json()["passages"]
    assert len(passages) == 1
    assert passages[0]["document_id"] == "kickstarter-3"
    assert passages[0]["chunk_text"].startswith("Hans walked")
    assert "Title:" not in passages[0]["chunk_text"]


def test_per_document_cap_keeps_other_sources(client, monkeypatch):
    rows = [
        _row("transcript", "Staffing Part 1 of 9", "Part one.", chunk_index=0),
        _row("transcript", "Staffing Part 2 of 9", "Part two.", chunk_index=1),
        _row("transcript", "Staffing Part 3 of 9", "Part three.", chunk_index=2),
        _row("transcript", "Staffing Part 4 of 9", "Part four.", chunk_index=3),
        _row("transcript", "Staffing Part 5 of 9", "Part five.", chunk_index=4),
        _row("official", "EMS Top 10 Questions", "Minimum of two trucks.", chunk_index=0),
        _row("portal", "Waste lines of business", "Industrial, commercial, residential.", chunk_index=0),
    ]
    RecordingDB(lambda kind: rows if kind == "vector" else []).install(monkeypatch)
    response = client.post(
        "/retrieve",
        headers=AUTH,
        json={"question": "minimum ambulances on a shift", "top_k": 4},
    )
    ids = [item["document_id"] for item in response.json()["passages"]]
    assert ids == ["transcript", "transcript", "official", "portal"]
    assert ids.count("transcript") == 2


def test_eval_answer_keys_are_excluded_even_if_the_database_returns_them(client, monkeypatch):
    rows = [
        _row(
            "eval-key",
            "RC Brain Agent Evaluation Q&A — question 10",
            "The answer key says $75k.",
            chunk_index=0,
        ),
        _row(
            "bread",
            "Bread Acquisition Strategy",
            "Working capital of seventy five thousand.",
            chunk_index=0,
        ),
    ]
    db = RecordingDB(lambda kind: rows if kind == "vector" else []).install(monkeypatch)
    response = client.post(
        "/retrieve",
        headers=AUTH,
        json={
            "question": "bread working capital",
            "exclude_titles": "draft notes",
            "exclude_categories": ["Archive"],
        },
    )
    ids = [item["document_id"] for item in response.json()["passages"]]
    assert ids == ["bread"]
    vector_sql = " ".join(next(sql for sql in db.statements if "embedding <=>" in sql).split())
    assert "NOT ILIKE ALL" in vector_sql
    assert "category <> ALL" in vector_sql
    flat = repr(db.params)
    assert "RC Brain Agent Evaluation Q&A" in flat
    assert "draft notes" in flat
    assert "Archive" in flat


def test_single_document_corpus_still_fills_the_requested_count(client, monkeypatch):
    rows = [
        _row("only", "Hans notes", f"Chunk {i} mentions Hans.", chunk_index=i)
        for i in range(6)
    ]
    RecordingDB(lambda kind: rows if kind == "vector" else []).install(monkeypatch)
    response = client.post(
        "/retrieve",
        headers=AUTH,
        json={"question": "Hans revenue examples", "top_k": 4},
    )
    passages = response.json()["passages"]
    assert len(passages) == 4
    assert {item["document_id"] for item in passages} == {"only"}


CHECKLIST = (
    "1. Inspect the vehicle before the shift starts. "
    "2. Check the tires, lights, and fuel. "
    "3. Confirm the route sheet matches the scanner. "
    "4. Record the odometer. "
    "5. Repeat this opening checklist every morning before the first stop."
)


def test_retrieve_collapses_drive_copies_and_backfills_unique_passages(client, monkeypatch):
    rows = []
    for index in range(1, 9):
        suffix = f" ({index})" if index < 8 else ""
        doc_id = "original" if suffix == "" else f"copy-{index}"
        title = f"Opening Checklist{suffix}"
        rows.append(_row(doc_id, title, CHECKLIST, chunk_index=4))
        rows.append(
            _row(
                doc_id,
                title,
                "After the checklist, fuel every truck before noon.",
                chunk_index=5,
            )
        )
    for index in range(10):
        rows.append(
            _row(
                f"memo-{index}",
                f"Memo {index}",
                f"Distinct memo {index} about a different topic.",
                chunk_index=0,
            )
        )

    def rows_for(kind):
        if kind == "vector":
            return rows
        return []

    RecordingDB(rows_for).install(monkeypatch)
    response = client.post(
        "/retrieve",
        headers=AUTH,
        json={"question": "opening checklist before the first stop", "top_k": 12},
    )
    assert response.status_code == 200
    passages = response.json()["passages"]
    texts = [item["chunk_text"] for item in passages]
    assert len(passages) == 12
    assert len(set(texts)) == 12
    assert texts.count(CHECKLIST) == 1
    checklist = [item for item in passages if item["title"].startswith("Opening Checklist")]
    assert len(checklist) == 2
    assert {item["document_id"] for item in checklist} == {"original"}


def test_expand_neighbors_includes_the_rest_of_a_split_list(client, monkeypatch):
    middle = _row(
        "checklist",
        "Opening Checklist",
        "4. Delta 5. Echo 6. Foxtrot",
        chunk_index=2,
    )
    memos = [
        _row(f"memo-{index}", f"Memo {index}", f"Memo body {index}.", chunk_index=0)
        for index in range(10)
    ]
    neighbors = [
        _row("checklist", "Opening Checklist", "1. Alpha 2. Bravo 3. Charlie", chunk_index=1),
        _row("checklist", "Opening Checklist", "7. Golf 8. Hotel 9. India", chunk_index=3),
    ]

    def rows_for(kind):
        if kind == "vector":
            return [middle, *memos]
        if kind == "neighbor":
            return neighbors
        return []

    db = RecordingDB(rows_for).install(monkeypatch)
    response = client.post(
        "/retrieve",
        headers=AUTH,
        json={"question": "opening checklist numbered steps", "top_k": 8},
    )
    texts = [item["chunk_text"] for item in response.json()["passages"]]
    assert texts[:3] == [
        "1. Alpha 2. Bravo 3. Charlie",
        "4. Delta 5. Echo 6. Foxtrot",
        "7. Golf 8. Hotel 9. India",
    ]
    assert len(texts) == 8
    assert len(set(texts)) == 8
    assert any("neighbor_chunks" in sql for sql in db.statements)

    db = RecordingDB(rows_for).install(monkeypatch)
    off = client.post(
        "/retrieve",
        headers=AUTH,
        json={
            "question": "opening checklist numbered steps",
            "top_k": 8,
            "expand_neighbors": False,
        },
    )
    off_texts = [item["chunk_text"] for item in off.json()["passages"]]
    assert off_texts[0] == "4. Delta 5. Echo 6. Foxtrot"
    assert "1. Alpha 2. Bravo 3. Charlie" not in off_texts
    assert not any("neighbor_chunks" in sql for sql in db.statements)


def test_duplicate_chunk_endpoint_is_report_only_until_apply(client, monkeypatch):
    class Cur:
        def __init__(self):
            self.statements = []
            self.rowcount = 2

        def execute(self, sql, params=None):
            self.statements.append(sql)

        def fetchall(self):
            sql = self.statements[-1]
            if "duplicate_identical" in sql:
                return [("doc-1", "Notes (1)", 0, 3, "alpha passage")]
            if "duplicate_cross_doc" in sql:
                return [(8, 8, "alpha passage", ["Notes", "Notes (1)"])]
            return []

    class Conn:
        def __init__(self):
            self.cur = Cur()
            self.committed = False
            self.rolled_back = False

        def cursor(self):
            return self.cur

        def commit(self):
            self.committed = True

        def rollback(self):
            self.rolled_back = True

        def close(self):
            return None

    holder = {}

    @contextmanager
    def fake_db():
        conn = Conn()
        holder["conn"] = conn
        yield conn

    monkeypatch.setattr(api, "get_db", fake_db)
    denied = client.get("/maintenance/duplicate-chunks")
    assert denied.status_code == 401

    report = client.get("/maintenance/duplicate-chunks", headers=AUTH)
    assert report.status_code == 200
    body = report.json()
    assert body["mode"] == "report"
    assert body["deleted"] == 0
    assert body["extra_identical_rows"] == 2
    assert body["cross_document_text_groups"] == 1
    assert holder["conn"].rolled_back is True
    assert holder["conn"].committed is False
    assert not any("duplicate_delete" in sql for sql in holder["conn"].cur.statements)

    empty = client.post("/maintenance/duplicate-chunks", headers=AUTH)
    assert empty.status_code == 200
    assert empty.json()["mode"] == "report"
    assert holder["conn"].committed is False

    applied = client.post(
        "/maintenance/duplicate-chunks",
        headers=AUTH,
        json={"apply": True},
    )
    assert applied.status_code == 200
    assert applied.json()["deleted"] == 2
    assert applied.json()["mode"] == "deleted"
    assert holder["conn"].committed is True
    assert any("duplicate_delete" in sql for sql in holder["conn"].cur.statements)


def test_health_shape(client, monkeypatch):
    class Cur:
        def execute(self, sql, params=None):
            self.sql = sql

        def fetchone(self):
            if "knowledge_chunks" in self.sql:
                return (8284,)
            return (1267,)

    class Conn:
        def cursor(self):
            return Cur()

        def close(self):
            return None

    @contextmanager
    def fake_db():
        yield Conn()

    monkeypatch.setattr(api, "get_db", fake_db)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "chunk_count": 8284,
        "doc_count": 1267,
        "embed_model": api.VOYAGE_EMBED_MODEL,
        "vector_index": "pending",
    }
    api.set_ann_index_ready(True)
    ready = client.get("/health")
    assert ready.status_code == 200
    assert ready.json()["vector_index"] == "ready"
    assert ready.json()["chunk_count"] == 8284


def _vector_sql(statements):
    return " ".join(next(sql for sql in statements if "embedding <=>" in sql).split())


def test_ready_index_uses_hnsw_and_exact_search_forces_the_old_scan(client, monkeypatch):
    handbook = _row("doc-1", "Employee handbook", "PTO accrues each pay period.", score=0.2)
    db = RecordingDB(lambda kind: [handbook] if kind == "vector" else []).install(monkeypatch)
    api.set_ann_index_ready(True)

    fast = client.post(
        "/retrieve",
        headers=AUTH,
        json={"question": "How does PTO work?"},
    )
    assert fast.status_code == 200
    assert fast.json()["passages"][0]["chunk_text"] == "PTO accrues each pay period."
    fast_sql = _vector_sql(db.statements)
    assert "vector_ann" in fast_sql
    assert fast_sql.count("embedding <=>") == 2
    assert "ORDER BY embedding <=> %s::vector" in fast_sql
    assert "enable_indexscan = off" not in " ".join(db.statements)
    assert "SET LOCAL hnsw.ef_search = 1000" in " ".join(db.statements)
    vector_params = next(
        params
        for sql, params in zip(db.statements, db.params)
        if params and "embedding <=>" in sql
    )
    assert vector_params[-4] == vector_params[0]
    assert vector_params[-3] == 1000
    assert vector_params[-1] >= 80
    assert db.rerank_inputs == ["Title: Employee handbook\n\nPTO accrues each pay period."]

    db = RecordingDB(lambda kind: [handbook] if kind == "vector" else []).install(monkeypatch)
    exact = client.post(
        "/retrieve",
        headers=AUTH,
        json={"question": "How does PTO work?", "exact_search": True},
    )
    assert exact.status_code == 200
    assert exact.json()["passages"][0]["document_id"] == "doc-1"
    exact_sql = _vector_sql(db.statements)
    assert "vector_exact" in exact_sql
    assert "ORDER BY embedding <=>" not in exact_sql
    assert exact_sql.count("embedding <=>") == 1
    assert "enable_indexscan = off" in " ".join(db.statements)


def test_vector_search_stays_exact_until_the_hnsw_index_is_ready(client, monkeypatch):
    handbook = _row("doc-1", "Employee handbook", "PTO accrues each pay period.")
    db = RecordingDB(lambda kind: [handbook] if kind == "vector" else []).install(monkeypatch)
    api.set_ann_index_ready(False)
    response = client.post(
        "/retrieve",
        headers=AUTH,
        json={"question": "index is not ready yet", "exact_search": False},
    )
    assert response.status_code == 200
    sql = _vector_sql(db.statements)
    assert "vector_exact" in sql
    assert "enable_indexscan = off" in " ".join(db.statements)


def test_vector_exact_search_env_rolls_back_to_the_sequential_scan(client, monkeypatch):
    handbook = _row("doc-1", "Employee handbook", "PTO accrues each pay period.")
    db = RecordingDB(lambda kind: [handbook] if kind == "vector" else []).install(monkeypatch)
    api.set_ann_index_ready(True)
    monkeypatch.setenv("VECTOR_EXACT_SEARCH", "true")
    response = client.post(
        "/retrieve",
        headers=AUTH,
        json={"question": "rollback this request"},
    )
    assert response.status_code == 200
    assert "vector_exact" in _vector_sql(db.statements)
    assert "enable_indexscan = off" in " ".join(db.statements)


def test_retrieve_logs_stage_timings(client, monkeypatch, caplog):
    handbook = _row("doc-1", "Employee handbook", "PTO accrues each pay period.")
    RecordingDB(lambda kind: [handbook] if kind == "vector" else []).install(monkeypatch)
    with caplog.at_level(logging.INFO, logger="api"):
        response = client.post(
            "/retrieve",
            headers=AUTH,
            json={"question": "log the stages"},
        )
    assert response.status_code == 200
    assert any(
        "timings_ms" in record.message and "vector_search" in record.message
        for record in caplog.records
    )


def test_repeated_question_reuses_the_query_embedding(client, monkeypatch):
    calls = []

    class Embed:
        embeddings = [[0.1, 0.2, 0.3]]

    def embed(texts, model=None, input_type=None):
        calls.append(texts[0])
        return Embed()

    RecordingDB(
        lambda kind: [_row("doc-1", "Employee handbook", "PTO accrues each pay period.")]
        if kind == "vector"
        else []
    ).install(monkeypatch)
    monkeypatch.setattr(api.vo, "embed", embed)
    question = "How does PTO work?"
    first = client.post("/retrieve", headers=AUTH, json={"question": question})
    second = client.post("/retrieve", headers=AUTH, json={"question": question})
    assert first.status_code == 200
    assert second.status_code == 200
    assert calls == [question]
    other = client.post(
        "/retrieve",
        headers=AUTH,
        json={"question": "A different question"},
    )
    assert other.status_code == 200
    assert calls == [question, "A different question"]


def test_failed_query_embedding_is_not_cached(client, monkeypatch):
    calls = {"n": 0}

    class Embed:
        embeddings = [[0.1, 0.2, 0.3]]

    def embed(texts, model=None, input_type=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("voyage down")
        return Embed()

    RecordingDB(
        lambda kind: [_row("doc-1", "Employee handbook", "PTO accrues each pay period.")]
        if kind == "vector"
        else []
    ).install(monkeypatch)
    monkeypatch.setattr(api.vo, "embed", embed)
    question = "cache miss after an error"
    failed = client.post("/retrieve", headers=AUTH, json={"question": question})
    assert failed.status_code == 502
    ok = client.post("/retrieve", headers=AUTH, json={"question": question})
    assert ok.status_code == 200
    assert calls["n"] == 2


def test_embedding_cache_expires_and_evicts_the_least_recently_used():
    cache = api.QueryEmbeddingCache(maxsize=2, ttl_seconds=0.05)
    cache.put("a", (1,))
    cache.put("b", (2,))
    assert cache.get("a") == (1,)
    cache.put("c", (3,))
    assert cache.get("b") is None
    assert cache.get("a") == (1,)
    assert cache.get("c") == (3,)
    time.sleep(0.06)
    assert cache.get("a") is None
    assert cache.get("c") is None

    disabled = api.QueryEmbeddingCache(maxsize=4, ttl_seconds=0)
    disabled.put("a", (1,))
    assert disabled.get("a") is None


def test_get_db_reuses_a_pooled_connection_and_rolls_it_back(monkeypatch):
    class FakeConn:
        def __init__(self):
            self.closed = 0
            self.rollbacks = 0

        def rollback(self):
            self.rollbacks += 1

    conn = FakeConn()

    class FakePool:
        def __init__(self):
            self.puts = []

        def getconn(self):
            return conn

        def putconn(self, returned, close=False):
            self.puts.append((returned, close))

    pool = FakePool()
    monkeypatch.setattr(api, "_pool", lambda: pool)
    with api.get_db() as first:
        assert first is conn
    with api.get_db() as second:
        assert second is conn
    assert conn.rollbacks == 2
    assert pool.puts == [(conn, False), (conn, False)]


def test_get_db_discards_a_connection_that_cannot_be_reset(monkeypatch):
    class FakeConn:
        def __init__(self):
            self.closed = 0

        def rollback(self):
            raise psycopg2.OperationalError("ssl eof")

    conn = FakeConn()
    puts = []

    class FakePool:
        def getconn(self):
            return conn

        def putconn(self, returned, close=False):
            puts.append(close)

    monkeypatch.setattr(api, "_pool", lambda: FakePool())
    with pytest.raises(RuntimeError):
        with api.get_db():
            raise RuntimeError("query failed")
    assert puts == [True]


class _IndexConn:
    def __init__(self, fetches):
        self.autocommit = False
        self.statements = []
        self.fetches = list(fetches)
        self.closed = False
        self.fail_analyze = False

    def cursor(self):
        return self

    def execute(self, sql, params=None):
        assert self.autocommit is True
        text = " ".join(sql.split())
        self.statements.append(text)
        if self.fail_analyze and text.startswith("ANALYZE"):
            raise RuntimeError("analyze failed")
        if getattr(self, "fail_create", False) and text.startswith("CREATE INDEX"):
            raise RuntimeError("relation already exists")

    def fetchone(self):
        if not self.fetches:
            return None
        return self.fetches.pop(0)

    def close(self):
        self.closed = True


def test_hnsw_build_creates_the_index_outside_a_transaction():
    conn = _IndexConn([(False,), None, (True,)])
    assert api.build_hnsw_index(connect=lambda: conn) is True
    assert conn.closed is True
    text = "\n".join(conn.statements)
    assert "CREATE INDEX CONCURRENTLY IF NOT EXISTS knowledge_chunks_embedding_hnsw" in text
    assert "USING hnsw (embedding vector_cosine_ops)" in text
    assert "WITH (m = 16, ef_construction = 64)" in text
    assert "ANALYZE knowledge_chunks" in text
    lowered = text.lower()
    for forbidden in ("drop ", "delete ", "truncate ", "alter table"):
        assert forbidden not in lowered
    assert "maintenance_work_mem" not in lowered


def test_hnsw_build_skips_create_when_the_index_is_already_valid():
    conn = _IndexConn([(True,)])
    assert api.build_hnsw_index(connect=lambda: conn) is True
    assert conn.closed is True
    assert not any("CREATE INDEX" in sql for sql in conn.statements)


def test_hnsw_build_does_not_drop_an_invalid_index():
    conn = _IndexConn([(False,), (False, False), (False,)])
    assert api.build_hnsw_index(connect=lambda: conn) is False
    assert any("CREATE INDEX CONCURRENTLY" in sql for sql in conn.statements)
    assert not any(sql.lower().startswith("drop") for sql in conn.statements)


def test_hnsw_ready_check_falls_back_when_the_catalog_query_fails():
    class Conn(_IndexConn):
        def execute(self, sql, params=None):
            assert self.autocommit is True
            text = " ".join(sql.split())
            if "vector_cosine_ops" in text and text.startswith("SELECT"):
                self.statements.append(text)
                raise RuntimeError("catalog query failed")
            return super().execute(sql, params)

    conn = Conn([(True, True)])
    assert api.build_hnsw_index(connect=lambda: conn) is True
    assert any("vector_cosine_ops" in sql for sql in conn.statements)
    assert not any(sql.startswith("CREATE INDEX") for sql in conn.statements)


def test_hnsw_build_accepts_an_index_created_by_another_session():
    conn = _IndexConn([(False,), None, (True,)])
    conn.fail_create = True
    assert api.build_hnsw_index(connect=lambda: conn) is True
    assert any(sql.startswith("CREATE INDEX") for sql in conn.statements)


def test_hnsw_build_survives_analyze_failing():
    conn = _IndexConn([(False,), None, (True,)])
    conn.fail_analyze = True
    assert api.build_hnsw_index(connect=lambda: conn) is True
    assert any(sql.startswith("ANALYZE") for sql in conn.statements)


def test_hnsw_maintenance_work_mem_rejects_anything_that_is_not_a_size(monkeypatch):
    monkeypatch.setenv("HNSW_MAINTENANCE_WORK_MEM", "512MB; DROP TABLE knowledge_chunks")
    conn = _IndexConn([(False,), None, (True,)])
    assert api.build_hnsw_index(connect=lambda: conn) is True
    assert not any("drop table" in sql.lower() for sql in conn.statements)
    assert not any("maintenance_work_mem" in sql for sql in conn.statements)

    monkeypatch.setenv("HNSW_MAINTENANCE_WORK_MEM", "512MB")
    conn = _IndexConn([(False,), None, (True,)])
    assert api.build_hnsw_index(connect=lambda: conn) is True
    assert any(sql == "SET maintenance_work_mem = '512MB'" for sql in conn.statements)


def test_hnsw_thread_failure_leaves_exact_search_in_place(monkeypatch):
    def boom():
        raise RuntimeError("disk full")

    monkeypatch.setattr(api, "build_hnsw_index", boom)
    api._ann_build_started = False
    api.set_ann_index_ready(False)
    thread = api.start_ann_index_build()
    assert thread is not None
    thread.join(2)
    assert api.ann_index_ready() is False
    assert api.start_ann_index_build() is None


def test_hnsw_build_does_not_block_startup(monkeypatch):
    started = threading.Event()
    release = threading.Event()

    def slow():
        started.set()
        assert release.wait(2)
        return False

    monkeypatch.setattr(api, "build_hnsw_index", slow)
    api._ann_build_started = False
    monkeypatch.setenv("ANN_INDEX_ON_START", "true")

    class Cur:
        def execute(self, sql, params=None):
            self.sql = sql

        def fetchone(self):
            return (1,)

    class Conn:
        def cursor(self):
            return Cur()

        def close(self):
            return None

    @contextmanager
    def fake_db():
        yield Conn()

    monkeypatch.setattr(api, "get_db", fake_db)
    began = time.perf_counter()
    try:
        with TestClient(api.app) as local_client:
            assert time.perf_counter() - began < 1.0
            assert started.wait(1)
            response = local_client.get("/health")
            assert response.status_code == 200
            assert response.json()["status"] == "ok"
            assert response.json()["vector_index"] == "pending"
    finally:
        release.set()
        _join_hnsw_threads()
