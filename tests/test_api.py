import threading
from contextlib import contextmanager
from datetime import datetime
from zoneinfo import ZoneInfo

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

    clear()
    api._search_ready = True
    yield
    clear()
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
    assert set(body) == {"question", "passages"}
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
    assert "enable_indexscan = off" in " ".join(sqls)
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
    }
