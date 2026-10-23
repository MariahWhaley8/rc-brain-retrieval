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
    yield
    clear()


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


def test_retrieve_response_shape_is_unchanged(client, monkeypatch):
    executed = {}

    class Embed:
        embeddings = [[0.1, 0.2, 0.3]]

    class RerankItem:
        index = 0
        relevance_score = 0.42

    class Rerank:
        results = [RerankItem()]

    class Cur:
        def execute(self, sql, params=None):
            executed["sql"] = sql
            executed["params"] = params

        def fetchall(self):
            return [
                (
                    "doc-1",
                    "Employee handbook",
                    "hr",
                    "benefits",
                    "Drive",
                    "https://example.com/handbook",
                    "2026-06-01",
                    True,
                    "PTO accrues each pay period.",
                    0.2,
                )
            ]

    class Conn:
        def cursor(self):
            return Cur()

        def close(self):
            return None

    @contextmanager
    def fake_db():
        yield Conn()

    monkeypatch.setattr(api.vo, "embed", lambda *args, **kwargs: Embed())
    monkeypatch.setattr(api.vo, "rerank", lambda *args, **kwargs: Rerank())
    monkeypatch.setattr(api, "get_db", fake_db)

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
    assert passage["relevance_score"] == 0.42
    sql = " ".join(executed["sql"].split())
    assert "document_id, title, category, topic, source, source_url, source_date, is_authoritative, chunk_text" in sql
    assert "category = %s" in sql


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
