import json

import pytest
import requests

import indexer
from indexer import Base44Client, Base44Error


class Response:
    def __init__(self, status, payload, headers=None):
        self.status_code = status
        self._payload = payload
        self.headers = headers or {}
        self.text = payload if isinstance(payload, str) else json.dumps(payload)

    def json(self):
        if isinstance(self._payload, str):
            raise ValueError("not json")
        return self._payload


class Session:
    def __init__(self, handler):
        self.handler = handler
        self.calls = []

    def request(self, method, url, headers=None, params=None, timeout=None):
        call = {
            "method": method,
            "url": url,
            "headers": headers or {},
            "params": dict(params or {}),
            "timeout": timeout,
        }
        self.calls.append(call)
        return self.handler(call)


def client(handler, **kwargs):
    session = Session(handler)
    base = Base44Client(
        api_key="secret",
        app_id="app/1",
        base_url="https://app.base44.com",
        session=session,
        sleep=kwargs.pop("sleep", lambda _seconds: None),
        rpm=kwargs.pop("rpm", 0),
        page_size=kwargs.pop("page_size", 100),
        max_attempts=kwargs.pop("max_attempts", 4),
        rand=lambda: 0,
        **kwargs,
    )
    return base, session


def page(items, cursor=None, has_more=False):
    return {"items": items, "next_cursor": cursor, "has_more": has_more}


def test_pagination_walks_every_cursor_page_and_never_sends_skip():
    payloads = [
        page([{"id": f"d{i}", "title": "t"} for i in range(100)], cursor="c1", has_more=True),
        page([{"id": f"d{i}", "title": "t"} for i in range(100, 200)], cursor="c2", has_more=True),
        page([{"id": f"d{i}", "title": "t"} for i in range(200, 207)], cursor=None, has_more=False),
    ]

    def handler(call):
        assert "skip" not in call["params"]
        assert call["url"] == "https://app.base44.com/api/apps/app%2F1/entities/KnowledgeDocument/v2/list"
        assert "api.base44.com" not in call["url"]
        assert call["headers"]["User-Agent"].startswith("rc-brain-retrieval/")
        cursor = call["params"].get("cursor")
        if cursor is None:
            assert "content" not in call["params"]["fields"]
            assert call["params"]["sort"] == "created_date"
            return Response(200, payloads[0])
        assert "fields" not in call["params"]
        assert "sort" not in call["params"]
        return Response(200, payloads[1] if cursor == "c1" else payloads[2])

    base, session = client(handler)
    records = list(base.iter_records())
    assert len(records) == 207
    assert records[0]["id"] == "d0"
    assert records[-1]["id"] == "d206"
    assert [call["params"].get("cursor") for call in session.calls] == [None, "c1", "c2"]


def test_full_page_without_has_more_is_not_treated_as_the_catalog():
    def handler(call):
        return Response(200, {"items": [{"id": "only"}] * 100})

    base, _session = client(handler, page_size=100)
    with pytest.raises(Base44Error, match="has_more"):
        list(base.iter_records())


def test_bare_list_is_refused():
    base, _session = client(lambda call: Response(200, [{"id": "a"}]))
    with pytest.raises(Base44Error, match="bare list"):
        list(base.iter_records())


def test_repeated_cursor_stops_the_walk():
    def handler(call):
        if call["params"].get("cursor") is None:
            return Response(200, page([{"id": "a"}], cursor="again", has_more=True))
        return Response(200, page([{"id": "b"}], cursor="again", has_more=True))

    base, _session = client(handler)
    with pytest.raises(Base44Error, match="cursor repeated"):
        list(base.iter_records())


def test_retries_429_and_timeouts():
    sleeps = []

    def sleep(seconds):
        sleeps.append(seconds)

    state = {"n": 0}

    def handler(call):
        state["n"] += 1
        if state["n"] == 1:
            raise requests.Timeout("timed out")
        if state["n"] == 2:
            return Response(429, {"message": "slow"}, headers={"Retry-After": "0"})
        return Response(200, page([{"id": "a"}], has_more=False))

    base, _session = client(handler, sleep=sleep)
    assert [row["id"] for row in base.iter_records()] == ["a"]
    assert sleeps == [1.0, 0.0]


def test_auth_falls_back_from_bearer_to_legacy_headers():
    def handler(call):
        if "Authorization" in call["headers"]:
            return Response(401, {"message": "unauthorized"})
        assert call["headers"]["api_key"] == "secret"
        return Response(200, page([{"id": "a"}], has_more=False))

    base, session = client(handler)
    assert list(base.iter_records())[0]["id"] == "a"
    assert "Authorization" not in session.calls[1]["headers"]


def test_rejected_key_names_the_personal_access_token():
    base, _session = client(lambda call: Response(401, {"message": "no"}))
    with pytest.raises(Base44Error, match="personal access token"):
        list(base.iter_records())


def test_field_projection_400_retries_without_the_custom_fields():
    def handler(call):
        fields = call["params"].get("fields")
        if fields and "category" in fields:
            return Response(400, {"message": "unknown field category"})
        assert fields == "title,updated_date"
        return Response(200, page([{"id": "a", "title": "A", "updated_date": "2026-10-01"}], has_more=False))

    base, session = client(handler)
    rows = list(base.iter_records())
    assert rows[0]["id"] == "a"
    assert len(session.calls) == 2


def test_get_document_unwraps_a_nested_record():
    def handler(call):
        assert call["url"].endswith("/entities/KnowledgeDocument/doc-1")
        return Response(200, {"document": {"id": "doc-1", "content": "hello"}})

    base, _session = client(handler)
    assert base.get_document("doc-1")["content"] == "hello"
