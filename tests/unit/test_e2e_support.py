from types import SimpleNamespace

from langfuse.api.commons.errors.not_found_error import NotFoundError
from tests.support.api_wrapper import LangfuseAPI as SupportLangfuseAPI
from tests.support.utils import get_api, wait_for_trace


def test_get_api_retries_not_found(monkeypatch):
    monkeypatch.setattr("tests.support.retry.sleep", lambda _: None)

    attempts = {"count": 0}

    class FakeTraceService:
        def get(self, trace_id):
            attempts["count"] += 1

            if attempts["count"] < 3:
                raise NotFoundError(
                    body={
                        "error": "LangfuseNotFoundError",
                        "message": f"Trace {trace_id} not found within authorized project",
                    }
                )

            return {"id": trace_id}

    class FakeClient:
        trace = FakeTraceService()

    monkeypatch.setattr("tests.support.utils.LangfuseAPI", lambda **_: FakeClient())

    trace = get_api().trace.get("trace-123")

    assert trace == {"id": "trace-123"}
    assert attempts["count"] == 3


def test_get_api_retries_filtered_lists(monkeypatch):
    monkeypatch.setattr("tests.support.retry.sleep", lambda _: None)

    attempts = {"count": 0}

    class FakeTraceService:
        def list(self, **kwargs):
            attempts["count"] += 1

            if attempts["count"] < 3:
                return SimpleNamespace(data=[])

            return SimpleNamespace(data=[kwargs["name"]])

    class FakeClient:
        trace = FakeTraceService()

    monkeypatch.setattr("tests.support.utils.LangfuseAPI", lambda **_: FakeClient())

    response = get_api().trace.list(name="ready-trace")

    assert response.data == ["ready-trace"]
    assert attempts["count"] == 3


def test_get_api_retry_can_be_disabled(monkeypatch):
    attempts = {"count": 0}

    class FakeTraceService:
        def list(self, **kwargs):
            attempts["count"] += 1
            return SimpleNamespace(data=[])

    class FakeClient:
        trace = FakeTraceService()

    monkeypatch.setattr("tests.support.utils.LangfuseAPI", lambda **_: FakeClient())

    response = get_api(retry=False).trace.list(name="missing-trace")

    assert response.data == []
    assert attempts["count"] == 1


def test_raw_api_wrapper_retries_not_found_payload(monkeypatch):
    monkeypatch.setattr("tests.support.retry.sleep", lambda _: None)

    attempts = {"count": 0}

    class FakeResponse:
        def __init__(self, status_code, payload):
            self.status_code = status_code
            self._payload = payload
            self.headers = {}

        def json(self):
            return self._payload

    def fake_get(*args, **kwargs):
        attempts["count"] += 1

        if attempts["count"] < 3:
            return FakeResponse(
                404,
                {
                    "error": "LangfuseNotFoundError",
                    "message": "Trace trace-123 not found within authorized project",
                },
            )

        return FakeResponse(200, {"id": "trace-123", "observations": []})

    monkeypatch.setattr("tests.support.api_wrapper.httpx.get", fake_get)

    api = SupportLangfuseAPI(username="user", password="pass", base_url="http://test")
    trace = api.get_trace("trace-123")

    assert trace["id"] == "trace-123"
    assert attempts["count"] == 3


def test_wait_for_trace_retries_until_predicate_matches(monkeypatch):
    monkeypatch.setattr("tests.support.retry.sleep", lambda _: None)

    attempts = {"count": 0}

    class FakeTraceService:
        def get(self, trace_id):
            attempts["count"] += 1
            return {"id": trace_id, "observations": [1] * attempts["count"]}

    class FakeClient:
        trace = FakeTraceService()

    monkeypatch.setattr("tests.support.utils.LangfuseAPI", lambda **_: FakeClient())

    trace = wait_for_trace(
        "trace-123", is_result_ready=lambda trace: len(trace["observations"]) == 3
    )

    assert trace["id"] == "trace-123"
    assert len(trace["observations"]) == 3
    assert attempts["count"] == 3
