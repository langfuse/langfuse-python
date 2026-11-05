import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple, Union

import httpx
import pytest

from langfuse import aiter_observation_pages, iter_observation_pages
from langfuse.api import (
    AsyncLangfuseAPI,
    LangfuseAPI,
    ObservationsV2Response,
    ObservationV2,
)
from langfuse.api.core import ApiError

MODES = pytest.mark.parametrize("mode", ["sync", "async"])


def page(ids: str, cursor: Optional[str] = None) -> ObservationsV2Response:
    return ObservationsV2Response(
        data=[
            ObservationV2(
                id=i,
                start_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
                project_id="p",
                type="SPAN",
            )
            for i in ids
        ],
        meta={"cursor": cursor} if cursor else {},
    )


class ScriptedObservations:
    def __init__(self, *responses: Union[ObservationsV2Response, Exception]):
        self.responses = list(responses)
        self.calls: List[Dict[str, Any]] = []

    def _next(self, kwargs: Dict[str, Any]) -> ObservationsV2Response:
        self.calls.append(kwargs)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    def get_many(self, **kwargs: Any) -> ObservationsV2Response:
        return self._next(kwargs)


class AsyncScriptedObservations(ScriptedObservations):
    async def get_many(self, **kwargs: Any) -> ObservationsV2Response:  # type: ignore[override]
        return self._next(kwargs)


def scripted(mode: str, *responses: Any) -> ScriptedObservations:
    cls = ScriptedObservations if mode == "sync" else AsyncScriptedObservations
    return cls(*responses)


def collect(
    mode: str, observations: ScriptedObservations, **kwargs: Any
) -> Tuple[List[ObservationsV2Response], Optional[Exception]]:
    api: Any = SimpleNamespace(observations=observations)
    pages: List[ObservationsV2Response] = []

    if mode == "sync":
        try:
            for p in iter_observation_pages(api, **kwargs):
                pages.append(p)
        except Exception as e:
            return pages, e
        return pages, None

    async def run() -> Optional[Exception]:
        try:
            async for p in aiter_observation_pages(api, **kwargs):
                pages.append(p)
        except Exception as e:
            return e
        return None

    return pages, asyncio.run(run())


def ids(pages: List[ObservationsV2Response]) -> List[str]:
    return [o.id for p in pages for o in p.data]


@MODES
def test_follows_cursor_through_empty_and_short_pages(mode: str) -> None:
    observations = scripted(
        mode,
        page("ab", "c1"),
        page("", "c2"),
        page("c", "c3"),
        page("de"),
    )

    pages, error = collect(mode, observations, limit=2, trace_id="t1")

    assert error is None
    assert [[o.id for o in p.data] for p in pages] == [
        ["a", "b"],
        [],
        ["c"],
        ["d", "e"],
    ]
    assert [c["cursor"] for c in observations.calls] == [None, "c1", "c2", "c3"]
    assert all(c["trace_id"] == "t1" and c["limit"] == 2 for c in observations.calls)


@MODES
def test_sends_per_request_timeout_and_keeps_caller_options(mode: str) -> None:
    observations = scripted(mode, page("a", "c1"), page("b"))
    collect(mode, observations, request_options={"max_retries": 0})
    assert [c["request_options"] for c in observations.calls] == [
        {"timeout_in_seconds": 60, "max_retries": 0}
    ] * 2

    observations = scripted(mode, page("a"))
    collect(mode, observations, request_options={"timeout_in_seconds": 120})
    assert observations.calls[0]["request_options"] == {"timeout_in_seconds": 120}


@MODES
def test_error_propagates_without_retry(mode: str) -> None:
    timeout_422 = ApiError(
        status_code=422, headers={}, body={"error": "Request timed out"}
    )
    observations = scripted(mode, page("a", "c1"), page("", "c2"), timeout_422)

    pages, error = collect(mode, observations)

    assert error is timeout_422
    assert len(observations.calls) == 3
    assert pages[-1].meta.cursor == observations.calls[-1]["cursor"] == "c2"


@MODES
def test_max_items_shrinks_the_last_request_so_the_cursor_resumes_after_it(
    mode: str,
) -> None:
    observations = scripted(mode, page("a", "c1"), page("bcd", "c2"), page("e", "c3"))

    pages, error = collect(mode, observations, limit=3, max_items=5)

    assert error is None
    assert ids(pages) == ["a", "b", "c", "d", "e"]
    assert [c["limit"] for c in observations.calls] == [3, 3, 1]
    assert pages[-1].meta.cursor == "c3"


@MODES
def test_generated_client_uses_60s_timeout_over_the_client_default(mode: str) -> None:
    bodies = [{"data": [], "meta": {"cursor": "c1"}}, {"data": [], "meta": {}}]
    requests: List[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=bodies[len(requests) - 1])

    client_kwargs: Dict[str, Any] = dict(
        base_url="http://langfuse.test", username="pk", password="sk", timeout=5
    )
    if mode == "sync":
        with httpx.Client(transport=httpx.MockTransport(handler)) as http:
            api = LangfuseAPI(**client_kwargs, httpx_client=http)
            pages = list(iter_observation_pages(api, trace_id="t1"))
    else:

        async def run() -> List[ObservationsV2Response]:
            transport = httpx.MockTransport(handler)
            async with httpx.AsyncClient(transport=transport) as http:
                api = AsyncLangfuseAPI(**client_kwargs, httpx_client=http)
                return [p async for p in aiter_observation_pages(api, trace_id="t1")]

        pages = asyncio.run(run())

    assert len(pages) == 2
    assert [r.url.params.get("cursor") for r in requests] == [None, "c1"]
    assert all(r.extensions["timeout"]["read"] == 60 for r in requests)
