"""Parsing of POST /api/public/scores responses into the CreateScoresResponse union.

The union is undiscriminated, so the model is picked from the response body. CI
also runs this file on pydantic 2.7, an older supported minor whose union matching
differs from current releases.
"""

import json
from typing import Any, List

import httpx
import pytest

from langfuse.api import AsyncLangfuseAPI, LangfuseAPI
from langfuse.api.scores.types.create_score_batch_response import (
    CreateScoreBatchResponse,
)
from langfuse.api.scores.types.create_score_batch_results import (
    CreateScoreBatchResults,
)
from langfuse.api.scores.types.create_score_request import CreateScoreRequest
from langfuse.api.scores.types.create_score_response import CreateScoreResponse

SINGLE_REQUEST = CreateScoreRequest(name="accuracy", value=0.9, trace_id="trace-1")
BATCH_REQUEST = [
    CreateScoreRequest(name="accuracy", value=0.9, trace_id="trace-1"),
    CreateScoreRequest(name="tone", value="friendly", trace_id="trace-1"),
]

CASES = [
    pytest.param(SINGLE_REQUEST, 200, {"id": "score-1"}, id="single-200"),
    pytest.param(BATCH_REQUEST, 202, {"message": "Accepted"}, id="batch-202"),
    pytest.param(
        BATCH_REQUEST,
        207,
        {
            "accepted": 1,
            "rejected": 1,
            "errors": [{"message": "tone: value must be numeric"}],
        },
        id="batch-207",
    ),
]


def _handler(status: int, body: dict, requests: List[httpx.Request]) -> Any:
    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(status, json=body)

    return handle


def _assert_parsed(response: Any, status: int, body: dict) -> None:
    if status == 200:
        assert isinstance(response, CreateScoreResponse)
        assert response.id == body["id"]
    elif status == 202:
        assert isinstance(response, CreateScoreBatchResponse)
        assert response.message == "Accepted"
    else:
        assert isinstance(response, CreateScoreBatchResults)
        assert (response.accepted, response.rejected) == (1, 1)
        assert [error.message for error in response.errors] == [
            "tone: value must be numeric"
        ]


def _assert_sent(requests: List[httpx.Request], request: Any) -> None:
    assert len(requests) == 1
    assert requests[0].url.path == "/api/public/scores"
    sent = json.loads(requests[0].content)
    assert isinstance(sent, list) == isinstance(request, list)


@pytest.mark.parametrize(("request_body", "status", "body"), CASES)
def test_sync_create_parses_each_response_shape(request_body, status, body):
    requests: List[httpx.Request] = []
    client = LangfuseAPI(
        base_url="http://langfuse.test",
        username="pk",
        password="sk",
        httpx_client=httpx.Client(
            transport=httpx.MockTransport(_handler(status, body, requests))
        ),
    )

    response = client.scores.create(request=request_body)

    _assert_parsed(response, status, body)
    _assert_sent(requests, request_body)


@pytest.mark.asyncio
@pytest.mark.parametrize(("request_body", "status", "body"), CASES)
async def test_async_create_parses_each_response_shape(request_body, status, body):
    requests: List[httpx.Request] = []
    client = AsyncLangfuseAPI(
        base_url="http://langfuse.test",
        username="pk",
        password="sk",
        httpx_client=httpx.AsyncClient(
            transport=httpx.MockTransport(_handler(status, body, requests))
        ),
    )

    response = await client.scores.create(request=request_body)

    _assert_parsed(response, status, body)
    _assert_sent(requests, request_body)
