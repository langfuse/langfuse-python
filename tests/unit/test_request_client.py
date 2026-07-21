"""Unit tests for LangfuseClient (langfuse/_utils/request.py) against a real,
local HTTP server.

This exercises the low-level HTTP layer's behavior under real failure modes
that mocked-response tests do not catch: a malformed/non-JSON response body,
an error status code with a body Langfuse doesn't recognize, and a slow/hung
dependency that must be bounded by the client's configured timeout rather
than hanging indefinitely.
"""

import time

import httpx
import pytest
from werkzeug.wrappers import Response as WerkzeugResponse

from langfuse._utils.request import APIError, APIErrors, LangfuseClient


@pytest.fixture
def client(httpserver):
    return LangfuseClient(
        public_key="test-public-key",
        secret_key="test-secret-key",
        base_url=httpserver.url_for("/"),
        version="test-version",
        timeout=5,
        session=httpx.Client(),
    )


def test_batch_post_success(client, httpserver):
    httpserver.expect_request("/api/public/ingestion").respond_with_json(
        {"successes": [], "errors": []}
    )

    response = client.batch_post(batch=[])

    assert response.status_code == 200


def test_batch_post_malformed_json_response_raises_api_error(client, httpserver):
    # A 200 response whose body isn't valid JSON must not crash with a raw
    # json.JSONDecodeError; it should surface as the SDK's own APIError.
    httpserver.expect_request("/api/public/ingestion").respond_with_data(
        "not valid json", status=200, content_type="application/json"
    )

    with pytest.raises(APIError, match="Invalid JSON response received"):
        client._process_response(
            httpx.get(httpserver.url_for("/api/public/ingestion")),
            success_message="ok",
            return_json=True,
        )


def test_batch_post_207_with_errors_raises_api_errors(client, httpserver):
    httpserver.expect_request("/api/public/ingestion").respond_with_json(
        {
            "errors": [
                {"status": 400, "message": "Bad request", "error": "Invalid event"}
            ]
        },
        status=207,
    )

    with pytest.raises(APIErrors):
        client.batch_post(batch=[{"bad": "event"}])


def test_batch_post_207_malformed_json_raises_api_error(client, httpserver):
    httpserver.expect_request("/api/public/ingestion").respond_with_data(
        "not valid json", status=207, content_type="application/json"
    )

    with pytest.raises(APIError, match="Invalid JSON response received"):
        client.batch_post(batch=[{"some": "event"}])


def test_batch_post_unauthorized_raises_api_error_with_status(client, httpserver):
    httpserver.expect_request("/api/public/ingestion").respond_with_json(
        {"message": "Unauthorized"}, status=401
    )

    with pytest.raises(APIError) as exc_info:
        client.batch_post(batch=[{"some": "event"}])

    assert exc_info.value.status == 401


def test_batch_post_generic_error_with_non_json_body_raises_api_error(
    client, httpserver
):
    # A 500 with a plain-text (non-JSON) body must not crash trying to parse
    # it as JSON; it should fall back to the raw response text.
    httpserver.expect_request("/api/public/ingestion").respond_with_data(
        "Internal Server Error", status=500, content_type="text/plain"
    )

    with pytest.raises(APIError) as exc_info:
        client.batch_post(batch=[{"some": "event"}])

    assert exc_info.value.status == 500
    assert "Internal Server Error" in str(exc_info.value)


def test_batch_post_times_out_on_slow_server_instead_of_hanging(httpserver):
    # A dependency that never responds must not hang the caller forever: the
    # configured timeout has to actually be enforced end-to-end, not just
    # accepted as a constructor argument.
    def slow_handler(_request):
        time.sleep(2)
        return WerkzeugResponse("late", status=200)

    httpserver.expect_request("/api/public/ingestion").respond_with_handler(
        slow_handler
    )

    slow_client = LangfuseClient(
        public_key="test-public-key",
        secret_key="test-secret-key",
        base_url=httpserver.url_for("/"),
        version="test-version",
        timeout=0.2,
        session=httpx.Client(),
    )

    start = time.monotonic()
    with pytest.raises(httpx.TimeoutException):
        slow_client.batch_post(batch=[{"some": "event"}])
    elapsed = time.monotonic() - start

    # Bounded well below the handler's 2s sleep: the client's own timeout
    # fired rather than eventually receiving the late response.
    assert elapsed < 1.0
