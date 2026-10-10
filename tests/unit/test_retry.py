import random
import time

import httpx
import pytest

from langfuse._utils.retry import (
    MAX_SERVER_RETRY_DELAY_SECONDS,
    call_with_retries,
    retry_delay_seconds,
)
from langfuse.api.core.api_error import ApiError


def _status_error(status_code: int, headers: dict) -> httpx.HTTPStatusError:
    request = httpx.Request("PUT", "https://example.com/upload")
    response = httpx.Response(status_code, headers=headers, request=request)
    return httpx.HTTPStatusError("failed", request=request, response=response)


def test_retry_delay_caps_retry_after():
    error = ApiError(status_code=429, headers={"retry-after": "3600"})

    assert retry_delay_seconds(error, 0, 10) == MAX_SERVER_RETRY_DELAY_SECONDS


def test_retry_delay_uses_x_ratelimit_reset_without_retry_after(monkeypatch):
    monkeypatch.setattr(time, "time", lambda: 1000.0)
    error = ApiError(status_code=429, headers={"x-ratelimit-reset": "1005"})

    assert retry_delay_seconds(error, 0, 10) == 5


def test_retry_delay_reads_headers_from_http_status_errors():
    error = _status_error(503, {"Retry-After": "7"})

    assert retry_delay_seconds(error, 0, 10) == 7


def test_retry_delay_falls_back_to_capped_exponential_backoff(monkeypatch):
    monkeypatch.setattr(random, "uniform", lambda _low, high: high)
    error = ApiError(status_code=503, headers={})

    assert [retry_delay_seconds(error, retries, 10) for retries in range(5)] == [
        1,
        2,
        4,
        8,
        10,
    ]


@pytest.mark.parametrize("status_code", [400, 401, 403, 404, 422])
def test_call_with_retries_does_not_retry_non_retryable_api_errors(
    monkeypatch, status_code
):
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)
    calls = []

    def fail() -> None:
        calls.append(1)
        raise ApiError(status_code=status_code)

    with pytest.raises(ApiError):
        call_with_retries(fail, max_retries=3)

    assert len(calls) == 1
