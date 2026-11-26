"""Retries for calls made through the generated API client.

Callers pass ``request_options={"max_retries": 0}`` to the API client and wrap
the call in :func:`call_with_retries`, so network errors and retryable HTTP
responses share a single retry budget instead of multiplying each other.
"""

import random
import time
from typing import Callable, Optional, TypeVar

import httpx

from langfuse.api.core.api_error import ApiError
from langfuse.api.core.http_client import (
    _add_positive_jitter,
    _parse_retry_after,
    _parse_x_ratelimit_reset,
)

T = TypeVar("T")

RETRYABLE_STATUS_CODES = frozenset({408, 409, 429})

# Matches the cap the generated API client applies to server-provided delays.
MAX_SERVER_RETRY_DELAY_SECONDS = 60.0


def is_retryable_status(status_code: int) -> bool:
    return status_code >= 500 or status_code in RETRYABLE_STATUS_CODES


def is_retryable_api_call_error(error: Exception) -> bool:
    """Return True for network errors and 408/409/429/5xx API errors."""
    if isinstance(error, httpx.TransportError):
        return True
    if isinstance(error, ApiError) and error.status_code is not None:
        return is_retryable_status(error.status_code)

    return False


def _response_headers(error: Exception) -> Optional[httpx.Headers]:
    if isinstance(error, ApiError) and error.headers:
        return httpx.Headers(error.headers)
    if isinstance(error, httpx.HTTPStatusError) and error.response is not None:
        return error.response.headers

    return None


def retry_delay_seconds(
    error: Exception, retries: int, max_backoff_seconds: float
) -> float:
    """Honor Retry-After-Ms / Retry-After / X-RateLimit-Reset, else exponential backoff with full jitter."""
    headers = _response_headers(error)
    if headers is not None:
        # The generated _parse_retry_after compares the raw string header to an int
        # and silently falls through to Retry-After, so parse the ms variant here.
        retry_after_ms = headers.get("retry-after-ms")
        if retry_after_ms is not None:
            try:
                milliseconds = float(retry_after_ms)
            except (TypeError, ValueError):
                milliseconds = 0
            if milliseconds > 0:
                return min(milliseconds / 1000, MAX_SERVER_RETRY_DELAY_SECONDS)

        retry_after = _parse_retry_after(headers)
        if retry_after is not None and retry_after > 0:
            return min(retry_after, MAX_SERVER_RETRY_DELAY_SECONDS)

        # Like the generated client, jitter the reset delay so clients sharing a
        # rate-limited key don't all retry at the same instant.
        ratelimit_reset = _parse_x_ratelimit_reset(headers)
        if ratelimit_reset is not None and ratelimit_reset > 0:
            return _add_positive_jitter(
                min(ratelimit_reset, MAX_SERVER_RETRY_DELAY_SECONDS)
            )

    return random.uniform(0, min(2.0**retries, max_backoff_seconds))


def call_with_retries(
    func: Callable[[], T],
    *,
    max_retries: int,
    should_retry: Callable[[Exception], bool] = is_retryable_api_call_error,
    max_backoff_seconds: float = 10.0,
) -> T:
    """Call ``func``, retrying up to ``max_retries`` times on retryable errors."""
    retries = 0
    while True:
        try:
            return func()
        except Exception as e:
            if retries >= max_retries or not should_retry(e):
                raise

            time.sleep(retry_delay_seconds(e, retries, max_backoff_seconds))
            retries += 1
