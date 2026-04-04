from __future__ import annotations

import os
from time import monotonic, sleep
from typing import Callable, TypeVar

from langfuse.api.commons.errors.not_found_error import NotFoundError
from langfuse.api.core.api_error import ApiError

T = TypeVar("T")

DEFAULT_RETRY_TIMEOUT_SECONDS = float(
    os.environ.get("LANGFUSE_E2E_READ_TIMEOUT_SECONDS", "12")
)
DEFAULT_RETRY_INTERVAL_SECONDS = float(
    os.environ.get("LANGFUSE_E2E_READ_INTERVAL_SECONDS", "0.25")
)


def is_eventual_consistency_error(error: Exception) -> bool:
    if isinstance(error, NotFoundError):
        return True

    if not isinstance(error, ApiError):
        return False

    body = error.body
    return isinstance(body, dict) and body.get("error") == "LangfuseNotFoundError"


def is_not_found_payload(payload: object) -> bool:
    return isinstance(payload, dict) and payload.get("error") == "LangfuseNotFoundError"


def retry_until_ready(
    operation: Callable[[], T],
    *,
    is_retryable_error: Callable[[Exception], bool] = is_eventual_consistency_error,
    is_result_ready: Callable[[T], bool] | None = None,
    timeout_seconds: float = DEFAULT_RETRY_TIMEOUT_SECONDS,
    interval_seconds: float = DEFAULT_RETRY_INTERVAL_SECONDS,
) -> T:
    deadline = monotonic() + timeout_seconds
    last_error: Exception | None = None

    while True:
        try:
            result = operation()
        except Exception as error:
            if not is_retryable_error(error) or monotonic() >= deadline:
                raise

            last_error = error
        else:
            if is_result_ready is None or is_result_ready(result):
                return result

            if monotonic() >= deadline:
                return result

        sleep(interval_seconds)

        if monotonic() >= deadline and last_error is not None:
            raise last_error
