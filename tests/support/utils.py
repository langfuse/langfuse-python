import base64
import os
from typing import Any, Callable, TypeVar
from uuid import uuid4

from langfuse.api import LangfuseAPI
from tests.support.retry import (
    DEFAULT_RETRY_INTERVAL_SECONDS,
    DEFAULT_RETRY_TIMEOUT_SECONDS,
    retry_until_ready,
)

READ_METHOD_NAMES = {"get", "get_by_id", "get_many", "get_run", "list"}
PAGINATION_ARGUMENTS = {"limit", "page"}
T = TypeVar("T")


def _has_filters(kwargs: dict[str, Any]) -> bool:
    return any(
        key not in PAGINATION_ARGUMENTS and value is not None
        for key, value in kwargs.items()
    )


class _RetryingApiProxy:
    def __init__(self, target: Any):
        self._target = target

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._target, name)

        if callable(attr):
            if name not in READ_METHOD_NAMES:
                return attr

            def _call(*args: Any, **kwargs: Any) -> Any:
                return retry_until_ready(
                    lambda: attr(*args, **kwargs),
                    is_result_ready=_result_ready(name, kwargs),
                )

            return _call

        if isinstance(attr, (str, bytes, int, float, bool, list, dict, tuple, set)):
            return attr

        if attr is None:
            return None

        return _RetryingApiProxy(attr)


def _result_ready(method_name: str, kwargs: dict[str, Any]):
    if method_name not in {"get_many", "list"} or not _has_filters(kwargs):
        return None

    def _has_data(result: Any) -> bool:
        data = getattr(result, "data", None)
        return data is None or len(data) > 0

    return _has_data


def create_uuid():
    return str(uuid4())


def get_api(*, retry: bool = True):
    client = LangfuseAPI(
        username=os.environ.get("LANGFUSE_PUBLIC_KEY"),
        password=os.environ.get("LANGFUSE_SECRET_KEY"),
        base_url=os.environ.get("LANGFUSE_BASE_URL"),
    )
    return _RetryingApiProxy(client) if retry else client


def wait_for_result(
    operation: Callable[[], T],
    *,
    is_result_ready: Callable[[T], bool] | None = None,
    timeout_seconds: float = DEFAULT_RETRY_TIMEOUT_SECONDS,
    interval_seconds: float = DEFAULT_RETRY_INTERVAL_SECONDS,
) -> T:
    return retry_until_ready(
        operation,
        is_result_ready=is_result_ready,
        timeout_seconds=timeout_seconds,
        interval_seconds=interval_seconds,
    )


def wait_for_trace(
    trace_id: str,
    *,
    is_result_ready: Callable[[Any], bool] | None = None,
    timeout_seconds: float = DEFAULT_RETRY_TIMEOUT_SECONDS,
    interval_seconds: float = DEFAULT_RETRY_INTERVAL_SECONDS,
):
    api = get_api(retry=False)
    return wait_for_result(
        lambda: api.trace.get(trace_id),
        is_result_ready=is_result_ready,
        timeout_seconds=timeout_seconds,
        interval_seconds=interval_seconds,
    )


def encode_file_to_base64(image_path) -> str:
    with open(image_path, "rb") as file:
        return base64.b64encode(file.read()).decode("utf-8")
