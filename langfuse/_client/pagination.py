"""Page through `GET /api/public/v2/observations`."""

from typing import Any, AsyncIterator, Iterator, Optional

from langfuse.api import AsyncLangfuseAPI, LangfuseAPI, ObservationsV2Response
from langfuse.api.core import RequestOptions

# The server scans for up to 45 s per request; the client-wide default is 5 s.
_FETCH_TIMEOUT_SECONDS = 60


def iter_observation_pages(
    api: LangfuseAPI,
    *,
    limit: int = 50,
    cursor: Optional[str] = None,
    max_items: Optional[int] = None,
    request_options: Optional[RequestOptions] = None,
    **params: Any,
) -> Iterator[ObservationsV2Response]:
    """Yield pages of `api.observations.get_many` until the server returns no cursor.

    A page can be short or empty while more data remains, so only a missing
    `meta.cursor` ends the loop. Every page is yielded, including empty ones. To
    resume later, save `page.meta.cursor` after processing a page and pass it
    back as `cursor` with the same other arguments.

    Args:
        api: `Langfuse.api` or a standalone `LangfuseAPI`.
        limit: Rows per request, at most 1000.
        cursor: Cursor to resume from.
        max_items: Stop after this many rows. The last request asks only for the
            remaining rows, so the last page's cursor resumes right after them.
        request_options: Passed to every request. `timeout_in_seconds` defaults
            to 60 because the server may scan for up to 45 s per request.
        **params: Other `get_many` arguments, e.g. `trace_id`, `fields`,
            `from_start_time`, `filter`.

    Errors propagate and end the iteration. A 422 means the server hit a resource
    limit, usually its time budget; the same request tends to fail again, so
    narrow the time range or add filters instead of retrying it unchanged.

    Without a lower time bound the server searches back only to 2020-01-01. A
    `startTime` condition in `filter` makes the server ignore `from_start_time`
    and `to_start_time`, so put both bounds in `filter`. Rows are not
    deduplicated; dedupe by `id` if you need unique observations.

    Example:
        ```python
        from datetime import datetime, timedelta, timezone

        from langfuse import get_client, iter_observation_pages

        langfuse = get_client()
        since = datetime.now(timezone.utc) - timedelta(days=7)

        for page in iter_observation_pages(
            langfuse.api, trace_id="abc123", from_start_time=since, limit=1000
        ):
            for observation in page.data:
                print(observation.id, observation.name)
        ```
    """
    options: RequestOptions = {
        "timeout_in_seconds": _FETCH_TIMEOUT_SECONDS,
        **(request_options or {}),
    }
    fetched = 0
    while max_items is None or fetched < max_items:
        page = api.observations.get_many(
            limit=limit if max_items is None else min(limit, max_items - fetched),
            cursor=cursor,
            request_options=options,
            **params,
        )
        yield page
        fetched += len(page.data)
        cursor = page.meta.cursor
        if not cursor:
            return


async def aiter_observation_pages(
    api: AsyncLangfuseAPI,
    *,
    limit: int = 50,
    cursor: Optional[str] = None,
    max_items: Optional[int] = None,
    request_options: Optional[RequestOptions] = None,
    **params: Any,
) -> AsyncIterator[ObservationsV2Response]:
    """Async version of `iter_observation_pages`; pass `Langfuse.async_api`."""
    options: RequestOptions = {
        "timeout_in_seconds": _FETCH_TIMEOUT_SECONDS,
        **(request_options or {}),
    }
    fetched = 0
    while max_items is None or fetched < max_items:
        page = await api.observations.get_many(
            limit=limit if max_items is None else min(limit, max_items - fetched),
            cursor=cursor,
            request_options=options,
            **params,
        )
        yield page
        fetched += len(page.data)
        cursor = page.meta.cursor
        if not cursor:
            return
