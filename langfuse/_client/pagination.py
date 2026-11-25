"""Page through `GET /api/public/v2/observations`."""

from typing import Any, AsyncIterator, Iterator, Optional

from langfuse.api import AsyncLangfuseAPI, LangfuseAPI, ObservationsV2Response
from langfuse.api.core import RequestOptions

# The server scans for up to 45 s per request; the client-wide default is 5 s.
_FETCH_TIMEOUT_SECONDS = 60


def _with_fetch_timeout(request_options: Optional[RequestOptions]) -> RequestOptions:
    return {"timeout_in_seconds": _FETCH_TIMEOUT_SECONDS, **(request_options or {})}


def _next_limit(limit: int, max_items: Optional[int], fetched: int) -> int:
    return limit if max_items is None else min(limit, max_items - fetched)


def iter_observation_pages(
    api: LangfuseAPI,
    *,
    limit: int,
    cursor: Optional[str],
    max_items: Optional[int],
    request_options: Optional[RequestOptions],
    **params: Any,
) -> Iterator[ObservationsV2Response]:
    options = _with_fetch_timeout(request_options)
    fetched = 0
    while max_items is None or fetched < max_items:
        page = api.observations.get_many(
            limit=_next_limit(limit, max_items, fetched),
            cursor=cursor,
            request_options=options,
            **params,
        )
        fetched += len(page.data)
        yield page
        cursor = page.meta.cursor
        if not cursor:
            return


async def aiter_observation_pages(
    api: AsyncLangfuseAPI,
    *,
    limit: int,
    cursor: Optional[str],
    max_items: Optional[int],
    request_options: Optional[RequestOptions],
    **params: Any,
) -> AsyncIterator[ObservationsV2Response]:
    options = _with_fetch_timeout(request_options)
    fetched = 0
    while max_items is None or fetched < max_items:
        page = await api.observations.get_many(
            limit=_next_limit(limit, max_items, fetched),
            cursor=cursor,
            request_options=options,
            **params,
        )
        fetched += len(page.data)
        yield page
        cursor = page.meta.cursor
        if not cursor:
            return
