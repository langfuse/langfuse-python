"""Unit tests for BatchEvaluationRunner._fetch_batch_with_retry.

These tests cover the v2 observations API path used by `batch_evaluation`,
so the SDK works on Langfuse platform v4 events_only deployments where the
legacy `/api/public/traces` and `/api/public/observations` endpoints are
unavailable. See langfuse/langfuse#1861.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from langfuse.batch_evaluation import BatchEvaluationRunner


def _v2_response(*, items: list[Any], cursor: str | None = None) -> MagicMock:
    response = MagicMock()
    response.data = items
    response.meta.cursor = cursor
    return response


def _obs(
    *,
    id: str,
    trace_id: str,
    input: Any = None,
    output: Any = None,
    is_root: bool = False,
) -> MagicMock:
    obs = MagicMock()
    obs.id = id
    obs.trace_id = trace_id
    obs.input = input
    obs.output = output
    obs.is_root_observation = is_root
    return obs


class _StubRunner(BatchEvaluationRunner):
    """Runner whose v3 endpoints raise and whose _process_batch_evaluation_item
    is replaced so the unit test focuses on the fetch path."""

    def __init__(self) -> None:
        self.client = MagicMock()
        self.client.api.trace.list.side_effect = AssertionError(
            "v3 GET /api/public/traces must not be called on v4 events_only"
        )
        self.client.api.legacy.observations_v1.get_many.side_effect = AssertionError(
            "v1 GET /api/public/observations must not be called on v4 events_only"
        )
        self.client.flush = MagicMock()


@pytest.mark.asyncio
async def test_fetch_batch_uses_v2_observations_api_for_observations_scope() -> None:
    runner = _StubRunner()
    runner._process_batch_evaluation_item = MagicMock(  # type: ignore[method-assign]
        return_value=(0, 0, 0, [])
    )

    runner.client.api.observations.get_many.side_effect = [
        _v2_response(items=[_obs(id="obs-1", trace_id="t-1")], cursor="c-2"),
        _v2_response(items=[_obs(id="obs-2", trace_id="t-2")], cursor=None),
    ]

    items: list = []
    cursor: Any = None
    for _ in range(3):
        batch, cursor = await runner._fetch_batch_with_retry(
            scope="observations",
            filter=None,
            cursor=cursor,
            limit=50,
            max_retries=1,
            fields=None,
        )
        items.extend(batch)
        if cursor is None:
            break

    assert [item.id for item in items] == ["obs-1", "obs-2"]
    calls = runner.client.api.observations.get_many.call_args_list
    assert len(calls) == 2
    assert calls[0].kwargs["cursor"] is None
    assert calls[0].kwargs["limit"] == 50
    assert calls[1].kwargs["cursor"] == "c-2"


@pytest.mark.asyncio
async def test_fetch_batch_groups_observations_per_trace_for_traces_scope() -> None:
    runner = _StubRunner()
    runner._process_batch_evaluation_item = MagicMock(  # type: ignore[method-assign]
        return_value=(0, 0, 0, [])
    )

    runner.client.api.observations.get_many.side_effect = [
        _v2_response(
            items=[
                _obs(id="root-a", trace_id="ta", is_root=True),
                _obs(id="child-a", trace_id="ta", is_root=False),
                _obs(id="root-b", trace_id="tb", is_root=True),
            ],
        )
    ]

    items, _cursor = await runner._fetch_batch_with_retry(
        scope="traces",
        filter=None,
        cursor=None,
        limit=50,
        max_retries=1,
        fields=None,
    )

    # Each trace appears once, with its root observation preferred.
    assert [item.id for item in items] == ["root-a", "root-b"]
    assert runner.client.api.observations.get_many.call_count == 1
    kwargs = runner.client.api.observations.get_many.call_args.kwargs
    assert kwargs["cursor"] is None
    assert kwargs["limit"] == 50


@pytest.mark.asyncio
async def test_fetch_batch_rejects_unknown_scope() -> None:
    runner = _StubRunner()
    runner.client.api.observations.get_many.side_effect = AssertionError(
        "v2 observations must not be called for an unknown scope"
    )

    with pytest.raises(ValueError, match="bogus"):
        await runner._fetch_batch_with_retry(
            scope="bogus",
            filter=None,
            cursor=None,
            limit=10,
            max_retries=1,
            fields=None,
        )


@pytest.mark.asyncio
async def test_fetch_batch_falls_back_when_trace_has_no_root_observation() -> None:
    """When scope='traces' but no observation on a page is marked as the
    root, the helper must still collapse to one item per trace using the
    first observation seen."""

    runner = _StubRunner()
    runner._process_batch_evaluation_item = MagicMock(  # type: ignore[method-assign]
        return_value=(0, 0, 0, [])
    )

    runner.client.api.observations.get_many.side_effect = [
        _v2_response(
            items=[
                _obs(id="first-t", trace_id="t1", is_root=False),
                _obs(id="second-t", trace_id="t1", is_root=False),
            ],
        )
    ]

    items, _cursor = await runner._fetch_batch_with_retry(
        scope="traces",
        filter=None,
        cursor=None,
        limit=50,
        max_retries=1,
        fields=None,
    )

    assert [item.id for item in items] == ["first-t"]


@pytest.mark.asyncio
async def test_fetch_batch_filters_out_observations_without_trace_id() -> None:
    runner = _StubRunner()
    runner._process_batch_evaluation_item = MagicMock(  # type: ignore[method-assign]
        return_value=(0, 0, 0, [])
    )

    no_trace = MagicMock()
    no_trace.id = "orphan"
    no_trace.trace_id = None
    no_trace.is_root_observation = False

    runner.client.api.observations.get_many.side_effect = [
        _v2_response(
            items=[
                no_trace,
                _obs(id="kept", trace_id="t1", is_root=True),
            ],
        )
    ]

    items, _cursor = await runner._fetch_batch_with_retry(
        scope="traces",
        filter=None,
        cursor=None,
        limit=50,
        max_retries=1,
        fields=None,
    )

    assert [item.id for item in items] == ["kept"]
