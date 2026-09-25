"""Unit tests for BatchEvaluationRunner._fetch_batch_with_retry.

These tests cover the v2 observations API path used by `batch_evaluation`,
so the SDK works on Langfuse platform v4 events_only deployments where the
legacy `/api/public/traces` and `/api/public/observations` endpoints are
unavailable. See langfuse/langfuse#1861.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from langfuse.batch_evaluation import (
    BatchEvaluationRunner,
    _v2_observations_fields,
)


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
        super().__init__(client=MagicMock())
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


@pytest.mark.asyncio
async def test_fetch_batch_prefers_root_observation_regardless_of_page_order() -> None:
    """When the root is not the first observation on the page, the helper
    must still pick it as the trace's representative."""

    runner = _StubRunner()
    runner._process_batch_evaluation_item = MagicMock(  # type: ignore[method-assign]
        return_value=(0, 0, 0, [])
    )

    runner.client.api.observations.get_many.side_effect = [
        _v2_response(
            items=[
                _obs(id="child-ta", trace_id="ta", is_root=False),
                _obs(id="root-ta", trace_id="ta", is_root=True),
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

    assert [item.id for item in items] == ["root-ta"]


def test_v2_observations_fields_defaults() -> None:
    assert (
        _v2_observations_fields(None)
        == "core,basic,io,metadata,model,usage,trace_context"
    )


def test_v2_observations_fields_merges_user_supplied_group_with_defaults() -> None:
    # ``io`` is the legacy default for ``fetch_trace_fields``; the user supply
    # is preserved and the v2 default groups are added. The merged set is
    # sorted alphabetically to produce a stable comma-separated string.
    result = _v2_observations_fields("io")
    assert result == "basic,core,io,metadata,model,trace_context,usage"


def test_v2_observations_fields_drops_legacy_only_groups() -> None:
    result = _v2_observations_fields("observations,scores,io")
    assert "observations" not in result.split(",")
    assert "scores" not in result.split(",")
    assert "io" in result.split(",")
    assert "metadata" in result.split(",")


def test_v2_observations_fields_falls_back_when_user_supply_is_all_legacy() -> None:
    # If the caller passes only legacy-only groups, we cannot satisfy them on
    # v2 and we fall back to the full default set. The merged set is then
    # sorted alphabetically to produce a stable comma-separated string.
    result = _v2_observations_fields("observations,scores")
    assert result == "basic,core,io,metadata,model,trace_context,usage"


@pytest.mark.asyncio
async def test_fetch_batch_does_not_re_evaluate_traces_already_seen() -> None:
    """When a trace's observations span cursor pages, the trace-collapse state
    must remember it on the first page so the same trace is not evaluated
    twice on a later page."""

    runner = _StubRunner()
    runner._process_batch_evaluation_item = MagicMock(  # type: ignore[method-assign]
        return_value=(0, 0, 0, [])
    )

    # Page 1: trace ``ta`` appears with its root. Page 2 also returns
    # observations on ``ta`` (e.g. via ``order_by``), but the trace has
    # already been processed.
    runner.client.api.observations.get_many.side_effect = [
        _v2_response(
            items=[_obs(id="root-ta", trace_id="ta", is_root=True)],
            cursor="c-2",
        ),
        _v2_response(
            items=[_obs(id="child-ta", trace_id="ta", is_root=False)],
            cursor=None,
        ),
    ]

    seen: list = []
    cursor: Any = None
    for _ in range(3):
        batch, cursor = await runner._fetch_batch_with_retry(
            scope="traces",
            filter=None,
            cursor=cursor,
            limit=50,
            max_retries=1,
            fields=None,
        )
        seen.extend(item.id for item in batch)
        if cursor is None:
            break

    assert seen == ["root-ta"]


@pytest.mark.asyncio
async def test_fetch_batch_translates_trace_filter_to_v2_columns() -> None:
    """For ``scope='traces'`` the v3 trace-level filter columns ``name`` and
    ``timestamp`` are rewritten to ``traceName`` / ``startTime`` so the v2
    endpoint returns the same traces."""

    runner = _StubRunner()
    runner._process_batch_evaluation_item = MagicMock(  # type: ignore[method-assign]
        return_value=(0, 0, 0, [])
    )

    runner.client.api.observations.get_many.side_effect = [
        _v2_response(items=[_obs(id="o1", trace_id="t1")], cursor=None),
    ]

    filter_json = (
        '[{"type":"string","column":"name","operator":"=","value":"checkout"},'
        '{"type":"datetime","column":"timestamp","operator":">","value":"2026-01-01T00:00:00Z"},'
        '{"type":"string","column":"id","operator":"=","value":"trace-abc"}]'
    )

    await runner._fetch_batch_with_retry(
        scope="traces",
        filter=filter_json,
        cursor=None,
        limit=50,
        max_retries=1,
        fields=None,
    )

    kwargs = runner.client.api.observations.get_many.call_args.kwargs
    import json as _json

    sent_filter = _json.loads(kwargs["filter"])
    columns = {c["column"] for c in sent_filter}
    assert "traceName" in columns
    assert "startTime" in columns
    assert "traceId" in columns
    assert "name" not in columns
    assert "timestamp" not in columns
    assert "id" not in columns


@pytest.mark.asyncio
async def test_fetch_batch_does_not_translate_filter_for_observations_scope() -> None:
    runner = _StubRunner()
    runner._process_batch_evaluation_item = MagicMock(  # type: ignore[method-assign]
        return_value=(0, 0, 0, [])
    )

    runner.client.api.observations.get_many.side_effect = [
        _v2_response(items=[_obs(id="o1", trace_id="t1")], cursor=None),
    ]

    filter_json = (
        '[{"type":"string","column":"name","operator":"=","value":"checkout"}]'
    )

    await runner._fetch_batch_with_retry(
        scope="observations",
        filter=filter_json,
        cursor=None,
        limit=50,
        max_retries=1,
        fields=None,
    )

    kwargs = runner.client.api.observations.get_many.call_args.kwargs
    import json as _json

    sent_filter = _json.loads(kwargs["filter"])
    assert sent_filter[0]["column"] == "name"


def test_get_item_id_returns_trace_id_for_scope_traces() -> None:
    """When ``scope='traces'`` the item is now an ``ObservationV2`` and its
    ``id`` is the observation ID; downstream score-create calls need the
    trace ID, which is ``trace_id``."""
    obs = _obs(id="obs-id", trace_id="trace-id")
    assert BatchEvaluationRunner._get_item_id(obs, "traces") == "trace-id"
    assert BatchEvaluationRunner._get_item_id(obs, "observations") == "obs-id"


def test_get_item_timestamp_uses_observation_start_time() -> None:
    obs = _obs(id="o", trace_id="t")
    obs.start_time = datetime(2026, 9, 24, 7, 0, 0)
    assert BatchEvaluationRunner._get_item_timestamp(obs, "traces") == (
        "2026-09-24T07:00:00"
    )
    assert BatchEvaluationRunner._get_item_timestamp(obs, "observations") == (
        "2026-09-24T07:00:00"
    )


def test_get_timestamp_field_for_scope_uses_v2_start_time() -> None:
    """The v2 observations filter column for resume-by-timestamp is
    ``startTime``; the trace's timestamp is approximated by the root
    observation's start time."""
    assert BatchEvaluationRunner._get_timestamp_field_for_scope("traces") == "startTime"
    assert (
        BatchEvaluationRunner._get_timestamp_field_for_scope("observations")
        == "startTime"
    )


def _loop_stub_runner(pages: list[Any]) -> _StubRunner:
    """Runner whose fetch and process layers are stubbed so the test drives
    the ``run_async`` pagination loop itself."""
    runner = _StubRunner()

    async def fake_fetch(**kwargs: Any) -> Any:
        return pages.pop(0)

    async def fake_process(*args: Any, **kwargs: Any) -> Any:
        return (0, 0, 0, [])

    runner._fetch_batch_with_retry = fake_fetch  # type: ignore[method-assign]
    runner._process_batch_evaluation_item = fake_process  # type: ignore[method-assign]
    return runner


@pytest.mark.asyncio
async def test_run_async_reports_done_when_max_items_reached_on_last_page() -> None:
    """Scenario: the server returns ``cursor=None`` on the page where
    ``max_items`` is reached. The run must report ``completed=True`` and
    ``has_more_items=False`` — not claim more work exists."""

    obs_page = [_obs(id="o1", trace_id="t1")]
    runner = _loop_stub_runner([(obs_page, None)])

    result = await runner.run_async(
        scope="observations",
        mapper=lambda **kw: None,
        evaluators=[],
        max_items=1,
    )

    assert result.total_items_fetched == 1
    assert result.completed is True
    assert result.has_more_items is False


@pytest.mark.asyncio
async def test_run_async_reports_more_when_max_items_reached_mid_stream() -> None:
    """Scenario: ``max_items`` is reached while the server still has pages
    (``cursor`` is set). The run reports ``has_more_items=True`` so callers
    know there is remaining work."""

    page1 = [_obs(id="o1", trace_id="t1")]
    runner = _loop_stub_runner([(page1, "next-cursor")])

    result = await runner.run_async(
        scope="observations",
        mapper=lambda **kw: None,
        evaluators=[],
        max_items=1,
    )

    assert result.completed is True
    assert result.has_more_items is True


@pytest.mark.asyncio
async def test_run_async_completed_on_empty_first_page() -> None:
    """Scenario: no items at all. The run completes with zero processed items
    and ``completed=True``."""
    runner = _loop_stub_runner([([], None)])

    result = await runner.run_async(
        scope="observations",
        mapper=lambda **kw: None,
        evaluators=[],
    )

    assert result.total_items_processed == 0
    assert result.completed is True
    assert result.has_more_items is False


@pytest.mark.asyncio
async def test_run_async_clears_seen_trace_ids_between_runs() -> None:
    """``_seen_trace_ids`` must be reset at the start of every ``run_async``
    call so consecutive runs on the same runner instance do not skip traces."""

    runner = _loop_stub_runner(
        [
            ([_obs(id="root-a", trace_id="ta", is_root=True)], None),
            ([_obs(id="root-a2", trace_id="ta", is_root=True)], None),
        ]
    )

    first = await runner.run_async(
        scope="traces",
        mapper=lambda **kw: None,
        evaluators=[],
    )
    assert first.total_items_fetched == 1

    second = await runner.run_async(
        scope="traces",
        mapper=lambda **kw: None,
        evaluators=[],
    )
    # The second run sees the same trace again — it must not be filtered out
    # by state left over from the first run.
    assert second.total_items_fetched == 1


@pytest.mark.asyncio
async def test_run_async_flushes_before_returning_resume_token() -> None:
    """When a fetch fails after retries, scores already created for earlier
    pages must be flushed before the early return, not left in the buffer."""

    async def failing_fetch(**kwargs: Any) -> Any:
        raise RuntimeError("fetch exploded")

    runner = _StubRunner()
    runner._process_batch_evaluation_item = (  # type: ignore[method-assign]
        AsyncMock(return_value=(0, 0, 0, []))
    )
    runner._fetch_batch_with_retry = failing_fetch  # type: ignore[method-assign]

    result = await runner.run_async(
        scope="observations",
        mapper=lambda **kw: None,
        evaluators=[],
        max_retries=1,
    )

    assert result.completed is False
    assert result.resume_token is not None
    runner.client.flush.assert_called()
