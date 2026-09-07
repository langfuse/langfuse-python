import json
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from langfuse.api import (
    NotFoundError,
    ObservationsView,
    ObservationV2,
    TraceWithFullDetails,
)
from langfuse.batch_evaluation import (
    BatchEvaluationRunner,
    EvaluatorInputs,
)
from langfuse.experiment import Evaluation


def _observation(
    *,
    observation_id: str = "observation-id",
    trace_id: str = "trace-id",
    is_root: bool = False,
) -> ObservationV2:
    return ObservationV2(
        id=observation_id,
        trace_id=trace_id,
        start_time=datetime(2026, 1, 2, tzinfo=timezone.utc),
        project_id="project-id",
        parent_observation_id=None,
        type="SPAN",
        is_root_observation=is_root,
        name="root-span",
        trace_name="trace-name",
        input='{"question": "hello"}',
        output='"answer"',
        metadata={"source": "test"},
        tags=["production"],
        environment="production",
    )


@pytest.mark.asyncio
async def test_fetches_traces_as_root_observations_via_v2_api() -> None:
    client = MagicMock()
    client.api.observations.get_many.return_value = SimpleNamespace(
        data=[_observation(is_root=True)],
        meta=SimpleNamespace(cursor="next-cursor"),
    )
    runner = BatchEvaluationRunner(client)

    items, cursor = await runner._fetch_batch_with_retry(
        scope="traces",
        filter='[{"type":"string","column":"user_id","operator":"=","value":"user"}]',
        page=1,
        cursor=None,
        limit=10,
        max_retries=2,
        fields="io",
    )

    assert cursor == "next-cursor"
    assert len(items) == 1
    trace = items[0]
    assert isinstance(trace, TraceWithFullDetails)
    assert trace.id == "trace-id"
    assert trace.timestamp == datetime(2026, 1, 2, tzinfo=timezone.utc)
    assert trace.name == "trace-name"
    assert trace.input == {"question": "hello"}
    assert trace.output == "answer"

    kwargs = client.api.observations.get_many.call_args.kwargs
    assert kwargs["cursor"] is None
    assert kwargs["request_options"] == {"max_retries": 2}
    assert set(kwargs["fields"].split(",")) == {
        "basic",
        "io",
        "metadata",
        "time",
        "trace_context",
    }
    assert json.loads(kwargs["filter"]) == [
        {
            "type": "string",
            "column": "userId",
            "operator": "=",
            "value": "user",
        },
        {
            "type": "boolean",
            "column": "isRootObservation",
            "operator": "=",
            "value": True,
        },
    ]


@pytest.mark.asyncio
async def test_fetches_observations_via_v2_api() -> None:
    client = MagicMock()
    client.api.observations.get_many.return_value = SimpleNamespace(
        data=[_observation()],
        meta=SimpleNamespace(cursor=None),
    )
    runner = BatchEvaluationRunner(client)

    items, cursor = await runner._fetch_batch_with_retry(
        scope="observations",
        filter=None,
        page=1,
        cursor="current-cursor",
        limit=25,
        max_retries=3,
        fields=None,
    )

    assert cursor is None
    assert len(items) == 1
    observation = items[0]
    assert isinstance(observation, ObservationsView)
    assert observation.id == "observation-id"
    assert observation.trace_id == "trace-id"
    assert observation.input == {"question": "hello"}
    assert observation.output == "answer"

    kwargs = client.api.observations.get_many.call_args.kwargs
    assert kwargs["cursor"] == "current-cursor"
    assert kwargs["filter"] == "[]"
    assert "io" in kwargs["fields"].split(",")


def test_resume_filter_uses_v2_start_time_column() -> None:
    assert BatchEvaluationRunner._get_timestamp_field_for_scope("traces") == "startTime"
    assert (
        BatchEvaluationRunner._get_timestamp_field_for_scope("observations")
        == "startTime"
    )


@pytest.mark.asyncio
async def test_falls_back_to_v3_read_api_when_v2_is_unavailable() -> None:
    client = MagicMock()
    client.api.observations.get_many.side_effect = NotFoundError(body="not found")
    legacy_observation = MagicMock(spec=ObservationsView)
    client.api.legacy.observations_v1.get_many.return_value = SimpleNamespace(
        data=[legacy_observation]
    )
    runner = BatchEvaluationRunner(client)

    items, cursor = await runner._fetch_batch_with_retry(
        scope="observations",
        filter=None,
        page=1,
        cursor=None,
        limit=1,
        max_retries=3,
        fields=None,
    )

    assert items == [legacy_observation]
    assert cursor == runner._LEGACY_PAGINATION_CURSOR

    client.api.observations.get_many.reset_mock()
    await runner._fetch_batch_with_retry(
        scope="observations",
        filter=None,
        page=2,
        cursor=cursor,
        limit=1,
        max_retries=3,
        fields=None,
    )

    client.api.observations.get_many.assert_not_called()
    assert client.api.legacy.observations_v1.get_many.call_args.kwargs["page"] == 2


@pytest.mark.asyncio
async def test_run_uses_v2_cursor_for_next_batch() -> None:
    client = MagicMock()
    client.api.observations.get_many.side_effect = [
        SimpleNamespace(
            data=[_observation(observation_id="first")],
            meta=SimpleNamespace(cursor="next-cursor"),
        ),
        SimpleNamespace(
            data=[_observation(observation_id="second")],
            meta=SimpleNamespace(cursor=None),
        ),
    ]
    runner = BatchEvaluationRunner(client)

    result = await runner.run_async(
        scope="observations",
        mapper=lambda *, item: EvaluatorInputs(
            input=item.input,
            output=item.output,
        ),
        evaluators=[
            lambda **kwargs: Evaluation(name="quality", value=1.0),
        ],
        fetch_batch_size=1,
    )

    assert result.total_items_processed == 2
    assert result.completed is True
    assert [
        call.kwargs["cursor"]
        for call in client.api.observations.get_many.call_args_list
    ] == [None, "next-cursor"]
