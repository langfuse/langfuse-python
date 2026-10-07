"""Unit tests for batch evaluation on the v2 observations API."""

import inspect
import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest

from langfuse import Langfuse
from langfuse.api import ObservationsV2Response, ObservationV2
from langfuse.batch_evaluation import (
    DEFAULT_BATCH_EVALUATION_FIELDS,
    BatchEvaluationResumeToken,
    BatchEvaluationRunner,
    EvaluatorInputs,
)
from langfuse.experiment import Evaluation

BASE_TIME = datetime(2026, 1, 1, tzinfo=timezone.utc)


def make_observation(
    index: int,
    *,
    trace_id: Optional[str] = "default",
    is_root: bool = False,
    start_time: Optional[datetime] = None,
    observation_id: Optional[str] = None,
    parent_observation_id: Optional[str] = "default",
) -> ObservationV2:
    return ObservationV2(
        id=observation_id or f"obs-{index}",
        trace_id=f"trace-{index}" if trace_id == "default" else trace_id,
        start_time=start_time or BASE_TIME + timedelta(seconds=index),
        end_time=None,
        project_id="project",
        parent_observation_id=(
            (None if is_root else f"parent-{index}")
            if parent_observation_id == "default"
            else parent_observation_id
        ),
        type="SPAN",
        is_root_observation=is_root,
        input=json.dumps({"question": index}),
        output=f"answer {index}",
    )


class FakeObservationsApi:
    """Serves observations newest first with an index cursor, like the v2 API."""

    def __init__(self, observations: List[ObservationV2]):
        self.observations = sorted(
            observations, key=lambda o: (o.start_time, o.id), reverse=True
        )
        self.calls: List[Dict[str, Any]] = []
        self.fail_on_call: Optional[int] = None

    def get_many(self, **kwargs: Any) -> ObservationsV2Response:
        self.calls.append(kwargs)
        if self.fail_on_call is not None and len(self.calls) == self.fail_on_call:
            raise ConnectionError("fetch failed")

        matching = [o for o in self.observations if self._matches(o, kwargs["filter"])]
        start = int(kwargs["cursor"]) if kwargs["cursor"] else 0
        page = matching[start : start + kwargs["limit"]]
        has_more = start + kwargs["limit"] < len(matching)
        meta = {"cursor": str(start + kwargs["limit"])} if has_more else {}
        return ObservationsV2Response(data=page, meta=meta)

    @staticmethod
    def _matches(observation: ObservationV2, filter_json: Optional[str]) -> bool:
        for condition in json.loads(filter_json) if filter_json else []:
            if condition["column"] == "isRootObservation":
                if observation.is_root_observation is not condition["value"]:
                    return False
            elif condition["column"] == "startTime":
                assert condition["operator"] == "<="
                if observation.start_time > datetime.fromisoformat(condition["value"]):
                    return False
        return True


def make_runner(observations: List[ObservationV2]):
    api = FakeObservationsApi(observations)
    client = SimpleNamespace(
        api=SimpleNamespace(observations=api),
        create_score=MagicMock(),
        flush=MagicMock(),
    )
    return BatchEvaluationRunner(client), api, client  # type: ignore[arg-type]


def mapper(*, item: ObservationV2) -> EvaluatorInputs:
    return EvaluatorInputs(input=item.input, output=item.output, metadata={})


def length_evaluator(*, input, output, **kwargs):
    return Evaluation(name="length", value=len(output))


async def run(runner: BatchEvaluationRunner, **kwargs: Any):
    kwargs.setdefault("scope", "observations")
    kwargs.setdefault("mapper", mapper)
    kwargs.setdefault("evaluators", [length_evaluator])
    return await runner.run_async(**kwargs)


@pytest.mark.asyncio
async def test_observations_scope_paginates_with_cursor_and_scores_observations():
    runner, api, client = make_runner([make_observation(i) for i in range(5)])

    result = await run(runner, fetch_batch_size=2)

    assert [call["cursor"] for call in api.calls] == [None, "2", "4"]
    assert all("page" not in call for call in api.calls)
    assert all(
        call["fields"] == DEFAULT_BATCH_EVALUATION_FIELDS and call["filter"] is None
        for call in api.calls
    )
    assert result.completed is True
    assert result.has_more_items is False
    assert result.resume_token is None
    assert result.total_items_fetched == 5
    assert result.total_items_processed == 5
    assert result.total_scores_created == 5
    assert set(result.item_evaluations) == {f"obs-{i}" for i in range(5)}
    client.create_score.assert_any_call(
        observation_id="obs-3",
        trace_id="trace-3",
        name="length",
        value=len("answer 3"),
        comment=None,
        metadata={},
        data_type=None,
        config_id=None,
    )
    client.flush.assert_called_once()


@pytest.mark.asyncio
async def test_root_observations_scope_filters_roots_and_scores_traces():
    observations = [make_observation(i, is_root=i % 2 == 0) for i in range(6)]
    runner, api, client = make_runner(observations)
    user_filter = [
        {"type": "string", "column": "traceName", "operator": "=", "value": "chat"}
    ]

    result = await run(
        runner,
        scope="root_observations",
        filter=json.dumps(user_filter),
        metadata={"run": "nightly"},
    )

    assert json.loads(api.calls[0]["filter"]) == user_filter + [
        {
            "type": "boolean",
            "column": "isRootObservation",
            "operator": "=",
            "value": True,
        }
    ]
    assert set(result.item_evaluations) == {"obs-0", "obs-2", "obs-4"}
    scored = [call.kwargs for call in client.create_score.call_args_list]
    assert sorted(kwargs["trace_id"] for kwargs in scored) == [
        "trace-0",
        "trace-2",
        "trace-4",
    ]
    assert all("observation_id" not in kwargs for kwargs in scored)
    assert all(kwargs["metadata"] == {"run": "nightly"} for kwargs in scored)


@pytest.mark.asyncio
async def test_mapper_receives_raw_string_io_and_requested_fields():
    runner, api, _ = make_runner([make_observation(1)])
    seen: List[ObservationV2] = []

    def recording_mapper(*, item):
        seen.append(item)
        return mapper(item=item)

    await run(runner, mapper=recording_mapper, fields="core,io")

    assert api.calls[0]["fields"] == "core,io"
    assert isinstance(seen[0], ObservationV2)
    assert seen[0].input == '{"question": 1}'


def test_client_and_runner_share_default_fields():
    client_default = (
        inspect.signature(Langfuse.run_batched_evaluation).parameters["fields"].default
    )
    runner_default = (
        inspect.signature(BatchEvaluationRunner.run_async).parameters["fields"].default
    )

    assert client_default == runner_default == DEFAULT_BATCH_EVALUATION_FIELDS
    assert "io" in DEFAULT_BATCH_EVALUATION_FIELDS.split(",")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"scope": "traces"}, "Invalid scope"),
        ({"filter": "not json"}, "JSON array"),
        ({"filter": '{"tags": ["a"]}'}, "JSON array"),
        (
            {
                "resume_from": BatchEvaluationResumeToken(
                    scope="root_observations",
                    filter=None,
                    last_processed_timestamp="",
                    last_processed_id="",
                    items_processed=0,
                )
            },
            "scope",
        ),
    ],
)
async def test_invalid_arguments_raise_before_fetching(kwargs, message):
    runner, api, _ = make_runner([make_observation(1)])

    with pytest.raises(ValueError, match=message):
        await run(runner, **kwargs)

    assert api.calls == []


@pytest.mark.asyncio
async def test_max_items_aligns_pages_and_resume_continues_without_gaps():
    # Shared start times exercise ties that a timestamp-only resume would skip.
    observations = [
        make_observation(i, start_time=BASE_TIME + timedelta(seconds=i // 3))
        for i in range(10)
    ]
    runner, api, _ = make_runner(observations)

    first = await run(runner, max_items=5, fetch_batch_size=3)

    assert [call["limit"] for call in api.calls] == [3, 2]
    assert first.completed is True
    assert first.has_more_items is True
    assert first.total_items_fetched == 5
    token = first.resume_token
    assert token is not None
    assert token.cursor == "5"
    assert token.items_processed == 5
    assert token.scope == "observations"

    second = await run(runner, resume_from=token, fetch_batch_size=3)

    assert second.resume_token is None
    assert second.has_more_items is False
    assert set(first.item_evaluations).isdisjoint(second.item_evaluations)
    assert set(first.item_evaluations) | set(second.item_evaluations) == {
        f"obs-{i}" for i in range(10)
    }


@pytest.mark.asyncio
async def test_max_items_reached_on_last_page_has_no_resume_token():
    runner, _, _ = make_runner([make_observation(i) for i in range(4)])

    result = await run(runner, max_items=4, fetch_batch_size=2)

    assert result.has_more_items is False
    assert result.resume_token is None


@pytest.mark.asyncio
async def test_fetch_failure_returns_resume_token_for_failed_page():
    runner, api, _ = make_runner([make_observation(i) for i in range(6)])
    api.fail_on_call = 2

    failed = await run(runner, fetch_batch_size=2, max_retries=0)

    assert failed.completed is False
    assert failed.total_items_processed == 2
    token = failed.resume_token
    assert token is not None
    assert token.cursor == "2"
    assert token.last_processed_id == "obs-4"

    api.fail_on_call = None
    resumed = await run(runner, fetch_batch_size=2, resume_from=token)

    assert resumed.completed is True
    assert set(resumed.item_evaluations) == {"obs-3", "obs-2", "obs-1", "obs-0"}
    assert api.calls[-1]["request_options"] == {"max_retries": 3}


@pytest.mark.asyncio
async def test_resume_without_cursor_falls_back_to_start_time_bound():
    tied_time = BASE_TIME + timedelta(seconds=3)
    runner, api, _ = make_runner(
        [make_observation(i) for i in range(5)]
        + [make_observation(9, observation_id="obs-tied", start_time=tied_time)]
    )
    token = BatchEvaluationResumeToken(
        scope="observations",
        filter=None,
        last_processed_timestamp=(BASE_TIME + timedelta(seconds=3)).isoformat(),
        last_processed_id="obs-3",
        items_processed=2,
    )

    result = await run(runner, resume_from=token)

    assert json.loads(api.calls[0]["filter"]) == [
        {
            "type": "datetime",
            "column": "startTime",
            "operator": "<=",
            "value": token.last_processed_timestamp,
        }
    ]
    assert api.calls[0]["cursor"] is None
    assert set(result.item_evaluations) == {"obs-tied", "obs-2", "obs-1", "obs-0"}


@pytest.mark.asyncio
async def test_resume_reuses_the_token_filter_and_rejects_a_different_one():
    runner, api, _ = make_runner([make_observation(i) for i in range(4)])
    token_filter = json.dumps(
        [{"type": "string", "column": "name", "operator": "=", "value": "x"}]
    )
    token = BatchEvaluationResumeToken(
        scope="observations",
        filter=token_filter,
        cursor="2",
        last_processed_timestamp=BASE_TIME.isoformat(),
        last_processed_id="obs-2",
        items_processed=2,
    )
    api._matches = lambda observation, filter_json: True  # type: ignore[method-assign]

    await run(runner, resume_from=token)
    assert json.loads(api.calls[0]["filter"]) == json.loads(token_filter)

    with pytest.raises(ValueError, match="different filter"):
        await run(runner, resume_from=token, filter="[]")


@pytest.mark.asyncio
async def test_root_observations_scope_prefers_the_physical_root_of_a_trace():
    runner, _, client = make_runner(
        [
            make_observation(0, trace_id="t1", is_root=True),
            # SDK-marked app root below the physical root of the same trace.
            make_observation(
                1, trace_id="t1", is_root=True, parent_observation_id="obs-0"
            ),
        ]
    )

    result = await run(runner, scope="root_observations")

    assert [c.kwargs["trace_id"] for c in client.create_score.call_args_list] == ["t1"]
    assert set(result.item_evaluations) == {"obs-0"}


@pytest.mark.asyncio
async def test_root_observations_scope_scores_a_trace_once_across_pages():
    # Sibling app roots under a parent that was not exported, on separate pages.
    runner, _, client = make_runner(
        [
            make_observation(
                i, trace_id="t2", is_root=True, parent_observation_id="hidden"
            )
            for i in range(3)
        ]
    )

    result = await run(runner, scope="root_observations", fetch_batch_size=1)

    assert [c.kwargs["trace_id"] for c in client.create_score.call_args_list] == ["t2"]
    assert set(result.item_evaluations) == {"obs-2"}


@pytest.mark.asyncio
async def test_observation_scores_can_be_duplicated_onto_trace():
    runner, _, client = make_runner([make_observation(1)])

    result = await run(runner, _add_observation_scores_to_trace=True)

    assert result.total_scores_created == 2
    targets = [
        (call.kwargs.get("observation_id"), call.kwargs["trace_id"])
        for call in client.create_score.call_args_list
    ]
    assert targets == [("obs-1", "trace-1"), (None, "trace-1")]


@pytest.mark.asyncio
async def test_item_failures_and_evaluator_failures_are_tracked():
    observations = [make_observation(1), make_observation(2, trace_id=None)]
    runner, _, client = make_runner(observations)

    def failing_evaluator(**kwargs):
        raise RuntimeError("boom")

    def composite(*, evaluations, **kwargs):
        return Evaluation(name="composite", value=len(evaluations))

    result = await run(
        runner,
        evaluators=[length_evaluator, failing_evaluator],
        composite_evaluator=composite,
    )

    assert result.total_items_processed == 1
    assert result.failed_item_ids == ["obs-2"]
    assert result.error_summary == {"ValueError": 1}
    assert result.total_evaluations_failed == 1
    assert result.total_composite_scores_created == 1
    assert [e.name for e in result.item_evaluations["obs-1"]] == [
        "length",
        "composite",
    ]
    stats = {s.name: s for s in result.evaluator_stats}
    assert stats["failing_evaluator"].failed_runs == 1
    assert stats["length_evaluator"].successful_runs == 1
    assert client.create_score.call_count == 2


@pytest.mark.asyncio
async def test_async_mapper_and_evaluator_are_awaited():
    runner, _, _ = make_runner([make_observation(1)])

    async def async_mapper(*, item):
        return mapper(item=item)

    async def async_evaluator(*, output, **kwargs):
        return [Evaluation(name="a", value=1), Evaluation(name="b", value=2)]

    result = await run(runner, mapper=async_mapper, evaluators=[async_evaluator])

    assert result.total_scores_created == 2
