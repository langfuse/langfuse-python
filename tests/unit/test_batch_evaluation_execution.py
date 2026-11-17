"""Unit tests for execution policies on batch evaluation.

Uses the local fakes from ``tests/unit/test_batch_evaluation.py``
(``make_runner`` / ``make_observation``); no Langfuse server required.
"""

import asyncio
from typing import Any

import pytest

from langfuse._experiment.policy import (
    ExecutionPolicy,
    FailurePolicy,
    RetryPolicy,
    TimeoutPolicy,
)
from langfuse.batch_evaluation import BatchEvaluationRunner
from langfuse.experiment import Evaluation

from .test_batch_evaluation import make_observation, make_runner, mapper


async def run(runner: BatchEvaluationRunner, **kwargs: Any):
    kwargs.setdefault("mapper", mapper)
    kwargs.setdefault("evaluators", [_length_evaluator])
    return await runner.run_async(**kwargs)


def _length_evaluator(*, output, **kwargs):
    return Evaluation(name="length", value=len(output))


class TestEvaluatorRetry:
    @pytest.mark.asyncio
    async def test_evaluator_retried_after_transient_errors_then_succeeds(self):
        runner, _, client = make_runner([make_observation(1)])
        calls = {"n": 0}

        def flaky_evaluator(*, output, **kwargs):
            calls["n"] += 1
            if calls["n"] < 3:
                raise ConnectionError("transient")
            return Evaluation(name="length", value=len(output))

        result = await run(
            runner,
            evaluators=[flaky_evaluator],
            execution=ExecutionPolicy(
                evaluator_retry=RetryPolicy(
                    max_attempts=5, initial_delay=0, jitter=False
                )
            ),
        )

        assert calls["n"] == 3
        assert result.total_items_processed == 1
        assert result.total_scores_created == 1
        assert result.failures == []
        stats = {s.name: s for s in result.evaluator_stats}
        # Stats keep their per-call (not per-attempt) accounting.
        assert stats["flaky_evaluator"].successful_runs == 1
        assert stats["flaky_evaluator"].failed_runs == 0
        assert result.execution is not None
        # 1 mapper attempt + 3 evaluator attempts.
        assert result.execution.total_attempts == 4
        assert result.execution.retried_items == 1
        assert result.execution.aborted is False
        client.create_score.assert_called_once()

    @pytest.mark.asyncio
    async def test_evaluator_non_retryable_fails_once(self):
        runner, _, _ = make_runner([make_observation(1)])
        calls = {"n": 0}

        def broken_evaluator(*, output, **kwargs):
            calls["n"] += 1
            raise ValueError("bad input")

        result = await run(
            runner,
            evaluators=[broken_evaluator],
            execution=ExecutionPolicy(
                evaluator_retry=RetryPolicy(
                    max_attempts=5, initial_delay=0, jitter=False
                )
            ),
        )

        assert calls["n"] == 1
        assert result.total_evaluations_failed == 1
        assert len(result.failures) == 1
        failure = result.failures[0]
        assert failure.stage == "evaluator"
        assert failure.error_type == "ValueError"
        assert failure.evaluator_name == "broken_evaluator"
        assert result.execution is not None
        # 1 mapper attempt + 1 evaluator attempt.
        assert result.execution.total_attempts == 2


class TestEvaluatorTimeout:
    @pytest.mark.asyncio
    async def test_evaluator_timeout_recorded_item_still_scored(self):
        runner, _, client = make_runner([make_observation(1)])

        async def slow_evaluator(*, output, **kwargs):
            await asyncio.sleep(5)
            return Evaluation(name="slow", value=1)

        result = await run(
            runner,
            evaluators=[slow_evaluator, _length_evaluator],
            execution=ExecutionPolicy(timeouts=TimeoutPolicy(evaluator_timeout=0.05)),
        )

        assert result.total_items_processed == 1
        # The slow evaluator failed, the fast one still scored the item.
        assert result.total_scores_created == 1
        assert [e.name for e in result.item_evaluations["obs-1"]] == ["length"]
        assert len(result.failures) == 1
        failure = result.failures[0]
        assert failure.stage == "evaluator"
        assert failure.error_type == "TimeoutError"
        assert failure.evaluator_name == "slow_evaluator"
        assert client.create_score.call_count == 1


class TestMapperFailures:
    @pytest.mark.asyncio
    async def test_mapper_failure_recorded(self):
        runner, _, _ = make_runner([make_observation(1)])
        calls = {"n": 0}

        def failing_mapper(*, item):
            calls["n"] += 1
            raise ConnectionError("mapper boom")

        result = await run(
            runner,
            mapper=failing_mapper,
            execution=ExecutionPolicy(
                evaluator_retry=RetryPolicy(
                    max_attempts=3, initial_delay=0, jitter=False
                )
            ),
        )

        assert calls["n"] == 3  # mapper retried per the policy
        assert result.total_items_failed == 1
        assert result.failed_item_ids == ["obs-1"]
        assert len(result.failures) == 1
        failure = result.failures[0]
        assert failure.stage == "mapper"
        assert failure.error_type == "ConnectionError"
        assert failure.item_key == "obs-1"

    @pytest.mark.asyncio
    async def test_missing_trace_id_recorded_as_validation(self):
        runner, _, _ = make_runner([make_observation(2, trace_id=None)])

        result = await run(runner, execution=ExecutionPolicy())

        assert result.total_items_failed == 1
        assert result.error_summary == {"ValueError": 1}
        assert len(result.failures) == 1
        assert result.failures[0].stage == "validation"

    @pytest.mark.asyncio
    async def test_composite_failure_recorded_item_still_processed(self):
        runner, _, client = make_runner([make_observation(1)])

        def failing_composite(*, evaluations, **kwargs):
            raise RuntimeError("composite boom")

        result = await run(
            runner,
            composite_evaluator=failing_composite,
            execution=ExecutionPolicy(),
        )

        assert result.total_items_processed == 1
        assert result.total_scores_created == 1
        assert result.total_composite_scores_created == 0
        assert len(result.failures) == 1
        assert result.failures[0].stage == "composite"


class TestFailureBudget:
    @pytest.mark.asyncio
    async def test_budget_stops_paging_and_token_resumes_from_cursor(self):
        runner, api, _ = make_runner([make_observation(i) for i in range(6)])

        def failing_evaluator(*, output, **kwargs):
            raise ConnectionError("judge down")

        first = await run(
            runner,
            evaluators=[failing_evaluator],
            fetch_batch_size=2,
            execution=ExecutionPolicy(failures=FailurePolicy(max_failures=2)),
        )

        # Two items on the first page; the evaluator fails on both, the items
        # themselves still count as processed (evaluator failures are
        # tolerated), then the budget stops further paging.
        assert first.completed is False
        assert first.total_items_fetched == 2
        assert first.total_items_processed == 2
        assert len(api.calls) == 1
        assert first.execution is not None
        assert first.execution.aborted is True
        assert "failure budget" in (first.execution.abort_reason or "")
        token = first.resume_token
        assert token is not None
        assert token.cursor == "2"
        assert token.last_processed_id == "obs-4"
        # Both items were processed (evaluator failures do not fail items).
        assert token.items_processed == 2

        # The token resumes after the aborted page.
        resumed = await run(
            runner,
            fetch_batch_size=2,
            resume_from=token,
            execution=ExecutionPolicy(failures=FailurePolicy(max_failures=100)),
        )

        assert resumed.completed is True
        assert set(resumed.item_evaluations) == {"obs-3", "obs-2", "obs-1", "obs-0"}

    @pytest.mark.asyncio
    async def test_budget_counts_evaluator_failures_across_pages(self):
        runner, api, _ = make_runner([make_observation(i) for i in range(6)])

        def half_failing_evaluator(*, output, **kwargs):
            if int(output.split()[-1]) % 2 == 0:
                raise ConnectionError("flaky judge")
            return Evaluation(name="length", value=len(output))

        result = await run(
            runner,
            evaluators=[half_failing_evaluator],
            fetch_batch_size=2,
            execution=ExecutionPolicy(
                failures=FailurePolicy(max_failures=2, on_threshold="stop"),
            ),
        )

        # Newest first: page 1 = obs-5 (ok) + obs-4 (fail); page 2 = obs-3
        # (ok) + obs-2 (fail) -> 2 evaluator failures -> stop before page 3.
        # Evaluator failures do not fail the item, so all 4 attempted items
        # count as processed.
        assert result.execution is not None
        assert result.execution.aborted is True
        assert len(result.failures) == 2
        assert result.total_items_processed == 4
        assert len(api.calls) == 2


class TestRunTimeout:
    @pytest.mark.asyncio
    async def test_run_timeout_stops_between_pages(self):
        runner, api, _ = make_runner([make_observation(i) for i in range(6)])

        async def slow_evaluator(*, output, **kwargs):
            await asyncio.sleep(0.4)
            return Evaluation(name="length", value=len(output))

        result = await run(
            runner,
            evaluators=[slow_evaluator],
            fetch_batch_size=2,
            execution=ExecutionPolicy(timeouts=TimeoutPolicy(run_timeout=0.2)),
        )

        # The first page takes ~0.4s, which already exceeds the 0.2s budget,
        # so the run stops before fetching the second page.
        assert result.execution is not None
        assert result.execution.aborted is True
        assert "run timeout" in (result.execution.abort_reason or "")
        assert result.completed is False
        assert result.resume_token is not None
        assert result.total_items_processed == 2
        assert len(api.calls) == 1


class TestDefaultBehaviour:
    @pytest.mark.asyncio
    async def test_no_policy_matches_previous_behaviour(self):
        runner, api, client = make_runner([make_observation(i) for i in range(5)])

        result = await run(runner, fetch_batch_size=2)

        assert result.completed is True
        assert result.total_items_processed == 5
        assert result.total_scores_created == 5
        assert result.failures == []
        assert result.execution is not None
        assert result.execution.aborted is False
        assert result.execution.total_attempts == 0
        assert result.execution.succeeded == 5
        assert result.execution.failures_by_stage == {}
        assert [call["cursor"] for call in api.calls] == [None, "2", "4"]
        client.flush.assert_called_once()

    @pytest.mark.asyncio
    async def test_fetch_failure_without_policy_unchanged(self):
        runner, api, _ = make_runner([make_observation(i) for i in range(6)])
        api.fail_on_call = 2

        result = await run(runner, fetch_batch_size=2, max_retries=0)

        assert result.completed is False
        assert result.resume_token is not None
        assert result.execution is not None
        assert result.execution.aborted is False
        assert result.failures == []
