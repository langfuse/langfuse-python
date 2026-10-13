"""Tests for ``langfuse.experiment`` — ``RunnerContext`` and ``RegressionError``."""

import asyncio
import inspect
import json
import sqlite3
import typing
from contextlib import closing
from datetime import datetime, timezone
from functools import partial, wraps
from threading import Barrier, get_ident
from typing import get_type_hints
from unittest.mock import MagicMock

import pytest
from opentelemetry import context as otel_context
from opentelemetry import trace as otel_trace_api

from langfuse import Evaluation, RegressionError, RunnerContext
from langfuse._client.attributes import LangfuseOtelSpanAttributes
from langfuse._client.client import Langfuse
from langfuse.api import DatasetItem, DatasetStatus
from langfuse.batch_evaluation import CompositeEvaluatorFunction
from langfuse.experiment import ExperimentItemResult, ExperimentResult, _run_task


def _noop_task(*, item, **kwargs):  # pragma: no cover - never invoked via mock
    return None


def _make_ctx(**kwargs) -> RunnerContext:
    client = MagicMock(spec=Langfuse)
    client.run_experiment.return_value = "result-sentinel"
    return RunnerContext(client=client, **kwargs)


def _task_form(task, form):
    if "callable" in form:
        wrapped_task = task

        class CallableTask:
            @wraps(wrapped_task)
            def __call__(self, **kwargs):
                return wrapped_task(**kwargs)

        task = CallableTask()

    if "partial" in form:
        task = partial(task, prefix="partial")

    return task


@pytest.mark.asyncio
async def test_async_task_stays_on_experiment_event_loop():
    loop = asyncio.get_running_loop()

    async def task(*, item):
        assert asyncio.get_running_loop() is loop
        return item["input"]

    assert await _run_task(task, {"input": "async-result"}) == "async-result"


@pytest.mark.asyncio
async def test_sync_task_returning_coroutine_is_awaited():
    loop = asyncio.get_running_loop()

    async def async_result():
        assert asyncio.get_running_loop() is loop
        return "coroutine-result"

    def task(*, item):
        return async_result()

    assert await _run_task(task, {"input": "question"}) == "coroutine-result"


@pytest.mark.asyncio
@pytest.mark.timeout(3)
@pytest.mark.parametrize("cycle", ["wrapped", "partial"])
async def test_cyclic_wrapper_metadata_keeps_sync_task_execution(cycle):
    caller_thread = get_ident()
    worker_threads = []

    def task(*, item):
        worker_threads.append(get_ident())
        return item["input"]

    task.__wrapped__ = task if cycle == "wrapped" else partial(task)

    assert await _run_task(task, {"input": "cycle-result"}) == "cycle-result"
    assert len(worker_threads) == 1
    assert worker_threads[0] != caller_thread


class TestRunnerContextDefaults:
    def test_context_defaults_flow_through(self):
        ctx_data = [{"input": "a"}]
        ctx_version = datetime(2026, 1, 1)
        ctx = _make_ctx(
            data=ctx_data,
            dataset_version=ctx_version,
            metadata={"sha": "abc123"},
        )

        result = ctx.run_experiment(name="exp", task=_noop_task)

        assert result == "result-sentinel"
        ctx.client.run_experiment.assert_called_once()
        kwargs = ctx.client.run_experiment.call_args.kwargs
        assert kwargs["name"] == "exp"
        assert kwargs["data"] is ctx_data
        assert kwargs["metadata"] == {"sha": "abc123"}
        assert kwargs["_dataset_version"] == ctx_version
        assert kwargs["task"] is _noop_task

    def test_call_overrides_win(self):
        ctx = _make_ctx(
            data=[{"input": "ctx"}],
            dataset_version=datetime(2026, 1, 1),
        )

        override_data = [{"input": "override"}]
        override_version = datetime(2026, 6, 6)
        ctx.run_experiment(
            name="exp",
            task=_noop_task,
            run_name="call-run",
            data=override_data,
            _dataset_version=override_version,
        )

        kwargs = ctx.client.run_experiment.call_args.kwargs
        assert kwargs["name"] == "exp"
        assert kwargs["run_name"] == "call-run"
        assert kwargs["data"] is override_data
        assert kwargs["_dataset_version"] == override_version


class TestRunnerContextMetadataMerge:
    def test_user_keys_win_on_collision(self):
        ctx = _make_ctx(
            data=[{"input": "a"}],
            metadata={"sha": "abc", "branch": "main"},
        )
        ctx.run_experiment(
            name="exp", task=_noop_task, metadata={"sha": "def", "pr": "42"}
        )
        assert ctx.client.run_experiment.call_args.kwargs["metadata"] == {
            "sha": "def",
            "branch": "main",
            "pr": "42",
        }

    def test_context_metadata_only(self):
        ctx = _make_ctx(data=[{"input": "a"}], metadata={"sha": "abc"})
        ctx.run_experiment(name="exp", task=_noop_task)
        assert ctx.client.run_experiment.call_args.kwargs["metadata"] == {"sha": "abc"}

    def test_call_metadata_only(self):
        ctx = _make_ctx(data=[{"input": "a"}])
        ctx.run_experiment(name="exp", task=_noop_task, metadata={"pr": "1"})
        assert ctx.client.run_experiment.call_args.kwargs["metadata"] == {"pr": "1"}

    def test_both_none_stays_none(self):
        ctx = _make_ctx(data=[{"input": "a"}])
        ctx.run_experiment(name="exp", task=_noop_task)
        assert ctx.client.run_experiment.call_args.kwargs["metadata"] is None


class TestRunnerContextLocalItems:
    def test_local_items_pass_through_as_context_default(self):
        items = [{"input": "x", "expected_output": "y"}]
        ctx = _make_ctx(data=items)
        ctx.run_experiment(name="exp", task=_noop_task)
        assert ctx.client.run_experiment.call_args.kwargs["data"] is items

    def test_local_items_pass_through_as_call_override(self):
        ctx = _make_ctx()
        items = [{"input": "x"}]
        ctx.run_experiment(name="exp", task=_noop_task, data=items)
        assert ctx.client.run_experiment.call_args.kwargs["data"] is items


class TestRunnerContextValidation:
    def test_missing_data_raises(self):
        ctx = _make_ctx()
        with pytest.raises(ValueError, match="data"):
            ctx.run_experiment(name="exp", task=_noop_task)


class TestRegressionError:
    def test_is_exception(self):
        result = MagicMock()
        exc = RegressionError(result=result)
        assert isinstance(exc, Exception)
        assert exc.result is result

    def test_default_message(self):
        exc = RegressionError(result=MagicMock())
        assert str(exc) == "Experiment regression detected"
        assert exc.metric is None
        assert exc.value is None
        assert exc.threshold is None

    def test_structured_message(self):
        exc = RegressionError(
            result=MagicMock(), metric="avg_accuracy", value=0.78, threshold=0.9
        )
        assert exc.metric == "avg_accuracy"
        assert exc.value == 0.78
        assert exc.threshold == 0.9
        assert "avg_accuracy" in str(exc)
        assert "0.78" in str(exc)
        assert "0.9" in str(exc)

    def test_free_form_message(self):
        exc = RegressionError(
            result=MagicMock(),
            message="custom explanation",
        )
        assert str(exc) == "custom explanation"

    def test_message_wins_over_structured(self):
        exc = RegressionError(
            result=MagicMock(),
            metric="avg_accuracy",
            value=0.5,
            threshold=0.9,
            message="custom explanation",
        )
        assert str(exc) == "custom explanation"
        assert exc.metric == "avg_accuracy"
        assert exc.value == 0.5
        assert exc.threshold == 0.9

    def test_partial_structured_falls_back_to_default(self):
        """The structured overload requires ``metric`` and ``value`` together.

        If a caller bypasses the type checker and passes only one, we fall
        back to the default message rather than rendering misleading
        ``None`` placeholders in the PR comment.
        """
        exc = RegressionError(result=MagicMock(), metric="avg_accuracy")  # type: ignore[call-overload]
        assert str(exc) == "Experiment regression detected"


class TestSignatureDriftGuard:
    """Fails loudly if ``Langfuse.run_experiment`` grows a parameter that is
    not threaded through ``RunnerContext.run_experiment``.

    ``data`` is the only genuinely relaxed parameter: it is required on the
    client but optional on the RunnerContext so the action can inject it.
    ``run_name`` and ``_dataset_version`` are already ``Optional`` on the
    client and must match as-is. ``name`` is required on both — the action
    supports a directory of experiments, so each script must name itself.
    """

    RELAXED_PARAMS = {"data"}

    # `CompositeEvaluatorFunction` is only imported under TYPE_CHECKING in
    # ``langfuse.experiment`` to break the circular dependency with
    # ``langfuse.batch_evaluation``, so its forward-ref must be resolved
    # explicitly when inspecting annotations.
    LOCALNS = {"CompositeEvaluatorFunction": CompositeEvaluatorFunction}

    def test_no_divergence(self):
        client_param_names = self._param_names(Langfuse.run_experiment)
        ctx_param_names = self._param_names(RunnerContext.run_experiment)

        assert client_param_names == ctx_param_names, (
            "RunnerContext.run_experiment params do not match "
            "Langfuse.run_experiment. Missing: "
            f"{client_param_names - ctx_param_names}. "
            f"Extra: {ctx_param_names - client_param_names}."
        )

        client_hints = get_type_hints(Langfuse.run_experiment)
        ctx_hints = get_type_hints(RunnerContext.run_experiment, localns=self.LOCALNS)

        for name in client_param_names:
            client_ann = client_hints.get(name, inspect.Parameter.empty)
            ctx_ann = ctx_hints.get(name, inspect.Parameter.empty)

            if name in self.RELAXED_PARAMS:
                # RunnerContext version must be Optional[<client_ann>].
                # Already-optional client annotations (``run_name``,
                # ``_dataset_version``) just need to match as-is.
                if self._is_optional(client_ann):
                    assert ctx_ann == client_ann, (
                        f"param `{name}`: expected {client_ann}, got {ctx_ann}"
                    )
                else:
                    assert ctx_ann == typing.Optional[client_ann], (
                        f"param `{name}`: expected Optional[{client_ann}], "
                        f"got {ctx_ann}"
                    )
            else:
                assert ctx_ann == client_ann, (
                    f"param `{name}`: annotation drift — "
                    f"client={client_ann}, context={ctx_ann}"
                )

    @staticmethod
    def _param_names(func) -> set:
        return {name for name in inspect.signature(func).parameters if name != "self"}

    @staticmethod
    def _is_optional(annotation) -> bool:
        origin = typing.get_origin(annotation)
        args = typing.get_args(annotation)
        return origin is typing.Union and type(None) in args


class TestExperimentObservationTree:
    def test_thread_affine_sync_task_records_item_failures(
        self, langfuse_memory_client, caplog
    ):
        with closing(sqlite3.connect(":memory:")) as connection:

            def task(*, item, **kwargs):
                return connection.execute("SELECT ?", (item["input"],)).fetchone()[0]

            result = langfuse_memory_client.run_experiment(
                name="thread-affine-sync-task",
                data=[{"input": "A"}, {"input": "B"}],
                task=task,
                max_concurrency=2,
            )

        assert result.item_results == []
        assert "SQLite objects created in a thread" in caplog.text

    def test_sync_task_can_create_its_own_thread_affine_resource(
        self, langfuse_memory_client, tmp_path
    ):
        database = str(tmp_path / "experiment.sqlite")
        with closing(sqlite3.connect(database)):
            pass

        def task(*, item, **kwargs):
            with closing(sqlite3.connect(database)) as connection:
                return connection.execute("SELECT ?", (item["input"],)).fetchone()[0]

        result = langfuse_memory_client.run_experiment(
            name="worker-owned-sync-task-resource",
            data=[{"input": "A"}, {"input": "B"}],
            task=task,
            max_concurrency=2,
        )

        assert [item.output for item in result.item_results] == ["A", "B"]

    @pytest.mark.skipif(
        not hasattr(inspect, "markcoroutinefunction"),
        reason="explicit coroutine markers require Python 3.12+",
    )
    @pytest.mark.parametrize("task_form", ["marked", "partial", "wrapped"])
    def test_marked_coroutine_factory_runs_on_experiment_loop(
        self, task_form, langfuse_memory_client
    ):
        def sync_target(*, item, prefix):
            return "the wrapped target must not be invoked"

        @wraps(sync_target)
        def factory(*, item, prefix="marked"):
            asyncio.get_running_loop()

            async def result():
                return f"{prefix}:{item['input']}"

            return result()

        marked = inspect.markcoroutinefunction(factory)
        task = marked
        expected_prefix = "marked"
        if task_form == "partial":
            task = partial(marked, prefix="partial")
            expected_prefix = "partial"
        elif task_form == "wrapped":

            @wraps(marked)
            def wrapped(**kwargs):
                return marked(**kwargs)

            task = wrapped

        result = langfuse_memory_client.run_experiment(
            name="marked-coroutine-factory",
            data=[{"input": "question"}],
            task=task,
            max_concurrency=1,
        )
        assert [item.output for item in result.item_results] == [
            f"{expected_prefix}:question"
        ]

    @pytest.mark.parametrize(
        "task_form,expected_prefix",
        [
            ("wrapped", "wrapped"),
            ("partial", "partial"),
            ("callable", "wrapped"),
            ("partial-callable", "partial"),
        ],
    )
    def test_wrapped_async_task_invokes_wrapper_on_event_loop(
        self, task_form, expected_prefix, langfuse_memory_client
    ):
        wrapper_calls = []

        async def native_async(*, item, prefix):
            return f"{prefix}:{item['input']}"

        @wraps(native_async)
        def wrapped(*, item, prefix="wrapped"):
            wrapper_calls.append(asyncio.get_running_loop().is_running())
            return native_async(item=item, prefix=prefix)

        result = langfuse_memory_client.run_experiment(
            name="wrapped-async-loop",
            data=[{"input": "question"}],
            task=_task_form(wrapped, task_form),
            max_concurrency=1,
        )

        assert [item.output for item in result.item_results] == [
            f"{expected_prefix}:question"
        ]
        assert wrapper_calls == [True]

    @pytest.mark.parametrize(
        "task_form,expected_prefix",
        [
            ("wrapped", "wrapped"),
            ("partial", "partial"),
            ("callable", "wrapped"),
            ("partial-callable", "partial"),
        ],
    )
    def test_wrapped_async_task_preserves_wrapper_otel_context(
        self, task_form, expected_prefix, langfuse_memory_client, get_span
    ):
        key = otel_context.create_key("wrapped-task-context")

        async def read_context(*, item, prefix):
            await asyncio.sleep(0)
            with langfuse_memory_client.start_as_current_observation(
                name="wrapped-async-child"
            ):
                return f"{prefix}:{otel_context.get_value(key)}"

        @wraps(read_context)
        def contextual_wrapper(*, item, prefix="wrapped"):
            token = otel_context.attach(
                otel_context.set_value(key, "wrapped-otel-context")
            )

            async def finish():
                try:
                    return await read_context(item=item, prefix=prefix)
                finally:
                    otel_context.detach(token)

            return finish()

        result = langfuse_memory_client.run_experiment(
            name="wrapped-async-context",
            data=[{"input": "question"}],
            task=_task_form(contextual_wrapper, task_form),
            max_concurrency=1,
        )

        assert [item.output for item in result.item_results] == [
            f"{expected_prefix}:wrapped-otel-context"
        ]
        child = get_span("wrapped-async-child")
        parent = get_span("experiment-item-task")
        assert child.parent.span_id == parent.context.span_id
        assert child.context.trace_id == parent.context.trace_id

    def test_sync_tasks_run_concurrently_with_trace_context(
        self, langfuse_memory_client, memory_exporter
    ):
        barrier = Barrier(2, timeout=3)

        def task(*, item):
            with langfuse_memory_client.start_as_current_observation(
                name=f"worker-{item['input']}"
            ):
                barrier.wait()
                return item["input"].upper()

        result = langfuse_memory_client.run_experiment(
            name="sync-task-concurrency",
            data=[{"input": "first"}, {"input": "second"}],
            task=task,
            max_concurrency=2,
        )

        assert [item.output for item in result.item_results] == ["FIRST", "SECOND"]

        spans = memory_exporter.get_finished_spans()
        tasks_by_id = {
            span.context.span_id: span
            for span in spans
            if span.name == "experiment-item-task"
        }
        workers = [span for span in spans if span.name.startswith("worker-")]
        assert len(workers) == 2
        assert len({span.parent.span_id for span in workers}) == 2
        for worker in workers:
            parent = tasks_by_id[worker.parent.span_id]
            assert worker.context.trace_id == parent.context.trace_id
            assert (
                worker.attributes[LangfuseOtelSpanAttributes.EXPERIMENT_ID]
                == result.experiment_id
            )

    def test_failed_task_preserves_experiment_attributes_on_item_run(
        self,
        langfuse_memory_client,
        get_span,
    ):
        def failing_task(**kwargs):
            raise RuntimeError("task failed")

        result = langfuse_memory_client.run_experiment(
            name="task-error",
            data=[{"input": "question", "metadata": {"item": "metadata"}}],
            task=failing_task,
            metadata={"run": "metadata"},
            max_concurrency=1,
        )

        item_run = get_span("experiment-item-run")
        task_span = get_span("experiment-item-task")
        task_span_id = otel_trace_api.format_span_id(task_span.context.span_id)

        assert (
            item_run.attributes[LangfuseOtelSpanAttributes.EXPERIMENT_ID]
            == result.experiment_id
        )
        assert (
            item_run.attributes[LangfuseOtelSpanAttributes.EXPERIMENT_NAME]
            == result.run_name
        )
        assert json.loads(
            item_run.attributes[LangfuseOtelSpanAttributes.EXPERIMENT_METADATA]
        ) == {"run": "metadata"}
        assert json.loads(
            item_run.attributes[LangfuseOtelSpanAttributes.EXPERIMENT_ITEM_METADATA]
        ) == {"item": "metadata"}
        assert (
            item_run.attributes[
                LangfuseOtelSpanAttributes.EXPERIMENT_ITEM_ROOT_OBSERVATION_ID
            ]
            == task_span_id
        )
        assert (
            task_span.attributes[LangfuseOtelSpanAttributes.OBSERVATION_LEVEL]
            == "ERROR"
        )
        assert (
            task_span.attributes[LangfuseOtelSpanAttributes.OBSERVATION_STATUS_MESSAGE]
            == "task failed"
        )
        assert (
            task_span.attributes[LangfuseOtelSpanAttributes.OBSERVATION_OUTPUT]
            == "Error: task failed"
        )

    def test_task_is_metric_root_and_evaluators_are_wrapped(
        self,
        langfuse_memory_client,
        get_span,
        json_attr,
        monkeypatch,
    ):
        create_score = MagicMock()
        monkeypatch.setattr(langfuse_memory_client, "create_score", create_score)

        def task(*, item):
            with langfuse_memory_client.start_as_current_observation(name="task-child"):
                return f"answer:{item['input']}"

        def quality_evaluator(*, input, output, expected_output, metadata):
            with langfuse_memory_client.start_as_current_observation(
                name="evaluator-child"
            ):
                assert input == "question"
                assert output == "answer:question"
                assert expected_output == "answer:question"
                assert metadata == {"item": "metadata"}
                return {"name": "quality", "value": 1.0}

        def aggregate_evaluator(
            *, input, output, expected_output, metadata, evaluations
        ):
            assert len(evaluations) == 1
            return {"name": "aggregate", "value": 1.0}

        langfuse_memory_client.run_experiment(
            name="observation-tree",
            data=[
                {
                    "input": "question",
                    "expected_output": "answer:question",
                    "metadata": {"item": "metadata"},
                }
            ],
            task=task,
            evaluators=[quality_evaluator],
            composite_evaluator=aggregate_evaluator,
            max_concurrency=1,
        )

        item_run = get_span("experiment-item-run")
        task_span = get_span("experiment-item-task")
        task_child = get_span("task-child")
        evaluation_span = get_span("experiment-item-evaluation")
        evaluator_span = get_span("quality_evaluator")
        evaluator_child = get_span("evaluator-child")
        composite_span = get_span("aggregate_evaluator")
        task_span_id = otel_trace_api.format_span_id(task_span.context.span_id)

        assert task_span.parent.span_id == item_run.context.span_id
        assert task_child.parent.span_id == task_span.context.span_id
        assert evaluation_span.parent.span_id == item_run.context.span_id
        assert evaluator_span.parent.span_id == evaluation_span.context.span_id
        assert evaluator_child.parent.span_id == evaluator_span.context.span_id
        assert composite_span.parent.span_id == evaluation_span.context.span_id

        for span in (
            item_run,
            task_span,
            task_child,
            evaluation_span,
            evaluator_span,
            evaluator_child,
            composite_span,
        ):
            assert (
                span.attributes[
                    LangfuseOtelSpanAttributes.EXPERIMENT_ITEM_ROOT_OBSERVATION_ID
                ]
                == task_span_id
            )

        assert (
            item_run.attributes[LangfuseOtelSpanAttributes.OBSERVATION_INPUT]
            == "question"
        )
        assert (
            item_run.attributes[LangfuseOtelSpanAttributes.OBSERVATION_OUTPUT]
            == "answer:question"
        )
        assert (
            task_span.attributes[LangfuseOtelSpanAttributes.OBSERVATION_INPUT]
            == "question"
        )
        assert (
            task_span.attributes[LangfuseOtelSpanAttributes.OBSERVATION_OUTPUT]
            == "answer:question"
        )
        assert (
            task_span.attributes[
                LangfuseOtelSpanAttributes.EXPERIMENT_ITEM_EXPECTED_OUTPUT
            ]
            == "answer:question"
        )
        assert task_span.end_time <= evaluation_span.start_time
        assert evaluation_span.end_time <= item_run.end_time

        assert (
            evaluator_span.attributes[LangfuseOtelSpanAttributes.OBSERVATION_TYPE]
            == "evaluator"
        )
        assert json_attr(
            evaluator_span, LangfuseOtelSpanAttributes.OBSERVATION_INPUT
        ) == {
            "input": "question",
            "output": "answer:question",
            "expected_output": "answer:question",
            "metadata": {"item": "metadata"},
        }
        assert json_attr(
            evaluator_span, LangfuseOtelSpanAttributes.OBSERVATION_OUTPUT
        ) == [{"name": "quality", "value": 1.0}]
        assert json_attr(
            composite_span, LangfuseOtelSpanAttributes.OBSERVATION_OUTPUT
        ) == [{"name": "aggregate", "value": 1.0}]
        assert json_attr(
            evaluation_span, LangfuseOtelSpanAttributes.OBSERVATION_OUTPUT
        ) == {
            "evaluator_count": 2,
            "evaluation_count": 2,
            "failed_evaluator_count": 0,
            "skipped_evaluator_count": 0,
        }

        assert create_score.call_count == 2
        for score_call in create_score.call_args_list:
            assert score_call.kwargs["trace_id"] == otel_trace_api.format_trace_id(
                item_run.context.trace_id
            )
            assert score_call.kwargs["observation_id"] == task_span_id

    def test_failed_evaluator_wrapper_is_marked_error(
        self,
        langfuse_memory_client,
        get_span,
        json_attr,
        monkeypatch,
    ):
        create_score = MagicMock()
        monkeypatch.setattr(langfuse_memory_client, "create_score", create_score)

        def task(*, item):
            return item["input"]

        def failing_evaluator(**kwargs):
            raise RuntimeError("evaluation unavailable")

        def passing_evaluator(**kwargs):
            return Evaluation(name="passing", value=1.0)

        langfuse_memory_client.run_experiment(
            name="evaluator-error",
            data=[{"input": "question"}],
            task=task,
            evaluators=[failing_evaluator, passing_evaluator],
            max_concurrency=1,
        )

        failing_span = get_span("failing_evaluator")
        evaluation_span = get_span("experiment-item-evaluation")

        assert (
            failing_span.attributes[LangfuseOtelSpanAttributes.OBSERVATION_LEVEL]
            == "ERROR"
        )
        assert (
            failing_span.attributes[
                LangfuseOtelSpanAttributes.OBSERVATION_STATUS_MESSAGE
            ]
            == "evaluation unavailable"
        )
        assert json_attr(
            failing_span, LangfuseOtelSpanAttributes.OBSERVATION_OUTPUT
        ) == {"error": "evaluation unavailable"}
        assert json_attr(
            evaluation_span, LangfuseOtelSpanAttributes.OBSERVATION_OUTPUT
        ) == {
            "evaluator_count": 2,
            "evaluation_count": 1,
            "failed_evaluator_count": 1,
            "skipped_evaluator_count": 0,
        }
        create_score.assert_called_once()

    def test_configured_composite_evaluator_is_counted_when_skipped(
        self,
        langfuse_memory_client,
        get_span,
        json_attr,
        monkeypatch,
    ):
        monkeypatch.setattr(langfuse_memory_client, "create_score", MagicMock())
        composite_evaluator = MagicMock()

        def failing_evaluator(**kwargs):
            raise RuntimeError("evaluation unavailable")

        langfuse_memory_client.run_experiment(
            name="skipped-composite-evaluator",
            data=[{"input": "question"}],
            task=lambda *, item: item["input"],
            evaluators=[failing_evaluator],
            composite_evaluator=composite_evaluator,
            max_concurrency=1,
        )

        evaluation_span = get_span("experiment-item-evaluation")

        composite_evaluator.assert_not_called()
        assert json_attr(
            evaluation_span, LangfuseOtelSpanAttributes.OBSERVATION_OUTPUT
        ) == {
            "evaluator_count": 2,
            "evaluation_count": 0,
            "failed_evaluator_count": 1,
            "skipped_evaluator_count": 1,
        }


def _dataset_item(*, item_id: str, dataset_id: str, input: str) -> DatasetItem:
    return DatasetItem(
        id=item_id,
        status=DatasetStatus.ACTIVE,
        input=input,
        expected_output=f"expected {input}",
        metadata=None,
        source_trace_id=None,
        source_observation_id=None,
        dataset_id=dataset_id,
        dataset_name="dataset",
        media_references=[],
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        updated_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


def _run_scores(create_score: MagicMock) -> list:
    return [
        call.kwargs
        for call in create_score.call_args_list
        if call.kwargs.get("dataset_run_id") is not None
    ]


class TestExperimentRunIdentity:
    @pytest.mark.parametrize(
        ("project_id", "dataset_id", "run_name", "expected"),
        [
            ("p", "d", "r", "eab007015b1c6f77"),
            ("proj-ü", "ds", 'Run – ✓ "q"', "8d4425c67052ced3"),
            # Shared with the JS SDK and the platform; keep these in sync.
            (
                "7a88fb47-b4e2-43b8-a06c-a5ce950dc53a",
                "cm9x1dataset0000000000001",
                "my-run",
                "a1164c0f238e4173",
            ),
            (
                "7a88fb47-b4e2-43b8-a06c-a5ce950dc53a",
                "cm9x1dataset0000000000001",
                'Läufe "v2" 🚀 – 2026-10-06T12:00:00.000Z',
                "dce21641128b88a8",
            ),
        ],
    )
    def test_dataset_experiment_id_matches_server_derivation(
        self, langfuse_memory_client, project_id, dataset_id, run_name, expected
    ):
        assert (
            langfuse_memory_client._create_experiment_id(
                project_id=project_id, dataset_id=dataset_id, run_name=run_name
            )
            == expected
        )

    def test_dataset_experiment_without_project_id_uses_random_id(
        self, langfuse_memory_client, caplog
    ):
        ids = {
            langfuse_memory_client._create_experiment_id(
                project_id=None, dataset_id="d", run_name="r"
            )
            for _ in range(2)
        }

        assert len(ids) == 2
        assert all(len(i) == 16 for i in ids)
        assert "random experiment id" in caplog.text

    def test_dataset_run_uses_stable_id_url_and_version(
        self, langfuse_memory_client, find_spans, monkeypatch
    ):
        create_score = MagicMock()
        monkeypatch.setattr(langfuse_memory_client, "create_score", create_score)
        monkeypatch.setattr(langfuse_memory_client, "_get_project_id", lambda: "p")
        version = datetime(2026, 2, 3, 4, 5, 6, tzinfo=timezone.utc)

        result = langfuse_memory_client.run_experiment(
            name="exp",
            run_name="r",
            data=[
                _dataset_item(item_id="i1", dataset_id="d", input="a"),
                _dataset_item(item_id="i2", dataset_id="d", input="b"),
            ],
            task=lambda *, item, **kwargs: item.input,
            run_evaluators=[lambda **kwargs: Evaluation(name="run", value=1.0)],
            _dataset_version=version,
        )
        langfuse_memory_client.flush()

        assert result.experiment_id == "eab007015b1c6f77"
        assert (
            result.experiment_url
            == "http://test-host/project/p/experiments/results?baseline=eab007015b1c6f77"
        )
        assert {r.experiment_id for r in result.item_results} == {result.experiment_id}

        spans = find_spans("experiment-item-run") + find_spans("experiment-item-task")
        assert len(spans) == 4
        for span in spans:
            assert (
                span.attributes[LangfuseOtelSpanAttributes.EXPERIMENT_ID]
                == result.experiment_id
            )
            assert (
                span.attributes[LangfuseOtelSpanAttributes.EXPERIMENT_ITEM_VERSION]
                == "2026-02-03T04:05:06Z"
            )

        run_scores = _run_scores(create_score)
        assert len(run_scores) == 1
        assert run_scores[0]["dataset_run_id"] == result.experiment_id
        assert run_scores[0]["name"] == "run"
        assert "trace_id" not in run_scores[0]
        assert "observation_id" not in run_scores[0]
        assert "session_id" not in run_scores[0]

    def test_local_run_persists_run_scores_without_item_version(
        self, langfuse_memory_client, find_spans, monkeypatch
    ):
        create_score = MagicMock()
        monkeypatch.setattr(langfuse_memory_client, "create_score", create_score)
        monkeypatch.setattr(langfuse_memory_client, "_get_project_id", lambda: "p")

        result = langfuse_memory_client.run_experiment(
            name="exp",
            run_name="r",
            data=[{"input": "a"}],
            task=lambda *, item, **kwargs: item["input"],
            run_evaluators=[lambda **kwargs: Evaluation(name="run", value=1.0)],
            _dataset_version=datetime(2026, 2, 3, tzinfo=timezone.utc),
        )
        langfuse_memory_client.flush()

        assert len(result.experiment_id) == 16
        assert result.experiment_id != "eab007015b1c6f77"
        assert result.experiment_url == (
            f"http://test-host/project/p/experiments/results?baseline={result.experiment_id}"
        )
        assert [s["dataset_run_id"] for s in _run_scores(create_score)] == [
            result.experiment_id
        ]
        for span in find_spans("experiment-item-run") + find_spans(
            "experiment-item-task"
        ):
            assert LangfuseOtelSpanAttributes.EXPERIMENT_ITEM_VERSION not in (
                span.attributes or {}
            )

    def test_unresolvable_project_id_yields_no_url(
        self, langfuse_memory_client, monkeypatch
    ):
        def fail():
            raise RuntimeError("no network")

        monkeypatch.setattr(langfuse_memory_client, "create_score", MagicMock())
        monkeypatch.setattr(langfuse_memory_client, "_get_project_id", fail)

        result = langfuse_memory_client.run_experiment(
            name="exp",
            data=[_dataset_item(item_id="i1", dataset_id="d", input="a")],
            task=lambda *, item, **kwargs: item.input,
        )

        assert len(result.item_results) == 1
        assert len(result.experiment_id) == 16
        assert result.experiment_url is None


class TestExperimentResultAliases:
    def test_dataset_run_fields_are_deprecated_aliases(self):
        item_result = ExperimentItemResult(
            item={"input": "a"},
            output="a",
            evaluations=[],
            trace_id="t",
            experiment_id="e",
        )
        result = ExperimentResult(
            name="exp",
            run_name="r",
            description=None,
            item_results=[item_result],
            run_evaluations=[],
            experiment_id="e",
            experiment_url="http://host/project/p/experiments/results?baseline=e",
        )

        with pytest.warns(DeprecationWarning, match="experiment_id"):
            assert result.dataset_run_id == "e"
        with pytest.warns(DeprecationWarning, match="experiment_url"):
            assert result.dataset_run_url == result.experiment_url
        with pytest.warns(DeprecationWarning, match="experiment_id"):
            assert item_result.dataset_run_id == "e"
        assert f"Experiment:\n   {result.experiment_url}" in result.format()

    def test_legacy_constructor_arguments_still_populate_new_fields(self):
        item_result = ExperimentItemResult(
            item={"input": "a"},
            output="a",
            evaluations=[],
            trace_id="t",
            dataset_run_id="legacy",
        )
        result = ExperimentResult(
            name="exp",
            run_name="r",
            description=None,
            item_results=[item_result],
            run_evaluations=[],
            experiment_id="e",
            dataset_run_url="http://legacy",
        )

        assert item_result.experiment_id == "legacy"
        assert result.experiment_url == "http://legacy"


class TestExperimentRunScoreGating:
    def test_run_scores_skipped_when_all_items_are_sampled_out(
        self, langfuse_memory_client, monkeypatch
    ):
        from opentelemetry.sdk.trace.sampling import ALWAYS_OFF

        create_score = MagicMock()
        monkeypatch.setattr(langfuse_memory_client, "create_score", create_score)
        monkeypatch.setattr(langfuse_memory_client, "_get_project_id", lambda: "p")
        monkeypatch.setattr(
            langfuse_memory_client._resources.tracer_provider, "sampler", ALWAYS_OFF
        )
        monkeypatch.setattr(langfuse_memory_client._otel_tracer, "sampler", ALWAYS_OFF)

        result = langfuse_memory_client.run_experiment(
            name="exp",
            data=[{"input": "a"}, {"input": "b"}],
            task=lambda *, item, **kwargs: item["input"],
            run_evaluators=[lambda **kwargs: Evaluation(name="run", value=1.0)],
        )

        assert [e.name for e in result.run_evaluations] == ["run"]
        assert _run_scores(create_score) == []

    def test_tracing_disabled_skips_project_id_lookup(self, monkeypatch):
        client = Langfuse(
            public_key="pk", secret_key="sk", base_url="http://x", tracing_enabled=False
        )
        get_project_id = MagicMock(return_value="p")
        monkeypatch.setattr(client, "_get_project_id", get_project_id)

        result = client.run_experiment(
            name="exp",
            data=[{"input": "a"}],
            task=lambda *, item, **kwargs: item["input"],
        )

        get_project_id.assert_not_called()
        assert result.experiment_url is None
