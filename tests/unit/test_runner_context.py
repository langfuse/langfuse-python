"""Tests for ``RunnerContext`` and ``RegressionError`` in ``langfuse.experiment``."""

import inspect
import typing
from datetime import datetime
from typing import get_type_hints
from unittest.mock import MagicMock

import pytest

from langfuse import RegressionError, RunnerContext
from langfuse._client.client import Langfuse
from langfuse.batch_evaluation import CompositeEvaluatorFunction


def _noop_task(*, item, **kwargs):  # pragma: no cover - never invoked via mock
    return None


def _make_ctx(**kwargs) -> RunnerContext:
    client = MagicMock(spec=Langfuse)
    client.run_experiment.return_value = "result-sentinel"
    return RunnerContext(client=client, **kwargs)


class TestRunnerContextDefaults:
    def test_context_defaults_flow_through(self):
        ctx_data = [{"input": "a"}]
        ctx_version = datetime(2026, 1, 1)
        ctx = _make_ctx(
            data=ctx_data,
            dataset_version=ctx_version,
            name="ctx-name",
            run_name="ctx-run",
            metadata={"sha": "abc123"},
        )

        result = ctx.run_experiment(task=_noop_task)

        assert result == "result-sentinel"
        ctx.client.run_experiment.assert_called_once()
        kwargs = ctx.client.run_experiment.call_args.kwargs
        assert kwargs["name"] == "ctx-name"
        assert kwargs["run_name"] == "ctx-run"
        assert kwargs["data"] is ctx_data
        assert kwargs["metadata"] == {"sha": "abc123"}
        assert kwargs["_dataset_version"] == ctx_version
        assert kwargs["task"] is _noop_task

    def test_call_overrides_win(self):
        ctx = _make_ctx(
            data=[{"input": "ctx"}],
            dataset_version=datetime(2026, 1, 1),
            name="ctx-name",
            run_name="ctx-run",
        )

        override_data = [{"input": "override"}]
        override_version = datetime(2026, 6, 6)
        ctx.run_experiment(
            task=_noop_task,
            name="call-name",
            run_name="call-run",
            data=override_data,
            _dataset_version=override_version,
        )

        kwargs = ctx.client.run_experiment.call_args.kwargs
        assert kwargs["name"] == "call-name"
        assert kwargs["run_name"] == "call-run"
        assert kwargs["data"] is override_data
        assert kwargs["_dataset_version"] == override_version


class TestRunnerContextMetadataMerge:
    def test_user_keys_win_on_collision(self):
        ctx = _make_ctx(
            data=[{"input": "a"}],
            name="n",
            metadata={"sha": "abc", "branch": "main"},
        )
        ctx.run_experiment(task=_noop_task, metadata={"sha": "def", "pr": "42"})
        assert ctx.client.run_experiment.call_args.kwargs["metadata"] == {
            "sha": "def",
            "branch": "main",
            "pr": "42",
        }

    def test_context_metadata_only(self):
        ctx = _make_ctx(
            data=[{"input": "a"}], name="n", metadata={"sha": "abc"}
        )
        ctx.run_experiment(task=_noop_task)
        assert ctx.client.run_experiment.call_args.kwargs["metadata"] == {"sha": "abc"}

    def test_call_metadata_only(self):
        ctx = _make_ctx(data=[{"input": "a"}], name="n")
        ctx.run_experiment(task=_noop_task, metadata={"pr": "1"})
        assert ctx.client.run_experiment.call_args.kwargs["metadata"] == {"pr": "1"}

    def test_both_none_stays_none(self):
        ctx = _make_ctx(data=[{"input": "a"}], name="n")
        ctx.run_experiment(task=_noop_task)
        assert ctx.client.run_experiment.call_args.kwargs["metadata"] is None


class TestRunnerContextLocalItems:
    def test_local_items_pass_through_as_context_default(self):
        items = [{"input": "x", "expected_output": "y"}]
        ctx = _make_ctx(data=items, name="n")
        ctx.run_experiment(task=_noop_task)
        assert ctx.client.run_experiment.call_args.kwargs["data"] is items

    def test_local_items_pass_through_as_call_override(self):
        ctx = _make_ctx(name="n")
        items = [{"input": "x"}]
        ctx.run_experiment(task=_noop_task, data=items)
        assert ctx.client.run_experiment.call_args.kwargs["data"] is items


class TestRunnerContextValidation:
    def test_missing_name_raises(self):
        ctx = _make_ctx(data=[{"input": "a"}])
        with pytest.raises(ValueError, match="name"):
            ctx.run_experiment(task=_noop_task)

    def test_missing_data_raises(self):
        ctx = _make_ctx(name="n")
        with pytest.raises(ValueError, match="data"):
            ctx.run_experiment(task=_noop_task)


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

    def test_user_message_wins(self):
        exc = RegressionError(
            result=MagicMock(),
            metric="avg_accuracy",
            value=0.5,
            threshold=0.9,
            message="custom explanation",
        )
        assert str(exc) == "custom explanation"


class TestSignatureDriftGuard:
    """Fails loudly if ``Langfuse.run_experiment`` grows a parameter that is
    not threaded through ``RunnerContext.run_experiment``.

    The four action-relaxed params (``name``, ``run_name``, ``data``,
    ``_dataset_version``) are allowed to diverge: the RunnerContext variant
    must be the ``Optional[...]`` of the client annotation so the action can
    inject them.
    """

    RELAXED_PARAMS = {"name", "run_name", "data", "_dataset_version"}

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
        ctx_hints = get_type_hints(
            RunnerContext.run_experiment, localns=self.LOCALNS
        )

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
        return {
            name
            for name in inspect.signature(func).parameters
            if name != "self"
        }

    @staticmethod
    def _is_optional(annotation) -> bool:
        origin = typing.get_origin(annotation)
        args = typing.get_args(annotation)
        return origin is typing.Union and type(None) in args
