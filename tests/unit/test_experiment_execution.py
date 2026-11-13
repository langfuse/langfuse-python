"""Unit tests for the experiment execution policies and executor.

All tests are deterministic and run against the in-memory span exporter; no
Langfuse server or model provider is contacted.
"""

import asyncio
import json
import os
import random
import re
import time
from datetime import datetime, timezone
from typing import List
from unittest.mock import MagicMock

import pytest

from langfuse import (
    AttemptRecord,
    Evaluation,
    ExecutionPolicy,
    FailurePolicy,
    RetryPolicy,
    RunnerContext,
    TimeoutPolicy,
    is_retryable_error,
)
from langfuse._client.attributes import LangfuseOtelSpanAttributes
from langfuse._client.client import Langfuse
from langfuse._client.datasets import DatasetClient
from langfuse._experiment.checkpoint import (
    CheckpointItem,
    ExperimentCheckpoint,
    ExperimentFingerprint,
    FileCheckpointStore,
    InMemoryCheckpointStore,
    compute_data_digest,
    experiment_checkpoint_key,
)
from langfuse._experiment.errors import (
    CheckpointCorruptedError,
    CheckpointMismatchError,
)
from langfuse._experiment.executor import (
    _run_with_retry,
)
from langfuse._experiment.item_key import (
    experiment_item_id_for,
    local_item_input_hash,
    make_item_keys,
)
from langfuse._experiment.policy import _parse_retry_after, _retry_after_from_exc
from langfuse.api import DatasetItem, DatasetStatus

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _dataset_item(*, item_id: str, dataset_id: str, input: str) -> DatasetItem:
    return DatasetItem(
        id=item_id,
        status=DatasetStatus.ACTIVE,
        input=input,
        expected_output="expected %s" % input,
        metadata=None,
        source_trace_id=None,
        source_observation_id=None,
        dataset_id=dataset_id,
        dataset_name="dataset",
        media_references=[],
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        updated_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


class _FakeStatus:
    """Minimal exception carrying a ``status_code`` attribute."""

    def __init__(self, status_code: int):
        self.status_code = status_code


class _FakeResponse:
    def __init__(self, status_code: int, headers=None):
        self.status_code = status_code
        self.headers = headers or {}


class _FakeHTTPError(Exception):
    def __init__(self, status_code: int, headers=None):
        self.response = _FakeResponse(status_code, headers)
        super().__init__("http %s" % status_code)


class _NamedAPIError(Exception):
    pass


# Register a retryable class name via MRO.
class APIConnectionError(_NamedAPIError):
    pass


class _NeverRetryValue(ValueError):
    pass


# ---------------------------------------------------------------------------
# Retry predicate
# ---------------------------------------------------------------------------


class TestIsRetryableError:
    @pytest.mark.parametrize(
        "exc,expected",
        [
            # HTTP status codes via status_code attr.
            (_FakeStatus(429), True),
            (_FakeStatus(408), True),
            (_FakeStatus(409), True),
            (_FakeStatus(425), True),
            (_FakeStatus(500), True),
            (_FakeStatus(503), True),
            (_FakeStatus(599), True),
            (_FakeStatus(400), False),
            (_FakeStatus(404), False),
            # HTTP status via response.status_code.
            (_FakeHTTPError(429), True),
            (_FakeHTTPError(500), True),
            (_FakeHTTPError(400), False),
            # Timeout / connection errors.
            (asyncio.TimeoutError(), True),
            (TimeoutError(), True),
            (ConnectionError(), True),
            # Retryable class name via MRO.
            (APIConnectionError(), True),
            # Never retry cancellation / keyboard / user Value/Type errors.
            (asyncio.CancelledError(), False),
            (KeyboardInterrupt(), False),
            (ValueError("bad"), False),
            (TypeError("bad"), False),
            (_NeverRetryValue("bad"), False),
            # Plain non-retryable.
            (RuntimeError("boom"), False),
            (KeyError("missing"), False),
        ],
    )
    def test_predicate_table(self, exc, expected):
        assert is_retryable_error(exc) is expected

    def test_retry_after_seconds_header(self):
        exc = _FakeHTTPError(429, headers={"Retry-After": "5"})
        assert _retry_after_from_exc(exc) == 5.0

    def test_retry_after_http_date_header(self):
        # A future HTTP date should yield a non-negative delay.
        import datetime as _dt
        import email.utils

        future = _dt.datetime.now(_dt.timezone.utc) + _dt.timedelta(seconds=10)
        exc = _FakeHTTPError(
            429, headers={"Retry-After": email.utils.format_datetime(future)}
        )
        delay = _retry_after_from_exc(exc)
        assert delay is not None and 0 <= delay <= 10

    def test_retry_after_missing_header(self):
        exc = _FakeHTTPError(429, headers={})
        assert _retry_after_from_exc(exc) is None

    def test_parse_retry_after_int(self):
        assert _parse_retry_after("3") == 3.0

    def test_parse_retry_after_invalid(self):
        assert _parse_retry_after("not-a-date") is None


# ---------------------------------------------------------------------------
# Backoff math
# ---------------------------------------------------------------------------


class TestBackoffMath:
    def test_no_jitter_uses_exponential_backoff(self):
        policy = RetryPolicy(
            max_attempts=4, initial_delay=1.0, backoff_multiplier=2.0, jitter=False
        )
        rng = random.Random(0)
        # attempt 1 fails -> delay before attempt 2 = 1.0
        assert policy.delay_for(1, None, rng) == 1.0
        assert policy.delay_for(2, None, rng) == 2.0
        assert policy.delay_for(3, None, rng) == 4.0

    def test_max_delay_caps_backoff(self):
        policy = RetryPolicy(
            max_attempts=5,
            initial_delay=10.0,
            backoff_multiplier=10.0,
            max_delay=15.0,
            jitter=False,
        )
        rng = random.Random(0)
        assert policy.delay_for(1, None, rng) == 10.0
        assert policy.delay_for(2, None, rng) == 15.0

    def test_retry_after_overrides_backoff(self):
        policy = RetryPolicy(max_attempts=3, jitter=False)
        rng = random.Random(0)
        assert policy.delay_for(1, 7.0, rng) == 7.0
        # Capped by max_delay.
        assert policy.delay_for(1, 100.0, rng) == 30.0

    def test_jitter_is_within_bounds(self):
        policy = RetryPolicy(
            max_attempts=3, initial_delay=4.0, backoff_multiplier=1.0, jitter=True
        )
        rng = random.Random(123)
        for _ in range(20):
            delay = policy.delay_for(1, None, rng)
            assert 0.0 <= delay <= 4.0

    def test_invalid_max_attempts_raises(self):
        with pytest.raises(ValueError, match="max_attempts"):
            RetryPolicy(max_attempts=0)

    def test_invalid_failure_rate_raises(self):
        with pytest.raises(ValueError, match="max_failure_rate"):
            FailurePolicy(max_failure_rate=1.5)

    def test_resume_requires_checkpoint(self):
        with pytest.raises(ValueError, match="checkpoint"):
            ExecutionPolicy(resume=True)

    def test_resume_force_accepted(self):
        store = InMemoryCheckpointStore()
        ExecutionPolicy(checkpoint=store, resume="force")


# ---------------------------------------------------------------------------
# _run_with_retry helper
# ---------------------------------------------------------------------------


class TestRunWithRetry:
    @pytest.mark.asyncio
    async def test_retries_then_succeeds(self):
        calls = {"n": 0}

        async def factory():
            calls["n"] += 1
            if calls["n"] < 3:
                raise _FakeHTTPError(429)
            return "ok"

        policy = RetryPolicy(max_attempts=5, initial_delay=0, jitter=False)
        rng = random.Random(0)

        sleeps = []

        async def fake_sleep(d):
            sleeps.append(d)

        result, attempts = await _run_with_retry(
            factory,
            policy=policy,
            timeout=None,
            clock=time.monotonic,
            sleep=fake_sleep,
            rng=rng,
            attempt_hook=None,
        )
        assert result == "ok"
        assert calls["n"] == 3
        assert len(attempts) == 3
        assert attempts[0].will_retry is True
        assert attempts[1].will_retry is True
        assert attempts[2].will_retry is False
        assert attempts[2].error_type is None

    @pytest.mark.asyncio
    async def test_non_retryable_fails_once(self):
        calls = {"n": 0}

        async def factory():
            calls["n"] += 1
            raise ValueError("nope")

        policy = RetryPolicy(max_attempts=5, initial_delay=0, jitter=False)
        rng = random.Random(0)
        with pytest.raises(ValueError):
            await _run_with_retry(
                factory,
                policy=policy,
                timeout=None,
                clock=time.monotonic,
                sleep=asyncio.sleep,
                rng=rng,
                attempt_hook=None,
            )
        assert calls["n"] == 1

    @pytest.mark.asyncio
    async def test_timeout_raises_timeout_error(self):
        async def factory():
            await asyncio.sleep(10)
            return "never"

        policy = RetryPolicy(max_attempts=3, initial_delay=0, jitter=False)
        rng = random.Random(0)
        with pytest.raises(asyncio.TimeoutError):
            await _run_with_retry(
                factory,
                policy=policy,
                timeout=0.05,
                clock=time.monotonic,
                sleep=asyncio.sleep,
                rng=rng,
                attempt_hook=None,
            )

    @pytest.mark.asyncio
    async def test_attempt_hook_invoked(self):
        records: List[AttemptRecord] = []

        def hook(record):
            records.append(record)

        async def factory():
            raise _FakeHTTPError(503)

        policy = RetryPolicy(max_attempts=2, initial_delay=0, jitter=False)
        rng = random.Random(0)
        with pytest.raises(_FakeHTTPError):
            await _run_with_retry(
                factory,
                policy=policy,
                timeout=None,
                clock=time.monotonic,
                sleep=asyncio.sleep,
                rng=rng,
                attempt_hook=hook,
            )
        assert len(records) == 2
        assert records[0].attempt == 1 and records[0].will_retry is True
        assert records[1].attempt == 2 and records[1].will_retry is False


# ---------------------------------------------------------------------------
# Item keys
# ---------------------------------------------------------------------------


class TestItemKeys:
    def test_local_item_input_hash_matches_experiment_item_id(self):
        item = {"input": "hello"}
        assert local_item_input_hash("hello") == experiment_item_id_for(item)

    def test_dataset_item_key_is_dataset_item_id(self):
        item = _dataset_item(item_id="i1", dataset_id="d", input="a")
        assert experiment_item_id_for(item) == "i1"

    def test_duplicate_input_keys_disambiguated(self):
        data = [
            {"input": "same", "expected_output": "x"},
            {"input": "same", "expected_output": "y"},
        ]
        keys = make_item_keys(data)
        assert len(keys) == 2
        assert keys[0] != keys[1]

    def test_distinct_inputs_keep_distinct_keys(self):
        data = [{"input": "a"}, {"input": "b"}]
        keys = make_item_keys(data)
        assert len(set(keys)) == 2

    def test_dataset_version_pinned_into_key(self):
        item = _dataset_item(item_id="i1", dataset_id="d", input="a")
        keys = make_item_keys([item], dataset_version="2026-01-01T00:00:00Z")
        assert keys[0] == "i1@v2026-01-01T00:00:00Z"


# ---------------------------------------------------------------------------
# Failure policy
# ---------------------------------------------------------------------------


class TestFailurePolicy:
    def test_max_failures_exceeded(self):
        p = FailurePolicy(max_failures=3)
        assert not p.is_exceeded(2, 10)
        assert p.is_exceeded(3, 10)

    def test_max_failure_rate_respects_min_items(self):
        p = FailurePolicy(max_failure_rate=0.3, min_items_for_rate=10)
        # 50% failure but below min_items_for_rate -> not exceeded.
        assert not p.is_exceeded(2, 4)
        # At min_items_for_rate with > 30% -> exceeded.
        assert p.is_exceeded(4, 10)

    def test_no_budget_never_exceeded(self):
        p = FailurePolicy()
        assert not p.is_exceeded(100, 100)


# ---------------------------------------------------------------------------
# Checkpoint serialization
# ---------------------------------------------------------------------------


class TestCheckpointSerialization:
    def test_roundtrip(self):
        cp = ExperimentCheckpoint(
            experiment_id="exp1",
            experiment_name="exp",
            run_name="r",
            fingerprint=ExperimentFingerprint(
                task_name="task",
                evaluator_names=["e1"],
                data_size=2,
                data_digest="abc",
            ),
            items={
                "k1": CheckpointItem(
                    state="completed",
                    output_json='"out"',
                    evaluations=[{"name": "e1", "value": 1.0}],
                    trace_id="t1",
                    attempts=1,
                ),
                "k2": CheckpointItem(state="failed", error="boom", attempts=2),
            },
            created_at=1.0,
            updated_at=2.0,
            status="running",
        )
        raw = cp.serialize()
        restored = ExperimentCheckpoint.deserialize(raw)
        assert restored.experiment_id == "exp1"
        assert restored.items["k1"].state == "completed"
        assert restored.items["k1"].output_json == '"out"'
        assert restored.items["k2"].state == "failed"
        assert restored.items["k2"].error == "boom"
        assert restored.fingerprint is not None
        assert restored.fingerprint.matches(cp.fingerprint)  # type: ignore[arg-type]

    def test_corrupted_integrity_detected(self):
        cp = ExperimentCheckpoint(experiment_id="exp1")
        raw = cp.serialize()
        # Tamper with the payload (remove the integrity key and re-add a bad one).
        data = json.loads(raw)
        data["experiment_id"] = "tampered"
        bad = json.dumps(data)
        with pytest.raises(CheckpointCorruptedError):
            ExperimentCheckpoint.deserialize(bad)

    def test_fingerprint_mismatch_raises(self):
        cp = ExperimentCheckpoint(
            experiment_id="exp1",
            fingerprint=ExperimentFingerprint(
                task_name="task",
                evaluator_names=[],
                data_size=1,
                data_digest="d1",
            ),
        )
        other = ExperimentFingerprint(
            task_name="other",
            evaluator_names=[],
            data_size=1,
            data_digest="d2",
        )
        with pytest.raises(CheckpointMismatchError):
            cp.verify_fingerprint(other)


class TestFileCheckpointStore:
    def test_atomic_write_and_load(self, tmp_path):
        store = FileCheckpointStore(str(tmp_path))
        cp = ExperimentCheckpoint(experiment_id="exp1", experiment_name="n")
        store.save(cp, "exp1")
        loaded = store.load("exp1")
        assert loaded is not None
        assert loaded.experiment_id == "exp1"

    def test_rejects_path_traversal_in_filename(self, tmp_path):
        store = FileCheckpointStore(str(tmp_path))
        # The key is hashed + sanitized, so traversal characters cannot escape.
        store.save(ExperimentCheckpoint(experiment_id="exp-evil"), "../evil")
        files = os.listdir(str(tmp_path))
        # No file should contain ".." or "/".
        for f in files:
            assert ".." not in f and "/" not in f

    def test_rejects_nul_in_key(self, tmp_path):
        store = FileCheckpointStore(str(tmp_path))
        with pytest.raises(ValueError, match="NUL"):
            store.save(ExperimentCheckpoint(experiment_id="exp"), "ex\x00p")

    def test_survives_exception_mid_write(self, tmp_path, monkeypatch):
        store = FileCheckpointStore(str(tmp_path))
        cp = ExperimentCheckpoint(experiment_id="exp1")

        original_replace = os.replace
        call_state = {"called": False}

        def boom(src, dst):
            call_state["called"] = True
            # Simulate the first os.replace failing; the temp file is cleaned.
            raise OSError("disk full")

        monkeypatch.setattr(os, "replace", boom)
        with pytest.raises(OSError):
            store.save(cp, "exp1")
        assert call_state["called"]
        # No partial checkpoint should be loadable.
        monkeypatch.setattr(os, "replace", original_replace)
        assert store.load("exp1") is None

    def test_debounce_flush(self, tmp_path):
        store = FileCheckpointStore(str(tmp_path), flush_interval_s=1.0)
        assert not store.should_flush(0.0)
        assert store.should_flush(1.0)
        store.mark_flushed(1.0)
        assert not store.should_flush(1.5)
        assert store.should_flush(2.5)


# ---------------------------------------------------------------------------
# Integration: task retry / timeout / failures recorded
# ---------------------------------------------------------------------------


def _run_scores(create_score: MagicMock) -> list:
    return [
        call.kwargs
        for call in create_score.call_args_list
        if call.kwargs.get("dataset_run_id") is not None
    ]


class TestTaskRetryIntegration:
    def test_task_retried_then_succeeds(
        self, langfuse_memory_client, monkeypatch, get_span, find_spans
    ):
        monkeypatch.setattr(langfuse_memory_client, "_get_project_id", lambda: "p")
        monkeypatch.setattr(langfuse_memory_client, "create_score", MagicMock())
        calls = {"n": 0}

        def task(*, item):
            calls["n"] += 1
            if calls["n"] < 3:
                raise _FakeHTTPError(429)
            return "ok:%s" % item["input"]

        policy = ExecutionPolicy(
            task_retry=RetryPolicy(max_attempts=5, initial_delay=0, jitter=False)
        )
        result = langfuse_memory_client.run_experiment(
            name="retry",
            data=[{"input": "a"}],
            task=task,
            execution=policy,
            max_concurrency=1,
        )
        assert calls["n"] == 3
        assert len(result.item_results) == 1
        assert result.item_results[0].output == "ok:a"
        assert result.failures == []
        # Attempt observations emitted for the two failed retries.
        attempt_spans = find_spans("experiment-item-task-attempt")
        assert len(attempt_spans) == 2
        for s in attempt_spans:
            assert (
                s.attributes.get(LangfuseOtelSpanAttributes.OBSERVATION_LEVEL)
                == "WARNING"
            )

    def test_non_retryable_error_fails_once(self, langfuse_memory_client, monkeypatch):
        monkeypatch.setattr(langfuse_memory_client, "_get_project_id", lambda: "p")
        monkeypatch.setattr(langfuse_memory_client, "create_score", MagicMock())
        calls = {"n": 0}

        def task(*, item):
            calls["n"] += 1
            raise ValueError("user error")

        policy = ExecutionPolicy(
            task_retry=RetryPolicy(max_attempts=5, initial_delay=0, jitter=False)
        )
        result = langfuse_memory_client.run_experiment(
            name="noretry",
            data=[{"input": "a"}],
            task=task,
            execution=policy,
            max_concurrency=1,
        )
        assert calls["n"] == 1
        assert result.item_results == []
        assert len(result.failures) == 1
        assert result.failures[0].stage == "task"
        assert result.failures[0].error_type == "ValueError"

    def test_task_timeout_recorded(self, langfuse_memory_client, monkeypatch):
        monkeypatch.setattr(langfuse_memory_client, "_get_project_id", lambda: "p")
        monkeypatch.setattr(langfuse_memory_client, "create_score", MagicMock())

        async def task(*, item):
            await asyncio.sleep(5)
            return "never"

        policy = ExecutionPolicy(
            task_retry=RetryPolicy(max_attempts=1),
            timeouts=TimeoutPolicy(task_timeout=0.05),
        )
        result = langfuse_memory_client.run_experiment(
            name="timeout",
            data=[{"input": "a"}],
            task=task,
            execution=policy,
            max_concurrency=1,
        )
        assert result.item_results == []
        assert len(result.failures) == 1
        assert result.failures[0].stage == "timeout"
        assert result.failures[0].error_type == "TaskTimeoutError"


class TestEvaluatorRetryIntegration:
    def test_evaluator_retry_and_timeout_recorded_item_succeeds(
        self, langfuse_memory_client, monkeypatch
    ):
        monkeypatch.setattr(langfuse_memory_client, "_get_project_id", lambda: "p")
        monkeypatch.setattr(langfuse_memory_client, "create_score", MagicMock())
        calls = {"n": 0}

        def failing_evaluator(*, input, output, expected_output, metadata):
            calls["n"] += 1
            if calls["n"] < 3:
                raise _FakeHTTPError(503)
            raise asyncio.TimeoutError()

        policy = ExecutionPolicy(
            evaluator_retry=RetryPolicy(max_attempts=5, initial_delay=0, jitter=False),
            timeouts=TimeoutPolicy(evaluator_timeout=0.05),
        )
        result = langfuse_memory_client.run_experiment(
            name="evalretry",
            data=[{"input": "a"}],
            task=lambda *, item: item["input"],
            evaluators=[failing_evaluator],
            execution=policy,
            max_concurrency=1,
        )
        # Item still succeeds; evaluator failure recorded.
        assert len(result.item_results) == 1
        assert result.item_results[0].output == "a"
        assert len(result.evaluator_failures) == 1
        assert result.evaluator_failures[0].error_type == "EvaluatorTimeoutError"
        assert result.evaluator_failures[0].evaluator_name == "failing_evaluator"


class TestFailuresRecordedDefaultPath:
    def test_failures_recorded_item_results_unchanged(
        self, langfuse_memory_client, monkeypatch
    ):
        monkeypatch.setattr(langfuse_memory_client, "_get_project_id", lambda: "p")
        monkeypatch.setattr(langfuse_memory_client, "create_score", MagicMock())

        def task(*, item):
            if item["input"] == "fail":
                raise RuntimeError("boom")
            return item["input"]

        result = langfuse_memory_client.run_experiment(
            name="mixed",
            data=[{"input": "ok"}, {"input": "fail"}],
            task=task,
            max_concurrency=2,
        )
        # Only the successful item appears in item_results.
        assert [r.output for r in result.item_results] == ["ok"]
        # The failure is recorded.
        assert len(result.failures) == 1
        assert result.failures[0].error_type == "RuntimeError"
        assert result.execution is not None
        assert result.execution.succeeded == 1
        assert result.execution.failed == 1


class TestRunEvaluatorFailuresKwarg:
    def test_run_evaluator_receives_failures_kwarg(
        self, langfuse_memory_client, monkeypatch
    ):
        monkeypatch.setattr(langfuse_memory_client, "_get_project_id", lambda: "p")
        monkeypatch.setattr(langfuse_memory_client, "create_score", MagicMock())

        def task(*, item):
            if item["input"] == "fail":
                raise RuntimeError("boom")
            return item["input"]

        seen = {}

        def run_eval(*, item_results, failures=None, **kwargs):
            seen["failures"] = failures
            return Evaluation(name="count", value=len(item_results))

        langfuse_memory_client.run_experiment(
            name="kwarg",
            data=[{"input": "ok"}, {"input": "fail"}],
            task=task,
            run_evaluators=[run_eval],
            max_concurrency=2,
        )
        assert seen["failures"] is not None
        assert len(seen["failures"]) == 1

    def test_run_evaluator_without_failures_kwarg_unaffected(
        self, langfuse_memory_client, monkeypatch
    ):
        monkeypatch.setattr(langfuse_memory_client, "_get_project_id", lambda: "p")
        monkeypatch.setattr(langfuse_memory_client, "create_score", MagicMock())

        def task(*, item):
            return item["input"]

        def run_eval(*, item_results, **kwargs):
            # No `failures` should be passed.
            assert "failures" not in kwargs
            return Evaluation(name="count", value=len(item_results))

        result = langfuse_memory_client.run_experiment(
            name="nokwarg",
            data=[{"input": "a"}],
            task=task,
            run_evaluators=[run_eval],
            max_concurrency=1,
        )
        assert len(result.run_evaluations) == 1


# ---------------------------------------------------------------------------
# Failure budget / run timeout / cancellation / resume
# ---------------------------------------------------------------------------


class TestFailureBudget:
    def test_max_failures_stops_scheduling(self, langfuse_memory_client, monkeypatch):
        monkeypatch.setattr(langfuse_memory_client, "_get_project_id", lambda: "p")
        monkeypatch.setattr(langfuse_memory_client, "create_score", MagicMock())
        started = {"n": 0}

        def task(*, item):
            started["n"] += 1
            raise RuntimeError("boom")

        policy = ExecutionPolicy(
            failures=FailurePolicy(max_failures=1, on_threshold="stop"),
        )
        result = langfuse_memory_client.run_experiment(
            name="budget",
            data=[{"input": "a"}, {"input": "b"}, {"input": "c"}],
            task=task,
            execution=policy,
            max_concurrency=1,
        )
        assert result.execution is not None
        assert result.execution.aborted is True
        assert "failure budget" in (result.execution.abort_reason or "")
        # With max_concurrency=1 and max_failures=1, only 1 item is attempted.
        assert started["n"] == 1

    def test_max_failure_rate_stops_scheduling(
        self, langfuse_memory_client, monkeypatch
    ):
        monkeypatch.setattr(langfuse_memory_client, "_get_project_id", lambda: "p")
        monkeypatch.setattr(langfuse_memory_client, "create_score", MagicMock())
        started = {"n": 0}

        def task(*, item):
            started["n"] += 1
            raise RuntimeError("boom")

        policy = ExecutionPolicy(
            failures=FailurePolicy(max_failure_rate=0.3, min_items_for_rate=2),
        )
        result = langfuse_memory_client.run_experiment(
            name="ratebudget",
            data=[{"input": "a"}, {"input": "b"}, {"input": "c"}],
            task=task,
            execution=policy,
            max_concurrency=1,
        )
        assert result.execution is not None
        assert result.execution.aborted is True


class TestRunTimeout:
    def test_run_timeout_aborts(self, langfuse_memory_client, monkeypatch):
        monkeypatch.setattr(langfuse_memory_client, "_get_project_id", lambda: "p")
        monkeypatch.setattr(langfuse_memory_client, "create_score", MagicMock())

        async def task(*, item):
            await asyncio.sleep(0.2)
            return item["input"]

        policy = ExecutionPolicy(timeouts=TimeoutPolicy(run_timeout=0.05))
        result = langfuse_memory_client.run_experiment(
            name="runtimeout",
            data=[{"input": "a"}, {"input": "b"}],
            task=task,
            execution=policy,
            max_concurrency=1,
        )
        assert result.execution is not None
        assert result.execution.aborted is True
        assert "run timeout" in (result.execution.abort_reason or "")


class TestCancellation:
    @pytest.mark.asyncio
    async def test_cancellation_writes_checkpoint_and_reraises(
        self, langfuse_memory_client, monkeypatch, tmp_path
    ):
        monkeypatch.setattr(langfuse_memory_client, "_get_project_id", lambda: "p")
        monkeypatch.setattr(langfuse_memory_client, "create_score", MagicMock())
        store = FileCheckpointStore(str(tmp_path))
        # asyncio.Event (not threading.Event): a threading.Event.wait() awaited
        # in a coroutine blocks the event loop, so the task coroutine would
        # never get a chance to set it and the test deadlocks.
        started = asyncio.Event()

        async def task(*, item):
            started.set()
            await asyncio.sleep(5)
            return item["input"]

        policy = ExecutionPolicy(
            checkpoint=store,
            resume=True,
            task_retry=RetryPolicy(max_attempts=1),
        )

        async def main():
            coro = langfuse_memory_client._run_experiment_async(
                name="cancel",
                run_name="r",
                description=None,
                data=[{"input": "a"}],
                task=task,
                evaluators=[],
                composite_evaluator=None,
                run_evaluators=[],
                max_concurrency=1,
                execution=policy,
            )
            task_obj = asyncio.ensure_future(coro)
            await started.wait()
            task_obj.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task_obj

        await main()
        # The checkpoint is stored under the derived name+run_name key; the
        # experiment id is random for local data and cannot serve as the key.
        cp = store.load(experiment_checkpoint_key("cancel", "r"))
        assert cp is not None
        files = [f for f in os.listdir(str(tmp_path)) if f.endswith(".json")]
        assert len(files) >= 1


class TestResume:
    def test_resume_skips_completed_retries_failed(
        self, langfuse_memory_client, monkeypatch
    ):
        monkeypatch.setattr(langfuse_memory_client, "_get_project_id", lambda: "p")
        monkeypatch.setattr(langfuse_memory_client, "create_score", MagicMock())
        store = InMemoryCheckpointStore()

        # Pre-populate a checkpoint with one completed and one failed item.
        data = [
            {"input": "done", "expected_output": "done"},
            {"input": "fail", "expected_output": "fail"},
            {"input": "new", "expected_output": "new"},
        ]
        keys = make_item_keys(data)
        cp = ExperimentCheckpoint(
            experiment_id="exp-resume",
            experiment_name="resume",
            run_name="r",
            fingerprint=ExperimentFingerprint(
                task_name="task",
                evaluator_names=[],
                data_size=3,
                data_digest=compute_data_digest(keys),
            ),
            items={
                keys[0]: CheckpointItem(
                    state="completed",
                    output_json='"done"',
                    evaluations=[],
                    trace_id="t-done",
                    attempts=1,
                ),
                keys[1]: CheckpointItem(
                    state="failed",
                    error="boom",
                    attempts=1,
                ),
            },
            created_at=1.0,
            updated_at=2.0,
            status="running",
        )
        store.save(cp, "exp-resume")

        task_calls = {"n": 0}

        def task(*, item):
            task_calls["n"] += 1
            return item["input"]

        policy = ExecutionPolicy(checkpoint=store, resume=True)
        result = langfuse_memory_client.run_experiment(
            name="resume",
            run_name="r",
            experiment_id="exp-resume",
            data=data,
            task=task,
            execution=policy,
            max_concurrency=1,
        )
        # The completed item is NOT re-executed; the failed and new ones are.
        assert task_calls["n"] == 2
        assert len(result.item_results) == 3
        # The resumed completed item is rebuilt from the checkpoint.
        resumed = [r for r in result.item_results if r.output == "done"]
        assert len(resumed) == 1
        assert resumed[0].trace_id == "t-done"
        assert result.execution is not None
        assert result.execution.skipped_resumed == 1

    def test_fingerprint_mismatch_raises(self, langfuse_memory_client, monkeypatch):
        monkeypatch.setattr(langfuse_memory_client, "_get_project_id", lambda: "p")
        monkeypatch.setattr(langfuse_memory_client, "create_score", MagicMock())
        store = InMemoryCheckpointStore()
        data = [{"input": "a"}]
        keys = make_item_keys(data)
        cp = ExperimentCheckpoint(
            experiment_id="exp-mismatch",
            experiment_name="m",
            run_name="r",
            fingerprint=ExperimentFingerprint(
                task_name="other_task",
                evaluator_names=[],
                data_size=1,
                data_digest=compute_data_digest(keys),
            ),
            items={},
            created_at=1.0,
            updated_at=2.0,
        )
        store.save(cp, "exp-mismatch")

        def task(*, item):
            return item["input"]

        policy = ExecutionPolicy(checkpoint=store, resume=True)
        with pytest.raises(CheckpointMismatchError):
            langfuse_memory_client.run_experiment(
                name="m",
                run_name="r",
                experiment_id="exp-mismatch",
                data=data,
                task=task,
                execution=policy,
                max_concurrency=1,
            )

    def test_resume_force_ignores_mismatch(self, langfuse_memory_client, monkeypatch):
        monkeypatch.setattr(langfuse_memory_client, "_get_project_id", lambda: "p")
        monkeypatch.setattr(langfuse_memory_client, "create_score", MagicMock())
        store = InMemoryCheckpointStore()
        data = [{"input": "a"}]
        keys = make_item_keys(data)
        cp = ExperimentCheckpoint(
            experiment_id="exp-force",
            experiment_name="m",
            run_name="r",
            fingerprint=ExperimentFingerprint(
                task_name="other_task",
                evaluator_names=[],
                data_size=1,
                data_digest=compute_data_digest(keys),
            ),
            items={},
            created_at=1.0,
            updated_at=2.0,
        )
        store.save(cp, "exp-force")

        def task(*, item):
            return item["input"]

        policy = ExecutionPolicy(checkpoint=store, resume="force")
        result = langfuse_memory_client.run_experiment(
            name="m",
            run_name="r",
            experiment_id="exp-force",
            data=data,
            task=task,
            execution=policy,
            max_concurrency=1,
        )
        assert len(result.item_results) == 1


class TestLocalDataResume:
    """Resume must work for local (non-dataset) data too.

    Local data gets a random experiment id per run, so the checkpoint is
    keyed by a digest of (experiment name, run name) instead, and a resumed
    run adopts the id recorded in the checkpoint.
    """

    @pytest.mark.asyncio
    async def test_crash_mid_run_then_resume_skips_completed_and_adopts_id(
        self, langfuse_memory_client, monkeypatch
    ):
        monkeypatch.setattr(langfuse_memory_client, "_get_project_id", lambda: "p")
        monkeypatch.setattr(langfuse_memory_client, "create_score", MagicMock())
        store = InMemoryCheckpointStore()
        data = [{"input": "a"}, {"input": "b"}, {"input": "c"}]

        started_item3 = asyncio.Event()
        block_item3 = True
        calls = {"n": 0}

        async def task(*, item):
            calls["n"] += 1
            if item["input"] == "c" and block_item3:
                started_item3.set()
                await asyncio.sleep(30)
            return item["input"]

        policy = ExecutionPolicy(checkpoint=store, resume=True)

        async def run_and_cancel():
            # Keyboard-interrupt-style cancellation of the whole run while
            # item "c" is in flight; items "a" and "b" already completed.
            run_task = asyncio.ensure_future(
                langfuse_memory_client._run_experiment_async(
                    name="local-resume",
                    run_name="r",
                    description=None,
                    data=data,
                    task=task,
                    evaluators=[],
                    composite_evaluator=None,
                    run_evaluators=[],
                    max_concurrency=1,
                    execution=policy,
                )
            )
            await started_item3.wait()
            run_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await run_task

        await run_and_cancel()
        calls_after_run1 = calls["n"]
        assert calls_after_run1 == 3

        # The checkpoint lives under the derived name+run_name key, not under
        # the random per-run experiment id.
        key = experiment_checkpoint_key("local-resume", "r")
        cp = store.load(key)
        assert cp is not None
        run1_experiment_id = cp.experiment_id
        assert run1_experiment_id
        assert set(cp.items) == set(make_item_keys(data)[:2])
        assert all(cp.items[k].state == "completed" for k in cp.items)

        # Second run with the same run_name: completed items are skipped
        # (only item "c" executes) and both runs share one experiment id.
        block_item3 = False
        result = langfuse_memory_client.run_experiment(
            name="local-resume",
            run_name="r",
            data=data,
            task=task,
            execution=ExecutionPolicy(checkpoint=store, resume=True),
            max_concurrency=1,
        )
        assert calls["n"] - calls_after_run1 == 1
        assert [r.output for r in result.item_results] == ["a", "b", "c"]
        assert result.experiment_id == run1_experiment_id
        assert {r.experiment_id for r in result.item_results} == {run1_experiment_id}
        assert result.execution is not None
        assert result.execution.skipped_resumed == 2
        assert result.execution.failed == 0

    def test_local_resume_without_run_name_or_experiment_id_raises(
        self, langfuse_memory_client, monkeypatch
    ):
        monkeypatch.setattr(langfuse_memory_client, "_get_project_id", lambda: "p")
        monkeypatch.setattr(langfuse_memory_client, "create_score", MagicMock())
        store = InMemoryCheckpointStore()
        policy = ExecutionPolicy(checkpoint=store, resume=True)

        def task(*, item):
            return item["input"] if isinstance(item, dict) else item.input

        with pytest.raises(ValueError, match="run_name"):
            langfuse_memory_client.run_experiment(
                name="local-resume",
                data=[{"input": "a"}],
                task=task,
                execution=policy,
            )

        # Dataset items have a deterministic experiment id, so resuming them
        # does not require an explicit run_name.
        result = langfuse_memory_client.run_experiment(
            name="dataset-resume",
            data=[_dataset_item(item_id="i1", dataset_id="d", input="a")],
            task=task,
            execution=policy,
        )
        assert len(result.item_results) == 1
        assert result.failures == []

    def test_checkpoint_key_is_stable_and_distinct(self):
        key = experiment_checkpoint_key("exp", "r")
        assert key == experiment_checkpoint_key("exp", "r")
        assert key != experiment_checkpoint_key("exp", "other")
        assert key != experiment_checkpoint_key("other", "r")
        assert len(key) == 32


class TestRetriedItemsCounting:
    def test_item_retried_three_times_counts_once(
        self, langfuse_memory_client, monkeypatch
    ):
        monkeypatch.setattr(langfuse_memory_client, "_get_project_id", lambda: "p")
        monkeypatch.setattr(langfuse_memory_client, "create_score", MagicMock())
        calls = {"n": 0}

        def task(*, item):
            calls["n"] += 1
            if calls["n"] <= 3:
                raise _FakeHTTPError(429)
            return "ok"

        policy = ExecutionPolicy(
            task_retry=RetryPolicy(max_attempts=5, initial_delay=0, jitter=False)
        )
        result = langfuse_memory_client.run_experiment(
            name="retried-count",
            data=[{"input": "a"}],
            task=task,
            execution=policy,
            max_concurrency=1,
        )
        assert result.item_results[0].output == "ok"
        assert result.execution is not None
        # One item was retried, no matter how many retry attempts it used.
        assert result.execution.retried_items == 1
        assert result.execution.total_attempts == 4

    def test_two_items_each_retried_once_count_two(
        self, langfuse_memory_client, monkeypatch
    ):
        monkeypatch.setattr(langfuse_memory_client, "_get_project_id", lambda: "p")
        monkeypatch.setattr(langfuse_memory_client, "create_score", MagicMock())
        calls = {"n": 0}

        def task(*, item):
            calls["n"] += 1
            # First call per item fails, second succeeds (concurrency 1).
            if calls["n"] % 2 == 1:
                raise _FakeHTTPError(429)
            return item["input"]

        policy = ExecutionPolicy(
            task_retry=RetryPolicy(max_attempts=5, initial_delay=0, jitter=False)
        )
        result = langfuse_memory_client.run_experiment(
            name="retried-count-two",
            data=[{"input": "a"}, {"input": "b"}],
            task=task,
            execution=policy,
            max_concurrency=1,
        )
        assert result.execution is not None
        assert result.execution.retried_items == 2
        assert result.execution.total_attempts == 4

    def test_terminal_failure_counts_item_once(
        self, langfuse_memory_client, monkeypatch
    ):
        monkeypatch.setattr(langfuse_memory_client, "_get_project_id", lambda: "p")
        monkeypatch.setattr(langfuse_memory_client, "create_score", MagicMock())

        def task(*, item):
            raise _FakeHTTPError(503)

        policy = ExecutionPolicy(
            task_retry=RetryPolicy(max_attempts=3, initial_delay=0, jitter=False)
        )
        result = langfuse_memory_client.run_experiment(
            name="retried-count-terminal",
            data=[{"input": "a"}],
            task=task,
            execution=policy,
            max_concurrency=1,
        )
        assert result.execution is not None
        assert result.execution.retried_items == 1
        assert result.execution.total_attempts == 3
        assert len(result.failures) == 1


class TestDefaultPathSpanTree:
    def test_no_policy_span_names_and_count_match_pre_change_code(
        self, langfuse_memory_client, monkeypatch, memory_exporter
    ):
        """With no execution policy the run must emit exactly the span tree
        the pre-policy code emitted (same names, same count, no attempt
        observations) and the same item-level score calls."""
        monkeypatch.setattr(langfuse_memory_client, "_get_project_id", lambda: "p")
        create_score = MagicMock()
        monkeypatch.setattr(langfuse_memory_client, "create_score", create_score)

        def task(*, item):
            return item["input"]

        def eval1(*, input, output, expected_output, metadata):
            return Evaluation(name="acc", value=1.0)

        result = langfuse_memory_client.run_experiment(
            name="default",
            data=[{"input": "a"}],
            task=task,
            evaluators=[eval1],
            max_concurrency=1,
        )
        langfuse_memory_client.flush()

        assert len(result.item_results) == 1
        names = [s.name for s in memory_exporter.get_finished_spans()]
        # Expected list pinned against the pre-change (main) behavior.
        assert sorted(names) == sorted(
            [
                "experiment-item-run",
                "experiment-item-task",
                "experiment-item-evaluation",
                "eval1",
            ]
        )
        assert len(names) == 4
        assert "experiment-item-task-attempt" not in names
        item_scores = [
            c.kwargs
            for c in create_score.call_args_list
            if c.kwargs.get("trace_id") is not None
        ]
        assert len(item_scores) == 1
        assert item_scores[0]["name"] == "acc"


class TestResumeFailureAccounting:
    def test_resumed_item_success_not_counted_as_failed(
        self, langfuse_memory_client, monkeypatch
    ):
        """A resumed run that retries a previously-failed checkpoint item and
        succeeds must not count it in ``ExecutionSummary.failed``."""
        monkeypatch.setattr(langfuse_memory_client, "_get_project_id", lambda: "p")
        monkeypatch.setattr(langfuse_memory_client, "create_score", MagicMock())
        store = InMemoryCheckpointStore()
        data = [
            {"input": "done", "expected_output": "done"},
            {"input": "flaky", "expected_output": "flaky"},
        ]
        keys = make_item_keys(data)
        cp = ExperimentCheckpoint(
            experiment_id="exp-accounting",
            experiment_name="accounting",
            run_name="r",
            fingerprint=ExperimentFingerprint(
                task_name="task",
                evaluator_names=[],
                data_size=2,
                data_digest=compute_data_digest(keys),
            ),
            items={
                keys[0]: CheckpointItem(
                    state="completed",
                    output_json='"done"',
                    evaluations=[],
                    trace_id="t-done",
                    attempts=1,
                ),
                keys[1]: CheckpointItem(
                    state="failed",
                    error="boom",
                    attempts=2,
                ),
            },
            created_at=1.0,
            updated_at=2.0,
            status="running",
        )
        store.save(cp, "exp-accounting")

        def task(*, item):
            return item["input"]

        result = langfuse_memory_client.run_experiment(
            name="accounting",
            run_name="r",
            experiment_id="exp-accounting",
            data=data,
            task=task,
            execution=ExecutionPolicy(checkpoint=store, resume=True),
            max_concurrency=1,
        )
        assert result.execution is not None
        # The previously-failed item was retried and succeeded: it counts as
        # succeeded, not failed.
        assert result.execution.failed == 0
        assert result.execution.succeeded == 2
        assert result.execution.skipped_resumed == 1
        assert result.failures == []
        # The checkpoint no longer marks the item as failed.
        reloaded = store.load("exp-accounting")
        assert reloaded is not None
        assert reloaded.items[keys[1]].state == "completed"


class TestNoFunctionLevelImports:
    """Function-level imports added by the execution-policy change must live
    at module top (repo style; ruff does not currently flag them)."""

    def test_experiment_package_has_no_function_level_imports(self):
        import ast
        from pathlib import Path

        import langfuse._experiment as experiment_pkg

        package_dir = Path(experiment_pkg.__file__).parent
        offenders = []
        for path in sorted(package_dir.glob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    for child in ast.walk(node):
                        if child is node:
                            continue
                        if isinstance(child, (ast.Import, ast.ImportFrom)):
                            offenders.append("%s:%d" % (path.name, child.lineno))
        assert offenders == []

    def test_client_has_no_local_time_or_random_imports(self):
        from pathlib import Path

        import langfuse._client.client as client_module

        source = Path(client_module.__file__).read_text(encoding="utf-8")
        assert not re.search(r"^\s+import (time|random) as ", source, re.MULTILINE)


class TestDefaultPathIdentical:
    def test_no_policy_identical_span_tree_and_scores(
        self, langfuse_memory_client, monkeypatch, find_spans
    ):
        monkeypatch.setattr(langfuse_memory_client, "_get_project_id", lambda: "p")
        create_score = MagicMock()
        monkeypatch.setattr(langfuse_memory_client, "create_score", create_score)

        def task(*, item):
            return item["input"]

        def eval1(*, input, output, expected_output, metadata):
            return Evaluation(name="acc", value=1.0)

        result = langfuse_memory_client.run_experiment(
            name="default",
            data=[{"input": "a"}],
            task=task,
            evaluators=[eval1],
            max_concurrency=1,
        )
        # No retry-attempt spans by default.
        assert find_spans("experiment-item-task-attempt") == []
        # One item-level score.
        item_scores = [
            c.kwargs
            for c in create_score.call_args_list
            if c.kwargs.get("trace_id") is not None
        ]
        assert len(item_scores) == 1
        # Failures empty, execution summary present.
        assert result.failures == []
        assert result.evaluator_failures == []
        assert result.execution is not None
        assert result.execution.succeeded == 1
        assert result.execution.total_attempts == 1


# ---------------------------------------------------------------------------
# Signature drift guard: execution param threaded through
# ---------------------------------------------------------------------------


class TestExecutionParamForwarding:
    def test_runner_context_forwards_execution(self):
        from unittest.mock import MagicMock

        client = MagicMock(spec=Langfuse)
        client.run_experiment.return_value = "sentinel"
        ctx = RunnerContext(client=client, data=[{"input": "a"}])
        policy = ExecutionPolicy()
        ctx.run_experiment(name="exp", task=lambda *, item: None, execution=policy)
        kwargs = client.run_experiment.call_args.kwargs
        assert kwargs["execution"] is policy

    def test_dataset_client_forwards_execution(self):
        langfuse_client = MagicMock()
        item = _dataset_item(item_id="i1", dataset_id="d", input="a")
        dataset = DatasetClient(
            dataset=MagicMock(),
            items=[item],
            langfuse_client=langfuse_client,
        )
        policy = ExecutionPolicy()
        dataset.run_experiment(name="exp", task=lambda *, item: None, execution=policy)
        assert langfuse_client.run_experiment.call_args.kwargs["execution"] is policy
