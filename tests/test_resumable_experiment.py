"""Unit tests for the resumable experiment runner.

No containers: everything runs against InMemoryResultStore / FlakyStore. Covers
resume, honest failures, idempotency, timeout, rate limiting, and the fingerprint
guard.
"""

import asyncio
import time

import pytest

from langfuse import (
    ClientError,
    Evaluation,
    ItemResult,
    ItemState,
    RateLimitError,
    ServerError,
    arun_experiment,
)
from langfuse.experiment_resume import is_retryable
from langfuse.experiment_store import summarize
from tests.experiment_doubles import FlakyStore, InMemoryResultStore

DATA = [
    {"input": "france", "expected_output": "PARIS"},
    {"input": "germany", "expected_output": "BERLIN"},
    {"input": "italy", "expected_output": "ROME"},
    {"input": "spain", "expected_output": "MADRID"},
]
ANSWERS = {"france": "PARIS", "germany": "BERLIN", "italy": "ROME", "spain": "MADRID"}


def good_task(*, item, **_):
    return ANSWERS[item["input"]]


def accuracy(*, input, output, expected_output, **_):
    ok = bool(output) and output == expected_output
    return Evaluation(name="accuracy", value=1.0 if ok else 0.0)


# ---------------------------------------------------------------- check 1: skip
@pytest.mark.asyncio
async def test_resume_skips_completed_items_no_repay():
    store = InMemoryResultStore()
    calls = []

    def counting_task(*, item, **_):
        calls.append(item["input"])
        return ANSWERS[item["input"]]

    # First run: complete only the first two items, then simulate a crash by
    # only feeding two items.
    r1 = await arun_experiment(
        name="geo", data=DATA[:2], task=counting_task, evaluators=[accuracy],
        store=store, task_version="v1",
    )
    assert r1.succeeded == 2
    assert sorted(calls) == ["france", "germany"]

    # Resume with the FULL dataset and the SAME code (same fingerprint/run_id).
    calls.clear()
    r2 = await arun_experiment(
        name="geo", data=DATA, task=counting_task, evaluators=[accuracy],
        store=store, task_version="v1",
    )
    # Only the two NEW items were executed — no re-pay for the first two.
    assert sorted(calls) == ["italy", "spain"]
    assert r2.total == 4
    assert r2.succeeded == 4
    assert r2.is_complete
    assert r2.counts_reconcile


# ------------------------------------------------- check 2: retry then surface
@pytest.mark.asyncio
async def test_retry_succeeds_and_permanent_failure_is_surfaced():
    store = InMemoryResultStore()
    attempts = {"italy": 0}

    def flaky_task(*, item, **_):
        key = item["input"]
        if key == "italy":
            attempts["italy"] += 1
            if attempts["italy"] < 3:
                raise RateLimitError("429")  # retryable, succeeds on 3rd try
            return ANSWERS[key]
        if key == "spain":
            raise ValueError("permanent boom")  # non-retryable
        return ANSWERS[key]

    r = await arun_experiment(
        name="retry", data=DATA, task=flaky_task, evaluators=[accuracy],
        store=store, task_version="v1", max_retries=5,
    )
    assert attempts["italy"] == 3
    assert r.succeeded == 3  # france, germany, italy
    assert r.failed == 1  # spain
    assert r.total == 4
    assert r.counts_reconcile
    # failure is visible, not silently dropped
    assert any("spain" in f["item_id"] or "permanent" in f["reason"] for f in r.failures)


# ------------------------------------------------------- check 3: idempotency
@pytest.mark.asyncio
async def test_duplicate_upsert_is_noop():
    store = InMemoryResultStore()
    run_id, _ = await store.create_or_resume_run(
        name="idem", run_name=None, dataset_ref=None, fingerprint="fp", config=None
    )
    res = ItemResult(item_id="sha256:x", state=ItemState.SUCCEEDED, output="PARIS",
                     evaluations=[Evaluation(name="accuracy", value=1.0)], attempts=1)
    first = await store.upsert_items(run_id, [res])
    second = await store.upsert_items(run_id, [res])
    assert first[0]["applied"] == "created"
    assert second[0]["applied"] == "noop_duplicate"
    summary = await store.complete_run(run_id, "complete")
    assert summary["rows"] == 1  # not doubled
    assert summary["succeeded"] == 1


@pytest.mark.asyncio
async def test_flaky_store_retried_and_stays_consistent():
    # Writes fail ~half the time (write-through-then-raise). The runner retries;
    # idempotency means the eventual state has exactly one row per item.
    store = FlakyStore(InMemoryResultStore(), fail_rate=0.5, seed=1)
    r = await arun_experiment(
        name="flaky", data=DATA, task=good_task, evaluators=[accuracy],
        store=store, task_version="v1", max_retries=8,
    )
    assert store.injected > 0  # we actually exercised the failure path
    assert r.total == 4
    assert r.succeeded == 4
    assert r.counts_reconcile


# ----------------------------------------------------------- check 4: timeout
@pytest.mark.asyncio
async def test_hung_item_times_out_and_run_completes():
    store = InMemoryResultStore()

    async def sometimes_hangs(*, item, **_):
        if item["input"] == "italy":
            await asyncio.sleep(10)  # longer than item_timeout
        return ANSWERS[item["input"]]

    r = await arun_experiment(
        name="timeout", data=DATA, task=sometimes_hangs, evaluators=[accuracy],
        store=store, task_version="v1", item_timeout=0.2, max_retries=1,
    )
    assert r.timed_out == 1
    assert r.succeeded == 3
    assert r.total == 4
    assert r.counts_reconcile
    assert any("timeout" in f["reason"] for f in r.failures)


# ------------------------------------------------- fingerprint guard
@pytest.mark.asyncio
async def test_changed_task_version_starts_fresh_run():
    store = InMemoryResultStore()
    r1 = await arun_experiment(name="fp", data=DATA, task=good_task, store=store, task_version="v1")
    r2 = await arun_experiment(name="fp", data=DATA, task=good_task, store=store, task_version="v2")
    assert r1.run_id != r2.run_id  # new code -> new run, no silent mixing
    r1b = await arun_experiment(name="fp", data=DATA, task=good_task, store=store, task_version="v1")
    assert r1b.run_id == r1.run_id  # same code -> same run (resumes)


# --------------------------------------------- honest reconciliation invariant
@pytest.mark.asyncio
async def test_counts_reconcile_and_scores_carry_n():
    store = InMemoryResultStore()
    r = await arun_experiment(
        name="honest", data=DATA, task=good_task, evaluators=[accuracy],
        store=store, task_version="v1",
    )
    assert r.succeeded + r.failed + r.timed_out + r.skipped == r.total
    assert r.scores["accuracy"]["n"] == 4


def test_summarize_pure_function():
    rows = [
        {"item_id": "a", "state": "succeeded", "evaluations": [{"name": "s", "value": 1.0}]},
        {"item_id": "b", "state": "succeeded", "evaluations": [{"name": "s", "value": 0.0}]},
        {"item_id": "c", "state": "failed", "error": "boom", "attempts": 2},
        {"item_id": "d", "state": "timed_out", "error": "timeout", "attempts": 1},
    ]
    s = summarize(rows)
    assert s["succeeded"] == 2 and s["failed"] == 1 and s["timed_out"] == 1
    assert s["rows"] == 4
    assert s["scores"]["s"] == {"mean": 0.5, "n": 2}
    assert len(s["failures"]) == 2


# ------------------------------------------------ rate limiter (symptom #2c)
@pytest.mark.asyncio
async def test_rate_limiter_paces_calls():
    """target_rpm must spread calls over time, even with spare concurrency.

    With rpm=600 (0.1s apart) and 6 instant items run at full concurrency, the
    limiter alone should stretch the run to >= ~5*0.1s. A control run without
    the limiter must be far faster — proving the delay is the pacing, not
    overhead.
    """
    rpm = 600
    interval = 60.0 / rpm       # 0.1s
    n = 6
    data = [{"input": f"q{i}", "expected_output": f"Q{i}"} for i in range(n)]

    def fast_task(*, item, **_):
        return "Q" + item["input"][1:]  # instant, always correct

    # paced run
    t0 = time.perf_counter()
    paced = await arun_experiment(
        name="rate-paced", data=data, task=fast_task, store=InMemoryResultStore(),
        task_version="v1", target_rpm=rpm, max_concurrency=n,
    )
    paced_elapsed = time.perf_counter() - t0

    # control: no limiter, same work
    t1 = time.perf_counter()
    await arun_experiment(
        name="rate-free", data=data, task=fast_task, store=InMemoryResultStore(),
        task_version="v1", max_concurrency=n,
    )
    control_elapsed = time.perf_counter() - t1

    floor = (n - 1) * interval          # 0.5s
    assert paced.succeeded == n and paced.counts_reconcile
    assert paced_elapsed >= floor * 0.8, f"expected >= {floor * 0.8:.2f}s, got {paced_elapsed:.2f}s"
    assert control_elapsed < floor * 0.5, f"control should be fast, got {control_elapsed:.2f}s"


# ------------------------------------ a persistent checkpoint failure must be
#                                       surfaced, never silently dropped
class _CheckpointAlwaysFails(InMemoryResultStore):
    async def upsert_items(self, run_id, results):  # type: ignore[override]
        raise ServerError("simulated unwritable store (e.g. disk full)")


@pytest.mark.asyncio
async def test_checkpoint_failure_is_surfaced_not_dropped():
    store = _CheckpointAlwaysFails()
    r = await arun_experiment(
        name="cpfail", data=DATA, task=good_task, evaluators=[accuracy],
        store=store, task_version="v1", max_retries=2,
    )
    assert r.succeeded == 0  # nothing could be persisted
    assert r.is_complete is False  # must not report complete
    assert len(r.failures) == len(DATA)
    assert all("checkpoint_failed" in f["reason"] for f in r.failures)
    assert r.counts_reconcile  # the unpersisted items fall into `skipped`


# ------------------------------------ 4xx is non-retryable, 429/5xx retryable
def test_error_classification():
    assert is_retryable(ServerError("503")) is True
    assert is_retryable(RateLimitError("429")) is True
    assert is_retryable(ClientError("400 bad request")) is False
    assert is_retryable(ValueError("boom")) is False


# ------------------------------------ boolean evaluations aggregate as a rate
@pytest.mark.asyncio
async def test_boolean_evaluations_are_aggregated():
    store = InMemoryResultStore()

    def bool_eval(*, input, output, expected_output, **_):
        return Evaluation(name="passed", value=(output == expected_output))

    r = await arun_experiment(
        name="boolagg", data=DATA, task=good_task, evaluators=[bool_eval],
        store=store, task_version="v1",
    )
    assert "passed" in r.scores  # would be absent before the fix (bool dropped)
    assert r.scores["passed"]["n"] == len(DATA)
    assert r.scores["passed"]["mean"] == 1.0  # all correct -> pass rate 1.0


# ------------------------------------ a failing evaluator is surfaced, not dropped
@pytest.mark.asyncio
async def test_failing_evaluator_is_surfaced_not_dropped():
    store = InMemoryResultStore()

    def broken_eval(*, input, output, expected_output, **_):
        raise ValueError("evaluator boom")

    r = await arun_experiment(
        name="everr", data=DATA, task=good_task, evaluators=[broken_eval],
        store=store, task_version="v1",
    )
    err_keys = [k for k in r.scores if "evaluator error" in k]
    assert err_keys, f"expected an evaluator-error metric, got {list(r.scores)}"
    assert r.scores[err_keys[0]]["n"] == len(DATA)


# ------------------------------------ client-centric DX: method on the client
@pytest.mark.asyncio
async def test_client_method_runs_with_no_url_or_key():
    from langfuse import Langfuse

    lf = Langfuse(public_key="pk-x", secret_key="sk-x",
                  host="http://localhost:9", tracing_enabled=False)
    # store injected so no network is touched; the point is the call site needs
    # no base URL / key / store beyond the client config.
    r = await lf.arun_experiment(
        name="viaclient", data=DATA, task=good_task, evaluators=[accuracy],
        store=InMemoryResultStore(), task_version="v1",
    )
    assert r.succeeded == len(DATA) and r.is_complete and r.counts_reconcile
