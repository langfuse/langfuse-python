"""Resumable experiment runs.

Reuses ``experiment.py``'s ``Evaluation``, ``_run_task``, ``_run_evaluator`` and
``_normalize_evaluator_result``. Public names are re-exported from
``langfuse/__init__.py``.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    runtime_checkable,
)

from langfuse.experiment import (
    Evaluation,
    _normalize_evaluator_result,
    _run_evaluator,
    _run_task,
)

# Baked into the run fingerprint below: bumping it changes the fingerprint, which
# forces a brand-new run instead of resuming. That is the guard against merging
# results produced by an incompatible engine change — and also the trade-off:
# bump it ONLY on a change that genuinely breaks resume compatibility, never on a
# routine release, or every SDK upgrade would orphan in-progress runs and re-pay
# their completed items.
RUNNER_VERSION = "2.0.0"


# --- identity ----------------------------------------------------
def _canonical(value: Any) -> str:
    """Deterministic JSON for hashing: sorted keys, compact, str-fallback."""
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), default=str, ensure_ascii=True
    )


def _sha(*parts: Any) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(_canonical(p).encode("utf-8"))
        h.update(b"\x1f")
    return h.hexdigest()


def item_input(item: Any) -> Any:
    if isinstance(item, Mapping):
        return item.get("input")
    return getattr(item, "input", None)


def item_expected_output(item: Any) -> Any:
    if isinstance(item, Mapping):
        return item.get("expected_output")
    return getattr(item, "expected_output", None)


def item_metadata(item: Any) -> Any:
    if isinstance(item, Mapping):
        return item.get("metadata")
    return getattr(item, "metadata", None)


def item_explicit_id(item: Any) -> Optional[str]:
    v = item.get("id") if isinstance(item, Mapping) else getattr(item, "id", None)
    return str(v) if v is not None else None


def item_id(item: Any) -> str:
    """Explicit id if given, else sha256 of the canonical input."""
    explicit = item_explicit_id(item)
    return explicit if explicit else "sha256:" + _sha(item_input(item))[:32]


def fingerprint(
    *,
    task_version: Optional[str],
    params: Optional[Mapping[str, Any]],
    evaluator_names: Sequence[str],
    runner_version: str,
    dataset_id: Optional[str] = None,
) -> str:
    """Comparability key. A change here starts a fresh run."""
    return "fp_" + _sha(
        task_version or "unversioned",
        params or {},
        list(evaluator_names),
        runner_version,
        dataset_id or "local",
    )[:24]


def derive_run_id(*, name: str, run_name: Optional[str], fingerprint: str) -> str:
    """run_id = hash(name, run_name|derived, fingerprint): same code resumes,
    changed fingerprint = fresh run with no silent mixing."""
    return "run_" + _sha(name, run_name or "", fingerprint)[:20]


# --- retry / backoff ----------------------------------
class RetryableError(Exception):
    """Base for transient failures worth retrying."""


class RateLimitError(RetryableError):
    """Provider said 429 / slow down."""


class ServerError(RetryableError):
    """Transient 5xx / transport hiccup."""


class ClientError(Exception):
    """Non-retryable client error (a 4xx other than 429). Fails fast — retrying
    it would never succeed."""


def is_retryable(exc: BaseException) -> bool:
    return isinstance(exc, RetryableError)


def backoff_delay(attempt: int, *, base: float = 0.1, cap: float = 10.0, jitter: bool = True) -> float:
    """Exponential backoff with full jitter. attempt is 1-based."""
    raw = min(cap, base * (2 ** (attempt - 1)))
    return random.uniform(0, raw) if jitter else raw


# --- client-side rate limiting -----------------------------------
class RateLimiter:
    """Async pacer: at most target_rpm acquisitions/min, shared across workers."""

    def __init__(self, target_rpm: Optional[float]):
        self._rpm = target_rpm
        self._interval = 60.0 / target_rpm if target_rpm else 0.0
        self._lock = asyncio.Lock()
        self._next_at = 0.0

    async def acquire(self) -> None:
        if not self._rpm:
            return
        loop = asyncio.get_event_loop()
        async with self._lock:
            now = loop.time()
            wait = max(0.0, self._next_at - now)
            self._next_at = max(now, self._next_at) + self._interval
        if wait > 0:
            await asyncio.sleep(wait)


# --- models --------------------------------------------------
class ItemState(str, Enum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    TIMED_OUT = "timed_out"


@dataclass
class ItemResult:
    """One item's terminal outcome — the unit that gets checkpointed."""

    item_id: str
    state: ItemState
    output: Any = None
    evaluations: List[Evaluation] = field(default_factory=list)
    error: Optional[str] = None
    attempts: int = 0
    started_at: Optional[str] = None
    ended_at: Optional[str] = None

    def to_wire(self) -> Dict[str, Any]:
        return {
            "item_id": self.item_id,
            "state": self.state.value,
            "output": self.output,
            "evaluations": [
                {"name": e.name, "value": e.value, "comment": e.comment}
                for e in self.evaluations
            ],
            "error": self.error,
            "attempts": self.attempts,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
        }


@dataclass
class ResumableExperimentResult:
    """Honest result for a resumable run: counts reconcile to total,
    aggregates carry their n, failures are listed."""

    run_id: str
    run_name: Optional[str]
    fingerprint: str
    status: str
    total: int
    succeeded: int
    failed: int
    timed_out: int
    skipped: int
    scores: Dict[str, Dict[str, float]]
    failures: List[Dict[str, Any]]
    run_evaluations: List[Evaluation] = field(default_factory=list)
    is_complete: bool = False

    @property
    def counts_reconcile(self) -> bool:
        return self.succeeded + self.failed + self.timed_out + self.skipped == self.total

    def format(self) -> str:
        lines = [
            "─" * 50,
            f"🧪 Experiment run: {self.run_name or self.run_id}  [{self.status}]",
            f"   run_id={self.run_id}  fingerprint={self.fingerprint}",
            f"   {self.total} items → "
            f"✓{self.succeeded}  ✗{self.failed}  ⏱{self.timed_out}  ↷{self.skipped}"
            f"   (complete={self.is_complete})",
        ]
        if self.scores:
            lines.append("   Scores (over succeeded):")
            for name, agg in sorted(self.scores.items()):
                lines.append(f"     • {name}: {agg['mean']:.3f}  (n={int(agg['n'])})")
        if self.run_evaluations:
            lines.append("   Run evaluations:")
            for e in self.run_evaluations:
                val = f"{e.value:.3f}" if isinstance(e.value, (int, float)) else str(e.value)
                lines.append(f"     • {e.name}: {val}")
        if self.failures:
            shown = self.failures[:5]
            lines.append(f"   Failures ({len(self.failures)} total, showing {len(shown)}):")
            for f in shown:
                lines.append(f"     • {f['item_id']}: {f['reason']} (attempts={f['attempts']})")
        if not self.counts_reconcile:
            lines.append("   ⚠️  counts do not reconcile to total — BUG")
        return "\n".join(lines)


# --- store contract --------------------------------------------------------
# The runner depends only on this protocol
@runtime_checkable
class ResultStore(Protocol):
    async def create_or_resume_run(self, *, name: str, run_name: Optional[str],
                                   dataset_ref: Optional[Dict[str, Any]], fingerprint: str,
                                   config: Optional[Dict[str, Any]]) -> Tuple[str, str]: ...
    async def gap_query(self, run_id: str, item_ids: Sequence[str]) -> Dict[str, str]: ...
    async def upsert_items(self, run_id: str, results: Sequence["ItemResult"]) -> List[Dict[str, str]]: ...
    async def upsert_run_evaluations(self, run_id: str, evaluations: List[Dict[str, Any]]) -> None: ...
    async def complete_run(self, run_id: str, status: str) -> Dict[str, Any]: ...


# --- the resumable runner ----------------------------------------
def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


async def _run_one(item: Any, iid: str, task: Callable, evaluators: Sequence[Callable],
                   item_timeout: Optional[float], max_retries: int) -> ItemResult:
    """Execute one item with per-item timeout and retry/backoff.

    Reuses this module's existing `_run_task` (sync/async handling) and
    `_run_evaluator` (evaluator invocation + normalization) rather than
    reimplementing them.
    """
    started = _now()
    last_state = ItemState.FAILED
    last_err: Optional[str] = None
    for attempt in range(1, max_retries + 1):
        try:
            coro = _run_task(task, item)  # reuse existing task runner
            output = await (asyncio.wait_for(coro, item_timeout) if item_timeout is not None else coro)
        except asyncio.TimeoutError:
            last_state, last_err = ItemState.TIMED_OUT, f"timeout>{item_timeout}s"
            if attempt < max_retries:
                await asyncio.sleep(backoff_delay(attempt))
                continue
            break
        except Exception as e:
            if is_retryable(e) and attempt < max_retries:
                last_err = repr(e)
                await asyncio.sleep(backoff_delay(attempt))
                continue
            return ItemResult(item_id=iid, state=ItemState.FAILED, error=repr(e),
                              attempts=attempt, started_at=started, ended_at=_now())
        # success — run each item evaluator; a failing evaluator must be visible,
        # not silently dropped, so record it as an explicit error evaluation.
        evals: List[Evaluation] = []
        for ev in evaluators:
            try:
                evals.extend(await _run_evaluator(
                    ev, _raise_on_error=True, input=item_input(item), output=output,
                    expected_output=item_expected_output(item), metadata=item_metadata(item),
                ))
            except Exception as e:
                name = getattr(ev, "__name__", "evaluator")
                evals.append(Evaluation(
                    name=f"{name} (evaluator error)", value=0.0, comment=repr(e)[:200],
                ))
        return ItemResult(item_id=iid, state=ItemState.SUCCEEDED, output=output,
                          evaluations=evals, attempts=attempt, started_at=started, ended_at=_now())
    return ItemResult(item_id=iid, state=last_state, error=last_err,
                      attempts=max_retries, started_at=started, ended_at=_now())


async def _with_retry(thunk: Callable[[], Any], max_retries: int) -> Any:
    """The single retry layer for store operations. Retries only retryable
    errors (ServerError/RateLimitError) with backoff; a non-retryable error
    (e.g. ClientError, a 4xx) fails fast. The store does NOT retry internally —
    this is the one place transport retry lives, so there is no double-nesting."""
    for attempt in range(1, max_retries + 1):
        try:
            return await thunk()
        except Exception as e:
            if not is_retryable(e) or attempt >= max_retries:
                raise
            await asyncio.sleep(backoff_delay(attempt))


async def arun_experiment(
    *,
    name: str,
    data: Sequence[Any],
    task: Callable,
    store: ResultStore,
    evaluators: Sequence[Callable] = (),
    run_evaluators: Sequence[Callable] = (),
    run_name: Optional[str] = None,
    resume: bool = True,
    task_version: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    max_concurrency: int = 50,
    target_rpm: Optional[float] = None,
    item_timeout: Optional[float] = None,
    max_retries: int = 5,
    abort_if_failed_fraction: Optional[float] = None,
    retry_failed_on_resume: bool = True,
) -> ResumableExperimentResult:
    """Resumable experiment run. Completed items are never re-invoked; every
    item reaches a terminal state; each result is checkpointed as it finishes."""
    data = list(data)
    fp = fingerprint(
        task_version=task_version, params=metadata,
        evaluator_names=[getattr(e, "__name__", "eval") for e in evaluators],
        runner_version=RUNNER_VERSION, dataset_id=None,
    )
    run_id, _status = await _with_retry(
        lambda: store.create_or_resume_run(
            name=name, run_name=run_name,
            dataset_ref={"type": "local", "item_count": len(data)},
            fingerprint=fp, config={"max_concurrency": max_concurrency, "target_rpm": target_rpm,
                                    "item_timeout": item_timeout, "max_retries": max_retries},
        ),
        max_retries,
    )

    pairs = [(item, item_id(item)) for item in data]
    _ids = [iid for _, iid in pairs]
    known: Dict[str, str] = (
        await _with_retry(lambda: store.gap_query(run_id, _ids), max_retries) if resume else {}
    )

    def _should_run(iid: str) -> bool:
        state = known.get(iid)
        if state is None:
            return True
        if state == ItemState.SUCCEEDED.value:
            return False
        return retry_failed_on_resume

    todo = [(item, iid) for item, iid in pairs if _should_run(iid)]

    sem = asyncio.Semaphore(max_concurrency)
    limiter = RateLimiter(target_rpm)
    abort = asyncio.Event()
    lock = asyncio.Lock()
    stats = {"processed": 0, "failed": 0}
    done_results: List[ItemResult] = []
    checkpoint_failed: List[Tuple[str, str]] = []

    async def process(item: Any, iid: str) -> None:
        async with sem:
            if abort.is_set():
                return  # leaves no row -> counted as skipped
            await limiter.acquire()
            result = await _run_one(item, iid, task, evaluators, item_timeout, max_retries)
            try:
                await _with_retry(lambda: store.upsert_items(run_id, [result]), max_retries)
            except Exception as e:
                # A result we could not persist must be surfaced, not dropped.
                async with lock:
                    checkpoint_failed.append((iid, repr(e)))
                return
            async with lock:
                stats["processed"] += 1
                done_results.append(result)
                if result.state is not ItemState.SUCCEEDED:
                    stats["failed"] += 1
                if (abort_if_failed_fraction is not None and stats["processed"] >= 1
                        and stats["failed"] / stats["processed"] >= abort_if_failed_fraction):
                    abort.set()

    await asyncio.gather(*(process(it, iid) for it, iid in todo), return_exceptions=True)

    # Run-level evaluators aggregate over items processed this invocation; the
    # per-metric scores in the summary are still over all stored rows. On resume,
    # previously-done items aren't re-fetched here.
    run_evals: List[Evaluation] = []
    for rev in run_evaluators:
        run_evals.extend(await _run_evaluator(rev, item_results=done_results))
    run_evals_wire = [{"name": e.name, "value": e.value, "comment": e.comment} for e in run_evals]

    status = "aborted" if abort.is_set() else "complete"
    await _with_retry(lambda: store.upsert_run_evaluations(run_id, run_evals_wire), max_retries)
    summary = await _with_retry(lambda: store.complete_run(run_id, status), max_retries)

    total = len(data)
    rows = summary["rows"]
    skipped = total - rows
    failures = summary["failures"]
    # Items whose result could not be persisted are surfaced here. They have no
    # stored row, so they also fall into `skipped`; listing them keeps the
    # failure visible and forces is_complete=False.
    for iid, err in checkpoint_failed:
        failures.append({"item_id": iid, "reason": f"checkpoint_failed: {err}", "attempts": max_retries})
    return ResumableExperimentResult(
        run_id=run_id, run_name=run_name, fingerprint=fp, status=status, total=total,
        succeeded=summary["succeeded"], failed=summary["failed"], timed_out=summary["timed_out"],
        skipped=skipped, scores=summary["scores"], failures=failures,
        run_evaluations=_normalize_evaluator_result(summary.get("run_evaluations") or []),
        is_complete=(status == "complete" and skipped == 0 and not checkpoint_failed),
    )


def run_experiment_resumable(**kwargs: Any) -> ResumableExperimentResult:
    """Synchronous wrapper around arun_experiment."""
    return asyncio.run(arun_experiment(**kwargs))
