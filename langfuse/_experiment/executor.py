"""@private Orchestrates one experiment run with retry, timeout, and resume.

The executor owns scheduling, semaphore, failure budget, run deadline,
cancellation, and checkpointing, and delegates per-item processing to a
caller-supplied coroutine. The ``Langfuse`` client wires its existing
``_process_experiment_item`` into that coroutine so the span tree and score
creation stay in the client.
"""

import asyncio
import json as _json
import random
import time
from dataclasses import dataclass, field
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
)

from langfuse._client.attributes import _serialize
from langfuse._experiment.checkpoint import (
    CheckpointItem,
    ExperimentCheckpoint,
    ExperimentFingerprint,
    FileCheckpointStore,
    compute_data_digest,
    experiment_checkpoint_key,
)
from langfuse._experiment.errors import (
    CheckpointMismatchError,
)
from langfuse._experiment.item_key import make_item_keys
from langfuse._experiment.policy import (
    ExecutionPolicy,
)
from langfuse._experiment.results import (
    AttemptRecord,
    EvaluatorFailure,
    ExecutionSummary,
    ItemFailure,
)
from langfuse._experiment.retrying import (
    _attempts_from_exception,
    _run_with_retry,
)
from langfuse.experiment import Evaluation, ExperimentItem, ExperimentItemResult

# processor(item, item_key, experiment_id, policy, attempt_hook,
# evaluator_failures). ``experiment_id`` is passed per item because a resumed
# local-data run adopts the id stored in the checkpoint.
ItemProcessor = Callable[
    [
        ExperimentItem,
        str,
        str,
        Optional[ExecutionPolicy],
        Optional[Callable[[AttemptRecord], None]],
        List[EvaluatorFailure],
    ],
    Awaitable[ExperimentItemResult],
]

# Grace period (seconds) when waiting for in-flight tasks to finish
# cancelling, so a task that swallows cancellation cannot hang the run.
_CANCEL_GRACE_S = 5.0


class ItemProcessingFailed(Exception):
    """Carrier exception raised by ``_process_experiment_item`` on terminal
    task failure. Carries the assembled ``ItemFailure``."""

    def __init__(self, failure: ItemFailure):
        self.failure = failure
        super().__init__(failure.error_message)


@dataclass
class ExecutorResult:
    item_results: List[ExperimentItemResult] = field(default_factory=list)
    failures: List[ItemFailure] = field(default_factory=list)
    evaluator_failures: List[EvaluatorFailure] = field(default_factory=list)
    summary: Optional[ExecutionSummary] = None
    aborted: bool = False
    abort_reason: Optional[str] = None
    # The effective experiment id: the constructor id, or the id adopted
    # from the checkpoint when a local-data run resumed.
    experiment_id: str = ""


def _evaluator_name(evaluator: Any) -> str:
    return getattr(evaluator, "__name__", "evaluator")


def _key_for_item(item: ExperimentItem, dataset_version: Optional[str]) -> str:
    keys = make_item_keys([item], dataset_version)
    return keys[0]


def _evaluation_to_dict(e: Evaluation) -> Dict[str, Any]:
    return {
        "name": e.name,
        "value": e.value,
        "comment": e.comment,
        "metadata": e.metadata,
        "data_type": e.data_type,
        "config_id": e.config_id,
    }


def _evaluation_from_dict(d: Dict[str, Any]) -> Evaluation:
    return Evaluation(
        name=str(d.get("name")),  # type: ignore[arg-type]
        value=d.get("value"),  # type: ignore[arg-type]
        comment=d.get("comment"),
        metadata=d.get("metadata"),
        data_type=d.get("data_type"),
        config_id=d.get("config_id"),
    )


class ExperimentExecutor:
    """Orchestrates a single experiment run."""

    def __init__(
        self,
        *,
        experiment_id: str,
        experiment_id_deterministic: bool,
        experiment_name: str,
        run_name: str,
        data: List[ExperimentItem],
        dataset_version: Optional[str],
        policy: Optional[ExecutionPolicy],
        processor: ItemProcessor,
        max_concurrency: int,
        clock: Optional[Callable[[], float]] = None,
        sleep: Optional[Callable[[float], Awaitable[None]]] = None,
        rng: Optional[random.Random] = None,
    ):
        if max_concurrency < 1:
            message = "max_concurrency must be >= 1, got %s" % max_concurrency
            raise ValueError(message)
        self.experiment_id = experiment_id
        # True when ``experiment_id`` is stable across runs (dataset runs, or
        # a caller-supplied experiment_id). Passed in explicitly by the
        # client; the executor does not infer it.
        self._experiment_id_deterministic = experiment_id_deterministic
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.data = data
        self.dataset_version = dataset_version
        self.policy = policy
        self.processor = processor
        self.max_concurrency = max_concurrency
        self._clock = clock or time.monotonic
        self._sleep = sleep or asyncio.sleep
        self._rng = rng or random.Random(
            policy.seed if policy is not None and policy.seed is not None else None
        )
        self._item_keys = make_item_keys(data, dataset_version)
        self._failures: List[ItemFailure] = []
        self._evaluator_failures: List[EvaluatorFailure] = []
        self._aborted = False
        self._abort_reason: Optional[str] = None
        self._skipped_resumed = 0
        self._retried_items = 0
        self._retried_keys: set = set()
        self._total_attempts = 0
        self._item_attempts: Dict[str, int] = {}
        self._checkpoint: Optional[ExperimentCheckpoint] = None

    def _fingerprint(self, task: Any, evaluators: List[Any]) -> ExperimentFingerprint:
        task_name = getattr(task, "__name__", "task")
        evaluator_names = [_evaluator_name(e) for e in evaluators]
        return ExperimentFingerprint(
            task_name=task_name,
            evaluator_names=evaluator_names,
            data_size=len(self.data),
            data_digest=compute_data_digest(self._item_keys),
        )

    def _checkpoint_key(self) -> str:
        """Store/load key for the checkpoint.

        Deterministic ids (dataset runs, caller-supplied ids) key the
        checkpoint by the id itself. Local-data runs draw a fresh random id
        per call, so their key is derived from the experiment name and run
        name instead; see ``experiment_checkpoint_key``.
        """
        if self._experiment_id_deterministic:
            return self.experiment_id
        return experiment_checkpoint_key(self.experiment_name, self.run_name)

    def _load_checkpoint(self, fingerprint: ExperimentFingerprint) -> None:
        if self.policy is None or self.policy.checkpoint is None:
            return
        if not self.policy.resume:
            return
        cp = self.policy.checkpoint.load(self._checkpoint_key())
        if cp is None:
            return
        if cp.experiment_id != self.experiment_id:
            if self._experiment_id_deterministic:
                raise CheckpointMismatchError(
                    "experiment_id", self.experiment_id, cp.experiment_id
                )
            # Local data: the id is random per run, so a resumed run adopts
            # the id recorded in the checkpoint. Resumed items and newly
            # executed items then share one experiment id.
            self.experiment_id = cp.experiment_id
        force = self.policy.resume == "force"
        if (
            not force
            and cp.fingerprint is not None
            and not cp.fingerprint.matches(fingerprint)
        ):
            raise CheckpointMismatchError(
                "fingerprint",
                fingerprint.to_dict(),
                cp.fingerprint.to_dict() if cp.fingerprint else None,
            )
        self._checkpoint = cp

    def _init_checkpoint(self, fingerprint: ExperimentFingerprint) -> None:
        if self.policy is None or self.policy.checkpoint is None:
            return
        if self._checkpoint is None:
            now = self._clock()
            self._checkpoint = ExperimentCheckpoint(
                schema_version=1,
                experiment_id=self.experiment_id,
                experiment_name=self.experiment_name,
                run_name=self.run_name,
                fingerprint=fingerprint,
                items={},
                created_at=now,
                updated_at=now,
                status="running",
            )

    async def _maybe_flush_checkpoint(self, force: bool = False) -> None:
        if (
            self._checkpoint is None
            or self.policy is None
            or self.policy.checkpoint is None
        ):
            return
        store = self.policy.checkpoint
        now = self._clock()
        if not force and isinstance(store, FileCheckpointStore):
            if not store.should_flush(now):
                return
        self._checkpoint.updated_at = now
        await asyncio.to_thread(store.save, self._checkpoint, self._checkpoint_key())
        if isinstance(store, FileCheckpointStore):
            store.mark_flushed(now)

    def _record_checkpoint_item(
        self,
        item_key: str,
        result: Optional[ExperimentItemResult],
        failure: Optional[ItemFailure],
    ) -> None:
        if self._checkpoint is None:
            return
        if result is not None:
            self._checkpoint.items[item_key] = CheckpointItem(
                state="completed",
                output_json=_serialize(result.output),
                evaluations=[_evaluation_to_dict(e) for e in result.evaluations],
                trace_id=result.trace_id,
                attempts=self._item_attempts.get(item_key, 1),
                error=None,
            )
        elif failure is not None:
            self._checkpoint.items[item_key] = CheckpointItem(
                state="failed",
                output_json=None,
                evaluations=[],
                trace_id=failure.trace_id,
                attempts=len(failure.attempts),
                error=failure.error_message,
            )

    def _make_attempt_hook(self, key: str) -> Callable[[AttemptRecord], None]:
        retried_keys = self._retried_keys

        def hook(record: AttemptRecord) -> None:
            self._total_attempts += 1
            self._item_attempts[key] = record.attempt
            # An item that consumed more than one attempt was retried; count
            # it at most once, no matter how many retry attempts it used.
            if record.attempt > 1 and key not in retried_keys:
                retried_keys.add(key)
                self._retried_items += 1

        return hook

    def _failure_budget_exceeded(self, failed: int, attempted: int) -> bool:
        if self.policy is None:
            return False
        return self.policy.failures.is_exceeded(failed, attempted)

    def _rebuild_result_from_checkpoint(
        self, item: ExperimentItem, key: str, cp_item: CheckpointItem
    ) -> ExperimentItemResult:
        output: Any = None
        if cp_item.output_json is not None:
            try:
                output = _json.loads(cp_item.output_json)
            except (ValueError, TypeError):
                output = cp_item.output_json
        evaluations = [_evaluation_from_dict(d) for d in cp_item.evaluations]
        return ExperimentItemResult(
            item=item,
            output=output,
            evaluations=evaluations,
            trace_id=cp_item.trace_id,
            experiment_id=self.experiment_id,
        )

    async def run(
        self,
        *,
        task: Any,
        evaluators: List[Any],
    ) -> ExecutorResult:
        fingerprint = self._fingerprint(task, evaluators)
        self._load_checkpoint(fingerprint)
        self._init_checkpoint(fingerprint)

        start = self._clock()
        run_timeout = (
            self.policy.timeouts.run_timeout if self.policy is not None else None
        )
        run_deadline = start + run_timeout if run_timeout is not None else None

        semaphore = asyncio.Semaphore(self.max_concurrency)

        skip_keys: set = set()
        if self._checkpoint is not None:
            for key, cp_item in self._checkpoint.items.items():
                if cp_item.state == "completed":
                    skip_keys.add(key)

        to_run: List[Tuple[int, ExperimentItem, str]] = []
        resumed_results: List[ExperimentItemResult] = []
        for idx, (item, key) in enumerate(zip(self.data, self._item_keys)):
            if key in skip_keys:
                assert self._checkpoint is not None
                cp_item = self._checkpoint.items[key]
                resumed_results.append(
                    self._rebuild_result_from_checkpoint(item, key, cp_item)
                )
                self._skipped_resumed += 1
            else:
                to_run.append((idx, item, key))

        in_flight: set = set()
        completed_results: List[ExperimentItemResult] = []
        attempted = 0
        failed = 0

        async def process_one(item: ExperimentItem, key: str) -> None:
            nonlocal attempted, failed
            async with semaphore:
                if self._aborted:
                    return
                attempted += 1
                try:
                    result = await self.processor(
                        item,
                        key,
                        self.experiment_id,
                        self.policy,
                        self._make_attempt_hook(key),
                        self._evaluator_failures,
                    )
                    completed_results.append(result)
                    self._record_checkpoint_item(key, result, None)
                    await self._maybe_flush_checkpoint()
                except ItemProcessingFailed as e:
                    self._failures.append(e.failure)
                    failed += 1
                    self._record_checkpoint_item(key, None, e.failure)
                    await self._maybe_flush_checkpoint()
                except asyncio.CancelledError:
                    raise
                except BaseException as e:  # noqa: BLE001
                    failure = ItemFailure(
                        item=item,
                        item_key=key,
                        stage="task",
                        error_type=type(e).__name__,
                        error_message=str(e),
                        attempts=_attempts_from_exception(e),
                        trace_id=None,
                    )
                    self._failures.append(failure)
                    failed += 1
                    self._record_checkpoint_item(key, None, failure)
                    await self._maybe_flush_checkpoint()

        next_index = 0

        def _spawn_next() -> bool:
            """Schedule the next pending item; return True if one was spawned."""
            nonlocal next_index
            if next_index >= len(to_run):
                return False
            _, item, key = to_run[next_index]
            next_index += 1
            t = asyncio.ensure_future(process_one(item, key))
            in_flight.add(t)
            t.add_done_callback(in_flight.discard)
            return True

        try:
            while not self._aborted:
                if run_deadline is not None and self._clock() >= run_deadline:
                    self._aborted = True
                    self._abort_reason = "run timeout"
                    break
                while len(in_flight) < self.max_concurrency and _spawn_next():
                    pass
                if not in_flight:
                    break
                wait_timeout = (
                    max(0.0, run_deadline - self._clock())
                    if run_deadline is not None
                    else None
                )
                await asyncio.wait(
                    in_flight,
                    timeout=wait_timeout,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if (
                    run_deadline is not None
                    and in_flight
                    and self._clock() >= run_deadline
                ):
                    self._aborted = True
                    self._abort_reason = "run timeout"
                if not self._aborted and self._failure_budget_exceeded(
                    failed, attempted
                ):
                    self._aborted = True
                    self._abort_reason = "failure budget exceeded"
        except asyncio.CancelledError:
            self._aborted = True
            self._abort_reason = "cancelled"
            for t in in_flight:
                t.cancel()
            if in_flight:
                # Bounded wait: never block the run on tasks that ignore
                # cancellation.
                await asyncio.wait(in_flight, timeout=_CANCEL_GRACE_S)
            await self._maybe_flush_checkpoint(force=True)
            raise
        finally:
            pending = [t for t in in_flight if not t.done()]
            for t in pending:
                t.cancel()
            if pending:
                # Bounded wait so a task that swallows cancellation cannot hang
                # the whole run.
                await asyncio.wait(pending, timeout=_CANCEL_GRACE_S)

        if self._aborted and self._abort_reason is None:
            self._abort_reason = "aborted"

        await self._maybe_flush_checkpoint(force=True)

        by_key: Dict[str, ExperimentItemResult] = {}
        for r in completed_results:
            by_key[_key_for_item(r.item, self.dataset_version)] = r
        for r in resumed_results:
            by_key[_key_for_item(r.item, self.dataset_version)] = r

        ordered: List[ExperimentItemResult] = []
        for item, key in zip(self.data, self._item_keys):
            if key in by_key:
                ordered.append(by_key[key])

        failures_by_stage: Dict[str, int] = {}
        for f in self._failures:
            failures_by_stage[f.stage] = failures_by_stage.get(f.stage, 0) + 1

        summary = ExecutionSummary(
            total_items=len(self.data),
            succeeded=len(completed_results) + len(resumed_results),
            failed=len(self._failures),
            skipped_resumed=self._skipped_resumed,
            retried_items=self._retried_items,
            total_attempts=self._total_attempts,
            aborted=self._aborted,
            abort_reason=self._abort_reason,
            duration_s=self._clock() - start,
            failures_by_stage=failures_by_stage,
        )

        return ExecutorResult(
            item_results=ordered,
            failures=list(self._failures),
            evaluator_failures=list(self._evaluator_failures),
            summary=summary,
            aborted=self._aborted,
            abort_reason=self._abort_reason,
            experiment_id=self.experiment_id,
        )


__all__ = [
    "ExperimentExecutor",
    "ExecutorResult",
    "ItemProcessingFailed",
    "ItemProcessor",
    "_run_with_retry",
]
