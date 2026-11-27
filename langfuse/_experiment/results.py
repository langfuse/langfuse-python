"""@private Result and summary types produced by the experiment executor."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional

if TYPE_CHECKING:
    pass

# Stage of the pipeline where a failure happened. ``task``/``validation``/
# ``timeout``/``cancelled`` are experiment-run stages; ``mapper``/``evaluator``/
# ``composite``/``score`` are batch-evaluation stages.
FailureStage = Literal[
    "validation",
    "task",
    "evaluator",
    "timeout",
    "cancelled",
    "mapper",
    "composite",
    "score",
]


class AttemptRecord:
    """Record of a single task or evaluator attempt."""

    __slots__ = (
        "attempt",
        "started_at",
        "duration_s",
        "error_type",
        "error_message",
        "retry_after",
        "will_retry",
    )

    def __init__(
        self,
        *,
        attempt: int,
        started_at: float,
        duration_s: float,
        error_type: Optional[str] = None,
        error_message: Optional[str] = None,
        retry_after: Optional[float] = None,
        will_retry: bool = False,
    ):
        self.attempt = attempt
        self.started_at = started_at
        self.duration_s = duration_s
        self.error_type = error_type
        self.error_message = error_message
        self.retry_after = retry_after
        self.will_retry = will_retry

    def to_dict(self) -> Dict[str, Any]:
        return {
            "attempt": self.attempt,
            "started_at": self.started_at,
            "duration_s": self.duration_s,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "retry_after": self.retry_after,
            "will_retry": self.will_retry,
        }


class ItemFailure:
    """Record of an item that could not be processed to completion."""

    __slots__ = (
        "item",
        "item_key",
        "stage",
        "error_type",
        "error_message",
        "attempts",
        "trace_id",
        "evaluator_name",
    )

    def __init__(
        self,
        *,
        item: Any,
        item_key: str,
        stage: FailureStage,
        error_type: str,
        error_message: str,
        attempts: List[AttemptRecord],
        trace_id: Optional[str] = None,
        evaluator_name: Optional[str] = None,
    ):
        self.item = item
        self.item_key = item_key
        self.stage = stage
        self.error_type = error_type
        self.error_message = error_message
        self.attempts = attempts
        self.trace_id = trace_id
        self.evaluator_name = evaluator_name

    def to_dict(self) -> Dict[str, Any]:
        return {
            "item_key": self.item_key,
            "stage": self.stage,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "attempts": [a.to_dict() for a in self.attempts],
            "trace_id": self.trace_id,
            "evaluator_name": self.evaluator_name,
        }


class EvaluatorFailure:
    """Record of an evaluator that failed for an item after all retries."""

    __slots__ = (
        "item_key",
        "evaluator_name",
        "error_type",
        "error_message",
        "attempts",
    )

    def __init__(
        self,
        *,
        item_key: str,
        evaluator_name: str,
        error_type: str,
        error_message: str,
        attempts: List[AttemptRecord],
    ):
        self.item_key = item_key
        self.evaluator_name = evaluator_name
        self.error_type = error_type
        self.error_message = error_message
        self.attempts = attempts

    def to_dict(self) -> Dict[str, Any]:
        return {
            "item_key": self.item_key,
            "evaluator_name": self.evaluator_name,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "attempts": [a.to_dict() for a in self.attempts],
        }


class ExecutionSummary:
    """Aggregate statistics for one experiment run."""

    def __init__(
        self,
        *,
        total_items: int,
        succeeded: int,
        failed: int,
        skipped_resumed: int,
        retried_items: int,
        total_attempts: int,
        aborted: bool,
        abort_reason: Optional[str],
        duration_s: float,
        failures_by_stage: Optional[Dict[str, int]] = None,
    ):
        self.total_items = total_items
        self.succeeded = succeeded
        self.failed = failed
        self.skipped_resumed = skipped_resumed
        self.retried_items = retried_items
        self.total_attempts = total_attempts
        self.aborted = aborted
        self.abort_reason = abort_reason
        self.duration_s = duration_s
        self.failures_by_stage: Dict[str, int] = failures_by_stage or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_items": self.total_items,
            "succeeded": self.succeeded,
            "failed": self.failed,
            "skipped_resumed": self.skipped_resumed,
            "retried_items": self.retried_items,
            "total_attempts": self.total_attempts,
            "aborted": self.aborted,
            "abort_reason": self.abort_reason,
            "duration_s": self.duration_s,
            "failures_by_stage": dict(self.failures_by_stage),
        }

    def format(self) -> str:
        lines = [
            "Execution Summary:",
            "  total items: %s" % self.total_items,
            "  succeeded: %s" % self.succeeded,
            "  failed: %s" % self.failed,
            "  skipped (resumed): %s" % self.skipped_resumed,
            "  retried items: %s" % self.retried_items,
            "  total attempts: %s" % self.total_attempts,
            "  duration: %.3fs" % self.duration_s,
        ]
        if self.failures_by_stage:
            lines.append("  failures by stage:")
            for stage, count in sorted(self.failures_by_stage.items()):
                lines.append("    %s: %s" % (stage, count))
        if self.aborted:
            reason = self.abort_reason or "unknown"
            lines.append("  aborted: %s" % reason)
        return "\n".join(lines)


__all__ = [
    "AttemptRecord",
    "ItemFailure",
    "EvaluatorFailure",
    "ExecutionSummary",
    "FailureStage",
]
