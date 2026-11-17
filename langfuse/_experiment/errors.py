"""@private Exception types raised by the experiment execution policies.

These are intentionally separate from the public ``langfuse.experiment`` types so
that callers can catch execution failures without importing the private package.
"""

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from langfuse._experiment.policy import FailurePolicy
    from langfuse.experiment import ExperimentResult


class ExperimentExecutionError(Exception):
    """Base class for all experiment execution policy errors."""


class TaskTimeoutError(ExperimentExecutionError):
    """Raised when a single task call exceeds its timeout budget."""

    def __init__(self, item_key: str, timeout: float):
        self.item_key = item_key
        self.timeout = timeout
        message = "Task for item %s timed out after %s seconds" % (item_key, timeout)
        super().__init__(message)


class EvaluatorTimeoutError(ExperimentExecutionError):
    """Raised when a single evaluator call exceeds its timeout budget."""

    def __init__(self, evaluator_name: str, item_key: str, timeout: float):
        self.evaluator_name = evaluator_name
        self.item_key = item_key
        self.timeout = timeout
        message = "Evaluator %s for item %s timed out after %s seconds" % (
            evaluator_name,
            item_key,
            timeout,
        )
        super().__init__(message)


class RunTimeoutError(ExperimentExecutionError):
    """Raised when the whole experiment run exceeds its run deadline."""

    def __init__(self, timeout: float):
        self.timeout = timeout
        message = "Experiment run timed out after %s seconds" % timeout
        super().__init__(message)


class FailureBudgetExceededError(ExperimentExecutionError):
    """Raised when the failure budget is exhausted.

    Carries the partial ``ExperimentResult`` so callers can inspect what
    completed before the budget tripped.
    """

    def __init__(
        self,
        failed: int,
        attempted: int,
        policy: "FailurePolicy",
        partial_result: Optional["ExperimentResult"] = None,
    ):
        self.failed = failed
        self.attempted = attempted
        self.policy = policy
        self.partial_result = partial_result
        message = "Failure budget exceeded: %s of %s attempted items failed" % (
            failed,
            attempted,
        )
        super().__init__(message)


class CheckpointMismatchError(ExperimentExecutionError):
    """Raised on resume when the checkpoint fingerprint differs."""

    def __init__(self, field: str, expected: object, actual: object):
        self.field = field
        self.expected = expected
        self.actual = actual
        message = "Checkpoint fingerprint mismatch for %r: expected %r, got %r" % (
            field,
            expected,
            actual,
        )
        super().__init__(message)


class CheckpointCorruptedError(ExperimentExecutionError):
    """Raised when a checkpoint payload fails integrity or schema checks."""

    def __init__(self, reason: str):
        self.reason = reason
        message = "Checkpoint corrupted: %s" % reason
        super().__init__(message)
