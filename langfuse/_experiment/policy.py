"""@private Execution policies for reliable experiment runs.

All policies are opt-in. With the defaults (``max_attempts=1``, no timeouts,
no failure budget, no checkpoint) the executor reproduces today's behavior
exactly, except that failures are additionally recorded on the result object.
"""

import asyncio
import datetime
import email.utils
import random
from dataclasses import dataclass, field
from typing import Callable, Optional, Union

from langfuse._experiment.checkpoint import CheckpointStore

# Status codes that should be retried by default.
_RETRYABLE_STATUS_CODES = {408, 409, 425, 429}
_RETRYABLE_STATUS_RANGES = ((500, 599),)

# Exception class names that are treated as retryable transient errors
# across provider SDKs (openai, anthropic, litellm, ...).
_RETRYABLE_CLASS_NAMES = {
    "APIConnectionError",
    "APITimeoutError",
    "RateLimitError",
    "InternalServerError",
    "ServiceUnavailableError",
}

# Exceptions that must never be retried unless an explicit ``retry_on``
# predicate says so.
_NEVER_RETRY_CLASS_NAMES = {"ValueError", "TypeError"}


def _coerce_status_code(exc: BaseException) -> Optional[int]:
    """Best-effort extraction of an HTTP status code from an exception."""
    for attr in ("status_code", "status"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value
    response = getattr(exc, "response", None)
    if response is not None:
        for attr in ("status_code", "status"):
            value = getattr(response, attr, None)
            if isinstance(value, int):
                return value
    return None


def _coerce_headers(exc: BaseException) -> Optional[object]:
    response = getattr(exc, "response", None)
    if response is None:
        return None
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    return headers  # type: ignore[no-any-return]


def _parse_retry_after(value: str) -> Optional[float]:
    """Parse a ``Retry-After`` header value (seconds or HTTP date)."""
    value = value.strip()
    if not value:
        return None
    # Try integer/float seconds first.
    try:
        return float(value)
    except ValueError:
        pass
    # Fall back to HTTP-date.
    try:
        parsed = email.utils.parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None

    now = datetime.datetime.now(tz=datetime.timezone.utc)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=datetime.timezone.utc)
    delta = (parsed - now).total_seconds()
    if delta < 0:
        return 0.0
    return delta


def _retry_after_from_exc(exc: BaseException) -> Optional[float]:
    headers = _coerce_headers(exc)
    if headers is None:
        return None
    getter = getattr(headers, "get", None)
    if callable(getter):
        raw = getter("Retry-After")
    else:
        try:
            raw = headers["Retry-After"]  # type: ignore[index]
        except (KeyError, TypeError):
            raw = None
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    if isinstance(raw, str):
        return _parse_retry_after(raw)
    return None


def _class_name_mro_retryable(exc: BaseException) -> bool:
    """Return True if any class in the exception's MRO has a retryable name."""
    for cls in type(exc).__mro__:
        if cls.__name__ in _RETRYABLE_CLASS_NAMES:
            return True
    return False


def _class_name_mro_never_retry(exc: BaseException) -> bool:
    for cls in type(exc).__mro__:
        if cls.__name__ in _NEVER_RETRY_CLASS_NAMES:
            return True
    return False


def is_retryable_error(exc: BaseException) -> bool:
    """Default retry predicate.

    Retries transient HTTP errors (408/409/425/429/5xx), timeout and
    connection errors, and provider SDK exception classes whose names match
    the openai/anthropic/litellm naming conventions. Never retries
    ``asyncio.CancelledError``, ``KeyboardInterrupt``, or user-code
    ``ValueError``/``TypeError``.
    """
    if isinstance(exc, asyncio.CancelledError):
        return False
    if isinstance(exc, KeyboardInterrupt):
        return False

    if isinstance(exc, (asyncio.TimeoutError, TimeoutError, ConnectionError)):
        return True

    if _class_name_mro_retryable(exc):
        return True

    status = _coerce_status_code(exc)
    if status is not None:
        if status in _RETRYABLE_STATUS_CODES:
            return True
        for low, high in _RETRYABLE_STATUS_RANGES:
            if low <= status <= high:
                return True

    if _class_name_mro_never_retry(exc):
        return False

    return False


def _default_retry_on(exc: BaseException) -> bool:
    return is_retryable_error(exc)


@dataclass(frozen=True)
class RetryPolicy:
    """Retry configuration for a task or evaluator call.

    ``max_attempts`` is the total number of attempts including the first one,
    so ``max_attempts=1`` disables retries (the default).
    """

    max_attempts: int = 1
    initial_delay: float = 1.0
    backoff_multiplier: float = 2.0
    max_delay: float = 30.0
    jitter: bool = True
    retry_on: Optional[Callable[[BaseException], bool]] = None

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            message = "max_attempts must be >= 1, got %s" % self.max_attempts
            raise ValueError(message)
        if self.initial_delay < 0:
            message = "initial_delay must be >= 0, got %s" % self.initial_delay
            raise ValueError(message)
        if self.backoff_multiplier < 0:
            message = (
                "backoff_multiplier must be >= 0, got %s" % self.backoff_multiplier
            )
            raise ValueError(message)
        if self.max_delay < 0:
            message = "max_delay must be >= 0, got %s" % self.max_delay
            raise ValueError(message)

    def should_retry(self, exc: BaseException) -> bool:
        predicate = self.retry_on if self.retry_on is not None else _default_retry_on
        try:
            return bool(predicate(exc))
        except Exception:
            return False

    def delay_for(
        self,
        attempt: int,
        retry_after: Optional[float],
        rng: random.Random,
    ) -> float:
        """Compute the delay (seconds) before the next attempt.

        ``attempt`` is the 1-based index of the attempt that just failed, so
        the delay precedes attempt ``attempt + 1``.
        """
        if retry_after is not None and retry_after >= 0:
            return min(retry_after, self.max_delay)
        if attempt < 1:
            return 0.0
        # Exponential backoff: initial_delay * backoff^(attempt-1).
        delay = self.initial_delay * (self.backoff_multiplier ** (attempt - 1))
        if self.max_delay is not None:
            delay = min(delay, self.max_delay)
        if self.jitter:
            # Full jitter: uniform in [0, delay].
            delay = rng.uniform(0.0, delay) if delay > 0 else 0.0
        return delay


@dataclass(frozen=True)
class TimeoutPolicy:
    """Per-call and per-run timeout budgets (seconds)."""

    task_timeout: Optional[float] = None
    evaluator_timeout: Optional[float] = None
    run_timeout: Optional[float] = None


@dataclass(frozen=True)
class FailurePolicy:
    """Failure budget that can abort a run early.

    When the budget is exceeded the executor stops scheduling new items,
    cancels in-flight ones, and marks the result ``aborted``. ``on_threshold``
    controls whether the run is stopped silently (``"stop"``) or whether a
    ``FailureBudgetExceededError`` is raised (``"raise"``).
    """

    max_failures: Optional[int] = None
    max_failure_rate: Optional[float] = None
    min_items_for_rate: int = 10
    on_threshold: str = "stop"

    def __post_init__(self) -> None:
        if self.max_failures is not None and self.max_failures < 0:
            message = "max_failures must be >= 0, got %s" % self.max_failures
            raise ValueError(message)
        if self.max_failure_rate is not None and not (
            0.0 <= self.max_failure_rate <= 1.0
        ):
            message = "max_failure_rate must be in [0, 1], got %s" % (
                self.max_failure_rate
            )
            raise ValueError(message)
        if self.min_items_for_rate < 0:
            message = "min_items_for_rate must be >= 0, got %s" % (
                self.min_items_for_rate
            )
            raise ValueError(message)
        if self.on_threshold not in ("stop", "raise"):
            message = "on_threshold must be 'stop' or 'raise', got %r" % (
                self.on_threshold
            )
            raise ValueError(message)

    def is_exceeded(self, failed: int, attempted: int) -> bool:
        if self.max_failures is not None and failed >= self.max_failures:
            return True
        if (
            self.max_failure_rate is not None
            and attempted >= self.min_items_for_rate
            and attempted > 0
            and (failed / attempted) > self.max_failure_rate
        ):
            return True
        return False


@dataclass(frozen=True)
class ExecutionPolicy:
    """Top-level execution policy for one experiment run."""

    task_retry: RetryPolicy = field(default_factory=RetryPolicy)
    evaluator_retry: RetryPolicy = field(default_factory=RetryPolicy)
    timeouts: TimeoutPolicy = field(default_factory=TimeoutPolicy)
    failures: FailurePolicy = field(default_factory=FailurePolicy)
    checkpoint: Optional[CheckpointStore] = None
    resume: Union[bool, str] = False
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        if isinstance(self.resume, str) and self.resume not in ("force",):
            message = "resume must be bool or 'force', got %r" % (self.resume,)
            raise ValueError(message)
        if self.resume and self.checkpoint is None:
            message = "resume=True requires a checkpoint store"
            raise ValueError(message)


__all__ = [
    "RetryPolicy",
    "TimeoutPolicy",
    "FailurePolicy",
    "ExecutionPolicy",
    "is_retryable_error",
]
