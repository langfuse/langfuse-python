"""@private Shared retry/timeout helper for experiment and batch evaluation.

Both ``ExperimentExecutor`` and ``BatchEvaluationRunner`` wrap their task,
evaluator, mapper, and composite calls in :func:`_run_with_retry` so retry
and timeout behavior is defined in exactly one place.
"""

import asyncio
import random
from typing import Any, Awaitable, Callable, List, Optional, Tuple

from langfuse._experiment.policy import RetryPolicy, _retry_after_from_exc
from langfuse._experiment.results import AttemptRecord

# Attribute name used to attach the recorded attempts to the terminal
# exception so failure records can carry them.
_ATTEMPTS_ATTR = "_langfuse_attempt_records"


async def _run_with_retry(
    coro_factory: Callable[[], Awaitable[Any]],
    *,
    policy: Optional[RetryPolicy],
    timeout: Optional[float],
    clock: Callable[[], float],
    sleep: Callable[[float], Awaitable[None]],
    rng: random.Random,
    attempt_hook: Optional[Callable[[AttemptRecord], None]],
) -> Tuple[Any, List[AttemptRecord]]:
    """Run ``coro_factory()`` with retry and timeout.

    Returns ``(result, attempts)`` on success. Raises the final exception on
    terminal failure (after ``max_attempts`` or a non-retryable error); the
    recorded attempts are attached to the raised exception under
    ``_langfuse_attempt_records``.
    """
    retry_policy = policy if policy is not None else RetryPolicy()
    attempts: List[AttemptRecord] = []
    max_attempts = retry_policy.max_attempts
    last_exc: Optional[BaseException] = None

    for attempt in range(1, max_attempts + 1):
        started_at = clock()
        attempt_exc: Optional[BaseException] = None
        result: Any = None
        succeeded = False
        try:
            if timeout is not None:
                result = await asyncio.wait_for(coro_factory(), timeout=timeout)
            else:
                result = await coro_factory()
            succeeded = True
        except asyncio.TimeoutError as e:
            attempt_exc = e
            last_exc = e
        except BaseException as e:  # noqa: BLE001
            attempt_exc = e
            last_exc = e
        duration = clock() - started_at

        record = AttemptRecord(
            attempt=attempt,
            started_at=started_at,
            duration_s=duration,
            error_type=(
                type(attempt_exc).__name__ if attempt_exc is not None else None
            ),
            error_message=(str(attempt_exc) if attempt_exc is not None else None),
            retry_after=_retry_after_from_exc_safe(attempt_exc),
            will_retry=False,
        )
        if succeeded:
            attempts.append(record)
            if attempt_hook is not None:
                attempt_hook(record)
            return result, attempts

        will_retry = attempt < max_attempts and retry_policy.should_retry(
            attempt_exc  # type: ignore[arg-type]
        )
        record.will_retry = will_retry
        attempts.append(record)
        if attempt_hook is not None:
            attempt_hook(record)

        if not will_retry:
            break

        delay = retry_policy.delay_for(
            attempt=attempt, retry_after=record.retry_after, rng=rng
        )
        if delay > 0:
            await sleep(delay)

    assert last_exc is not None
    try:
        setattr(last_exc, _ATTEMPTS_ATTR, attempts)
    except Exception:
        pass
    raise last_exc


def _attempts_from_exception(exc: Optional[BaseException]) -> List[AttemptRecord]:
    """Return the attempts recorded for a terminal failure, if any."""
    if exc is None:
        return []
    return list(getattr(exc, _ATTEMPTS_ATTR, None) or [])


def _retry_after_from_exc_safe(exc: Optional[BaseException]) -> Optional[float]:
    """Best-effort ``Retry-After`` extraction; never raises."""
    if exc is None:
        return None
    try:
        return _retry_after_from_exc(exc)
    except Exception:
        return None


__all__ = [
    "_run_with_retry",
    "_attempts_from_exception",
]
