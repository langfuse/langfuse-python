"""Regression tests for cross-context span detach (langfuse/langfuse#13590).

`start_as_current_observation(end_on_exit=False)` lets `__enter__` and `__exit__`
run in different execution contexts (a manual close on another asyncio task). The
OpenTelemetry span-activation token is then detached in a context it was not
created in, and OTel's own `context.detach` logs `Failed to detach context` on
every such close. The Langfuse span-activation detach must go through
`_detach_context_token_safely`, which swallows that expected mismatch.
"""

import asyncio
import logging

from langfuse import Langfuse


def _enter_and_close_across_contexts(client: Langfuse) -> None:
    # Attach happens here, in the current (root) context.
    observation = client.start_as_current_observation(
        name="cross-context-span", end_on_exit=False
    )
    observation.__enter__()

    async def _close_on_other_task() -> None:
        # create_task copies the context, so __exit__ (and its detach) run in a
        # different context than the one __enter__ attached in.
        await asyncio.create_task(_do_exit())

    async def _do_exit() -> None:
        observation.__exit__(None, None, None)

    asyncio.run(_close_on_other_task())


def test_cross_context_close_does_not_log_detach_error(
    langfuse_memory_client: Langfuse, caplog
) -> None:
    with caplog.at_level(logging.ERROR, logger="opentelemetry.context"):
        _enter_and_close_across_contexts(langfuse_memory_client)

    assert "Failed to detach context" not in caplog.text
