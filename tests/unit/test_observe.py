import asyncio
import contextvars
import gc
import inspect
import json
import sys
from typing import Any, AsyncGenerator, Generator, cast

import pytest

from langfuse import observe
from langfuse._client.attributes import LangfuseOtelSpanAttributes
from langfuse._client.observe import (
    _ContextPreservedAsyncGeneratorWrapper,
    _ContextPreservedAwaitable,
    _ContextPreservedSyncGeneratorWrapper,
)


class SpanRecorder:
    def __init__(self) -> None:
        self.ended = 0
        self.updates: list[dict[str, Any]] = []

    def update(self, **kwargs: Any) -> "SpanRecorder":
        self.updates.append(kwargs)
        return self

    def end(self) -> "SpanRecorder":
        self.ended += 1
        return self


def _finished_spans_by_name(memory_exporter: Any, name: str) -> list[Any]:
    return [span for span in memory_exporter.get_finished_spans() if span.name == name]


@pytest.mark.asyncio
async def test_capture_output_false_preserves_type_when_current_span_is_updated(
    langfuse_memory_client: Any, memory_exporter: Any
) -> None:
    @observe(name="guardrail_check", as_type="guardrail", capture_output=False)
    async def guardrail_check() -> bool:
        langfuse_memory_client.update_current_span(output={"verdict": "manually set"})
        return True

    assert await guardrail_check() is True

    langfuse_memory_client.flush()

    guardrail_span = _finished_spans_by_name(memory_exporter, "guardrail_check")[0]
    attributes = guardrail_span.attributes

    assert attributes[LangfuseOtelSpanAttributes.OBSERVATION_TYPE] == "guardrail"
    assert json.loads(attributes[LangfuseOtelSpanAttributes.OBSERVATION_OUTPUT]) == {
        "verdict": "manually set"
    }


def test_sync_generator_preserves_context_without_output_capture(
    langfuse_memory_client: Any, memory_exporter: Any
) -> None:
    @observe(name="child_step")
    def child_step(index: int) -> str:
        return f"item_{index}"

    @observe(name="root", capture_output=False)
    def root() -> Generator[str, None, None]:
        def body() -> Generator[str, None, None]:
            for index in range(2):
                yield child_step(index)

        return body()

    generator = root()

    assert memory_exporter.get_finished_spans() == []

    assert list(generator) == ["item_0", "item_1"]
    assert cast(Any, generator).items == []

    langfuse_memory_client.flush()

    root_span = _finished_spans_by_name(memory_exporter, "root")[0]
    child_spans = _finished_spans_by_name(memory_exporter, "child_step")

    assert len(child_spans) == 2
    assert all(child.parent is not None for child in child_spans)
    assert all(
        child.parent.span_id == root_span.context.span_id for child in child_spans
    )
    assert all(
        child.context.trace_id == root_span.context.trace_id for child in child_spans
    )
    assert LangfuseOtelSpanAttributes.OBSERVATION_OUTPUT not in root_span.attributes


@pytest.mark.asyncio
async def test_streaming_response_preserves_context_without_output_capture(
    langfuse_memory_client: Any, memory_exporter: Any
) -> None:
    class StreamingResponse:
        def __init__(self, body_iterator: AsyncGenerator[str, None]) -> None:
            self.body_iterator = body_iterator

    @observe(name="stream_step")
    async def stream_step(index: int) -> str:
        return f"chunk_{index}"

    async def body() -> AsyncGenerator[str, None]:
        for index in range(2):
            yield await stream_step(index)

    @observe(name="endpoint", capture_output=False)
    async def endpoint() -> StreamingResponse:
        return StreamingResponse(body())

    response = await endpoint()

    assert memory_exporter.get_finished_spans() == []

    assert [item async for item in response.body_iterator] == ["chunk_0", "chunk_1"]
    assert cast(Any, response.body_iterator).items == []

    langfuse_memory_client.flush()

    endpoint_span = _finished_spans_by_name(memory_exporter, "endpoint")[0]
    step_spans = _finished_spans_by_name(memory_exporter, "stream_step")

    assert len(step_spans) == 2
    assert all(step.parent is not None for step in step_spans)
    assert all(
        step.parent.span_id == endpoint_span.context.span_id for step in step_spans
    )
    assert all(
        step.context.trace_id == endpoint_span.context.trace_id for step in step_spans
    )
    assert LangfuseOtelSpanAttributes.OBSERVATION_OUTPUT not in endpoint_span.attributes


def test_sync_generator_wrapper_close_ends_span_without_exhaustion() -> None:
    def generator() -> Generator[str, None, None]:
        yield "item_0"
        yield "item_1"

    span = SpanRecorder()
    wrapper = _ContextPreservedSyncGeneratorWrapper(
        generator(),
        contextvars.copy_context(),
        cast(Any, span),
        False,
        None,
    )

    assert next(wrapper) == "item_0"

    wrapper.close()
    wrapper.close()

    assert span.ended == 1
    assert span.updates == []


def test_sync_generator_wrapper_close_preserves_context() -> None:
    marker = contextvars.ContextVar("marker", default="ambient")
    seen: list[str] = []

    def generator() -> Generator[str, None, None]:
        try:
            yield "item_0"
            yield "item_1"
        finally:
            seen.append(marker.get())

    span = SpanRecorder()
    context = contextvars.copy_context()
    context.run(marker.set, "preserved")
    wrapper = _ContextPreservedSyncGeneratorWrapper(
        generator(),
        context,
        cast(Any, span),
        False,
        None,
    )

    assert next(wrapper) == "item_0"
    marker.set("ambient-now")

    wrapper.close()

    assert seen == ["preserved"]
    assert span.ended == 1


def test_sync_generator_wrapper_del_ends_span_when_abandoned() -> None:
    def generator() -> Generator[str, None, None]:
        yield "item_0"
        yield "item_1"

    span = SpanRecorder()
    wrapper = _ContextPreservedSyncGeneratorWrapper(
        generator(),
        contextvars.copy_context(),
        cast(Any, span),
        False,
        None,
    )

    assert next(wrapper) == "item_0"

    del wrapper
    gc.collect()

    assert span.ended == 1
    assert span.updates == []


@pytest.mark.asyncio
async def test_async_generator_wrapper_aclose_ends_span_without_exhaustion() -> None:
    async def generator() -> AsyncGenerator[str, None]:
        yield "item_0"
        yield "item_1"

    span = SpanRecorder()
    wrapper = _ContextPreservedAsyncGeneratorWrapper(
        generator(),
        contextvars.copy_context(),
        cast(Any, span),
        False,
        None,
    )

    assert await wrapper.__anext__() == "item_0"

    await wrapper.aclose()
    await wrapper.close()

    assert span.ended == 1
    assert span.updates == []


@pytest.mark.asyncio
async def test_async_generator_wrapper_aclose_preserves_context() -> None:
    marker = contextvars.ContextVar("marker", default="ambient")
    seen: list[str] = []

    async def generator() -> AsyncGenerator[str, None]:
        try:
            yield "item_0"
            yield "item_1"
        finally:
            seen.append(marker.get())

    span = SpanRecorder()
    context = contextvars.copy_context()
    context.run(marker.set, "preserved")
    wrapper = _ContextPreservedAsyncGeneratorWrapper(
        generator(),
        context,
        cast(Any, span),
        False,
        None,
    )

    assert await wrapper.__anext__() == "item_0"
    marker.set("ambient-now")

    await wrapper.aclose()

    assert seen == ["preserved"]
    assert span.ended == 1


def test_sync_generator_wrapper_close_closes_generator_after_span_ended() -> None:
    marker = contextvars.ContextVar("marker", default="ambient")
    seen: list[str] = []

    def generator() -> Generator[str, None, None]:
        try:
            yield "item_0"
            yield "item_1"
        finally:
            seen.append(marker.get())

    span = SpanRecorder()
    context = contextvars.copy_context()
    context.run(marker.set, "preserved")
    wrapper = _ContextPreservedSyncGeneratorWrapper(
        generator(),
        context,
        cast(Any, span),
        False,
        None,
    )

    assert next(wrapper) == "item_0"

    # An error from __next__ that never resumed the generator ends the span.
    with pytest.raises(RuntimeError):
        context.run(lambda: next(wrapper))

    assert span.ended == 1
    assert seen == []

    marker.set("ambient-now")
    wrapper.close()

    assert seen == ["preserved"]
    assert span.ended == 1


@pytest.mark.asyncio
async def test_async_generator_wrapper_aclose_closes_generator_after_span_ended() -> (
    None
):
    marker = contextvars.ContextVar("marker", default="ambient")
    seen: list[str] = []

    async def generator() -> AsyncGenerator[str, None]:
        try:
            yield "item_0"
            yield "item_1"
        finally:
            seen.append(marker.get())

    span = SpanRecorder()
    context = contextvars.copy_context()
    context.run(marker.set, "preserved")
    wrapper = _ContextPreservedAsyncGeneratorWrapper(
        generator(),
        context,
        cast(Any, span),
        False,
        None,
    )

    assert await wrapper.__anext__() == "item_0"

    # Span ends while the generator is still suspended.
    wrapper._finalize_with_error(RuntimeError("ended early"))
    assert span.ended == 1
    assert seen == []

    marker.set("ambient-now")
    await wrapper.aclose()

    assert seen == ["preserved"]
    assert span.ended == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("without_inspect_api", [False, True])
async def test_async_generator_wrapper_cancel_runs_cleanup_before_span_end(
    monkeypatch: pytest.MonkeyPatch, without_inspect_api: bool
) -> None:
    if without_inspect_api:
        monkeypatch.delattr(inspect, "getasyncgenstate", raising=False)

    marker = contextvars.ContextVar("marker", default="ambient")
    seen: list[str] = []
    cleanup_span_states: list[int] = []
    waiting = asyncio.Event()

    async def generator() -> AsyncGenerator[str, None]:
        try:
            yield "item_0"
            waiting.set()
            await asyncio.Event().wait()
        finally:
            seen.append(marker.get())
            cleanup_span_states.append(span.ended)
            span.update(cleanup=True)

    span = SpanRecorder()
    context = contextvars.copy_context()
    context.run(marker.set, "preserved")
    raw = generator()
    wrapper = _ContextPreservedAsyncGeneratorWrapper(
        raw, context, cast(Any, span), False, None
    )

    async def consume() -> None:
        try:
            async for _ in wrapper:
                pass
        finally:
            assert marker.get() == "ambient"

    consumer = asyncio.create_task(consume())
    await waiting.wait()
    consumer.cancel()
    with pytest.raises(asyncio.CancelledError):
        await consumer

    assert raw.ag_frame is None
    assert seen == ["preserved"]
    assert cleanup_span_states == [0]
    assert span.ended == 1
    assert span.updates == [
        {"cleanup": True},
        {"level": "ERROR", "status_message": "CancelledError"},
    ]
    await wrapper.aclose()
    assert span.ended == 1


@pytest.mark.asyncio
async def test_async_generator_wrapper_aclose_propagates_cleanup_type_error() -> None:
    async def generator() -> AsyncGenerator[str, None]:
        try:
            yield "item_0"
        finally:
            raise TypeError("cleanup failed")

    span = SpanRecorder()
    wrapper = _ContextPreservedAsyncGeneratorWrapper(
        generator(),
        contextvars.copy_context(),
        cast(Any, span),
        False,
        None,
    )

    assert await wrapper.__anext__() == "item_0"

    with pytest.raises(TypeError, match="cleanup failed"):
        await wrapper.aclose()

    assert span.ended == 1
    assert span.updates[-1] == {"level": "ERROR", "status_message": "cleanup failed"}


@pytest.mark.asyncio
async def test_async_generator_wrapper_preserves_context_on_close() -> None:
    marker = contextvars.ContextVar("marker", default="ambient")
    seen: list[str] = []

    async def generator() -> AsyncGenerator[str, None]:
        try:
            yield marker.get()
            yield "item_1"
        finally:
            seen.append(marker.get())

    span = SpanRecorder()
    context = contextvars.copy_context()
    context.run(marker.set, "preserved")
    wrapper = _ContextPreservedAsyncGeneratorWrapper(
        generator(),
        context,
        cast(Any, span),
        False,
        None,
    )

    assert await wrapper.__anext__() == "preserved"
    marker.set("ambient-now")

    await wrapper.aclose()

    assert seen == ["preserved"]
    assert span.ended == 1


@pytest.mark.asyncio
async def test_async_generator_wrapper_del_ends_span_when_abandoned() -> None:
    async def generator() -> AsyncGenerator[str, None]:
        yield "item_0"
        yield "item_1"

    span = SpanRecorder()
    wrapper = _ContextPreservedAsyncGeneratorWrapper(
        generator(),
        contextvars.copy_context(),
        cast(Any, span),
        False,
        None,
    )

    assert await wrapper.__anext__() == "item_0"

    del wrapper
    gc.collect()
    await asyncio.sleep(0)

    assert span.ended == 1
    assert span.updates == []


@pytest.mark.asyncio
@pytest.mark.skipif(
    sys.version_info < (3, 11), reason="asyncio.timeout requires Python 3.11"
)
@pytest.mark.parametrize("expires", [False, True])
async def test_async_generator_wrapper_timeout_inside_generator(expires: bool) -> None:
    marker = contextvars.ContextVar("timeout-marker", default="ambient")
    tasks: list[Any] = []
    cleaned_up: list[str] = []

    async def generator() -> AsyncGenerator[str, None]:
        try:
            async with asyncio.timeout(0.01 if expires else 1):
                tasks.append(asyncio.current_task())
                yield "first"
                await asyncio.sleep(0.05 if expires else 0)
                tasks.append(asyncio.current_task())
                yield "second"
        finally:
            cleaned_up.append(marker.get())

    span = SpanRecorder()
    context = contextvars.copy_context()
    context.run(marker.set, "preserved")
    wrapper = _ContextPreservedAsyncGeneratorWrapper(
        generator(), context, cast(Any, span), True, None
    )
    assert await wrapper.__anext__() == "first"
    assert marker.get() == "ambient"
    if expires:
        with pytest.raises(asyncio.TimeoutError):
            await wrapper.__anext__()
        assert span.updates[-1] == {"level": "ERROR", "status_message": "TimeoutError"}
    else:
        assert await wrapper.__anext__() == "second"
        with pytest.raises(StopAsyncIteration):
            await wrapper.__anext__()
        assert span.updates == [{"output": "firstsecond"}]
    assert tasks == [asyncio.current_task()] * len(tasks)
    assert cleaned_up == ["preserved"]
    assert marker.get() == "ambient"
    assert span.ended == 1
    await wrapper.aclose()
    assert span.ended == 1


@pytest.mark.asyncio
async def test_async_generator_wrapper_preserves_context_tokens_across_yields() -> None:
    marker = contextvars.ContextVar("generator-marker", default="ambient")
    new_marker = contextvars.ContextVar("new-generator-marker", default="unset")
    seen: list[str] = []
    task = asyncio.current_task()

    async def generator() -> AsyncGenerator[str, None]:
        assert asyncio.current_task() is task
        token = marker.set("changed")
        new_token = new_marker.set("new-value")
        try:
            yield marker.get()
            await asyncio.sleep(0)
            yield new_marker.get()
        finally:
            assert asyncio.current_task() is task
            seen.append(marker.get())
            marker.reset(token)
            new_marker.reset(new_token)
            seen.append(marker.get())

    span = SpanRecorder()
    context = contextvars.copy_context()
    context.run(marker.set, "preserved")
    wrapper = _ContextPreservedAsyncGeneratorWrapper(
        generator(), context, cast(Any, span), False, None
    )
    assert await wrapper.__anext__() == "changed"
    assert marker.get() == "ambient"
    assert new_marker.get() == "unset"
    marker.set("caller-changed")
    assert await wrapper.__anext__() == "new-value"
    assert marker.get() == "caller-changed"
    assert new_marker.get() == "unset"
    await wrapper.aclose()
    assert seen == ["changed", "preserved"]
    assert marker.get() == "caller-changed"
    assert new_marker.get() == "unset"
    assert span.ended == 1


@pytest.mark.asyncio
async def test_context_preserved_awaitable_close_runs_cleanup_in_context() -> None:
    marker = contextvars.ContextVar("close-marker", default="ambient")
    seen: list[str] = []

    async def coroutine() -> None:
        token = marker.set("inside")
        try:
            await asyncio.sleep(0)
        finally:
            seen.append(marker.get())
            marker.reset(token)
            seen.append(marker.get())

    context = contextvars.copy_context()
    context.run(marker.set, "preserved")
    iterator = _ContextPreservedAwaitable(coroutine(), context).__await__()
    assert next(iterator) is None
    assert marker.get() == "ambient"
    iterator.close()
    assert seen == ["inside", "preserved"]
    assert marker.get() == "ambient"


@pytest.mark.asyncio
@pytest.mark.parametrize("close_early", [False, True])
async def test_observed_async_generator_keeps_child_span_across_yields(
    langfuse_memory_client: Any, memory_exporter: Any, close_early: bool
) -> None:
    task = asyncio.current_task()

    @observe(name="stream-child-step")
    async def child_step() -> str:
        await asyncio.sleep(0)
        return "second"

    @observe(name="stream-root")
    async def generator() -> AsyncGenerator[str, None]:
        with langfuse_memory_client.start_as_current_observation(name="stream-child"):
            assert asyncio.current_task() is task
            try:
                yield "first"
                yield await child_step()
            finally:
                await asyncio.sleep(0)
                langfuse_memory_client.update_current_span(output="cleanup")

    wrapper = generator()
    assert await wrapper.__anext__() == "first"
    with langfuse_memory_client.start_as_current_observation(name="caller") as caller:
        if close_early:
            await wrapper.aclose()
        else:
            assert await wrapper.__anext__() == "second"
            with pytest.raises(StopAsyncIteration):
                await wrapper.__anext__()
        assert langfuse_memory_client.get_current_observation_id() == caller.id

    langfuse_memory_client.flush()
    root_span = _finished_spans_by_name(memory_exporter, "stream-root")[0]
    child_span = _finished_spans_by_name(memory_exporter, "stream-child")[0]
    caller_span = _finished_spans_by_name(memory_exporter, "caller")[0]
    assert child_span.parent.span_id == root_span.context.span_id
    assert caller_span.parent is None
    assert (
        child_span.attributes[LangfuseOtelSpanAttributes.OBSERVATION_OUTPUT]
        == "cleanup"
    )
    assert len(_finished_spans_by_name(memory_exporter, "stream-root")) == 1
    if not close_early:
        step_span = _finished_spans_by_name(memory_exporter, "stream-child-step")[0]
        assert step_span.parent.span_id == child_span.context.span_id
