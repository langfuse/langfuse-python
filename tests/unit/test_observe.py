import asyncio
import contextvars
import gc
import sys
from typing import Any, AsyncGenerator, Generator, cast

import pytest

from langfuse import observe
from langfuse._client.attributes import LangfuseOtelSpanAttributes
from langfuse._client.observe import (
    _ContextPreservedAsyncGeneratorWrapper,
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
@pytest.mark.skipif(sys.version_info < (3, 11), reason="requires python3.11 or higher")
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
