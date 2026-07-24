import asyncio
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from openai.types.responses import ParsedResponseOutputMessage, ParsedResponseOutputText
from pydantic import BaseModel

import langfuse.openai as lf_openai_module
from langfuse._client.attributes import LangfuseOtelSpanAttributes
from langfuse.api import Prompt_Text
from langfuse.model import TextPromptClient
from langfuse.openai import openai as lf_openai


class StreamAnswer(BaseModel):
    answer: int


class DummySyncResponse:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class DummyAsyncResponse:
    def __init__(self) -> None:
        self.closed = False

    async def aclose(self) -> None:
        self.closed = True


class DummyOpenAIStream(lf_openai.Stream):
    def __init__(self, items, response) -> None:
        self.response = response
        self._iterator = iter(items)


class DummyOpenAIAsyncStream(lf_openai.AsyncStream):
    def __init__(self, items, response) -> None:
        self.response = response
        self._iterator = self._stream(items)

    async def _stream(self, items):
        for item in items:
            yield item


class DummyGeneration:
    def __init__(self) -> None:
        self.end_calls = 0

    def update(self, **kwargs):
        return self

    def end(self) -> None:
        self.end_calls += 1


class DummyFallbackAsyncResponse:
    def __init__(self) -> None:
        self.close_calls = 0
        self.aclose_calls = 0

    async def close(self) -> None:
        self.close_calls += 1

    async def aclose(self) -> None:
        self.aclose_calls += 1


def _make_chat_stream_chunks():
    usage = SimpleNamespace(prompt_tokens=3, completion_tokens=1, total_tokens=4)

    return [
        SimpleNamespace(
            model="gpt-4o-mini",
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(
                        role="assistant",
                        content="2",
                        function_call=None,
                        tool_calls=None,
                    ),
                    finish_reason=None,
                )
            ],
            usage=None,
        ),
        SimpleNamespace(
            model="gpt-4o-mini",
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(
                        role=None,
                        content=None,
                        function_call=None,
                        tool_calls=None,
                    ),
                    finish_reason="stop",
                )
            ],
            usage=usage,
        ),
    ]


def _make_chat_stream_chunks_with_trailing_content_filter_chunk():
    usage = SimpleNamespace(prompt_tokens=3, completion_tokens=1, total_tokens=4)

    return [
        SimpleNamespace(
            model="gpt-4o-mini",
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(
                        role="assistant",
                        content="2",
                        function_call=None,
                        tool_calls=None,
                    ),
                    finish_reason=None,
                )
            ],
            usage=None,
        ),
        SimpleNamespace(
            model="gpt-4o-mini",
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(
                        role=None,
                        content=None,
                        function_call=None,
                        tool_calls=None,
                    ),
                    finish_reason="stop",
                )
            ],
            usage=usage,
        ),
        SimpleNamespace(
            model="",
            choices=[
                SimpleNamespace(
                    delta=None,
                    finish_reason=None,
                    content_filter_offsets={
                        "check_offset": 44,
                        "start_offset": 44,
                        "end_offset": 121,
                    },
                    content_filter_results={
                        "hate": {"filtered": False, "severity": "safe"},
                        "self_harm": {"filtered": False, "severity": "safe"},
                        "sexual": {"filtered": False, "severity": "safe"},
                        "violence": {"filtered": False, "severity": "safe"},
                    },
                )
            ],
            usage=None,
        ),
    ]


def _make_chat_stream_chunks_with_content_before_tool_call():
    usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15)

    return [
        SimpleNamespace(
            model="gpt-4o-mini",
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(
                        role="assistant",
                        content="\n\n",
                        function_call=None,
                        tool_calls=None,
                    ),
                    finish_reason=None,
                )
            ],
            usage=None,
        ),
        SimpleNamespace(
            model="gpt-4o-mini",
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(
                        role=None,
                        content=None,
                        function_call=None,
                        tool_calls=[
                            SimpleNamespace(
                                index=0,
                                id="call_weather",
                                type="function",
                                function=SimpleNamespace(
                                    name="get_weather",
                                    arguments='{"city"',
                                ),
                            )
                        ],
                    ),
                    finish_reason=None,
                )
            ],
            usage=None,
        ),
        SimpleNamespace(
            model="gpt-4o-mini",
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(
                        role=None,
                        content=None,
                        function_call=None,
                        tool_calls=[
                            SimpleNamespace(
                                index=0,
                                id=None,
                                type=None,
                                function=SimpleNamespace(
                                    name=None,
                                    arguments=': "Berlin"}',
                                ),
                            )
                        ],
                    ),
                    finish_reason="tool_calls",
                )
            ],
            usage=usage,
        ),
    ]


def _make_single_chunk_stream():
    return SimpleNamespace(
        model="gpt-4o-mini",
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(
                    role="assistant",
                    content="2",
                    function_call=None,
                    tool_calls=None,
                ),
                finish_reason="stop",
            )
        ],
        usage=None,
    )


def test_chat_completion_exports_generation_span(
    langfuse_memory_client, get_span, json_attr
):
    openai_client = lf_openai.OpenAI(api_key="test")
    response = SimpleNamespace(
        model="gpt-4o-mini",
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    role="assistant",
                    content="2",
                    function_call=None,
                    tool_calls=None,
                    audio=None,
                )
            )
        ],
        usage=SimpleNamespace(prompt_tokens=3, completion_tokens=1, total_tokens=4),
    )

    with patch.object(openai_client.chat.completions, "_post", return_value=response):
        result = openai_client.chat.completions.create(
            name="unit-openai-chat",
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "1 + 1 = ?"}],
            temperature=0,
            metadata={"suite": "unit"},
        )

    assert result is response

    langfuse_memory_client.flush()
    span = get_span("unit-openai-chat")

    assert span.attributes[LangfuseOtelSpanAttributes.OBSERVATION_TYPE] == "generation"
    assert (
        span.attributes[LangfuseOtelSpanAttributes.OBSERVATION_MODEL] == "gpt-4o-mini"
    )
    assert span.attributes["langfuse.observation.metadata.suite"] == "unit"
    assert json_attr(span, LangfuseOtelSpanAttributes.OBSERVATION_INPUT) == [
        {"role": "user", "content": "1 + 1 = ?"}
    ]
    assert json_attr(span, LangfuseOtelSpanAttributes.OBSERVATION_OUTPUT) == {
        "role": "assistant",
        "content": "2",
    }
    assert json_attr(span, LangfuseOtelSpanAttributes.OBSERVATION_MODEL_PARAMETERS) == {
        "temperature": 0,
        "max_tokens": "Infinity",
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0,
    }
    assert json_attr(span, LangfuseOtelSpanAttributes.OBSERVATION_USAGE_DETAILS) == {
        "prompt_tokens": 3,
        "completion_tokens": 1,
        "total_tokens": 4,
    }


def test_chat_completion_with_none_choices_does_not_crash(
    langfuse_memory_client, get_span, json_attr
):
    openai_client = lf_openai.OpenAI(api_key="test")
    response = SimpleNamespace(
        model="gpt-4o-mini",
        choices=None,
        usage=SimpleNamespace(prompt_tokens=3, completion_tokens=1, total_tokens=4),
    )

    with patch.object(openai_client.chat.completions, "_post", return_value=response):
        result = openai_client.chat.completions.create(
            name="unit-openai-chat-none-choices",
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "1 + 1 = ?"}],
        )

    assert result is response

    langfuse_memory_client.flush()
    span = get_span("unit-openai-chat-none-choices")

    assert LangfuseOtelSpanAttributes.OBSERVATION_LEVEL not in span.attributes
    assert (
        span.attributes[LangfuseOtelSpanAttributes.OBSERVATION_MODEL] == "gpt-4o-mini"
    )
    assert json_attr(span, LangfuseOtelSpanAttributes.OBSERVATION_USAGE_DETAILS) == {
        "prompt_tokens": 3,
        "completion_tokens": 1,
        "total_tokens": 4,
    }


def test_openai_stream_with_none_choices_chunk_does_not_crash(
    langfuse_memory_client, get_span
):
    openai_client = lf_openai.OpenAI(api_key="test")
    chunks_with_none_choices = [
        SimpleNamespace(model="gpt-4o-mini", choices=None, usage=None),
        *_make_chat_stream_chunks(),
    ]
    raw_stream = DummyOpenAIStream(chunks_with_none_choices, DummySyncResponse())

    with patch.object(openai_client.chat.completions, "_post", return_value=raw_stream):
        stream = openai_client.chat.completions.create(
            name="unit-openai-stream-none-choices",
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "1 + 1 = ?"}],
            stream=True,
        )

    chunks = list(stream)
    stream.close()

    assert len(chunks) == 3

    langfuse_memory_client.flush()
    span = get_span("unit-openai-stream-none-choices")

    assert span.attributes[LangfuseOtelSpanAttributes.OBSERVATION_OUTPUT] == "2"
    assert span.attributes["langfuse.observation.metadata.finish_reason"] == "stop"


def test_streaming_chat_completion_preserves_tool_calls_after_content():
    model, completion, usage, metadata, _service_tier = (
        lf_openai_module._extract_streamed_openai_response(
            SimpleNamespace(type="chat"),
            _make_chat_stream_chunks_with_content_before_tool_call(),
        )
    )

    assert model == "gpt-4o-mini"
    assert completion == {
        "role": "assistant",
        "content": "\n\n",
        "tool_calls": [
            {
                "id": "call_weather",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": '{"city": "Berlin"}',
                },
            }
        ],
    }
    assert usage.prompt_tokens == 10
    assert metadata == {"finish_reason": "tool_calls"}


def test_response_api_output_serializes_openai_parsed_response_objects():
    class ParsedOutput(BaseModel):
        name: str

    _, completion, _, _ = lf_openai_module._get_langfuse_data_from_default_response(
        SimpleNamespace(type="chat", object="Responses"),
        {
            "model": "gpt-4.1-mini",
            "output": [
                ParsedResponseOutputMessage(
                    id="msg_1",
                    type="message",
                    role="assistant",
                    status="completed",
                    content=[
                        ParsedResponseOutputText(
                            annotations=[],
                            text='{"name":"dave"}',
                            type="output_text",
                            parsed=ParsedOutput(name="dave"),
                        )
                    ],
                )
            ],
            "usage": None,
        },
    )

    assert completion == {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "content": [
            {
                "annotations": [],
                "text": '{"name":"dave"}',
                "type": "output_text",
                "logprobs": None,
                "parsed": {"name": "dave"},
            }
        ],
        "phase": None,
    }


def test_streaming_chat_completion_exports_ttft(
    langfuse_memory_client, get_span, json_attr
):
    openai_client = lf_openai.OpenAI(api_key="test")
    usage = SimpleNamespace(prompt_tokens=3, completion_tokens=1, total_tokens=4)

    def fake_stream():
        yield SimpleNamespace(
            model="gpt-4o-mini",
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(
                        role="assistant",
                        content="2",
                        function_call=None,
                        tool_calls=None,
                    ),
                    finish_reason=None,
                )
            ],
            usage=None,
        )
        yield SimpleNamespace(
            model="gpt-4o-mini",
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(
                        role=None,
                        content=None,
                        function_call=None,
                        tool_calls=None,
                    ),
                    finish_reason="stop",
                )
            ],
            usage=usage,
        )

    with patch.object(
        openai_client.chat.completions, "_post", return_value=fake_stream()
    ):
        stream = openai_client.chat.completions.create(
            name="unit-openai-stream",
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "1 + 1 = ?"}],
            temperature=0,
            stream=True,
        )
        chunks = list(stream)

    assert len(chunks) == 2

    langfuse_memory_client.flush()
    span = get_span("unit-openai-stream")

    assert span.attributes[LangfuseOtelSpanAttributes.OBSERVATION_OUTPUT] == "2"
    assert (
        span.attributes[LangfuseOtelSpanAttributes.OBSERVATION_COMPLETION_START_TIME]
        is not None
    )
    assert span.attributes["langfuse.observation.metadata.finish_reason"] == "stop"
    assert json_attr(span, LangfuseOtelSpanAttributes.OBSERVATION_USAGE_DETAILS) == {
        "prompt_tokens": 3,
        "completion_tokens": 1,
        "total_tokens": 4,
    }


def test_chat_completion_error_marks_generation_error(langfuse_memory_client, get_span):
    openai_client = lf_openai.OpenAI(api_key="test")

    with patch.object(
        openai_client.chat.completions,
        "_post",
        side_effect=RuntimeError("boom"),
    ):
        with pytest.raises(RuntimeError, match="boom"):
            openai_client.chat.completions.create(
                name="unit-openai-error",
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "explode"}],
                temperature=0,
            )

    langfuse_memory_client.flush()
    span = get_span("unit-openai-error")

    assert span.attributes[LangfuseOtelSpanAttributes.OBSERVATION_LEVEL] == "ERROR"
    assert (
        "boom" in span.attributes[LangfuseOtelSpanAttributes.OBSERVATION_STATUS_MESSAGE]
    )
    assert LangfuseOtelSpanAttributes.OBSERVATION_OUTPUT not in span.attributes


def test_openai_stream_preserves_original_stream_contract(
    langfuse_memory_client, get_span, json_attr
):
    openai_client = lf_openai.OpenAI(api_key="test")
    raw_response = DummySyncResponse()
    raw_stream = DummyOpenAIStream(_make_chat_stream_chunks(), raw_response)

    with patch.object(openai_client.chat.completions, "_post", return_value=raw_stream):
        stream = openai_client.chat.completions.create(
            name="unit-openai-native-stream",
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "1 + 1 = ?"}],
            temperature=0,
            stream=True,
        )

    assert stream is raw_stream
    assert isinstance(stream, lf_openai.Stream)
    assert stream.response is raw_response

    chunks = list(stream)
    stream.close()

    assert len(chunks) == 2
    assert raw_response.closed is True

    langfuse_memory_client.flush()
    span = get_span("unit-openai-native-stream")

    assert span.attributes[LangfuseOtelSpanAttributes.OBSERVATION_OUTPUT] == "2"
    assert (
        span.attributes[LangfuseOtelSpanAttributes.OBSERVATION_COMPLETION_START_TIME]
        is not None
    )
    assert span.attributes["langfuse.observation.metadata.finish_reason"] == "stop"
    assert json_attr(span, LangfuseOtelSpanAttributes.OBSERVATION_USAGE_DETAILS) == {
        "prompt_tokens": 3,
        "completion_tokens": 1,
        "total_tokens": 4,
    }


def test_openai_stream_handles_trailing_azure_content_filter_chunk(
    langfuse_memory_client, get_span, json_attr
):
    openai_client = lf_openai.OpenAI(api_key="test")
    raw_stream = DummyOpenAIStream(
        _make_chat_stream_chunks_with_trailing_content_filter_chunk(),
        DummySyncResponse(),
    )

    with patch.object(openai_client.chat.completions, "_post", return_value=raw_stream):
        stream = openai_client.chat.completions.create(
            name="unit-openai-native-stream-azure-filter",
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "1 + 1 = ?"}],
            temperature=0,
            stream=True,
        )

    chunks = list(stream)
    stream.close()

    assert len(chunks) == 3

    langfuse_memory_client.flush()
    span = get_span("unit-openai-native-stream-azure-filter")

    assert span.attributes[LangfuseOtelSpanAttributes.OBSERVATION_OUTPUT] == "2"
    assert span.attributes["langfuse.observation.metadata.finish_reason"] == "stop"
    assert json_attr(span, LangfuseOtelSpanAttributes.OBSERVATION_USAGE_DETAILS) == {
        "prompt_tokens": 3,
        "completion_tokens": 1,
        "total_tokens": 4,
    }


def test_openai_stream_break_still_finalizes_generation(
    langfuse_memory_client, get_span
):
    openai_client = lf_openai.OpenAI(api_key="test")
    raw_response = DummySyncResponse()
    raw_stream = DummyOpenAIStream(_make_chat_stream_chunks(), raw_response)

    with patch.object(openai_client.chat.completions, "_post", return_value=raw_stream):
        stream = openai_client.chat.completions.create(
            name="unit-openai-native-stream-break",
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "1 + 1 = ?"}],
            temperature=0,
            stream=True,
        )

    for chunk in stream:
        assert chunk.choices[0].delta.content == "2"
        break

    assert raw_response.closed is False

    langfuse_memory_client.flush()
    span = get_span("unit-openai-native-stream-break")

    assert span.attributes[LangfuseOtelSpanAttributes.OBSERVATION_OUTPUT] == "2"
    assert (
        span.attributes[LangfuseOtelSpanAttributes.OBSERVATION_COMPLETION_START_TIME]
        is not None
    )


@pytest.mark.asyncio
async def test_async_chat_completion_exports_generation_span(
    langfuse_memory_client, get_span, json_attr
):
    openai_client = lf_openai.AsyncOpenAI(api_key="test")
    response = SimpleNamespace(
        model="gpt-4o-mini",
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    role="assistant",
                    content="async result",
                    function_call=None,
                    tool_calls=None,
                    audio=None,
                )
            )
        ],
        usage=SimpleNamespace(prompt_tokens=5, completion_tokens=2, total_tokens=7),
    )

    with patch.object(openai_client.chat.completions, "_post", return_value=response):
        result = await openai_client.chat.completions.create(
            name="unit-openai-async",
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "hello"}],
            temperature=0,
        )

    assert result is response

    langfuse_memory_client.flush()
    span = get_span("unit-openai-async")

    assert json_attr(span, LangfuseOtelSpanAttributes.OBSERVATION_OUTPUT) == {
        "role": "assistant",
        "content": "async result",
    }
    assert json_attr(span, LangfuseOtelSpanAttributes.OBSERVATION_USAGE_DETAILS) == {
        "prompt_tokens": 5,
        "completion_tokens": 2,
        "total_tokens": 7,
    }


@pytest.mark.asyncio
async def test_openai_async_stream_preserves_original_stream_contract(
    langfuse_memory_client, get_span, json_attr
):
    openai_client = lf_openai.AsyncOpenAI(api_key="test")
    raw_response = DummyAsyncResponse()
    raw_stream = DummyOpenAIAsyncStream(_make_chat_stream_chunks(), raw_response)

    with patch.object(openai_client.chat.completions, "_post", return_value=raw_stream):
        stream = await openai_client.chat.completions.create(
            name="unit-openai-native-async-stream",
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "1 + 1 = ?"}],
            temperature=0,
            stream=True,
        )

    assert stream is raw_stream
    assert isinstance(stream, lf_openai.AsyncStream)
    assert stream.response is raw_response
    assert hasattr(stream, "aclose")

    chunks = []
    async for chunk in stream:
        chunks.append(chunk)

    await stream.aclose()

    assert len(chunks) == 2
    assert raw_response.closed is True

    langfuse_memory_client.flush()
    span = get_span("unit-openai-native-async-stream")

    assert span.attributes[LangfuseOtelSpanAttributes.OBSERVATION_OUTPUT] == "2"
    assert (
        span.attributes[LangfuseOtelSpanAttributes.OBSERVATION_COMPLETION_START_TIME]
        is not None
    )
    assert span.attributes["langfuse.observation.metadata.finish_reason"] == "stop"
    assert json_attr(span, LangfuseOtelSpanAttributes.OBSERVATION_USAGE_DETAILS) == {
        "prompt_tokens": 3,
        "completion_tokens": 1,
        "total_tokens": 4,
    }


@pytest.mark.asyncio
async def test_openai_async_stream_supports_anext(
    langfuse_memory_client, get_span, json_attr
):
    openai_client = lf_openai.AsyncOpenAI(api_key="test")
    raw_stream = DummyOpenAIAsyncStream(
        _make_chat_stream_chunks(), DummyAsyncResponse()
    )

    with patch.object(openai_client.chat.completions, "_post", return_value=raw_stream):
        stream = await openai_client.chat.completions.create(
            name="unit-openai-native-async-anext",
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "1 + 1 = ?"}],
            temperature=0,
            stream=True,
        )

    first = await stream.__anext__()
    second = await stream.__anext__()

    assert first.choices[0].delta.content == "2"
    assert second.choices[0].finish_reason == "stop"

    with pytest.raises(StopAsyncIteration):
        await stream.__anext__()

    langfuse_memory_client.flush()
    span = get_span("unit-openai-native-async-anext")

    assert span.attributes[LangfuseOtelSpanAttributes.OBSERVATION_OUTPUT] == "2"
    assert (
        span.attributes[LangfuseOtelSpanAttributes.OBSERVATION_COMPLETION_START_TIME]
        is not None
    )
    assert span.attributes["langfuse.observation.metadata.finish_reason"] == "stop"
    assert json_attr(span, LangfuseOtelSpanAttributes.OBSERVATION_USAGE_DETAILS) == {
        "prompt_tokens": 3,
        "completion_tokens": 1,
        "total_tokens": 4,
    }


@pytest.mark.asyncio
async def test_openai_async_stream_break_still_finalizes_generation(
    langfuse_memory_client, get_span
):
    openai_client = lf_openai.AsyncOpenAI(api_key="test")
    raw_stream = DummyOpenAIAsyncStream(
        _make_chat_stream_chunks(), DummyAsyncResponse()
    )

    with patch.object(openai_client.chat.completions, "_post", return_value=raw_stream):
        stream = await openai_client.chat.completions.create(
            name="unit-openai-native-async-stream-break",
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "1 + 1 = ?"}],
            temperature=0,
            stream=True,
        )

    async for chunk in stream:
        assert chunk.choices[0].delta.content == "2"
        break

    # Async generator finalizers are scheduled across event-loop turns.
    for _ in range(5):
        await asyncio.sleep(0)

    langfuse_memory_client.flush()
    span = get_span("unit-openai-native-async-stream-break")

    assert span.attributes[LangfuseOtelSpanAttributes.OBSERVATION_OUTPUT] == "2"
    assert (
        span.attributes[LangfuseOtelSpanAttributes.OBSERVATION_COMPLETION_START_TIME]
        is not None
    )


def test_fallback_sync_stream_finalizes_once():
    resource = SimpleNamespace(object="Completions", type="chat")
    generation = DummyGeneration()

    def fallback_stream():
        yield _make_single_chunk_stream()

    wrapper = lf_openai_module.LangfuseResponseGeneratorSync(
        resource=resource,
        response=fallback_stream(),
        generation=generation,
    )

    list(wrapper)

    with pytest.raises(StopIteration):
        next(wrapper)

    assert generation.end_calls == 1


def test_fallback_sync_stream_exit_finalizes_once():
    resource = SimpleNamespace(object="Completions", type="chat")
    generation = DummyGeneration()

    def fallback_stream():
        yield _make_single_chunk_stream()

    wrapper = lf_openai_module.LangfuseResponseGeneratorSync(
        resource=resource,
        response=fallback_stream(),
        generation=generation,
    )

    wrapper.__exit__(None, None, None)

    assert generation.end_calls == 1


@pytest.mark.asyncio
async def test_fallback_async_stream_finalizes_once():
    resource = SimpleNamespace(object="Completions", type="chat")
    generation = DummyGeneration()

    async def fallback_stream():
        yield _make_single_chunk_stream()

    wrapper = lf_openai_module.LangfuseResponseGeneratorAsync(
        resource=resource,
        response=fallback_stream(),
        generation=generation,
    )

    async for _ in wrapper:
        pass

    with pytest.raises(StopAsyncIteration):
        await wrapper.__anext__()

    assert generation.end_calls == 1


@pytest.mark.asyncio
async def test_fallback_async_stream_close_and_exit_finalize_once():
    resource = SimpleNamespace(object="Completions", type="chat")
    generation = DummyGeneration()
    response = DummyFallbackAsyncResponse()

    wrapper = lf_openai_module.LangfuseResponseGeneratorAsync(
        resource=resource,
        response=response,
        generation=generation,
    )

    await wrapper.close()
    await wrapper.__aexit__(None, None, None)

    assert generation.end_calls == 1
    assert response.close_calls == 1
    assert response.aclose_calls == 1


@pytest.mark.asyncio
async def test_fallback_async_stream_aclose_finalizes_once():
    resource = SimpleNamespace(object="Completions", type="chat")
    generation = DummyGeneration()

    async def fallback_stream():
        yield _make_single_chunk_stream()

    wrapper = lf_openai_module.LangfuseResponseGeneratorAsync(
        resource=resource,
        response=fallback_stream(),
        generation=generation,
    )

    await wrapper.aclose()

    assert generation.end_calls == 1


def test_embedding_exports_dimensions_and_count(
    langfuse_memory_client, get_span, json_attr
):
    openai_client = lf_openai.OpenAI(api_key="test")
    response = SimpleNamespace(
        model="text-embedding-3-small",
        data=[SimpleNamespace(embedding=[0.1, 0.2, 0.3])],
        usage=SimpleNamespace(prompt_tokens=2, total_tokens=2),
    )

    with patch.object(openai_client.embeddings, "_post", return_value=response):
        result = openai_client.embeddings.create(
            name="unit-openai-embedding",
            model="text-embedding-3-small",
            input="hello world",
        )

    assert result is response

    langfuse_memory_client.flush()
    span = get_span("unit-openai-embedding")

    assert span.attributes[LangfuseOtelSpanAttributes.OBSERVATION_TYPE] == "embedding"
    assert json_attr(span, LangfuseOtelSpanAttributes.OBSERVATION_OUTPUT) == {
        "dimensions": 3,
        "count": 1,
    }
    assert json_attr(span, LangfuseOtelSpanAttributes.OBSERVATION_USAGE_DETAILS) == {
        "input": 2
    }


def _make_chat_response(**extra_response_fields):
    return SimpleNamespace(
        model="gpt-4o-mini",
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    role="assistant",
                    content="2",
                    function_call=None,
                    tool_calls=None,
                    audio=None,
                )
            )
        ],
        usage=SimpleNamespace(prompt_tokens=3, completion_tokens=1, total_tokens=4),
        **extra_response_fields,
    )


def test_chat_completion_captures_request_service_tier(
    langfuse_memory_client, get_span, json_attr
):
    openai_client = lf_openai.OpenAI(api_key="test")
    response = _make_chat_response()

    with patch.object(openai_client.chat.completions, "_post", return_value=response):
        openai_client.chat.completions.create(
            name="unit-openai-service-tier-request",
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "1 + 1 = ?"}],
            temperature=0,
            service_tier="flex",
        )

    langfuse_memory_client.flush()
    span = get_span("unit-openai-service-tier-request")

    model_parameters = json_attr(
        span, LangfuseOtelSpanAttributes.OBSERVATION_MODEL_PARAMETERS
    )
    assert model_parameters["service_tier"] == "flex"
    assert model_parameters["temperature"] == 0


def test_chat_completion_service_tier_not_given_is_absent(
    langfuse_memory_client, get_span, json_attr
):
    from openai._types import NOT_GIVEN

    openai_client = lf_openai.OpenAI(api_key="test")
    response = _make_chat_response()

    with patch.object(openai_client.chat.completions, "_post", return_value=response):
        openai_client.chat.completions.create(
            name="unit-openai-service-tier-not-given",
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "1 + 1 = ?"}],
            temperature=0,
            service_tier=NOT_GIVEN,
        )

    langfuse_memory_client.flush()
    span = get_span("unit-openai-service-tier-not-given")

    model_parameters = json_attr(
        span, LangfuseOtelSpanAttributes.OBSERVATION_MODEL_PARAMETERS
    )
    assert "service_tier" not in model_parameters


def test_chat_completion_service_tier_absent_by_default(
    langfuse_memory_client, get_span, json_attr
):
    openai_client = lf_openai.OpenAI(api_key="test")
    response = _make_chat_response()

    with patch.object(openai_client.chat.completions, "_post", return_value=response):
        openai_client.chat.completions.create(
            name="unit-openai-service-tier-absent",
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "1 + 1 = ?"}],
            temperature=0,
        )

    langfuse_memory_client.flush()
    span = get_span("unit-openai-service-tier-absent")

    model_parameters = json_attr(
        span, LangfuseOtelSpanAttributes.OBSERVATION_MODEL_PARAMETERS
    )
    assert "service_tier" not in model_parameters


def test_chat_completion_response_service_tier_overrides_request(
    langfuse_memory_client, get_span, json_attr
):
    openai_client = lf_openai.OpenAI(api_key="test")
    response = _make_chat_response(service_tier="default")

    with patch.object(openai_client.chat.completions, "_post", return_value=response):
        openai_client.chat.completions.create(
            name="unit-openai-service-tier-override",
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "1 + 1 = ?"}],
            temperature=0,
            service_tier="auto",
        )

    langfuse_memory_client.flush()
    span = get_span("unit-openai-service-tier-override")

    model_parameters = json_attr(
        span, LangfuseOtelSpanAttributes.OBSERVATION_MODEL_PARAMETERS
    )
    # Response value is authoritative: it reflects the tier actually used.
    assert model_parameters["service_tier"] == "default"
    # Request-side model parameters must be preserved (merge, not clobber).
    assert model_parameters["temperature"] == 0
    assert model_parameters["top_p"] == 1


@pytest.mark.asyncio
async def test_async_chat_completion_response_service_tier_overrides_request(
    langfuse_memory_client, get_span, json_attr
):
    openai_client = lf_openai.AsyncOpenAI(api_key="test")
    response = _make_chat_response(service_tier="priority")

    with patch.object(openai_client.chat.completions, "_post", return_value=response):
        await openai_client.chat.completions.create(
            name="unit-openai-async-service-tier-override",
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "1 + 1 = ?"}],
            temperature=0,
            service_tier="auto",
        )

    langfuse_memory_client.flush()
    span = get_span("unit-openai-async-service-tier-override")

    model_parameters = json_attr(
        span, LangfuseOtelSpanAttributes.OBSERVATION_MODEL_PARAMETERS
    )
    assert model_parameters["service_tier"] == "priority"
    assert model_parameters["temperature"] == 0


def test_openai_stream_captures_service_tier_from_chunks(
    langfuse_memory_client, get_span, json_attr
):
    openai_client = lf_openai.OpenAI(api_key="test")
    chunks = _make_chat_stream_chunks()
    for chunk in chunks:
        chunk.service_tier = "default"
    raw_stream = DummyOpenAIStream(chunks, DummySyncResponse())

    with patch.object(openai_client.chat.completions, "_post", return_value=raw_stream):
        stream = openai_client.chat.completions.create(
            name="unit-openai-stream-service-tier",
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "1 + 1 = ?"}],
            temperature=0,
            service_tier="auto",
            stream=True,
        )

    list(stream)
    stream.close()

    langfuse_memory_client.flush()
    span = get_span("unit-openai-stream-service-tier")

    model_parameters = json_attr(
        span, LangfuseOtelSpanAttributes.OBSERVATION_MODEL_PARAMETERS
    )
    assert model_parameters["service_tier"] == "default"
    assert model_parameters["temperature"] == 0


@pytest.mark.asyncio
async def test_openai_async_stream_captures_service_tier_from_chunks(
    langfuse_memory_client, get_span, json_attr
):
    openai_client = lf_openai.AsyncOpenAI(api_key="test")
    chunks = _make_chat_stream_chunks()
    for chunk in chunks:
        chunk.service_tier = "flex"
    raw_stream = DummyOpenAIAsyncStream(chunks, DummyAsyncResponse())

    with patch.object(openai_client.chat.completions, "_post", return_value=raw_stream):
        stream = await openai_client.chat.completions.create(
            name="unit-openai-async-stream-service-tier",
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "1 + 1 = ?"}],
            temperature=0,
            stream=True,
        )

    async for _ in stream:
        pass

    await stream.aclose()

    langfuse_memory_client.flush()
    span = get_span("unit-openai-async-stream-service-tier")

    model_parameters = json_attr(
        span, LangfuseOtelSpanAttributes.OBSERVATION_MODEL_PARAMETERS
    )
    assert model_parameters["service_tier"] == "flex"
    assert model_parameters["temperature"] == 0


def test_embedding_model_parameters_do_not_include_service_tier(
    langfuse_memory_client, get_span, json_attr
):
    openai_client = lf_openai.OpenAI(api_key="test")
    response = SimpleNamespace(
        model="text-embedding-3-small",
        data=[SimpleNamespace(embedding=[0.1, 0.2, 0.3])],
        usage=SimpleNamespace(prompt_tokens=2, total_tokens=2),
    )

    with patch.object(openai_client.embeddings, "_post", return_value=response):
        openai_client.embeddings.create(
            name="unit-openai-embedding-no-service-tier",
            model="text-embedding-3-small",
            input="hello world",
            dimensions=3,
        )

    langfuse_memory_client.flush()
    span = get_span("unit-openai-embedding-no-service-tier")

    model_parameters = json_attr(
        span, LangfuseOtelSpanAttributes.OBSERVATION_MODEL_PARAMETERS
    )
    assert model_parameters == {"dimensions": 3}


def test_default_response_extraction_returns_service_tier():
    (
        model,
        _completion,
        _usage,
        service_tier,
    ) = lf_openai_module._get_langfuse_data_from_default_response(
        SimpleNamespace(type="chat", object="Responses"),
        {
            "model": "gpt-4.1-mini",
            "output": [],
            "usage": None,
            "service_tier": "flex",
        },
    )

    assert model == "gpt-4.1-mini"
    assert service_tier == "flex"


def test_streamed_response_api_extraction_returns_service_tier():
    final_response = SimpleNamespace(
        model="gpt-4.1-mini",
        output=[],
        usage=SimpleNamespace(input_tokens=1, output_tokens=1, total_tokens=2),
        service_tier="priority",
        created_at=1700000000,
        text=None,
    )
    chunks = [
        SimpleNamespace(type="response.completed", response=final_response),
    ]

    (
        model,
        _completion,
        _usage,
        _metadata,
        service_tier,
    ) = lf_openai_module._extract_streamed_response_api_response(chunks)

    assert model == "gpt-4.1-mini"
    assert service_tier == "priority"


def _chat_completion_payload():
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "gpt-4o-mini-2024-07-18",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "2"},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 1,
            "total_tokens": 11,
            "prompt_tokens_details": {"cached_tokens": 4, "audio_tokens": 0},
        },
    }


def _chat_completion_chunk_sse_body(content: str = "2"):
    chunks = [
        {
            "id": "chatcmpl-test",
            "object": "chat.completion.chunk",
            "created": 1700000000,
            "model": "sample-model",
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": content},
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": "chatcmpl-test",
            "object": "chat.completion.chunk",
            "created": 1700000000,
            "model": "sample-model",
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop",
                }
            ],
        },
        {
            "id": "chatcmpl-test",
            "object": "chat.completion.chunk",
            "created": 1700000000,
            "model": "sample-model",
            "choices": [],
            "usage": {
                "prompt_tokens": 3,
                "completion_tokens": 1,
                "total_tokens": 4,
            },
        },
    ]

    return "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks) + (
        "data: [DONE]\n\n"
    )


def _response_stream_sse_body():
    return (
        'data: {"type":"response.created","sequence_number":0,"response":'
        '{"id":"resp-test","object":"response","created_at":1700000000,'
        '"model":"sample-model","output":[],"parallel_tool_calls":true,'
        '"tool_choice":"auto","tools":[],"status":"in_progress"}}\n\n'
        'data: {"type":"response.completed","sequence_number":1,"response":'
        '{"id":"resp-test","object":"response","created_at":1700000000,'
        '"model":"sample-model","output":[{"id":"msg-test","type":"message",'
        '"status":"completed","role":"assistant","content":[{"type":"output_text",'
        '"text":"Hello","annotations":[]}]}],"parallel_tool_calls":true,'
        '"tool_choice":"auto","tools":[],"status":"completed","usage":'
        '{"input_tokens":3,"input_tokens_details":{"cached_tokens":0},'
        '"output_tokens":1,"output_tokens_details":{"reasoning_tokens":0},'
        '"total_tokens":4}}}\n\n'
        "data: [DONE]\n\n"
    )


def _mock_transport_openai_client(
    async_client: bool = False, request_bodies=None, chat_stream_content: str = "2"
):
    import httpx

    def handler(request: httpx.Request) -> httpx.Response:
        if request_bodies is not None:
            request_bodies.append(json.loads(request.content))

        if request.url.path.endswith("/responses"):
            return httpx.Response(
                200,
                content=_response_stream_sse_body().encode(),
                headers={"content-type": "text/event-stream"},
            )

        if b'"stream": true' in request.content or b'"stream":true' in request.content:
            return httpx.Response(
                200,
                content=_chat_completion_chunk_sse_body(chat_stream_content).encode(),
                headers={"content-type": "text/event-stream"},
            )

        return httpx.Response(200, json=_chat_completion_payload())

    if async_client:
        return lf_openai.AsyncOpenAI(
            api_key="test",
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        )

    return lf_openai.OpenAI(
        api_key="test",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )


def _stream_manager(openai_client, resource, name):
    if resource == "chat":
        return openai_client.chat.completions.stream(
            name=name,
            model="sample-model",
            messages=[{"role": "user", "content": "Hello"}],
        )

    return openai_client.responses.stream(
        name=name,
        model="sample-model",
        input="Hello",
    )


def test_openai_stream_helpers_accept_langfuse_arguments(
    langfuse_memory_client, find_spans, get_span, json_attr
):
    request_bodies = []
    openai_client = _mock_transport_openai_client(
        request_bodies=request_bodies,
        chat_stream_content='{"answer":2}',
    )
    prompt = TextPromptClient(
        Prompt_Text(
            name="stream-helper-prompt",
            version=3,
            prompt="Hello",
            type="text",
            labels=[],
            config={},
            tags=[],
        )
    )
    trace_id = "1" * 32
    parent_observation_id = "2" * 16
    extra_body = {"custom_field": "kept"}

    with patch.object(
        lf_openai_module, "get_client", return_value=langfuse_memory_client
    ) as get_client_mock:
        with openai_client.chat.completions.stream(
            name="unit-openai-chat-stream-helper",
            langfuse_prompt=prompt,
            langfuse_public_key="stream-public-key",
            trace_id=trace_id,
            parent_observation_id=parent_observation_id,
            metadata={"stream": "helper"},
            extra_body=extra_body,
            model="sample-model",
            messages=[{"role": "user", "content": "Hello"}],
            response_format=StreamAnswer,
            stream_options={"include_usage": True},
        ) as stream:
            completion = stream.get_final_completion()

    get_client_mock.assert_called_once_with(public_key="stream-public-key")

    assert completion.model == "sample-model"
    assert completion.choices[0].message.content == '{"answer":2}'
    assert completion.choices[0].message.parsed == StreamAnswer(answer=2)
    assert extra_body == {"custom_field": "kept"}
    assert request_bodies[0]["custom_field"] == "kept"
    assert all(
        name not in request_bodies[0]
        for name in (
            "name",
            "langfuse_prompt",
            "langfuse_public_key",
            "trace_id",
            "parent_observation_id",
        )
    )

    with openai_client.responses.stream(
        name="unit-openai-responses-stream-helper",
        model="sample-model",
        input="Hello",
    ) as stream:
        response = stream.get_final_response()

    assert response.model == "sample-model"
    assert response.status == "completed"
    assert response.output_text == "Hello"

    langfuse_memory_client.flush()
    assert len(find_spans("unit-openai-chat-stream-helper")) == 1
    assert len(find_spans("unit-openai-responses-stream-helper")) == 1
    assert find_spans("OpenAI-generation") == []

    chat_span = get_span("unit-openai-chat-stream-helper")
    assert (
        chat_span.attributes[LangfuseOtelSpanAttributes.OBSERVATION_PROMPT_NAME]
        == "stream-helper-prompt"
    )
    assert (
        chat_span.attributes[LangfuseOtelSpanAttributes.OBSERVATION_PROMPT_VERSION] == 3
    )
    assert chat_span.attributes["langfuse.observation.metadata.stream"] == "helper"
    assert json_attr(chat_span, LangfuseOtelSpanAttributes.OBSERVATION_INPUT) == [
        {"role": "user", "content": "Hello"}
    ]
    assert json_attr(
        chat_span, LangfuseOtelSpanAttributes.OBSERVATION_MODEL_PARAMETERS
    ) == {
        "temperature": 1,
        "max_tokens": "Infinity",
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0,
    }
    assert (
        chat_span.attributes[LangfuseOtelSpanAttributes.OBSERVATION_OUTPUT]
        == '{"answer":2}'
    )
    assert (
        chat_span.attributes[
            LangfuseOtelSpanAttributes.OBSERVATION_COMPLETION_START_TIME
        ]
        is not None
    )
    assert json_attr(
        chat_span, LangfuseOtelSpanAttributes.OBSERVATION_USAGE_DETAILS
    ) == {
        "prompt_tokens": 3,
        "completion_tokens": 1,
        "total_tokens": 4,
        "prompt_tokens_details": None,
        "completion_tokens_details": None,
    }
    assert format(chat_span.context.trace_id, "032x") == trace_id
    assert chat_span.parent is not None
    assert format(chat_span.parent.span_id, "016x") == parent_observation_id

    responses_span = get_span("unit-openai-responses-stream-helper")
    assert (
        responses_span.attributes[LangfuseOtelSpanAttributes.OBSERVATION_INPUT]
        == "Hello"
    )
    assert json_attr(
        responses_span, LangfuseOtelSpanAttributes.OBSERVATION_MODEL_PARAMETERS
    ) == {
        "temperature": 1,
        "max_tokens": "Infinity",
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0,
    }
    assert json_attr(responses_span, LangfuseOtelSpanAttributes.OBSERVATION_OUTPUT) == {
        "id": "msg-test",
        "content": [
            {
                "annotations": [],
                "text": "Hello",
                "type": "output_text",
                "logprobs": None,
            }
        ],
        "role": "assistant",
        "status": "completed",
        "type": "message",
        "phase": None,
    }
    assert json_attr(
        responses_span, LangfuseOtelSpanAttributes.OBSERVATION_USAGE_DETAILS
    ) == {
        "input_tokens": 3,
        "input_tokens_details": {"cached_tokens": 0},
        "output_tokens": 1,
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": 4,
    }


@pytest.mark.asyncio
async def test_async_openai_stream_helpers_accept_langfuse_arguments(
    langfuse_memory_client, find_spans
):
    openai_client = _mock_transport_openai_client(async_client=True)

    async with openai_client.chat.completions.stream(
        name="unit-openai-async-chat-stream-helper",
        model="sample-model",
        messages=[{"role": "user", "content": "Hello"}],
    ) as stream:
        completion = await stream.get_final_completion()

    assert completion.model == "sample-model"
    assert completion.choices[0].message.content == "2"

    async with openai_client.responses.stream(
        name="unit-openai-async-responses-stream-helper",
        model="sample-model",
        input="Hello",
    ) as stream:
        response = await stream.get_final_response()

    assert response.model == "sample-model"
    assert response.status == "completed"

    langfuse_memory_client.flush()
    assert len(find_spans("unit-openai-async-chat-stream-helper")) == 1
    assert len(find_spans("unit-openai-async-responses-stream-helper")) == 1
    assert find_spans("OpenAI-generation") == []


@pytest.mark.parametrize("omitted_argument", ["response_id", "starting_after"])
@pytest.mark.asyncio
async def test_async_responses_stream_treats_openai_omit_as_absent(
    omitted_argument, langfuse_memory_client, find_spans
):
    openai_client = _mock_transport_openai_client(async_client=True)
    name = f"unit-openai-responses-stream-omit-{omitted_argument}"

    async with openai_client.responses.stream(
        name=name,
        model="sample-model",
        input="Hello",
        **{omitted_argument: lf_openai.omit},
    ) as stream:
        response = await stream.get_final_response()

    assert response.status == "completed"

    langfuse_memory_client.flush()
    assert len(find_spans(name)) == 1


@pytest.mark.parametrize("resource", ["chat", "responses"])
@pytest.mark.parametrize(
    "consume_one_event", [False, True], ids=["unconsumed", "partial"]
)
def test_stream_helpers_finalize_on_context_exit(
    resource, consume_one_event, langfuse_memory_client, find_spans
):
    openai_client = _mock_transport_openai_client()
    name = f"unit-openai-{resource}-stream-context-exit-{consume_one_event}"
    manager = _stream_manager(openai_client, resource, name)

    with manager as stream:
        if consume_one_event:
            next(stream)

    langfuse_memory_client.flush()
    assert len(find_spans(name)) == 1


@pytest.mark.parametrize("resource", ["chat", "responses"])
@pytest.mark.parametrize(
    "consume_one_event", [False, True], ids=["unconsumed", "partial"]
)
@pytest.mark.asyncio
async def test_async_stream_helpers_finalize_on_context_exit(
    resource, consume_one_event, langfuse_memory_client, find_spans
):
    openai_client = _mock_transport_openai_client(async_client=True)
    name = f"unit-openai-async-{resource}-stream-context-exit-{consume_one_event}"
    manager = _stream_manager(openai_client, resource, name)

    async with manager as stream:
        if consume_one_event:
            await stream.__anext__()

    langfuse_memory_client.flush()
    assert len(find_spans(name)) == 1


@pytest.mark.parametrize("resource", ["chat", "responses"])
def test_stream_manager_reuse_preserves_langfuse_arguments(
    resource, langfuse_memory_client, find_spans
):
    openai_client = _mock_transport_openai_client()
    name = f"unit-openai-reused-{resource}-stream-manager"
    manager = _stream_manager(openai_client, resource, name)

    with manager as stream:
        list(stream)

    with manager as stream:
        list(stream)

    langfuse_memory_client.flush()
    assert len(find_spans(name)) == 2


@pytest.mark.parametrize(
    ("openai_version", "expected_registrations"),
    [
        ("1.39.9", set()),
        (
            "1.40.0",
            {
                (
                    "openai.resources.beta.chat.completions",
                    "Completions.stream",
                ),
                (
                    "openai.resources.beta.chat.completions",
                    "AsyncCompletions.stream",
                ),
            },
        ),
        (
            "1.92.0",
            {
                ("openai.resources.chat.completions", "Completions.stream"),
                ("openai.resources.chat.completions", "AsyncCompletions.stream"),
            },
        ),
    ],
)
def test_chat_stream_registration_version_boundaries(
    monkeypatch, openai_version, expected_registrations
):
    registrations = set()

    def record_registration(module, name, _wrapper):
        if module in {
            "openai.resources.beta.chat.completions",
            "openai.resources.chat.completions",
        } and name.endswith(".stream"):
            registrations.add((module, name))

    monkeypatch.setattr(lf_openai_module.openai, "__version__", openai_version)
    monkeypatch.setattr(lf_openai_module, "wrap_function_wrapper", record_registration)

    lf_openai_module.register_tracing()

    assert registrations == expected_registrations


def test_legacy_beta_stream_forwards_langfuse_metadata():
    captured_args = {}

    def legacy_stream(*, model, messages, extra_body):
        extractor = lf_openai_module.OpenAiArgsExtractor(
            model=model,
            messages=messages,
            extra_body=extra_body,
        )
        captured_args.update(extractor.get_langfuse_args())
        return "stream-manager"

    resource = lf_openai_module.OpenAiDefinition(
        module="openai.resources.beta.chat.completions",
        object="Completions",
        method="stream",
        type="chat",
        sync=True,
    )
    wrapper = lf_openai_module._wrap_stream(resource)

    result = wrapper(
        legacy_stream,
        None,
        (),
        {
            "metadata": {"suite": "legacy-beta"},
            "model": "sample-model",
            "messages": [{"role": "user", "content": "Hello"}],
        },
    )

    assert result == "stream-manager"
    assert captured_args["metadata"] == {"suite": "legacy-beta"}


def test_responses_stream_retrieval_rejects_langfuse_arguments(
    langfuse_memory_client, find_spans
):
    openai_client = _mock_transport_openai_client()

    with pytest.raises(TypeError):
        openai_client.responses.stream(
            response_id="resp-test",
            name="unit-openai-responses-stream-retrieval",
        )

    langfuse_memory_client.flush()
    assert find_spans("unit-openai-responses-stream-retrieval") == []


def test_with_raw_response_chat_completion_captures_output_and_usage(
    langfuse_memory_client, get_span, json_attr
):
    openai_client = _mock_transport_openai_client()

    raw_response = openai_client.chat.completions.with_raw_response.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "1 + 1 = ?"}],
    )

    parsed = raw_response.parse()
    assert parsed.choices[0].message.content == "2"

    langfuse_memory_client.flush()
    span = get_span("OpenAI-generation")

    assert span.attributes[LangfuseOtelSpanAttributes.OBSERVATION_TYPE] == "generation"
    assert json_attr(span, LangfuseOtelSpanAttributes.OBSERVATION_OUTPUT) == {
        "role": "assistant",
        "content": "2",
    }

    usage = json_attr(span, LangfuseOtelSpanAttributes.OBSERVATION_USAGE_DETAILS)
    assert usage["prompt_tokens"] == 10
    assert usage["completion_tokens"] == 1
    assert usage["total_tokens"] == 11
    assert usage["prompt_tokens_details"] == {"cached_tokens": 4, "audio_tokens": 0}


@pytest.mark.asyncio
async def test_async_with_raw_response_chat_completion_captures_output_and_usage(
    langfuse_memory_client, get_span, json_attr
):
    openai_client = _mock_transport_openai_client(async_client=True)

    raw_response = await openai_client.chat.completions.with_raw_response.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "1 + 1 = ?"}],
    )

    parsed = raw_response.parse()
    assert parsed.choices[0].message.content == "2"

    langfuse_memory_client.flush()
    span = get_span("OpenAI-generation")

    assert json_attr(span, LangfuseOtelSpanAttributes.OBSERVATION_OUTPUT) == {
        "role": "assistant",
        "content": "2",
    }

    usage = json_attr(span, LangfuseOtelSpanAttributes.OBSERVATION_USAGE_DETAILS)
    assert usage["prompt_tokens_details"] == {"cached_tokens": 4, "audio_tokens": 0}


def test_with_raw_response_skip_flag_disables_instrumentation(
    langfuse_memory_client, memory_exporter, get_span, monkeypatch
):
    monkeypatch.setenv("LANGFUSE_OPENAI_SKIP_RAW_RESPONSES", "True")
    openai_client = _mock_transport_openai_client()

    raw_response = openai_client.chat.completions.with_raw_response.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "1 + 1 = ?"}],
    )
    assert raw_response.parse().choices[0].message.content == "2"

    langfuse_memory_client.flush()
    assert all(
        span.name != "OpenAI-generation"
        for span in memory_exporter.get_finished_spans()
    )

    openai_client.chat.completions.create(
        name="unit-openai-direct-with-skip-flag",
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "1 + 1 = ?"}],
    )

    langfuse_memory_client.flush()
    span = get_span("unit-openai-direct-with-skip-flag")
    assert span.attributes[LangfuseOtelSpanAttributes.OBSERVATION_TYPE] == "generation"


def test_with_raw_response_streaming_passes_through_untraced(
    langfuse_memory_client, memory_exporter
):
    openai_client = _mock_transport_openai_client()

    raw_response = openai_client.chat.completions.with_raw_response.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "1 + 1 = ?"}],
        stream=True,
    )

    chunks = list(raw_response.parse())
    assert chunks[0].choices[0].delta.content == "2"

    langfuse_memory_client.flush()
    assert all(
        span.name != "OpenAI-generation"
        for span in memory_exporter.get_finished_spans()
    )
