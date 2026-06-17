"""Unit tests for _extract_streamed_openai_response / get_response_for_chat.

Covers the bug where a non-empty content chunk (e.g. "\n\n") emitted before
tool-call deltas caused get_response_for_chat() to short-circuit and silently
drop all collected tool_calls from the logged generation output.

No real OpenAI API calls — chunks are built from SimpleNamespace objects that
mirror the __dict__ structure of openai-python v1 Pydantic models.
"""

import types
from dataclasses import dataclass
from typing import Any, List, Optional
from unittest.mock import patch

import pytest

from langfuse.openai import OpenAiDefinition, _extract_streamed_openai_response


# ---------------------------------------------------------------------------
# Helpers: fake OpenAI v1 streaming chunk objects
# ---------------------------------------------------------------------------

def _make_tool_call_delta(
    name: Optional[str] = None,
    arguments: str = "",
    index: int = 0,
    tool_id: Optional[str] = None,
    call_type: Optional[str] = None,
) -> Any:
    """Build a ChoiceDeltaToolCall-alike SimpleNamespace."""
    function = types.SimpleNamespace(name=name, arguments=arguments)
    return types.SimpleNamespace(
        index=index,
        id=tool_id,
        type=call_type,
        function=function,
    )


def _make_chunk(
    content: Optional[str] = None,
    tool_calls: Optional[List[Any]] = None,
    function_call: Any = None,
    role: Optional[str] = None,
    finish_reason: Optional[str] = None,
    model: str = "gpt-4o",
) -> Any:
    """Build a streaming chunk SimpleNamespace (mirrors chunk.__dict__ in v1)."""
    delta = types.SimpleNamespace(
        role=role,
        content=content,
        tool_calls=tool_calls,
        function_call=function_call,
    )
    choice = types.SimpleNamespace(delta=delta, finish_reason=finish_reason)
    return types.SimpleNamespace(model=model, choices=[choice], usage=None)


def _chat_resource() -> OpenAiDefinition:
    return OpenAiDefinition(
        module="openai",
        object="ChatCompletion",
        method="create",
        type="chat",
        sync=True,
    )


def _run(chunks: List[Any]) -> Any:
    """Run _extract_streamed_openai_response with is_openai_v1 patched to True."""
    with patch("langfuse.openai._is_openai_v1", return_value=True):
        _, response, _, _ = _extract_streamed_openai_response(_chat_resource(), iter(chunks))
    return response


# ---------------------------------------------------------------------------
# Bug reproduction: content chunk before tool_calls
# ---------------------------------------------------------------------------


class TestToolCallsWithPrecedingContentChunk:
    """
    Models like Qwen/DeepSeek sometimes emit a whitespace content chunk
    (e.g. "\n\n") before beginning to stream tool-call deltas. Previously
    get_response_for_chat() evaluated `completion["content"] or ...` and
    returned the content string immediately, dropping the tool_calls entirely.
    """

    def test_tool_calls_not_dropped_when_whitespace_content_precedes_them(self):
        chunks = [
            _make_chunk(role="assistant"),
            _make_chunk(content="\n\n"),   # spurious whitespace before tool call
            _make_chunk(
                tool_calls=[_make_tool_call_delta(name="get_weather", arguments="")],
            ),
            _make_chunk(
                tool_calls=[_make_tool_call_delta(name=None, arguments='{"city": "Paris"}')],
            ),
            _make_chunk(finish_reason="tool_calls"),
        ]
        result = _run(chunks)

        assert isinstance(result, dict), "Expected a dict, not a plain string"
        assert "tool_calls" in result, "tool_calls must not be dropped"
        assert result["tool_calls"][0]["function"]["name"] == "get_weather"
        assert result["tool_calls"][0]["function"]["arguments"] == '{"city": "Paris"}'

    def test_whitespace_only_content_not_included_in_result(self):
        """A leading "\n\n" is whitespace-only and should be omitted from output."""
        chunks = [
            _make_chunk(role="assistant"),
            _make_chunk(content="\n\n"),
            _make_chunk(
                tool_calls=[_make_tool_call_delta(name="search", arguments='{"q":"hi"}')],
            ),
            _make_chunk(finish_reason="tool_calls"),
        ]
        result = _run(chunks)

        assert "content" not in result or result.get("content") is None

    def test_meaningful_content_preserved_alongside_tool_calls(self):
        """When content has real text (not just whitespace), it should be kept."""
        chunks = [
            _make_chunk(role="assistant"),
            _make_chunk(content="Sure, let me check that. "),
            _make_chunk(
                tool_calls=[_make_tool_call_delta(name="lookup", arguments='{"id":1}')],
            ),
            _make_chunk(finish_reason="tool_calls"),
        ]
        result = _run(chunks)

        assert "tool_calls" in result
        assert result.get("content") == "Sure, let me check that. "

    def test_non_whitespace_content_before_tool_calls_preserves_both(self):
        chunks = [
            _make_chunk(role="assistant"),
            _make_chunk(content="I'll call"),
            _make_chunk(content=" the tool."),
            _make_chunk(
                tool_calls=[_make_tool_call_delta(name="do_thing", arguments="{}")],
            ),
            _make_chunk(finish_reason="tool_calls"),
        ]
        result = _run(chunks)

        assert result["tool_calls"][0]["function"]["name"] == "do_thing"
        assert result.get("content") == "I'll call the tool."


# ---------------------------------------------------------------------------
# Baseline: pure content response (no tools)
# ---------------------------------------------------------------------------


class TestPureContentResponse:
    def test_plain_text_response_returned_as_string(self):
        chunks = [
            _make_chunk(role="assistant"),
            _make_chunk(content="Hello, "),
            _make_chunk(content="world!"),
            _make_chunk(finish_reason="stop"),
        ]
        result = _run(chunks)
        assert result == "Hello, world!"

    def test_empty_stream_returns_none(self):
        result = _run([])
        assert result is None


# ---------------------------------------------------------------------------
# Pure tool-call response (no content at all)
# ---------------------------------------------------------------------------


class TestPureToolCallResponse:
    def test_tool_calls_returned_without_content(self):
        chunks = [
            _make_chunk(role="assistant"),
            _make_chunk(
                tool_calls=[_make_tool_call_delta(name="get_price", arguments='{"sku":"A1"}')],
            ),
            _make_chunk(finish_reason="tool_calls"),
        ]
        result = _run(chunks)

        assert isinstance(result, dict)
        assert "tool_calls" in result
        assert result["tool_calls"][0]["function"]["name"] == "get_price"
        assert "content" not in result

    def test_multiple_tool_calls_all_returned(self):
        chunks = [
            _make_chunk(role="assistant"),
            _make_chunk(
                tool_calls=[_make_tool_call_delta(name="tool_a", arguments='{"x":1}')],
            ),
            # second tool call — name triggers a new entry in the accumulator
            _make_chunk(
                tool_calls=[_make_tool_call_delta(name="tool_b", arguments='{"y":2}')],
            ),
            _make_chunk(finish_reason="tool_calls"),
        ]
        result = _run(chunks)

        assert len(result["tool_calls"]) == 2
        names = {tc["function"]["name"] for tc in result["tool_calls"]}
        assert names == {"tool_a", "tool_b"}


# ---------------------------------------------------------------------------
# Legacy function_call (OpenAI v0 format)
# ---------------------------------------------------------------------------


class TestFunctionCallResponse:
    def test_function_call_returned_when_no_tool_calls(self):
        # function_call uses a different delta key path; simulate with direct
        # injection via a SimpleNamespace that has function_call set
        chunks = [
            _make_chunk(role="assistant"),
        ]
        # Patch the completion dict after the fact is tricky; instead, build
        # a chunk that triggers the function_call accumulation path.
        fn_chunk = types.SimpleNamespace(
            model="gpt-3.5-turbo",
            choices=[
                types.SimpleNamespace(
                    delta=types.SimpleNamespace(
                        role=None,
                        content=None,
                        tool_calls=None,
                        function_call=types.SimpleNamespace(
                            name="old_fn",
                            arguments='{"a":1}',
                        ),
                    ),
                    finish_reason=None,
                )
            ],
            usage=None,
        )
        with patch("langfuse.openai._is_openai_v1", return_value=True):
            _, result, _, _ = _extract_streamed_openai_response(
                _chat_resource(), iter([fn_chunk])
            )

        assert isinstance(result, dict)
        assert "function_call" in result
        assert result["function_call"]["name"] == "old_fn"
