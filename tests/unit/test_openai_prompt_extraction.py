from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum

import pytest
from pydantic import BaseModel

try:
    # Compatibility across OpenAI SDK versions where NOT_GIVEN export moved.
    from openai import NOT_GIVEN
except ImportError:
    from openai._types import NOT_GIVEN

from openai._types import Omit

from langfuse.openai import (
    OpenAiArgsExtractor,
    OpenAiDefinition,
    _extract_chat_prompt,
    _extract_responses_prompt,
    _get_langfuse_data_from_kwargs,
    _serialize_openai_value,
)

OMIT = Omit()


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        ({"input": "Hello!"}, "Hello!"),
        (
            {"instructions": "You are helpful.", "input": "Hello!"},
            [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello!"},
            ],
        ),
        (
            {
                "instructions": "You are helpful.",
                "input": [{"role": "user", "content": "Hello!"}],
            },
            [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello!"},
            ],
        ),
        (
            {"instructions": "You are helpful."},
            {"instructions": "You are helpful."},
        ),
        (
            {"instructions": "You are helpful.", "input": NOT_GIVEN},
            {"instructions": "You are helpful."},
        ),
        ({"instructions": NOT_GIVEN, "input": "Hello!"}, "Hello!"),
        ({"instructions": NOT_GIVEN, "input": NOT_GIVEN}, None),
        (
            {"instructions": "You are helpful.", "input": OMIT},
            {"instructions": "You are helpful."},
        ),
        ({"instructions": OMIT, "input": "Hello!"}, "Hello!"),
        ({"instructions": OMIT, "input": OMIT}, None),
        (
            {
                "instructions": OMIT,
                "input": "Hello!",
                "tool_choice": OMIT,
                "parallel_tool_calls": OMIT,
                "tools": OMIT,
            },
            "Hello!",
        ),
        (
            {
                "input": "Search for the weather in Berlin.",
                "tools": [
                    {
                        "type": "function",
                        "name": "get_weather",
                        "parameters": {"type": "object"},
                    }
                ],
                "tool_choice": "auto",
                "parallel_tool_calls": False,
            },
            {
                "input": "Search for the weather in Berlin.",
                "tools": [
                    {
                        "type": "function",
                        "name": "get_weather",
                        "parameters": {"type": "object"},
                    }
                ],
                "tool_choice": "auto",
                "parallel_tool_calls": False,
            },
        ),
        (
            {
                "instructions": "You are helpful.",
                "input": "Hello!",
                "tools": [{"type": "web_search_preview"}],
            },
            {
                "input": [
                    {"role": "system", "content": "You are helpful."},
                    {"role": "user", "content": "Hello!"},
                ],
                "tools": [{"type": "web_search_preview"}],
            },
        ),
    ],
)
def test_extract_responses_prompt(kwargs, expected):
    assert _extract_responses_prompt(kwargs) == expected


def test_responses_parse_text_format_is_captured_as_metadata():
    class ResponseFormat(BaseModel):
        name: str

    resource = OpenAiDefinition(
        module="",
        object="Responses",
        method="parse",
        type="chat",
        sync=True,
    )
    args = OpenAiArgsExtractor(
        model="gpt-4.1",
        input="What is your name?",
        text_format=ResponseFormat,
    ).get_langfuse_args()

    langfuse_data = _get_langfuse_data_from_kwargs(resource, args)

    assert (
        langfuse_data["metadata"]["text_format"]["properties"]["name"]["type"]
        == "string"
    )


def test_openai_value_serialization_fallback_stays_json_safe():
    class UnknownLeaf:
        def __str__(self):
            return "<unknown>"

    class MetadataKind(Enum):
        EXAMPLE = "example"

    class FallbackModel(BaseModel):
        created_at: datetime
        amount: Decimal
        kind: MetadataKind

        def model_dump(self, *args, **kwargs):
            if kwargs.get("mode") == "json":
                raise RuntimeError("json mode unavailable")

            return super().model_dump(*args, **kwargs)

    value = FallbackModel(
        created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        amount=Decimal("1.23"),
        kind=MetadataKind.EXAMPLE,
    )

    assert _serialize_openai_value(value) == {
        "created_at": "2024-01-01T00:00:00Z",
        "amount": "1.23",
        "kind": "example",
    }
    assert _serialize_openai_value({"leaf": UnknownLeaf()}) == {"leaf": "<unknown>"}


def test_base_model_metadata_uses_json_safe_values():
    class Metadata(BaseModel):
        created_at: datetime

    resource = OpenAiDefinition(
        module="",
        object="Responses",
        method="create",
        type="chat",
        sync=True,
    )
    args = OpenAiArgsExtractor(
        metadata=Metadata(created_at=datetime(2024, 1, 1, tzinfo=timezone.utc)),
        model="gpt-4.1",
        input="Hello!",
    ).get_langfuse_args()

    langfuse_data = _get_langfuse_data_from_kwargs(resource, args)

    assert langfuse_data["metadata"] == {"created_at": "2024-01-01T00:00:00Z"}


def test_store_preserves_user_structured_output_metadata_keys_for_openai():
    metadata = {
        "purpose": "distillation",
        "response_format": "user-response-format",
        "text_format": "plain",
    }

    openai_args = OpenAiArgsExtractor(
        metadata=metadata,
        model="gpt-4.1",
        input="Hello!",
        store=True,
    ).get_openai_args()

    assert openai_args["metadata"] == metadata
    assert openai_args["metadata"] is not metadata


def test_serialize_openai_value_treats_omit_as_not_given():
    assert _serialize_openai_value(OMIT) is None
    assert _serialize_openai_value({"tool_choice": OMIT, "keep": 1}) == {"keep": 1}


@pytest.mark.parametrize("sentinel", [NOT_GIVEN, OMIT], ids=["not_given", "omit"])
def test_extract_chat_prompt_excludes_unset_sentinel_tool_fields(sentinel):
    messages = [{"role": "user", "content": "Hello!"}]

    prompt = _extract_chat_prompt(
        {
            "messages": messages,
            "functions": sentinel,
            "function_call": sentinel,
            "tools": sentinel,
        }
    )

    assert prompt == messages


@pytest.mark.parametrize("sentinel", [NOT_GIVEN, OMIT], ids=["not_given", "omit"])
def test_unset_sentinel_kwargs_do_not_leak_into_langfuse_data(sentinel):
    resource = OpenAiDefinition(
        module="",
        object="Completions",
        method="create",
        type="chat",
        sync=True,
    )
    args = OpenAiArgsExtractor(
        model="gpt-4.1",
        messages=[{"role": "user", "content": "Hello!"}],
        metadata=sentinel,
        temperature=sentinel,
        top_p=sentinel,
        max_tokens=sentinel,
        frequency_penalty=sentinel,
        presence_penalty=sentinel,
        seed=sentinel,
        n=sentinel,
        service_tier=sentinel,
        tools=sentinel,
    ).get_langfuse_args()

    langfuse_data = _get_langfuse_data_from_kwargs(resource, args)

    assert langfuse_data["input"] == [{"role": "user", "content": "Hello!"}]
    assert langfuse_data["metadata"] is None
    assert langfuse_data["model_parameters"] == {
        "temperature": 1,
        "max_tokens": float("inf"),
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0,
    }


def test_store_does_not_forward_instrumentation_structured_output_metadata():
    class ResponseFormat(BaseModel):
        name: str

    extractor = OpenAiArgsExtractor(
        metadata={"purpose": "distillation"},
        model="gpt-4.1",
        input="Hello!",
        store=True,
        text_format=ResponseFormat,
    )

    langfuse_args = extractor.get_langfuse_args()
    openai_args = extractor.get_openai_args()

    assert "text_format" in langfuse_args["metadata"]
    assert openai_args["metadata"] == {"purpose": "distillation"}
