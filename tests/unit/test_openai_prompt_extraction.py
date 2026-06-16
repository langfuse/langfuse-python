import pytest
from pydantic import BaseModel

try:
    # Compatibility across OpenAI SDK versions where NOT_GIVEN export moved.
    from openai import NOT_GIVEN
except ImportError:
    from openai._types import NOT_GIVEN

from langfuse.openai import (
    OpenAiArgsExtractor,
    OpenAiDefinition,
    _extract_responses_prompt,
    _get_langfuse_data_from_kwargs,
)


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
