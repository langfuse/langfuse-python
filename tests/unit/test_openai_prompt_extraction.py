import pytest

try:
    # Compatibility across OpenAI SDK versions where NOT_GIVEN export moved.
    from openai import NOT_GIVEN
except ImportError:
    from openai._types import NOT_GIVEN

from langfuse.openai import _extract_responses_prompt


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
    ],
)
def test_extract_responses_prompt(kwargs, expected):
    assert _extract_responses_prompt(kwargs) == expected
