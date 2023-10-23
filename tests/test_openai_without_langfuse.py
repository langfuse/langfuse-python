import openai
import pytest
from langfuse import Langfuse


def test_plain_openai():
    langfuse = Langfuse()
    assert langfuse is not None

    with pytest.raises(openai.InvalidRequestError, match="Unrecognized request argument supplied: metadata"):
        openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "1 + 1 = "}],
            temperature=0,
            metadata={"someKey": "someResponse"},
        )
