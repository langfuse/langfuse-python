import os
import pytest
from dotenv import load_dotenv
from langfuse.integrations import openai
from langfuse.api.client import FintoLangfuse


from tests.utils import create_uuid


load_dotenv()

api = FintoLangfuse(
    environment=os.environ["LANGFUSE_HOST"],
    username=os.environ["LANGFUSE_PUBLIC_KEY"],
    password=os.environ["LANGFUSE_SECRET_KEY"],
)


# @pytest.mark.skip(reason="inference cost")
def test_openai_chat_completion():
    generation_name = create_uuid()
    completion = openai.ChatCompletion.create(
        name=generation_name,
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "1 + 1 = "}],
        temperature=0,
        metadata={"someKey": "someResponse"},
    )

    openai.flush_langfuse()

    generation = api.observations.get_many(name=generation_name, type="GENERATION")

    assert len(generation.data) != 0
    assert generation.data[0].name == generation_name
    assert generation.data[0].metadata == {"someKey": "someResponse"}
    assert len(completion.choices) != 0
    assert completion.choices[0].message.content == generation.data[0].output
    assert generation.data[0].input == "1 + 1 = "
    assert generation.data[0].type == "GENERATION"
    assert generation.data[0].model == "gpt-3.5-turbo-0613"
    assert generation.data[0].start_time is not None
    assert generation.data[0].end_time is not None
    assert generation.data[0].start_time < generation.data[0].end_time
    assert generation.data[0].model_parameters == {
        "temperature": "0",
        "top_p": "1",
        "frequency_penalty": "0",
        "maxTokens": "inf",
        "presence_penalty": "0",
    }
    assert generation.data[0].prompt_tokens is not None
    assert generation.data[0].completion_tokens is not None
    assert generation.data[0].total_tokens is not None


def test_openai_chat_completion_two_calls():
    generation_name = create_uuid()
    completion = openai.ChatCompletion.create(
        name=generation_name,
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "1 + 1 = "}],
        temperature=0,
        metadata={"someKey": "someResponse"},
    )

    generation_name_2 = create_uuid()

    completion_2 = openai.ChatCompletion.create(
        name=generation_name_2,
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "2 + 2 = "}],
        temperature=0,
        metadata={"someKey": "someResponse"},
    )

    openai.flush_langfuse()

    generation = api.observations.get_many(name=generation_name, type="GENERATION")

    assert len(generation.data) != 0
    assert generation.data[0].name == generation_name
    assert len(completion.choices) != 0
    assert completion.choices[0].message.content == generation.data[0].output
    assert generation.data[0].input == "1 + 1 = "

    generation_2 = api.observations.get_many(name=generation_name_2, type="GENERATION")

    assert len(generation_2.data) != 0
    assert generation_2.data[0].name == generation_name_2
    assert len(completion_2.choices) != 0
    assert completion_2.choices[0].message.content == generation_2.data[0].output
    assert generation_2.data[0].input == "2 + 2 = "


# @pytest.mark.skip(reason="inference cost")
def test_openai_completion():
    generation_name = create_uuid()
    completion = openai.Completion.create(
        name=generation_name,
        model="gpt-3.5-turbo-instruct",
        prompt="1 + 1 = ",
        temperature=0,
        metadata={"someKey": "someResponse"},
    )

    openai.flush_langfuse()

    generation = api.observations.get_many(name=generation_name, type="GENERATION")

    assert len(generation.data) != 0
    assert generation.data[0].name == generation_name
    assert generation.data[0].metadata == {"someKey": "someResponse"}
    assert len(completion.choices) != 0
    assert completion.choices[0].text == generation.data[0].output
    assert generation.data[0].input == "1 + 1 = "
    assert generation.data[0].type == "GENERATION"
    assert generation.data[0].model == "gpt-3.5-turbo-instruct"
    assert generation.data[0].start_time is not None
    assert generation.data[0].end_time is not None
    assert generation.data[0].start_time < generation.data[0].end_time
    assert generation.data[0].model_parameters == {
        "temperature": "0",
        "top_p": "1",
        "frequency_penalty": "0",
        "maxTokens": "inf",
        "presence_penalty": "0",
    }
    assert generation.data[0].prompt_tokens is not None
    assert generation.data[0].completion_tokens is not None
    assert generation.data[0].total_tokens is not None
