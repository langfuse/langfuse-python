import os
from dotenv import load_dotenv
from langfuse.integrations import openai
from langfuse.api.client import FintoLangfuse
from langfuse.version import __version__ as version

from tests.utils import create_uuid


load_dotenv()

api = FintoLangfuse(
    environment=os.environ["LANGFUSE_HOST"],
    username=os.environ["LANGFUSE_PUBLIC_KEY"],
    password=os.environ["LANGFUSE_SECRET_KEY"],
)


def test_openai_chat_completion():
    trace_id = create_uuid()
    completion = openai.ChatCompletion.create(
        name=trace_id, model="gpt-3.5-turbo", messages=[{"role": "user", "content": "1 + 1 = "}], temperature=0
    )

    generation = api.generations.get(name=trace_id)

    assert len(generation.data) != 0
    assert len(completion.choices) != 0


def test_openai_completion():
    trace_id = create_uuid()
    completion = openai.Completion.create(
        name=trace_id, model="gpt-3.5-turbo-instruct", prompt="1 + 1 = ", temperature=0
    )

    generation = api.generations.get(name=trace_id)

    assert len(generation.data) != 0
    assert len(completion.choices) != 0
