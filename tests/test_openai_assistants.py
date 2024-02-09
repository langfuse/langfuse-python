import os
import pytest
import time
import inspect

from openai import APIConnectionError
from openai.resources.beta import assistants

from langfuse.client import Langfuse, FernLangfuse
from langfuse.openai import openai, OPENAI_METHODS_V1, OpenAiDefinition
from tests.utils import create_uuid, get_api


def call_resource(openai_resource: OpenAiDefinition):
    fn_path = (
        f"{openai_resource.module}.{openai_resource.object}.{openai_resource.method}"
    )
    fn_path_list = fn_path.split(".")
    fn = openai
    for x in fn_path_list[1:]:
        fn = getattr(fn, x)
    return fn


def test_call_resource():
    openai_resource = OPENAI_METHODS_V1[-1]
    fn = call_resource(openai_resource)
    assert callable(fn)


@pytest.fixture
def trace_id():
    return create_uuid()


@pytest.fixture
def openai_assistant():
    # TODO: no wrap
    assistant = openai.beta.assistants.create(model="gpt-3.5-turbo")
    yield assistant
    try:
        openai.beta.assistants.delete(assistant_id=assistant.id)
    except:
        pass


@pytest.fixture
def api():
    return FernLangfuse(
        username=os.environ.get("LANGFUSE_PUBLIC_KEY"),
        password=os.environ.get("LANGFUSE_SECRET_KEY"),
        base_url=os.environ.get("LANGFUSE_HOST"),
    )


def test_openai_assistant_create(api: FernLangfuse, trace_id):
    openai_kwargs = {"model": "gpt-3.5-turbo"}

    langfuse_kwargs = {
        "trace_id": trace_id,
    }
    fn = openai.beta.assistants.create

    openai_response_object = fn(**openai_kwargs, **langfuse_kwargs)
    openai.flush_langfuse()
    trace = api.trace.get(trace_id)
    observations = api.observations.get_many(
        trace_id=trace_id,
    )

    assert observations.data
    observation = observations.data[0]
    for key, value in openai_kwargs.items():
        assert observation.input[key] == value

    assert observation.output == dict(openai_response_object)


def test_openai_assistant_delete(api: FernLangfuse, trace_id, openai_assistant):
    openai_kwargs = {"assistant_id": openai_assistant.id}
    langfuse_kwargs = {
        "trace_id": trace_id,
    }

    openai_response_object = openai.beta.assistants.delete(**openai_kwargs)
    openai.flush_langfuse()

    trace = api.trace.get(trace_id)
    observations = api.observations.get_many(
        trace_id=trace_id,
    )

    assert observations.data
    observation = observations.data[0]
    for key, value in openai_kwargs.items():
        assert observation.input[key] == value

    assert observation.output == dict(openai_response_object)
