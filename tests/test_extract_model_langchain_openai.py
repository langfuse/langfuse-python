from langchain_openai import ChatOpenAI, OpenAI
import pytest
from langfuse.callback import CallbackHandler
from tests.utils import get_api


@pytest.mark.parametrize(  # noqa: F821
    "expected_model,model",
    [("gpt-3.5-turbo", ChatOpenAI()), ("gpt-3.5-turbo-instruct", OpenAI())],
)
def test_entire_llm_call_using_langchain_openai(expected_model, model):
    callback = CallbackHandler()
    model.invoke("Hello, how are you?", config={"callbacks": [callback]})

    callback.flush()
    api = get_api()

    trace = api.trace.get(callback.get_trace_id())

    assert len(trace.observations) == 1

    generation = list(filter(lambda o: o.type == "GENERATION", trace.observations))[0]
    assert generation.model == expected_model
