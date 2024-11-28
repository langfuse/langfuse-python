from langchain_openai import AzureChatOpenAI, ChatOpenAI, OpenAI
import pytest

from langfuse.callback import CallbackHandler
from tests.utils import get_api


@pytest.mark.parametrize(  # noqa: F821
    "expected_model,model",
    [
        ("gpt-3.5-turbo", ChatOpenAI()),
        ("gpt-3.5-turbo-instruct", OpenAI()),
        (
            "gpt-3.5-turbo",
            AzureChatOpenAI(
                openai_api_version="2023-05-15",
                azure_deployment="your-deployment-name",
                azure_endpoint="https://your-endpoint-name.azurewebsites.net",
            ),
        ),
        # # default model is now set a s azure-deployment since langchain > 0.3.0
        # (
        #     "gpt-3.5-turbo-instruct",
        #     AzureOpenAI(
        #         openai_api_version="2023-05-15",
        #         azure_deployment="your-deployment-name",
        #         azure_endpoint="https://your-endpoint-name.azurewebsites.net",
        #     ),
        # ),
    ],
)
def test_entire_llm_call_using_langchain_openai(expected_model, model):
    callback = CallbackHandler()
    try:
        # LLM calls are failing, because of missing API keys etc.
        # However, we are still able to extract the model names beforehand.
        model.invoke("Hello, how are you?", config={"callbacks": [callback]})
    except Exception as e:
        print(e)
        pass

    callback.flush()
    api = get_api()

    trace = api.trace.get(callback.get_trace_id())

    assert len(trace.observations) == 1

    generation = list(filter(lambda o: o.type == "GENERATION", trace.observations))[0]
    assert expected_model in generation.model
