from typing import Any
from unittest.mock import MagicMock

import pytest
from langchain.schema.messages import HumanMessage
from langchain_anthropic import Anthropic, ChatAnthropic
from langchain_aws import BedrockLLM, ChatBedrock
from langchain_community.chat_models import (
    ChatCohere,
    ChatTongyi,
)
from langchain_community.chat_models.fake import FakeMessagesListChatModel

# from langchain_huggingface.llms import HuggingFacePipeline
from langchain_community.llms.textgen import TextGen
from langchain_core.load.dump import default
from langchain_google_vertexai import ChatVertexAI
from langchain_groq import ChatGroq
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_ollama import ChatOllama, OllamaLLM
from langchain_openai import (
    AzureChatOpenAI,
    ChatOpenAI,
    OpenAI,
)

from langfuse.langchain import CallbackHandler
from langfuse.langchain.utils import _extract_model_name
from tests.utils import get_api


@pytest.mark.parametrize(
    "expected_model,model",
    [
        (
            "mixtral-8x7b-32768",
            ChatGroq(
                temperature=0, model_name="mixtral-8x7b-32768", groq_api_key="something"
            ),
        ),
        ("llama3", OllamaLLM(model="llama3")),
        ("llama3", ChatOllama(model="llama3")),
        (
            None,
            FakeMessagesListChatModel(responses=[HumanMessage("Hello, how are you?")]),
        ),
        (
            "mistralai",
            ChatMistralAI(mistral_api_key="mistral_api_key", model="mistralai"),
        ),
        (
            "text-gen",
            TextGen(model_url="some-url"),
        ),  # local deployments, does not have a model name
        ("claude-2", ChatAnthropic(model_name="claude-2")),
        (
            "claude-3-sonnet-20240229",
            ChatAnthropic(model="claude-3-sonnet-20240229"),
        ),
        ("claude-2", Anthropic()),
        ("claude-2", Anthropic()),
        ("command", ChatCohere(model="command", cohere_api_key="command")),
        (None, ChatTongyi(dashscope_api_key="dash")),
        (
            "amazon.titan-tg1-large",
            BedrockLLM(
                model="amazon.titan-tg1-large",
                region="us-east-1",
                client=MagicMock(),
            ),
        ),
        (
            "anthropic.claude-3-sonnet-20240229-v1:0",
            ChatBedrock(
                model_id="anthropic.claude-3-sonnet-20240229-v1:0",
                region_name="us-east-1",
                client=MagicMock(),
            ),
        ),
        (
            "claude-1",
            BedrockLLM(
                model="claude-1",
                region="us-east-1",
                client=MagicMock(),
            ),
        ),
    ],
)
def test_models(expected_model: str, model: Any):
    serialized = default(model)
    model_name = _extract_model_name(serialized)
    assert model_name == expected_model


# all models here need to be tested here because we take the model from the kwargs / invocation_params or we need to make an actual call for setup
@pytest.mark.parametrize(
    "expected_model,model",
    [
        ("gpt-3.5-turbo-0125", ChatOpenAI()),
        ("gpt-3.5-turbo-instruct", OpenAI()),
        (
            "gpt-3.5-turbo",
            AzureChatOpenAI(
                openai_api_version="2023-05-15",
                model="gpt-3.5-turbo",
                azure_deployment="your-deployment-name",
                azure_endpoint="https://your-endpoint-name.azurewebsites.net",
            ),
        ),
        # (
        #     "gpt2",
        #     HuggingFacePipeline(
        #         model_id="gpt2",
        #         model_kwargs={
        #             "max_new_tokens": 512,
        #             "top_k": 30,
        #             "temperature": 0.1,
        #             "repetition_penalty": 1.03,
        #         },
        #     ),
        # ),
        (
            "qwen-72b-chat",
            ChatTongyi(model="qwen-72b-chat", dashscope_api_key="dashscope"),
        ),
        (
            "gemini",
            ChatVertexAI(
                model_name="gemini", credentials=MagicMock(), project="some-project"
            ),
        ),
    ],
)
def test_entire_llm_call(expected_model, model):
    callback = CallbackHandler()

    with callback.client.start_as_current_span(name="parent") as span:
        trace_id = span.trace_id

        try:
            # LLM calls are failing, because of missing API keys etc.
            # However, we are still able to extract the model names beforehand.
            model.invoke("Hello, how are you?", config={"callbacks": [callback]})
        except Exception as e:
            print(e)
            pass

    callback.client.flush()
    api = get_api()

    trace = api.trace.get(trace_id)

    assert len(trace.observations) == 2

    generation = list(filter(lambda o: o.type == "GENERATION", trace.observations))[0]
    assert generation.model == expected_model
