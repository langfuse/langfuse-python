from typing import Any
from unittest.mock import MagicMock

from langchain_anthropic import ChatAnthropic
from langchain_google_vertexai import ChatVertexAI
from langchain_groq import ChatGroq
import pytest
from langfuse.callback import CallbackHandler

from langfuse.extract_model import _extract_model_name
from langchain_core.load.dump import default


from langchain_community.chat_models import (
    ChatOpenAI,
    AzureChatOpenAI,
    ChatTongyi,
    ChatCohere,
    BedrockChat,
    ChatOllama,
)

from langchain_community.chat_models.fake import FakeMessagesListChatModel
from langchain_community.llms.anthropic import Anthropic
from langchain_community.llms.bedrock import Bedrock
from langchain_community.llms.cohere import Cohere
from langchain_community.llms.huggingface_hub import HuggingFaceHub
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_community.llms.textgen import TextGen
from langchain_community.llms.openai import (
    AzureOpenAI,
    OpenAI,
)
from langchain_community.llms.ollama import Ollama
from langchain_mistralai.chat_models import ChatMistralAI
from langchain.schema.messages import HumanMessage


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
        ("llama3", Ollama(model="llama3")),
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
        ("anthropic", Anthropic()),
        ("anthropic", Anthropic()),
        ("command", ChatCohere(model="command", cohere_api_key="command")),
        ("command", Cohere(model="command", cohere_api_key="command")),
        (None, ChatTongyi(dashscope_api_key="dash")),
        (
            "amazon.titan-tg1-large",
            BedrockChat(
                model_id="amazon.titan-tg1-large",
                region_name="us-east-1",
                client=MagicMock(),
            ),
        ),
        (
            "amazon.titan-tg1-large",
            Bedrock(
                model_id="amazon.titan-tg1-large",
                region_name="us-east-1",
                client=MagicMock(),
            ),
        ),
        (
            "claude-1",
            BedrockChat(
                model_id="claude-1",
                region_name="us-east-1",
                client=MagicMock(),
            ),
        ),
        (
            "claude-1",
            Bedrock(
                model_id="claude-1",
                region_name="us-east-1",
                client=MagicMock(),
            ),
        ),
        (
            "HuggingFaceH4/zephyr-7b-beta",
            HuggingFaceHub(
                repo_id="HuggingFaceH4/zephyr-7b-beta",
                task="text-generation",
                model_kwargs={
                    "max_new_tokens": 512,
                    "top_k": 30,
                    "temperature": 0.1,
                    "repetition_penalty": 1.03,
                },
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
        (
            "gpt-3.5-turbo-instruct",
            AzureOpenAI(
                openai_api_version="2023-05-15",
                azure_deployment="your-deployment-name",
                azure_endpoint="https://your-endpoint-name.azurewebsites.net",
            ),
        ),
        (
            "gpt2",
            HuggingFacePipeline(
                model_id="gpt2",
                model_kwargs={
                    "max_new_tokens": 512,
                    "top_k": 30,
                    "temperature": 0.1,
                    "repetition_penalty": 1.03,
                },
            ),
        ),
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
    assert generation.model == expected_model
