from langchain_openai import AzureChatOpenAI, AzureOpenAI, ChatOpenAI, OpenAI
import pytest
import types
from langfuse.callback import CallbackHandler
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
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
        (
            "gpt-3.5-turbo-instruct",
            AzureOpenAI(
                openai_api_version="2023-05-15",
                azure_deployment="your-deployment-name",
                azure_endpoint="https://your-endpoint-name.azurewebsites.net",
            ),
        ),
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
    assert generation.model == expected_model


@pytest.mark.parametrize(  # noqa: F821
    "expected_model,model",
    [
        ("gpt-3.5-turbo", ChatOpenAI(streaming=True)),
        ("gpt-3.5-turbo-instruct", OpenAI(streaming=True)),
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
    ],
)
def test_simple_streaming_llm_call_with_langchain_openai(expected_model, model):
    callback = CallbackHandler()
    try:
        res = model.stream("Hello, how are you?", config={"callbacks": [callback]})
        for chunk in res:
            chunk
    except Exception as e:
        print(e)
        pass

    callback.flush()
    api = get_api()

    trace = api.trace.get(callback.get_trace_id())

    assert _is_streaming_response(res)
    assert len(trace.observations) == 1

    generation = list(filter(lambda o: o.type == "GENERATION", trace.observations))[0]
    assert generation.model == expected_model


def _is_streaming_response(response):
    return isinstance(response, types.GeneratorType) or isinstance(
        response, types.AsyncGeneratorType
    )


@pytest.mark.parametrize(  # noqa: F821
    "expected_model,model",
    [
        ("gpt-3.5-turbo", ChatOpenAI(streaming=True)),
        ("gpt-3.5-turbo-instruct", OpenAI(streaming=True)),
        ("gpt-4-0613", ChatOpenAI(streaming=True, model="gpt-4-0613")),
    ],
)
def test_chain_streaming_llm_call_with_langchain_openai(expected_model, model):
    trace_name = f"Chain Streaming {expected_model}"
    callback = CallbackHandler(trace_name=trace_name)
    try:
        prompt1 = ChatPromptTemplate.from_template(
            """You are a skilled writer tasked with crafting an engaging introduction for a blog post on the following topic:
        Topic: {topic}
        Introduction: This is an engaging introduction for the blog post on the topic above:"""
        )
        chain = prompt1 | model | StrOutputParser()
        res = chain.stream("Hello, how are you?", config={"callbacks": [callback]})
        for chunk in res:
            chunk
    except Exception as e:
        print(e)
        pass

    callback.flush()
    api = get_api()

    trace = api.trace.get(callback.get_trace_id())

    assert _is_streaming_response(res)
    assert len(trace.observations) == 4
    assert trace.name == trace_name

    generation = list(filter(lambda o: o.type == "GENERATION", trace.observations))[0]
    assert generation.model == expected_model


@pytest.mark.asyncio
@pytest.mark.parametrize(  # noqa: F821
    "expected_model,model",
    [
        ("gpt-3.5-turbo", ChatOpenAI(streaming=True)),
        ("gpt-3.5-turbo-instruct", OpenAI(streaming=True)),
        ("gpt-4-0613", ChatOpenAI(streaming=True, model="gpt-4-0613")),
    ],
)
async def test_chain_async_streaming_llm_call_with_langchain_openai(
    expected_model, model
):
    trace_name = f"Chain Streaming {expected_model}"
    callback = CallbackHandler(trace_name=trace_name)
    try:
        prompt1 = ChatPromptTemplate.from_template(
            """You are a skilled writer tasked with crafting an engaging introduction for a blog post on the following topic:
        Topic: {topic}
        Introduction: This is an engaging introduction for the blog post on the topic above:"""
        )
        chain = prompt1 | model | StrOutputParser()
        res = chain.astream("Hello, how are you?", config={"callbacks": [callback]})
        async for chunk in res:
            chunk
    except Exception as e:
        print(e)
        pass

    callback.flush()
    api = get_api()

    trace = api.trace.get(callback.get_trace_id())

    assert _is_streaming_response(res)
    assert len(trace.observations) == 4
    assert trace.name == trace_name

    generation = list(filter(lambda o: o.type == "GENERATION", trace.observations))[0]
    assert generation.model == expected_model


@pytest.mark.asyncio
@pytest.mark.parametrize(  # noqa: F821
    "expected_model,model",
    [
        ("gpt-3.5-turbo", ChatOpenAI(streaming=True)),
        ("gpt-3.5-turbo-instruct", OpenAI(streaming=True)),
        ("gpt-4-0613", ChatOpenAI(streaming=True, model="gpt-4-0613")),
    ],
)
async def test_chain_async_invoke_llm_call_with_langchain_openai(expected_model, model):
    trace_name = f"Chain Async Invoke {expected_model}"
    callback = CallbackHandler(trace_name=trace_name)
    try:
        prompt1 = ChatPromptTemplate.from_template(
            """You are a skilled writer tasked with crafting an engaging introduction for a blog post on the following topic:
        Topic: {topic}
        Introduction: This is an engaging introduction for the blog post on the topic above:"""
        )
        chain = prompt1 | model | StrOutputParser()
        res = await chain.ainvoke(
            "Hello, how are you?", config={"callbacks": [callback]}
        )
    except Exception as e:
        print(e)
        pass

    callback.flush()
    api = get_api()

    trace = api.trace.get(callback.get_trace_id())

    assert isinstance(res, str)
    assert len(trace.observations) == 4
    assert trace.name == trace_name

    generation = list(filter(lambda o: o.type == "GENERATION", trace.observations))[0]
    assert generation.model == expected_model


@pytest.mark.parametrize(  # noqa: F821
    "expected_model,model",
    [
        ("gpt-3.5-turbo", ChatOpenAI(streaming=True)),
        ("gpt-3.5-turbo-instruct", OpenAI(streaming=True)),
        ("gpt-4-0613", ChatOpenAI(streaming=True, model="gpt-4-0613")),
    ],
)
def test_invoke_llm_call_with_langchain_openai(expected_model, model):
    trace_name = f"Chain Invoke{expected_model}"
    callback = CallbackHandler(trace_name=trace_name)
    try:
        prompt1 = ChatPromptTemplate.from_template(
            """You are a skilled writer tasked with crafting an engaging introduction for a blog post on the following topic:
        Topic: {topic}
        Introduction: This is an engaging introduction for the blog post on the topic above:"""
        )
        chain = prompt1 | model | StrOutputParser()
        res = chain.invoke("Hello, how are you?", config={"callbacks": [callback]})
    except Exception as e:
        print(e)
        pass

    callback.flush()
    api = get_api()

    trace = api.trace.get(callback.get_trace_id())

    assert isinstance(res, str)
    assert len(trace.observations) == 4
    assert trace.name == trace_name

    generation = list(filter(lambda o: o.type == "GENERATION", trace.observations))[0]
    assert generation.model == expected_model
