import os
import random
import string
import time
from time import sleep
from typing import Any, Dict, List, Mapping, Optional

import pytest
from langchain.agents import AgentType, initialize_agent
from langchain.chains import (
    ConversationalRetrievalChain,
    ConversationChain,
    LLMChain,
    RetrievalQA,
    SimpleSequentialChain,
)
from langchain.chains.openai_functions import create_openai_fn_chain
from langchain.chains.summarize import load_summarize_chain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.schema import Document, HumanMessage, SystemMessage
from langchain.text_splitter import CharacterTextSplitter
from langchain_anthropic import Anthropic
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.llms.huggingface_hub import HuggingFaceHub
from langchain_community.vectorstores import Chroma
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.base import RunnableLambda
from langchain_core.tools import StructuredTool
from langchain_openai import AzureChatOpenAI, ChatOpenAI, OpenAI
from pydantic.v1 import BaseModel, Field

from langfuse.callback import CallbackHandler
from langfuse.client import Langfuse
from tests.api_wrapper import LangfuseAPI
from tests.utils import create_uuid, encode_file_to_base64, get_api


def test_callback_init():
    callback = CallbackHandler(release="something", session_id="session-id")
    assert callback.trace is None
    assert not callback.runs
    assert callback.langfuse.release == "something"
    assert callback.session_id == "session-id"
    assert callback._task_manager is not None


def test_callback_kwargs():
    callback = CallbackHandler(
        trace_name="trace-name",
        release="release",
        version="version",
        session_id="session-id",
        user_id="user-id",
        metadata={"key": "value"},
        tags=["tag1", "tag2"],
    )

    llm = OpenAI(openai_api_key=os.environ.get("OPENAI_API_KEY"), max_tokens=5)
    prompt_template = PromptTemplate(input_variables=["input"], template="""{input}""")
    test_chain = LLMChain(llm=llm, prompt=prompt_template)
    test_chain.run("Hi", callbacks=[callback])
    callback.flush()

    trace_id = callback.get_trace_id()

    trace = get_api().trace.get(trace_id)
    assert trace.input is not None
    assert trace.output is not None
    assert trace.metadata == {"key": "value"}
    assert trace.tags == ["tag1", "tag2"]
    assert trace.release == "release"
    assert trace.version == "version"
    assert trace.session_id == "session-id"
    assert trace.user_id == "user-id"


def test_langfuse_span():
    trace_id = create_uuid()
    span_id = create_uuid()
    langfuse = Langfuse(debug=False)
    trace = langfuse.trace(id=trace_id)
    span = trace.span(id=span_id)

    handler = span.get_langchain_handler()

    assert handler.get_trace_id() == trace_id
    assert handler.root_span.id == span_id
    assert handler._task_manager is not None


def test_callback_generated_from_trace_chain():
    langfuse = Langfuse(debug=True)

    trace_id = create_uuid()

    trace = langfuse.trace(id=trace_id, name=trace_id)

    handler = trace.get_langchain_handler()

    llm = OpenAI(openai_api_key=os.environ.get("OPENAI_API_KEY"))
    template = """You are a playwright. Given the title of play, it is your job to write a synopsis for that title.
        Title: {title}
        Playwright: This is a synopsis for the above play:"""

    prompt_template = PromptTemplate(input_variables=["title"], template=template)
    synopsis_chain = LLMChain(llm=llm, prompt=prompt_template)

    synopsis_chain.run("Tragedy at sunset on the beach", callbacks=[handler])

    langfuse.flush()

    trace = get_api().trace.get(trace_id)

    assert trace.input is None
    assert trace.output is None
    assert handler.get_trace_id() == trace_id

    assert len(trace.observations) == 2
    assert trace.id == trace_id

    langchain_span = list(
        filter(
            lambda o: o.type == "SPAN" and o.name == "LLMChain",
            trace.observations,
        )
    )[0]

    assert langchain_span.parent_observation_id is None
    assert langchain_span.input is not None
    assert langchain_span.output is not None

    langchain_generation_span = list(
        filter(
            lambda o: o.type == "GENERATION" and o.name == "OpenAI",
            trace.observations,
        )
    )[0]

    assert langchain_generation_span.parent_observation_id == langchain_span.id
    assert langchain_generation_span.usage_details["input"] > 0
    assert langchain_generation_span.usage_details["output"] > 0
    assert langchain_generation_span.usage_details["total"] > 0
    assert langchain_generation_span.input is not None
    assert langchain_generation_span.input != ""
    assert langchain_generation_span.output is not None
    assert langchain_generation_span.output != ""


def test_callback_generated_from_trace_chat():
    langfuse = Langfuse(debug=False)

    trace_id = create_uuid()

    trace = langfuse.trace(id=trace_id, name=trace_id)
    handler = trace.get_langchain_handler()

    chat = ChatOpenAI(temperature=0)

    messages = [
        SystemMessage(
            content="You are a helpful assistant that translates English to French."
        ),
        HumanMessage(
            content="Translate this sentence from English to French. I love programming."
        ),
    ]

    chat(messages, callbacks=[handler])

    langfuse.flush()

    trace = get_api().trace.get(trace_id)

    assert trace.input is None
    assert trace.output is None

    assert handler.get_trace_id() == trace_id
    assert trace.id == trace_id

    assert len(trace.observations) == 1

    langchain_generation_span = list(
        filter(
            lambda o: o.type == "GENERATION" and o.name == "ChatOpenAI",
            trace.observations,
        )
    )[0]

    assert langchain_generation_span.parent_observation_id is None
    assert langchain_generation_span.usage_details["input"] > 0
    assert langchain_generation_span.usage_details["output"] > 0
    assert langchain_generation_span.usage_details["total"] > 0
    assert langchain_generation_span.input is not None
    assert langchain_generation_span.input != ""
    assert langchain_generation_span.output is not None
    assert langchain_generation_span.output != ""


def test_callback_generated_from_lcel_chain():
    langfuse = Langfuse(debug=False)

    run_name_override = "This is a custom Run Name"
    handler = CallbackHandler(debug=False)

    prompt = ChatPromptTemplate.from_template("tell me a short joke about {topic}")
    model = ChatOpenAI(temperature=0)

    chain = prompt | model

    chain.invoke(
        {"topic": "ice cream"},
        config={
            "callbacks": [handler],
            "run_name": run_name_override,
        },
    )

    langfuse.flush()
    handler.flush()
    trace_id = handler.get_trace_id()
    trace = get_api().trace.get(trace_id)

    assert trace.name == run_name_override


def test_callback_generated_from_span_chain():
    langfuse = Langfuse(debug=False)

    trace_id = create_uuid()
    span_id = create_uuid()

    trace = langfuse.trace(id=trace_id, name=trace_id)
    span = trace.span(id=span_id, name=span_id)

    handler = span.get_langchain_handler()

    llm = OpenAI(openai_api_key=os.environ.get("OPENAI_API_KEY"))
    template = """You are a playwright. Given the title of play, it is your job to write a synopsis for that title.
        Title: {title}
        Playwright: This is a synopsis for the above play:"""

    prompt_template = PromptTemplate(input_variables=["title"], template=template)
    synopsis_chain = LLMChain(llm=llm, prompt=prompt_template)

    synopsis_chain.run("Tragedy at sunset on the beach", callbacks=[handler])

    langfuse.flush()

    trace = get_api().trace.get(trace_id)

    assert trace.input is None
    assert trace.output is None
    assert handler.get_trace_id() == trace_id

    assert len(trace.observations) == 3
    assert trace.id == trace_id

    user_span = list(
        filter(
            lambda o: o.id == span_id,
            trace.observations,
        )
    )[0]

    assert user_span.input is None
    assert user_span.output is None

    assert user_span.input is None
    assert user_span.output is None

    langchain_span = list(
        filter(
            lambda o: o.type == "SPAN" and o.name == "LLMChain",
            trace.observations,
        )
    )[0]

    assert langchain_span.parent_observation_id == user_span.id

    langchain_generation_span = list(
        filter(
            lambda o: o.type == "GENERATION" and o.name == "OpenAI",
            trace.observations,
        )
    )[0]

    assert langchain_generation_span.parent_observation_id == langchain_span.id
    assert langchain_generation_span.usage_details["input"] > 0
    assert langchain_generation_span.usage_details["output"] > 0
    assert langchain_generation_span.usage_details["total"] > 0
    assert langchain_generation_span.input is not None
    assert langchain_generation_span.input != ""
    assert langchain_generation_span.output is not None
    assert langchain_generation_span.output != ""


def test_callback_generated_from_span_chat():
    langfuse = Langfuse(debug=False)

    trace_id = create_uuid()
    span_id = create_uuid()

    trace = langfuse.trace(id=trace_id, name=trace_id)
    span = trace.span(id=span_id, name=span_id)

    handler = span.get_langchain_handler()

    chat = ChatOpenAI(temperature=0)

    messages = [
        SystemMessage(
            content="You are a helpful assistant that translates English to French."
        ),
        HumanMessage(
            content="Translate this sentence from English to French. I love programming."
        ),
    ]

    chat(messages, callbacks=[handler])

    langfuse.flush()

    trace = get_api().trace.get(trace_id)

    assert trace.input is None
    assert trace.output is None

    assert handler.get_trace_id() == trace_id
    assert trace.id == trace_id

    assert len(trace.observations) == 2

    user_span = list(
        filter(
            lambda o: o.id == span_id,
            trace.observations,
        )
    )[0]

    assert user_span.input is None
    assert user_span.output is None

    langchain_generation_span = list(
        filter(
            lambda o: o.type == "GENERATION" and o.name == "ChatOpenAI",
            trace.observations,
        )
    )[0]

    assert langchain_generation_span.parent_observation_id == user_span.id
    assert langchain_generation_span.usage_details["input"] > 0
    assert langchain_generation_span.usage_details["output"] > 0
    assert langchain_generation_span.usage_details["total"] > 0
    assert langchain_generation_span.input is not None
    assert langchain_generation_span.input != ""
    assert langchain_generation_span.output is not None
    assert langchain_generation_span.output != ""


@pytest.mark.skip(reason="missing api key")
def test_callback_generated_from_trace_azure_chat():
    api_wrapper = LangfuseAPI()
    langfuse = Langfuse(debug=False)

    trace_id = create_uuid()
    trace = langfuse.trace(id=trace_id)

    handler = trace.getNewHandler()

    llm = AzureChatOpenAI(
        openai_api_base="AZURE_OPENAI_ENDPOINT",
        openai_api_version="2023-05-15",
        deployment_name="gpt-4",
        openai_api_key="AZURE_OPENAI_API_KEY",
        openai_api_type="azure",
        model_version="0613",
        temperature=0,
    )
    template = """You are a playwright. Given the title of play, it is your job to write a synopsis for that title.
        Title: {title}
        Playwright: This is a synopsis for the above play:"""

    prompt_template = PromptTemplate(input_variables=["title"], template=template)
    synopsis_chain = LLMChain(llm=llm, prompt=prompt_template)

    synopsis_chain.run("Tragedy at sunset on the beach", callbacks=[handler])

    langfuse.flush()

    trace = api_wrapper.get_trace(trace_id)

    assert handler.get_trace_id() == trace_id
    assert len(trace["observations"]) == 2
    assert trace["id"] == trace_id


@pytest.mark.skip(reason="missing api key")
def test_mistral():
    from langchain_core.messages import HumanMessage
    from langchain_mistralai.chat_models import ChatMistralAI

    callback = CallbackHandler(debug=False)

    chat = ChatMistralAI(model="mistral-small", callbacks=[callback])
    messages = [HumanMessage(content="say a brief hello")]
    chat.invoke(messages)

    callback.flush()

    trace_id = callback.get_trace_id()

    trace = get_api().trace.get(trace_id)

    assert trace.id == trace_id
    assert len(trace.observations) == 2

    generation = list(filter(lambda o: o.type == "GENERATION", trace.observations))[0]
    assert generation.model == "mistral-small"


@pytest.mark.skip(reason="missing api key")
def test_vertx():
    from langchain.llms import VertexAI

    callback = CallbackHandler(debug=False)

    llm = VertexAI(callbacks=[callback])
    llm.predict("say a brief hello", callbacks=[callback])

    callback.flush()

    trace_id = callback.get_trace_id()

    trace = get_api().trace.get(trace_id)

    assert trace.id == trace_id
    assert len(trace.observations) == 2

    generation = list(filter(lambda o: o.type == "GENERATION", trace.observations))[0]
    assert generation.model == "text-bison"


@pytest.mark.skip(reason="rate limits")
def test_callback_generated_from_trace_anthropic():
    langfuse = Langfuse(debug=False)

    trace_id = create_uuid()
    trace = langfuse.trace(id=trace_id)

    handler = trace.getNewHandler()

    llm = Anthropic(
        model="claude-instant-1.2",
    )
    template = """You are a playwright. Given the title of play, it is your job to write a synopsis for that title.
        Title: {title}
        Playwright: This is a synopsis for the above play:"""

    prompt_template = PromptTemplate(input_variables=["title"], template=template)
    synopsis_chain = LLMChain(llm=llm, prompt=prompt_template)

    synopsis_chain.run("Tragedy at sunset on the beach", callbacks=[handler])

    langfuse.flush()

    trace = get_api().trace.get(trace_id)

    assert handler.get_trace_id() == trace_id
    assert len(trace.observations) == 2
    assert trace.id == trace_id
    for observation in trace.observations:
        if observation.type == "GENERATION":
            assert observation.usage_details["input"] > 0
            assert observation.usage_details["output"] > 0
            assert observation.usage_details["total"] > 0
            assert observation.output is not None
            assert observation.output != ""
            assert isinstance(observation.input, str) is True
            assert isinstance(observation.output, str) is True
            assert observation.input != ""
            assert observation.model == "claude-instant-1.2"


def test_basic_chat_openai():
    callback = CallbackHandler(debug=False)

    chat = ChatOpenAI(temperature=0)

    messages = [
        SystemMessage(
            content="You are a helpful assistant that translates English to French."
        ),
        HumanMessage(
            content="Translate this sentence from English to French. I love programming."
        ),
    ]

    chat(messages, callbacks=[callback])
    callback.flush()

    trace_id = callback.get_trace_id()

    trace = get_api().trace.get(trace_id)

    assert trace.id == trace_id
    assert len(trace.observations) == 1

    assert trace.output == trace.observations[0].output
    assert trace.input == trace.observations[0].input

    assert trace.observations[0].input == [
        {
            "role": "system",
            "content": "You are a helpful assistant that translates English to French.",
        },
        {
            "role": "user",
            "content": "Translate this sentence from English to French. I love programming.",
        },
    ]
    assert trace.observations[0].output["role"] == "assistant"


def test_basic_chat_openai_based_on_trace():
    from langchain.schema import HumanMessage, SystemMessage

    trace_id = create_uuid()

    langfuse = Langfuse(debug=False)
    trace = langfuse.trace(id=trace_id)

    callback = trace.get_langchain_handler()

    chat = ChatOpenAI(temperature=0)

    messages = [
        SystemMessage(
            content="You are a helpful assistant that translates English to French."
        ),
        HumanMessage(
            content="Translate this sentence from English to French. I love programming."
        ),
    ]

    chat(messages, callbacks=[callback])
    callback.flush()

    trace_id = callback.get_trace_id()

    trace = get_api().trace.get(trace_id)

    assert trace.id == trace_id
    assert len(trace.observations) == 1


def test_callback_from_trace_with_trace_update():
    langfuse = Langfuse(debug=False)

    trace_id = create_uuid()
    trace = langfuse.trace(id=trace_id)

    handler = trace.get_langchain_handler(update_parent=True)
    llm = OpenAI(openai_api_key=os.environ.get("OPENAI_API_KEY"))
    template = """You are a playwright. Given the title of play, it is your job to write a synopsis for that title.
        Title: {title}
        Playwright: This is a synopsis for the above play:"""

    prompt_template = PromptTemplate(input_variables=["title"], template=template)
    synopsis_chain = LLMChain(llm=llm, prompt=prompt_template)

    synopsis_chain.run("Tragedy at sunset on the beach", callbacks=[handler])

    langfuse.flush()

    trace_id = handler.get_trace_id()

    trace = get_api().trace.get(trace_id)

    assert trace.input is not None
    assert trace.output is not None

    assert len(trace.observations) == 2
    assert handler.get_trace_id() == trace_id
    assert trace.id == trace_id

    generations = list(filter(lambda x: x.type == "GENERATION", trace.observations))
    assert len(generations) > 0
    for generation in generations:
        assert generation.input is not None
        assert generation.output is not None
        assert generation.usage_details["total"] is not None
        assert generation.usage_details["input"] is not None
        assert generation.usage_details["output"] is not None


def test_callback_from_span_with_span_update():
    langfuse = Langfuse(debug=False)

    trace_id = create_uuid()
    span_id = create_uuid()
    trace = langfuse.trace(id=trace_id)
    span = trace.span(id=span_id)

    handler = span.get_langchain_handler(update_parent=True)
    llm = OpenAI(openai_api_key=os.environ.get("OPENAI_API_KEY"))
    template = """You are a playwright. Given the title of play, it is your job to write a synopsis for that title.
        Title: {title}
        Playwright: This is a synopsis for the above play:"""

    prompt_template = PromptTemplate(input_variables=["title"], template=template)
    synopsis_chain = LLMChain(llm=llm, prompt=prompt_template)

    synopsis_chain.run("Tragedy at sunset on the beach", callbacks=[handler])

    langfuse.flush()

    trace_id = handler.get_trace_id()

    trace = get_api().trace.get(trace_id)

    assert trace.input is None
    assert trace.output is None
    assert trace.metadata == {}

    assert len(trace.observations) == 3
    assert handler.get_trace_id() == trace_id
    assert trace.id == trace_id
    assert handler.root_span.id == span_id

    root_span_observation = [o for o in trace.observations if o.id == span_id][0]
    assert root_span_observation.input is not None
    assert root_span_observation.output is not None

    generations = list(filter(lambda x: x.type == "GENERATION", trace.observations))
    assert len(generations) > 0
    for generation in generations:
        assert generation.input is not None
        assert generation.output is not None
        assert generation.usage_details["total"] is not None
        assert generation.usage_details["input"] is not None
        assert generation.usage_details["output"] is not None


def test_callback_from_trace_simple_chain():
    langfuse = Langfuse(debug=False)

    trace_id = create_uuid()
    trace = langfuse.trace(id=trace_id)

    handler = trace.getNewHandler()
    llm = OpenAI(openai_api_key=os.environ.get("OPENAI_API_KEY"))
    template = """You are a playwright. Given the title of play, it is your job to write a synopsis for that title.
        Title: {title}
        Playwright: This is a synopsis for the above play:"""

    prompt_template = PromptTemplate(input_variables=["title"], template=template)
    synopsis_chain = LLMChain(llm=llm, prompt=prompt_template)

    synopsis_chain.run("Tragedy at sunset on the beach", callbacks=[handler])

    langfuse.flush()

    trace_id = handler.get_trace_id()

    trace = get_api().trace.get(trace_id)
    assert trace.input is None
    assert trace.output is None

    assert len(trace.observations) == 2
    assert handler.get_trace_id() == trace_id
    assert trace.id == trace_id

    generations = list(filter(lambda x: x.type == "GENERATION", trace.observations))
    assert len(generations) > 0
    for generation in generations:
        assert generation.input is not None
        assert generation.output is not None
        assert generation.usage_details["total"] is not None
        assert generation.usage_details["input"] is not None
        assert generation.usage_details["output"] is not None


def test_next_span_id_from_trace_simple_chain():
    api_wrapper = LangfuseAPI()
    langfuse = Langfuse()

    trace_id = create_uuid()
    trace = langfuse.trace(id=trace_id)

    handler = trace.getNewHandler()
    llm = OpenAI(openai_api_key=os.environ.get("OPENAI_API_KEY"))
    template = """You are a playwright. Given the title of play, it is your job to write a synopsis for that title.
        Title: {title}
        Playwright: This is a synopsis for the above play:"""

    prompt_template = PromptTemplate(input_variables=["title"], template=template)
    synopsis_chain = LLMChain(llm=llm, prompt=prompt_template)

    synopsis_chain.run("Tragedy at sunset on the beach", callbacks=[handler])

    next_span_id = create_uuid()
    handler.setNextSpan(next_span_id)

    synopsis_chain.run("Comedy at sunset on the beach", callbacks=[handler])

    langfuse.flush()

    trace_id = handler.get_trace_id()

    trace = api_wrapper.get_trace(trace_id)

    assert len(trace["observations"]) == 4
    assert handler.get_trace_id() == trace_id
    assert trace["id"] == trace_id

    assert any(
        observation["id"] == next_span_id for observation in trace["observations"]
    )
    for observation in trace["observations"]:
        if observation["type"] == "GENERATION":
            assert observation["promptTokens"] > 0
            assert observation["completionTokens"] > 0
            assert observation["totalTokens"] > 0
            assert observation["input"] is not None
            assert observation["input"] != ""
            assert observation["output"] is not None
            assert observation["output"] != ""


def test_callback_sequential_chain():
    handler = CallbackHandler(debug=False)

    llm = OpenAI(openai_api_key=os.environ.get("OPENAI_API_KEY"))
    template = """You are a playwright. Given the title of play, it is your job to write a synopsis for that title.
        Title: {title}
        Playwright: This is a synopsis for the above play:"""

    prompt_template = PromptTemplate(input_variables=["title"], template=template)
    synopsis_chain = LLMChain(llm=llm, prompt=prompt_template)

    template = """You are a play critic from the New York Times.
    Given the synopsis of play, it is your job to write a review for that play.

        Play Synopsis:
        {synopsis}
        Review from a New York Times play critic of the above play:"""
    prompt_template = PromptTemplate(input_variables=["synopsis"], template=template)
    review_chain = LLMChain(llm=llm, prompt=prompt_template)

    overall_chain = SimpleSequentialChain(
        chains=[synopsis_chain, review_chain],
    )
    overall_chain.run("Tragedy at sunset on the beach", callbacks=[handler])

    handler.flush()

    trace_id = handler.get_trace_id()

    trace = get_api().trace.get(trace_id)

    assert len(trace.observations) == 5
    assert trace.id == trace_id

    for observation in trace.observations:
        if observation.type == "GENERATION":
            assert observation.usage_details["input"] > 0
            assert observation.usage_details["output"] > 0
            assert observation.usage_details["total"] > 0
            assert observation.input is not None
            assert observation.input != ""
            assert observation.output is not None
            assert observation.output != ""


def test_stuffed_chain():
    with open("./static/state_of_the_union_short.txt", encoding="utf-8") as f:
        api_wrapper = LangfuseAPI()
        handler = CallbackHandler(debug=False)

        text = f.read()
        docs = [Document(page_content=text)]
        llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

        template = """
        Compose a concise and a brief summary of the following text:
        TEXT: `{text}`
        """

        prompt = PromptTemplate(input_variables=["text"], template=template)

        chain = load_summarize_chain(
            llm, chain_type="stuff", prompt=prompt, verbose=False
        )

        chain.run(docs, callbacks=[handler])

        handler.flush()

        trace_id = handler.get_trace_id()

        trace = api_wrapper.get_trace(trace_id)

        assert len(trace["observations"]) == 3
        for observation in trace["observations"]:
            if observation["type"] == "GENERATION":
                assert observation["promptTokens"] > 0
                assert observation["completionTokens"] > 0
                assert observation["totalTokens"] > 0
                assert observation["input"] is not None
                assert observation["input"] != ""
                assert observation["output"] is not None
                assert observation["output"] != ""


def test_callback_retriever():
    api_wrapper = LangfuseAPI()
    handler = CallbackHandler(debug=False)

    loader = TextLoader("./static/state_of_the_union.txt", encoding="utf8")
    llm = OpenAI(openai_api_key=os.environ.get("OPENAI_API_KEY"))

    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))
    docsearch = Chroma.from_documents(texts, embeddings)

    query = "What did the president say about Ketanji Brown Jackson"

    chain = RetrievalQA.from_chain_type(
        llm,
        retriever=docsearch.as_retriever(),
    )

    chain.run(query, callbacks=[handler])
    handler.flush()

    trace_id = handler.get_trace_id()

    trace = api_wrapper.get_trace(trace_id)

    assert len(trace["observations"]) == 5
    for observation in trace["observations"]:
        if observation["type"] == "GENERATION":
            assert observation["promptTokens"] > 0
            assert observation["completionTokens"] > 0
            assert observation["totalTokens"] > 0
            assert observation["input"] is not None
            assert observation["input"] != ""
            assert observation["output"] is not None
            assert observation["output"] != ""


def test_callback_retriever_with_sources():
    api_wrapper = LangfuseAPI()
    handler = CallbackHandler(debug=False)

    loader = TextLoader("./static/state_of_the_union.txt", encoding="utf8")
    llm = OpenAI(openai_api_key=os.environ.get("OPENAI_API_KEY"))

    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))
    docsearch = Chroma.from_documents(texts, embeddings)

    query = "What did the president say about Ketanji Brown Jackson"

    chain = RetrievalQA.from_chain_type(
        llm, retriever=docsearch.as_retriever(), return_source_documents=True
    )

    chain(query, callbacks=[handler])
    handler.flush()

    trace_id = handler.get_trace_id()

    trace = api_wrapper.get_trace(trace_id)

    assert len(trace["observations"]) == 5
    for observation in trace["observations"]:
        if observation["type"] == "GENERATION":
            assert observation["promptTokens"] > 0
            assert observation["completionTokens"] > 0
            assert observation["totalTokens"] > 0
            assert observation["input"] is not None
            assert observation["input"] != ""
            assert observation["output"] is not None
            assert observation["output"] != ""


def test_callback_retriever_conversational_with_memory():
    handler = CallbackHandler(debug=False)
    llm = OpenAI(openai_api_key=os.environ.get("OPENAI_API_KEY"))
    conversation = ConversationChain(
        llm=llm, verbose=True, memory=ConversationBufferMemory(), callbacks=[handler]
    )
    conversation.predict(input="Hi there!", callbacks=[handler])
    handler.flush()

    trace = get_api().trace.get(handler.get_trace_id())

    generations = list(filter(lambda x: x.type == "GENERATION", trace.observations))
    assert len(generations) == 1

    for generation in generations:
        assert generation.input is not None
        assert generation.output is not None
        assert generation.input != ""
        assert generation.output != ""
        assert generation.usage_details["total"] is not None
        assert generation.usage_details["input"] is not None
        assert generation.usage_details["output"] is not None


def test_callback_retriever_conversational():
    api_wrapper = LangfuseAPI()
    handler = CallbackHandler(debug=False)

    loader = TextLoader("./static/state_of_the_union.txt", encoding="utf8")

    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))
    docsearch = Chroma.from_documents(texts, embeddings)

    query = "What did the president say about Ketanji Brown Jackson"

    chain = ConversationalRetrievalChain.from_llm(
        ChatOpenAI(
            openai_api_key=os.environ.get("OPENAI_API_KEY"),
            temperature=0.5,
            model="gpt-3.5-turbo-16k",
        ),
        docsearch.as_retriever(search_kwargs={"k": 6}),
        return_source_documents=True,
    )

    chain({"question": query, "chat_history": []}, callbacks=[handler])
    handler.flush()

    trace_id = handler.get_trace_id()

    trace = api_wrapper.get_trace(trace_id)

    assert len(trace["observations"]) == 5
    for observation in trace["observations"]:
        if observation["type"] == "GENERATION":
            assert observation["promptTokens"] > 0
            assert observation["completionTokens"] > 0
            assert observation["totalTokens"] > 0
            assert observation["input"] is not None
            assert observation["input"] != ""
            assert observation["output"] is not None
            assert observation["output"] != ""


def test_callback_simple_openai():
    handler = CallbackHandler()

    llm = OpenAI(openai_api_key=os.environ.get("OPENAI_API_KEY"))

    text = "What would be a good company name for a company that makes colorful socks?"

    llm.predict(text, callbacks=[handler])

    handler.flush()

    trace_id = handler.get_trace_id()

    trace = get_api().trace.get(trace_id)

    assert len(trace.observations) == 1

    for observation in trace.observations:
        if observation.type == "GENERATION":
            print(observation.usage_details)
            assert observation.usage_details["input"] > 0
            assert observation.usage_details["output"] > 0
            assert observation.usage_details["total"] > 0
            assert observation.input is not None
            assert observation.input != ""
            assert observation.output is not None
            assert observation.output != ""


def test_callback_multiple_invocations_on_different_traces():
    handler = CallbackHandler(debug=False)

    llm = OpenAI(openai_api_key=os.environ.get("OPENAI_API_KEY"))

    text = "What would be a good company name for a company that makes colorful socks?"

    llm.predict(text, callbacks=[handler])

    trace_id_one = handler.get_trace_id()

    llm.predict(text, callbacks=[handler])

    trace_id_two = handler.get_trace_id()

    handler.flush()

    assert trace_id_one != trace_id_two

    trace_one = get_api().trace.get(trace_id_one)
    trace_two = get_api().trace.get(trace_id_two)

    for test_data in [
        {"trace": trace_one, "expected_trace_id": trace_id_one},
        {"trace": trace_two, "expected_trace_id": trace_id_two},
    ]:
        assert len(test_data["trace"].observations) == 1
        assert test_data["trace"].id == test_data["expected_trace_id"]
        for observation in test_data["trace"].observations:
            if observation.type == "GENERATION":
                assert observation.usage_details["input"] > 0
                assert observation.usage_details["output"] > 0
                assert observation.usage_details["total"] > 0
                assert observation.input is not None
                assert observation.input != ""
                assert observation.output is not None
                assert observation.output != ""


@pytest.mark.skip(reason="inference cost")
def test_callback_simple_openai_streaming():
    api_wrapper = LangfuseAPI()
    handler = CallbackHandler(debug=False)

    llm = OpenAI(openai_api_key=os.environ.get("OPENAI_API_KEY"), streaming=False)

    text = "What would be a good company name for a company that makes laptops?"

    llm.predict(text, callbacks=[handler])

    handler.flush()

    trace_id = handler.get_trace_id()

    trace = api_wrapper.get_trace(trace_id)

    generation = trace["observations"][1]

    assert generation["promptTokens"] is not None
    assert generation["completionTokens"] is not None
    assert generation["totalTokens"] is not None

    assert len(trace["observations"]) == 2
    for observation in trace["observations"]:
        if observation["type"] == "GENERATION":
            assert observation["promptTokens"] > 0
            assert observation["completionTokens"] > 0
            assert observation["totalTokens"] > 0
            assert observation["input"] is not None
            assert observation["input"] != ""
            assert observation["output"] is not None
            assert observation["output"] != ""


@pytest.mark.skip(reason="no serpapi setup in CI")
def test_tools():
    handler = CallbackHandler(debug=False)

    llm = ChatOpenAI(openai_api_key=os.environ.get("OPENAI_API_KEY"))

    tools = load_tools(["serpapi", "llm-math"], llm=llm)

    agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION)

    agent.run(
        "Who is Leo DiCaprio's girlfriend? What is her current age raised to the 0.43 power?",
        callbacks=[handler],
    )

    handler.flush()

    trace_id = handler.get_trace_id()

    trace = get_api().trace.get(trace_id)
    assert trace.id == trace_id
    assert len(trace.observations) > 2

    generations = list(filter(lambda x: x.type == "GENERATION", trace.observations))
    assert len(generations) > 0

    for generation in generations:
        assert generation.input is not None
        assert generation.output is not None
        assert generation.input != ""
        assert generation.output != ""
        assert generation.total_tokens is not None
        assert generation.prompt_tokens is not None
        assert generation.completion_tokens is not None


@pytest.mark.skip(reason="inference cost")
def test_callback_huggingface_hub():
    api_wrapper = LangfuseAPI()
    handler = CallbackHandler(debug=False)

    def initialize_huggingface_llm(prompt: PromptTemplate) -> LLMChain:
        repo_id = "google/flan-t5-small"
        # Experiment with the max_length parameter and temperature
        llm = HuggingFaceHub(
            repo_id=repo_id, model_kwargs={"temperature": 0.1, "max_length": 500}
        )
        return LLMChain(prompt=prompt, llm=llm)

    hugging_chain = initialize_huggingface_llm(
        prompt=PromptTemplate(
            input_variables=["title"],
            template="""
You are a playwright. Given the title of play, it is your job to write a synopsis for that title.
Title: {title}
        """,
        )
    )

    hugging_chain.run(title="Mission to Mars", callbacks=[handler])

    handler.langfuse.flush()

    trace_id = handler.get_trace_id()

    trace = api_wrapper.get_trace(trace_id)

    assert len(trace["observations"]) == 2
    for observation in trace["observations"]:
        if observation["type"] == "GENERATION":
            assert observation["promptTokens"] > 0
            assert observation["completionTokens"] > 0
            assert observation["totalTokens"] > 0
            assert observation["input"] is not None
            assert observation["input"] != ""
            assert observation["output"] is not None
            assert observation["output"] != ""


def test_callback_openai_functions_python():
    handler = CallbackHandler(debug=False)
    assert handler.langfuse.base_url == "http://localhost:3000"

    llm = ChatOpenAI(model="gpt-4", temperature=0)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a world class algorithm for extracting information in structured formats.",
            ),
            (
                "human",
                "Use the given format to extract information from the following input: {input}",
            ),
            ("human", "Tip: Make sure to answer in the correct format"),
        ]
    )

    class OptionalFavFood(BaseModel):
        """Either a food or null."""

        food: Optional[str] = Field(
            None,
            description="Either the name of a food or null. Should be null if the food isn't known.",
        )

    def record_person(name: str, age: int, fav_food: OptionalFavFood) -> str:
        """Record some basic identifying information about a person.

        Args:
            name: The person's name.
            age: The person's age in years.
            fav_food: An OptionalFavFood object that either contains the person's favorite food or a null value.
            Food should be null if it's not known.
        """
        return (
            f"Recording person {name} of age {age} with favorite food {fav_food.food}!"
        )

    def record_dog(name: str, color: str, fav_food: OptionalFavFood) -> str:
        """Record some basic identifying information about a dog.

        Args:
            name: The dog's name.
            color: The dog's color.
            fav_food: An OptionalFavFood object that either contains the dog's favorite food or a null value.
            Food should be null if it's not known.
        """
        return f"Recording dog {name} of color {color} with favorite food {fav_food}!"

    chain = create_openai_fn_chain(
        [record_person, record_dog], llm, prompt, callbacks=[handler]
    )
    chain.run(
        "I can't find my dog Henry anywhere, he's a small brown beagle. Could you send a message about him?",
        callbacks=[handler],
    )

    handler.langfuse.flush()

    trace = get_api().trace.get(handler.get_trace_id())

    assert len(trace.observations) == 2

    generations = list(filter(lambda x: x.type == "GENERATION", trace.observations))
    assert len(generations) > 0

    for generation in generations:
        assert generation.input is not None
        assert generation.output is not None
        assert generation.input == [
            {
                "role": "system",
                "content": "You are a world class algorithm for extracting information in structured formats.",
            },
            {
                "role": "user",
                "content": "Use the given format to extract information from the following input: I can't find my dog Henry anywhere, he's a small brown beagle. Could you send a message about him?",
            },
            {
                "role": "user",
                "content": "Tip: Make sure to answer in the correct format",
            },
        ]
        assert generation.output == {
            "role": "assistant",
            "content": "",
            "additional_kwargs": {
                "function_call": {
                    "arguments": '{\n  "name": "Henry",\n  "color": "brown",\n  "fav_food": {\n    "food": null\n  }\n}',
                    "name": "record_dog",
                },
                "refusal": None,
            },
        }
        assert generation.usage_details["total"] is not None
        assert generation.usage_details["input"] is not None
        assert generation.usage_details["output"] is not None


def test_agent_executor_chain():
    from langchain.agents import AgentExecutor, create_react_agent
    from langchain.tools import tool

    prompt = PromptTemplate.from_template("""
    Answer the following questions as best you can. You have access to the following tools:

    {tools}

    Use the following format:

    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question

    Begin!

    Question: {input}
    Thought:{agent_scratchpad}
    """)

    callback = CallbackHandler(debug=True)
    llm = OpenAI(temperature=0)

    @tool
    def get_word_length(word: str) -> int:
        """Returns the length of a word."""
        return len(word)

    tools = [get_word_length]
    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, handle_parsing_errors=True)

    agent_executor.invoke(
        {"input": "what is the length of the word LangFuse?"},
        config={"callbacks": [callback]},
    )

    callback.flush()

    trace = get_api().trace.get(callback.get_trace_id())

    generations = list(filter(lambda x: x.type == "GENERATION", trace.observations))
    assert len(generations) > 0

    for generation in generations:
        assert generation.input is not None
        assert generation.output is not None
        assert generation.input != ""
        assert generation.output != ""
        assert generation.usage_details["total"] is not None
        assert generation.usage_details["input"] is not None
        assert generation.usage_details["output"] is not None


# def test_create_extraction_chain():
#     import os
#     from uuid import uuid4

#     from langchain.chains import create_extraction_chain
#     from langchain.chat_models import ChatOpenAI
#     from langchain.document_loaders import TextLoader
#     from langchain.embeddings.openai import OpenAIEmbeddings
#     from langchain.text_splitter import CharacterTextSplitter
#     from langchain.vectorstores import Chroma

#     from langfuse.client import Langfuse

#     def create_uuid():
#         return str(uuid4())

#     langfuse = Langfuse(debug=False, host="http://localhost:3000")

#     trace_id = create_uuid()

#     trace = langfuse.trace(id=trace_id)
#     handler = trace.getNewHandler()

#     loader = TextLoader("./static/state_of_the_union.txt", encoding="utf8")

#     documents = loader.load()
#     text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
#     texts = text_splitter.split_documents(documents)

#     embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))
#     vector_search = Chroma.from_documents(texts, embeddings)

#     main_character = vector_search.similarity_search(
#         "Who is the main character and what is the summary of the text?"
#     )

#     llm = ChatOpenAI(
#         openai_api_key=os.getenv("OPENAI_API_KEY"),
#         temperature=0,
#         streaming=False,
#         model="gpt-3.5-turbo-16k-0613",
#     )

#     schema = {
#         "properties": {
#             "Main character": {"type": "string"},
#             "Summary": {"type": "string"},
#         },
#         "required": [
#             "Main character",
#             "Cummary",
#         ],
#     }
#     chain = create_extraction_chain(schema, llm)

#     chain.run(main_character, callbacks=[handler])

#     handler.flush()

#

#     trace = get_api().trace.get(handler.get_trace_id())

#     generations = list(filter(lambda x: x.type == "GENERATION", trace.observations))
#     assert len(generations) > 0

#     for generation in generations:
#         assert generation.input is not None
#         assert generation.output is not None
#         assert generation.input != ""
#         assert generation.output != ""
#         assert generation.usage_details["total"] is not None
#         assert generation.usage_details["input"] is not None
#         assert generation.usage_details["output"] is not None


@pytest.mark.skip(reason="inference cost")
def test_aws_bedrock_chain():
    import os

    import boto3
    from langchain.llms.bedrock import Bedrock

    api_wrapper = LangfuseAPI()
    handler = CallbackHandler(debug=False)

    bedrock_client = boto3.client(
        "bedrock-runtime",
        region_name="us-east-1",
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
        aws_session_token=os.environ.get("AWS_SESSION_TOKEN"),
    )

    llm = Bedrock(
        model_id="anthropic.claude-instant-v1",
        client=bedrock_client,
        model_kwargs={
            "max_tokens_to_sample": 1000,
            "temperature": 0.0,
        },
    )

    text = "What would be a good company name for a company that makes colorful socks?"

    llm.predict(text, callbacks=[handler])

    handler.flush()

    trace_id = handler.get_trace_id()

    trace = api_wrapper.get_trace(trace_id)

    generation = trace["observations"][1]

    assert generation["promptTokens"] is not None
    assert generation["completionTokens"] is not None
    assert generation["totalTokens"] is not None

    assert len(trace["observations"]) == 2
    for observation in trace["observations"]:
        if observation["type"] == "GENERATION":
            assert observation["promptTokens"] > 0
            assert observation["completionTokens"] > 0
            assert observation["totalTokens"] > 0
            assert observation["input"] is not None
            assert observation["input"] != ""
            assert observation["output"] is not None
            assert observation["output"] != ""
            assert observation["name"] == "Bedrock"
            assert observation["model"] == "claude"


def test_unimplemented_model():
    callback = CallbackHandler(debug=False)

    class CustomLLM(LLM):
        n: int

        @property
        def _llm_type(self) -> str:
            return "custom"

        def _call(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
        ) -> str:
            if stop is not None:
                raise ValueError("stop kwargs are not permitted.")
            return "This is a great text, which i can take characters from "[: self.n]

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"n": self.n}

    custom_llm = CustomLLM(n=10)

    llm = OpenAI(openai_api_key=os.environ.get("OPENAI_API_KEY"))
    template = """You are a playwright. Given the title of play, it is your job to write a synopsis for that title.
        Title: {title}
        Playwright: This is a synopsis for the above play:"""

    prompt_template = PromptTemplate(input_variables=["title"], template=template)
    synopsis_chain = LLMChain(llm=llm, prompt=prompt_template)

    template = """You are a play critic from the New York Times.
    Given the synopsis of play, it is your job to write a review for that play.

        Play Synopsis:
        {synopsis}
        Review from a New York Times play critic of the above play:"""
    prompt_template = PromptTemplate(input_variables=["synopsis"], template=template)
    custom_llm_chain = LLMChain(llm=custom_llm, prompt=prompt_template)

    sequential_chain = SimpleSequentialChain(chains=[custom_llm_chain, synopsis_chain])
    sequential_chain.run("This is a foobar thing", callbacks=[callback])

    callback.flush()

    trace = get_api().trace.get(callback.get_trace_id())

    assert len(trace.observations) == 5

    custom_generation = list(
        filter(
            lambda x: x.type == "GENERATION" and x.name == "CustomLLM",
            trace.observations,
        )
    )[0]

    assert custom_generation.output == "This is a"
    assert custom_generation.model is None


def test_names_on_spans_lcel():
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough
    from langchain_openai import OpenAIEmbeddings

    callback = CallbackHandler(debug=False)
    model = ChatOpenAI(temperature=0)

    template = """Answer the question based only on the following context:
    {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    loader = TextLoader("./static/state_of_the_union.txt", encoding="utf8")

    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))
    docsearch = Chroma.from_documents(texts, embeddings)

    retriever = docsearch.as_retriever()

    retrieval_chain = (
        {
            "context": retriever.with_config(run_name="Docs"),
            "question": RunnablePassthrough(),
        }
        | prompt
        | model.with_config(run_name="my_llm")
        | StrOutputParser()
    )

    retrieval_chain.invoke(
        "What did the president say about Ketanji Brown Jackson?",
        config={
            "callbacks": [callback],
        },
    )

    callback.flush()

    trace = get_api().trace.get(callback.get_trace_id())

    assert len(trace.observations) == 7

    assert (
        len(
            list(
                filter(
                    lambda x: x.type == "GENERATION" and x.name == "my_llm",
                    trace.observations,
                )
            )
        )
        == 1
    )

    assert (
        len(
            list(
                filter(
                    lambda x: x.type == "SPAN" and x.name == "Docs",
                    trace.observations,
                )
            )
        )
        == 1
    )


def test_openai_instruct_usage():
    from langchain_core.output_parsers.string import StrOutputParser
    from langchain_core.runnables import Runnable
    from langchain_openai import OpenAI

    lf_handler = CallbackHandler(debug=True)

    runnable_chain: Runnable = (
        PromptTemplate.from_template(
            """Answer the question based only on the following context:

            Question: {question}

            Answer in the following language: {language}
            """
        )
        | OpenAI(
            model="gpt-3.5-turbo-instruct",
            temperature=0,
            callbacks=[lf_handler],
            max_retries=3,
            timeout=30,
        )
        | StrOutputParser()
    )
    input_list = [
        {"question": "where did harrison work", "language": "english"},
        {"question": "how is your day", "language": "english"},
    ]
    runnable_chain.batch(input_list)

    lf_handler.flush()

    observations = get_api().trace.get(lf_handler.get_trace_id()).observations

    assert len(observations) == 2

    for observation in observations:
        assert observation.type == "GENERATION"
        assert observation.output is not None
        assert observation.output != ""
        assert observation.input is not None
        assert observation.input != ""
        assert observation.usage is not None
        assert observation.usage_details["input"] is not None
        assert observation.usage_details["output"] is not None
        assert observation.usage_details["total"] is not None


def test_get_langchain_prompt_with_jinja2():
    langfuse = Langfuse()

    prompt = 'this is a {{ template }} template that should remain unchanged: {{ handle_text(payload["Name"], "Name is") }}'
    langfuse.create_prompt(
        name="test_jinja2",
        prompt=prompt,
        labels=["production"],
    )

    langfuse_prompt = langfuse.get_prompt(
        "test_jinja2", fetch_timeout_seconds=1, max_retries=3
    )

    assert (
        langfuse_prompt.get_langchain_prompt()
        == 'this is a {template} template that should remain unchanged: {{ handle_text(payload["Name"], "Name is") }}'
    )


def test_get_langchain_prompt():
    langfuse = Langfuse()

    test_prompts = ["This is a {{test}}", "This is a {{test}}. And this is a {{test2}}"]

    for i, test_prompt in enumerate(test_prompts):
        langfuse.create_prompt(
            name=f"test_{i}",
            prompt=test_prompt,
            config={
                "model": "gpt-3.5-turbo-1106",
                "temperature": 0,
            },
            labels=["production"],
        )

        langfuse_prompt = langfuse.get_prompt(f"test_{i}")

        langchain_prompt = ChatPromptTemplate.from_template(
            langfuse_prompt.get_langchain_prompt()
        )

        if i == 0:
            assert langchain_prompt.format(test="test") == "Human: This is a test"
        else:
            assert (
                langchain_prompt.format(test="test", test2="test2")
                == "Human: This is a test. And this is a test2"
            )


def test_get_langchain_chat_prompt():
    langfuse = Langfuse()

    test_prompts = [
        [{"role": "system", "content": "This is a {{test}} with a {{test}}"}],
        [
            {"role": "system", "content": "This is a {{test}}."},
            {"role": "user", "content": "And this is a {{test2}}"},
        ],
    ]

    for i, test_prompt in enumerate(test_prompts):
        langfuse.create_prompt(
            name=f"test_chat_{i}",
            prompt=test_prompt,
            type="chat",
            config={
                "model": "gpt-3.5-turbo-1106",
                "temperature": 0,
            },
            labels=["production"],
        )

        langfuse_prompt = langfuse.get_prompt(f"test_chat_{i}", type="chat")
        langchain_prompt = ChatPromptTemplate.from_messages(
            langfuse_prompt.get_langchain_prompt()
        )

        if i == 0:
            assert (
                langchain_prompt.format(test="test")
                == "System: This is a test with a test"
            )
        else:
            assert (
                langchain_prompt.format(test="test", test2="test2")
                == "System: This is a test.\nHuman: And this is a test2"
            )


def test_disabled_langfuse():
    run_name_override = "This is a custom Run Name"
    handler = CallbackHandler(enabled=False, debug=False)

    prompt = ChatPromptTemplate.from_template("tell me a short joke about {topic}")
    model = ChatOpenAI(temperature=0)

    chain = prompt | model

    chain.invoke(
        {"topic": "ice cream"},
        config={
            "callbacks": [handler],
            "run_name": run_name_override,
        },
    )

    assert handler.langfuse.task_manager._ingestion_queue.empty()

    handler.flush()

    trace_id = handler.get_trace_id()

    with pytest.raises(Exception):
        get_api().trace.get(trace_id)


def test_link_langfuse_prompts_invoke():
    langfuse = Langfuse()
    trace_name = "test_link_langfuse_prompts_invoke"
    session_id = "session_" + create_uuid()[:8]
    user_id = "user_" + create_uuid()[:8]

    # Create prompts
    joke_prompt_name = "joke_prompt_" + create_uuid()[:8]
    joke_prompt_string = "Tell me a joke involving the animal {{animal}}"

    explain_prompt_name = "explain_prompt_" + create_uuid()[:8]
    explain_prompt_string = "Explain the joke to me like I'm a 5 year old {{joke}}"

    langfuse.create_prompt(
        name=joke_prompt_name,
        prompt=joke_prompt_string,
        labels=["production"],
    )

    langfuse.create_prompt(
        name=explain_prompt_name,
        prompt=explain_prompt_string,
        labels=["production"],
    )

    # Get prompts
    langfuse_joke_prompt = langfuse.get_prompt(joke_prompt_name)
    langfuse_explain_prompt = langfuse.get_prompt(explain_prompt_name)

    langchain_joke_prompt = PromptTemplate.from_template(
        langfuse_joke_prompt.get_langchain_prompt(),
        metadata={"langfuse_prompt": langfuse_joke_prompt},
    )

    langchain_explain_prompt = PromptTemplate.from_template(
        langfuse_explain_prompt.get_langchain_prompt(),
        metadata={"langfuse_prompt": langfuse_explain_prompt},
    )

    # Create chain
    parser = StrOutputParser()
    model = OpenAI()
    chain = (
        {"joke": langchain_joke_prompt | model | parser}
        | langchain_explain_prompt
        | model
        | parser
    )

    # Run chain
    langfuse_handler = CallbackHandler(debug=True)

    output = chain.invoke(
        {"animal": "dog"},
        config={
            "callbacks": [langfuse_handler],
            "run_name": trace_name,
            "tags": ["langchain-tag"],
            "metadata": {
                "langfuse_session_id": session_id,
                "langfuse_user_id": user_id,
            },
        },
    )

    langfuse_handler.flush()
    sleep(2)

    trace = get_api().trace.get(langfuse_handler.get_trace_id())

    assert trace.tags == ["langchain-tag"]
    assert trace.session_id == session_id
    assert trace.user_id == user_id

    observations = trace.observations

    generations = sorted(
        list(filter(lambda x: x.type == "GENERATION", observations)),
        key=lambda x: x.start_time,
    )

    assert len(generations) == 2
    assert generations[0].input == "Tell me a joke involving the animal dog"
    assert "Explain the joke to me like I'm a 5 year old" in generations[1].input

    assert generations[0].prompt_name == joke_prompt_name
    assert generations[1].prompt_name == explain_prompt_name

    assert generations[0].prompt_version == langfuse_joke_prompt.version
    assert generations[1].prompt_version == langfuse_explain_prompt.version

    assert generations[1].output == output.strip()


def test_link_langfuse_prompts_stream():
    langfuse = Langfuse(debug=True)
    trace_name = "test_link_langfuse_prompts_stream"
    session_id = "session_" + create_uuid()[:8]
    user_id = "user_" + create_uuid()[:8]

    # Create prompts
    joke_prompt_name = "joke_prompt_" + create_uuid()[:8]
    joke_prompt_string = "Tell me a joke involving the animal {{animal}}"

    explain_prompt_name = "explain_prompt_" + create_uuid()[:8]
    explain_prompt_string = "Explain the joke to me like I'm a 5 year old {{joke}}"

    langfuse.create_prompt(
        name=joke_prompt_name,
        prompt=joke_prompt_string,
        labels=["production"],
    )

    langfuse.create_prompt(
        name=explain_prompt_name,
        prompt=explain_prompt_string,
        labels=["production"],
    )

    # Get prompts
    langfuse_joke_prompt = langfuse.get_prompt(joke_prompt_name)
    langfuse_explain_prompt = langfuse.get_prompt(explain_prompt_name)

    langchain_joke_prompt = PromptTemplate.from_template(
        langfuse_joke_prompt.get_langchain_prompt(),
        metadata={"langfuse_prompt": langfuse_joke_prompt},
    )

    langchain_explain_prompt = PromptTemplate.from_template(
        langfuse_explain_prompt.get_langchain_prompt(),
        metadata={"langfuse_prompt": langfuse_explain_prompt},
    )

    # Create chain
    parser = StrOutputParser()
    model = OpenAI()
    chain = (
        {"joke": langchain_joke_prompt | model | parser}
        | langchain_explain_prompt
        | model
        | parser
    )

    # Run chain
    langfuse_handler = CallbackHandler()

    stream = chain.stream(
        {"animal": "dog"},
        config={
            "callbacks": [langfuse_handler],
            "run_name": trace_name,
            "tags": ["langchain-tag"],
            "metadata": {
                "langfuse_session_id": session_id,
                "langfuse_user_id": user_id,
            },
        },
    )

    output = ""
    for chunk in stream:
        output += chunk

    langfuse_handler.flush()
    sleep(2)

    trace = get_api().trace.get(langfuse_handler.get_trace_id())

    assert trace.tags == ["langchain-tag"]
    assert trace.session_id == session_id
    assert trace.user_id == user_id

    observations = trace.observations

    generations = sorted(
        list(filter(lambda x: x.type == "GENERATION", observations)),
        key=lambda x: x.start_time,
    )

    assert len(generations) == 2
    assert generations[0].input == "Tell me a joke involving the animal dog"
    assert "Explain the joke to me like I'm a 5 year old" in generations[1].input

    assert generations[0].prompt_name == joke_prompt_name
    assert generations[1].prompt_name == explain_prompt_name

    assert generations[0].prompt_version == langfuse_joke_prompt.version
    assert generations[1].prompt_version == langfuse_explain_prompt.version

    assert generations[0].time_to_first_token is not None
    assert generations[1].time_to_first_token is not None

    assert generations[1].output == output.strip() if output else None


def test_link_langfuse_prompts_batch():
    langfuse = Langfuse()
    trace_name = "test_link_langfuse_prompts_batch_" + create_uuid()[:8]

    # Create prompts
    joke_prompt_name = "joke_prompt_" + create_uuid()[:8]
    joke_prompt_string = "Tell me a joke involving the animal {{animal}}"

    explain_prompt_name = "explain_prompt_" + create_uuid()[:8]
    explain_prompt_string = "Explain the joke to me like I'm a 5 year old {{joke}}"

    langfuse.create_prompt(
        name=joke_prompt_name,
        prompt=joke_prompt_string,
        labels=["production"],
    )

    langfuse.create_prompt(
        name=explain_prompt_name,
        prompt=explain_prompt_string,
        labels=["production"],
    )

    # Get prompts
    langfuse_joke_prompt = langfuse.get_prompt(joke_prompt_name)
    langfuse_explain_prompt = langfuse.get_prompt(explain_prompt_name)

    langchain_joke_prompt = PromptTemplate.from_template(
        langfuse_joke_prompt.get_langchain_prompt(),
        metadata={"langfuse_prompt": langfuse_joke_prompt},
    )

    langchain_explain_prompt = PromptTemplate.from_template(
        langfuse_explain_prompt.get_langchain_prompt(),
        metadata={"langfuse_prompt": langfuse_explain_prompt},
    )

    # Create chain
    parser = StrOutputParser()
    model = OpenAI()
    chain = (
        {"joke": langchain_joke_prompt | model | parser}
        | langchain_explain_prompt
        | model
        | parser
    )

    # Run chain
    langfuse_handler = CallbackHandler(debug=True)

    chain.batch(
        [{"animal": "dog"}, {"animal": "cat"}, {"animal": "elephant"}],
        config={
            "callbacks": [langfuse_handler],
            "run_name": trace_name,
            "tags": ["langchain-tag"],
        },
    )

    langfuse_handler.flush()

    traces = get_api().trace.list(name=trace_name).data

    assert len(traces) == 3

    for trace in traces:
        trace = get_api().trace.get(trace.id)

        assert trace.tags == ["langchain-tag"]

        observations = trace.observations

        generations = sorted(
            list(filter(lambda x: x.type == "GENERATION", observations)),
            key=lambda x: x.start_time,
        )

        assert len(generations) == 2

        assert generations[0].prompt_name == joke_prompt_name
        assert generations[1].prompt_name == explain_prompt_name

        assert generations[0].prompt_version == langfuse_joke_prompt.version
        assert generations[1].prompt_version == langfuse_explain_prompt.version


def test_get_langchain_text_prompt_with_precompiled_prompt():
    langfuse = Langfuse()

    prompt_name = "test_precompiled_langchain_prompt"
    test_prompt = (
        "This is a {{pre_compiled_var}}. This is a langchain {{langchain_var}}"
    )

    langfuse.create_prompt(
        name=prompt_name,
        prompt=test_prompt,
        labels=["production"],
    )

    langfuse_prompt = langfuse.get_prompt(prompt_name)
    langchain_prompt = PromptTemplate.from_template(
        langfuse_prompt.get_langchain_prompt(pre_compiled_var="dog")
    )

    assert (
        langchain_prompt.format(langchain_var="chain")
        == "This is a dog. This is a langchain chain"
    )


def test_get_langchain_chat_prompt_with_precompiled_prompt():
    langfuse = Langfuse()

    prompt_name = "test_precompiled_langchain_chat_prompt"
    test_prompt = [
        {"role": "system", "content": "This is a {{pre_compiled_var}}."},
        {"role": "user", "content": "This is a langchain {{langchain_var}}."},
    ]

    langfuse.create_prompt(
        name=prompt_name,
        prompt=test_prompt,
        type="chat",
        labels=["production"],
    )

    langfuse_prompt = langfuse.get_prompt(prompt_name, type="chat")
    langchain_prompt = ChatPromptTemplate.from_messages(
        langfuse_prompt.get_langchain_prompt(pre_compiled_var="dog")
    )

    system_message, user_message = langchain_prompt.format_messages(
        langchain_var="chain"
    )

    assert system_message.content == "This is a dog."
    assert user_message.content == "This is a langchain chain."


def test_callback_openai_functions_with_tools():
    handler = CallbackHandler()

    llm = ChatOpenAI(model="gpt-4", temperature=0, callbacks=[handler])

    class StandardizedAddress(BaseModel):
        street: str = Field(description="The street name and number")
        city: str = Field(description="The city name")
        state: str = Field(description="The state or province")
        zip_code: str = Field(description="The postal code")

    class GetWeather(BaseModel):
        city: str = Field(description="The city name")
        state: str = Field(description="The state or province")
        zip_code: str = Field(description="The postal code")

    address_tool = StructuredTool.from_function(
        func=lambda **kwargs: StandardizedAddress(**kwargs),
        name="standardize_address",
        description="Standardize the given address",
        args_schema=StandardizedAddress,
    )

    weather_tool = StructuredTool.from_function(
        func=lambda **kwargs: GetWeather(**kwargs),
        name="get_weather",
        description="Get the weather for the given city",
        args_schema=GetWeather,
    )

    messages = [
        {
            "role": "user",
            "content": "Please standardize this address: 123 Main St, Springfield, IL 62701",
        }
    ]

    llm.bind_tools([address_tool, weather_tool]).invoke(messages)

    handler.flush()

    trace = get_api().trace.get(handler.get_trace_id())

    generations = list(filter(lambda x: x.type == "GENERATION", trace.observations))
    assert len(generations) > 0

    for generation in generations:
        assert generation.input is not None
        tool_messages = [msg for msg in generation.input if msg["role"] == "tool"]
        assert len(tool_messages) == 2
        assert any(
            "standardize_address" == msg["content"]["function"]["name"]
            for msg in tool_messages
        )
        assert any(
            "get_weather" == msg["content"]["function"]["name"] for msg in tool_messages
        )

        assert generation.output is not None


def test_langfuse_overhead():
    def _generate_random_dict(n: int, key_length: int = 8) -> Dict[str, Any]:
        result = {}
        value_generators = [
            lambda: "".join(
                random.choices(string.ascii_letters, k=random.randint(3, 15))
            ),
            lambda: random.randint(0, 1000),
            lambda: round(random.uniform(0, 100), 2),
            lambda: [random.randint(0, 100) for _ in range(random.randint(1, 5))],
            lambda: random.choice([True, False]),
        ]
        while len(result) < n:
            key = "".join(
                random.choices(string.ascii_letters + string.digits, k=key_length)
            )
            if key in result:
                continue
            value = random.choice(value_generators)()
            result[key] = value
        return result

    # Test performance overhead of langfuse tracing
    inputs = _generate_random_dict(10000, 20000)
    test_chain = RunnableLambda(lambda x: None)

    start = time.monotonic()
    test_chain.invoke(inputs)
    duration_without_langfuse = (time.monotonic() - start) * 1000

    start = time.monotonic()
    handler = CallbackHandler()
    test_chain.invoke(inputs, config={"callbacks": [handler]})
    duration_with_langfuse = (time.monotonic() - start) * 1000

    overhead = duration_with_langfuse - duration_without_langfuse
    print(f"Langfuse overhead: {overhead}ms")

    assert overhead < 50, f"Langfuse tracing overhead of {overhead}ms exceeds threshold"

    handler.flush()

    duration_full = (time.monotonic() - start) * 1000
    print(f"Full execution took {duration_full}ms")

    assert duration_full > 1000, "Full execution should take longer than 1 second"


def test_multimodal():
    handler = CallbackHandler()
    model = ChatOpenAI(model="gpt-4o-mini")

    image_data = encode_file_to_base64("static/puton.jpg")

    message = HumanMessage(
        content=[
            {"type": "text", "text": "What's in this image?"},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
            },
        ],
    )

    response = model.invoke([message], config={"callbacks": [handler]})

    print(response.content)

    handler.flush()

    trace = get_api().trace.get(handler.get_trace_id())

    assert len(trace.observations) == 1
    assert trace.observations[0].type == "GENERATION"

    print(trace.observations[0].input)

    assert (
        "@@@langfuseMedia:type=image/jpeg|id="
        in trace.observations[0].input[0]["content"][1]["image_url"]["url"]
    )
