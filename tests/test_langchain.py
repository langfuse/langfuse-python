import os
from langchain import Anthropic, ConversationChain, HuggingFaceHub
from langchain.llms import OpenAI
from langchain.chains import LLMChain, SimpleSequentialChain, RetrievalQA
from langchain.prompts import PromptTemplate
import pytest
from langfuse.callback import CallbackHandler
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.chat_models import ChatOpenAI
from langchain.schema import Document
from pydantic import BaseModel, Field
from langchain.chains.openai_functions import create_openai_fn_chain
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import AzureChatOpenAI
from typing import Optional
from langfuse.client import Langfuse
from langfuse.model import CreateSpan, CreateTrace
from langchain.chains.summarize import load_summarize_chain
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

from tests.api_wrapper import LangfuseAPI
from tests.utils import create_uuid, get_api


def test_langfuse_release_init():
    callback = CallbackHandler(release="something")
    assert callback.langfuse.release == "something"


def test_callback_init():
    callback = CallbackHandler(debug=False)
    assert callback.trace is None
    assert not callback.runs


def test_langfuse_span():
    trace_id = create_uuid()
    span_id = create_uuid()
    langfuse = Langfuse(debug=False)
    trace = langfuse.trace(CreateTrace(id=trace_id))
    span = trace.span(CreateSpan(id=span_id))

    handler = span.get_langchain_handler()

    assert handler.get_trace_id() == trace_id
    assert handler.rootSpan.id == span_id


def test_callback_generated_from_trace():
    api_wrapper = LangfuseAPI()
    langfuse = Langfuse(debug=False)

    trace_id = create_uuid()
    trace = langfuse.trace(CreateTrace(id=trace_id))

    handler = trace.getNewHandler()

    llm = OpenAI(openai_api_key=os.environ.get("OPENAI_API_KEY"))
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

    for observation in trace["observations"]:
        if observation["type"] == "GENERATION":
            assert observation["promptTokens"] > 0
            assert observation["completionTokens"] > 0
            assert observation["totalTokens"] > 0
            assert observation["input"] is not None
            assert observation["input"] != ""
            assert observation["output"] is not None
            assert observation["output"] != ""


@pytest.mark.skip(reason="inference cost")
def test_callback_generated_from_trace_azure_chat():
    api_wrapper = LangfuseAPI()
    langfuse = Langfuse(debug=False)

    trace_id = create_uuid()
    trace = langfuse.trace(CreateTrace(id=trace_id))

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


@pytest.mark.skip(reason="inference cost")
def test_callback_generated_from_trace_anthropic():
    api_wrapper = LangfuseAPI()
    langfuse = Langfuse(debug=False)

    trace_id = create_uuid()
    trace = langfuse.trace(CreateTrace(id=trace_id))

    handler = trace.getNewHandler()

    llm = Anthropic(anthropic_api_key=os.environ.get("OPENAI_API_KEY"), model="Claude-v1")
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
    for observation in trace["observations"]:
        if observation["type"] == "GENERATION":
            assert observation["promptTokens"] > 0
            assert observation["completionTokens"] > 0
            assert observation["totalTokens"] > 0
            assert observation["input"] is not None
            assert observation["input"] != ""
            assert observation["output"] is not None
            assert observation["output"] != ""


def test_callback_from_trace_simple_chain():
    langfuse = Langfuse(debug=False)

    trace_id = create_uuid()
    trace = langfuse.trace(CreateTrace(id=trace_id))

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

    api = get_api()
    trace = api.trace.get(trace_id)

    assert len(trace.observations) == 2
    assert handler.get_trace_id() == trace_id
    assert trace.id == trace_id

    generations = list(filter(lambda x: x.type == "GENERATION", trace.observations))
    assert len(generations) > 0
    for generation in generations:
        assert generation.input is not None
        assert generation.output is not None
        assert generation.total_tokens is not None
        assert generation.prompt_tokens is not None
        assert generation.completion_tokens is not None


def test_next_span_id_from_trace_simple_chain():
    api_wrapper = LangfuseAPI()
    langfuse = Langfuse()

    trace_id = create_uuid()
    trace = langfuse.trace(CreateTrace(id=trace_id))

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

    assert any(observation["id"] == next_span_id for observation in trace["observations"])
    for observation in trace["observations"]:
        if observation["type"] == "GENERATION":
            assert observation["promptTokens"] > 0
            assert observation["completionTokens"] > 0
            assert observation["totalTokens"] > 0
            assert observation["input"] is not None
            assert observation["input"] != ""
            assert observation["output"] is not None
            assert observation["output"] != ""


def test_callback_simple_chain():
    api_wrapper = LangfuseAPI()
    handler = CallbackHandler(debug=False)

    llm = ChatOpenAI(openai_api_key=os.environ.get("OPENAI_API_KEY"))
    template = """You are a playwright. Given the title of play, it is your job to write a synopsis for that title.
        Title: {title}
        Playwright: This is a synopsis for the above play:"""

    prompt_template = PromptTemplate(input_variables=["title"], template=template)
    synopsis_chain = LLMChain(llm=llm, prompt=prompt_template)

    synopsis_chain.run("Tragedy at sunset on the beach", callbacks=[handler])

    handler.flush()

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


def test_callback_sequential_chain():
    api_wrapper = LangfuseAPI()
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

        chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt, verbose=False)

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

    chain = RetrievalQA.from_chain_type(llm, retriever=docsearch.as_retriever(), return_source_documents=True)

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
    conversation = ConversationChain(llm=llm, verbose=True, memory=ConversationBufferMemory(), callbacks=[handler])
    conversation.predict(input="Hi there!", callbacks=[handler])
    handler.flush()

    api = get_api()

    trace = api.trace.get(handler.get_trace_id())

    generations = list(filter(lambda x: x.type == "GENERATION", trace.observations))
    assert len(generations) == 1

    for generation in generations:
        assert generation.input is not None
        assert generation.output is not None
        assert generation.input != ""
        assert generation.output != ""
        assert generation.total_tokens is not None
        assert generation.prompt_tokens is not None
        assert generation.completion_tokens is not None


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
        ChatOpenAI(openai_api_key=os.environ.get("OPENAI_API_KEY"), temperature=0.5, model="gpt-3.5-turbo-16k"),
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
    api_wrapper = LangfuseAPI()
    handler = CallbackHandler(debug=False)

    llm = OpenAI(openai_api_key=os.environ.get("OPENAI_API_KEY"))

    text = "What would be a good company name for a company that makes colorful socks?"

    llm.predict(text, callbacks=[handler])

    handler.flush()

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


@pytest.mark.skip(reason="inference cost")
def test_callback_simple_openai_streaming():
    api_wrapper = LangfuseAPI()
    handler = CallbackHandler(debug=False)

    llm = OpenAI(openai_api_key=os.environ.get("OPENAI_API_KEY"), streaming=True)

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
def test_callback_simple_llm_chat():
    handler = CallbackHandler()

    llm = OpenAI(openai_api_key=os.environ.get("OPENAI_API_KEY"))

    tools = load_tools(["serpapi", "llm-math"], llm=llm)

    agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION)

    agent.run("Who is Leo DiCaprio's girlfriend? What is her current age raised to the 0.43 power?", callbacks=[handler])

    handler.flush()

    trace_id = handler.get_trace_id()
    api = get_api()

    trace = api.trace.get(trace_id)
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
        llm = HuggingFaceHub(repo_id=repo_id, model_kwargs={"temperature": 0.1, "max_length": 500})
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
            ("system", "You are a world class algorithm for extracting information in structured formats."),
            ("human", "Use the given format to extract information from the following input: {input}"),
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
        return f"Recording person {name} of age {age} with favorite food {fav_food.food}!"

    def record_dog(name: str, color: str, fav_food: OptionalFavFood) -> str:
        """Record some basic identifying information about a dog.

        Args:
            name: The dog's name.
            color: The dog's color.
            fav_food: An OptionalFavFood object that either contains the dog's favorite food or a null value.
            Food should be null if it's not known.
        """
        return f"Recording dog {name} of color {color} with favorite food {fav_food}!"

    chain = create_openai_fn_chain([record_person, record_dog], llm, prompt, callbacks=[handler])
    chain.run(
        "I can't find my dog Henry anywhere, he's a small brown beagle. Could you send a message about him?",
        callbacks=[handler],
    )

    handler.langfuse.flush()

    api = get_api()
    trace = api.trace.get(handler.get_trace_id())

    assert len(trace.observations) == 2

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


def test_create_extraction_chain():
    import os

    from uuid import uuid4
    from langchain.chains import create_extraction_chain

    from langfuse.client import Langfuse
    from langfuse.model import CreateTrace
    from langchain.text_splitter import CharacterTextSplitter
    from langchain.embeddings.openai import OpenAIEmbeddings
    from langchain.vectorstores import Chroma
    from langchain.chat_models import ChatOpenAI
    from langchain.document_loaders import TextLoader

    def create_uuid():
        return str(uuid4())

    langfuse = Langfuse(debug=False, host="http://localhost:3000")

    trace_id = create_uuid()

    trace = langfuse.trace(CreateTrace(id=trace_id))
    handler = trace.getNewHandler()

    loader = TextLoader("./static/state_of_the_union.txt", encoding="utf8")

    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))
    vector_search = Chroma.from_documents(texts, embeddings)

    main_character = vector_search.similarity_search("Who is the main character and what is the summary of the text?")

    llm = ChatOpenAI(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0,
        streaming=False,
        model="gpt-3.5-turbo-16k-0613",
    )

    schema = {
        "properties": {
            "Main character": {"type": "string"},
            "Summary": {"type": "string"},
        },
        "required": [
            "Main character",
            "Cummary",
        ],
    }
    chain = create_extraction_chain(schema, llm)

    chain.run(main_character, callbacks=[handler])

    handler.flush()

    api = get_api()

    trace = api.trace.get(handler.get_trace_id())

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
def test_aws_bedrock_chain():
    import os

    from langchain.llms.bedrock import Bedrock
    import boto3

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
