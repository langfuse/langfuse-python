import os
from langchain import HuggingFaceHub
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
from langfuse.client import Langfuse
from langfuse.model import CreateTrace

from tests.api_wrapper import LangfuseAPI
from tests.utils import create_uuid


def test_callback_default_host():
    handler = CallbackHandler(os.environ.get("LF_PK"), os.environ.get("LF_SK"), debug=True)
    assert handler.langfuse.base_url == "https://cloud.langfuse.com"


def test_langfuse_init():
    callback = CallbackHandler(os.environ.get("LF_PK"), os.environ.get("LF_SK"), os.environ.get("HOST"), debug=True)
    assert callback.trace is None
    assert not callback.runs


@pytest.mark.skip(reason="inference cost")
def test_callback_generate_from_trace():
    api_wrapper = LangfuseAPI(os.environ.get("LF_PK"), os.environ.get("LF_SK"), os.environ.get("HOST"))
    langfuse = Langfuse(os.environ.get("LF_PK"), os.environ.get("LF_SK"), os.environ.get("HOST"), debug=True)

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

    assert len(trace["observations"]) == 2
    assert trace["id"] == trace_id


@pytest.mark.skip(reason="inference cost")
def test_callback_simple_chain():
    api_wrapper = LangfuseAPI(os.environ.get("LF_PK"), os.environ.get("LF_SK"), os.environ.get("HOST"))
    handler = CallbackHandler(os.environ.get("LF_PK"), os.environ.get("LF_SK"), os.environ.get("HOST"), debug=True)

    llm = OpenAI(openai_api_key=os.environ.get("OPENAI_API_KEY"))
    template = """You are a playwright. Given the title of play, it is your job to write a synopsis for that title.
        Title: {title}
        Playwright: This is a synopsis for the above play:"""

    prompt_template = PromptTemplate(input_variables=["title"], template=template)
    synopsis_chain = LLMChain(llm=llm, prompt=prompt_template)

    synopsis_chain.run("Tragedy at sunset on the beach", callbacks=[handler])

    handler.langfuse.flush()

    trace_id = handler.get_trace_id()

    trace = api_wrapper.get_trace(trace_id)

    assert len(trace["observations"]) == 2


@pytest.mark.skip(reason="inference cost")
def test_callback_sequential_chain():
    api_wrapper = LangfuseAPI(os.environ.get("LF_PK"), os.environ.get("LF_SK"), os.environ.get("HOST"))
    handler = CallbackHandler(os.environ.get("LF_PK"), os.environ.get("LF_SK"), os.environ.get("HOST"), debug=True)

    llm = OpenAI(openai_api_key=os.environ.get("OPENAI_API_KEY"))
    template = """You are a playwright. Given the title of play, it is your job to write a synopsis for that title.
        Title: {title}
        Playwright: This is a synopsis for the above play:"""

    prompt_template = PromptTemplate(input_variables=["title"], template=template)
    synopsis_chain = LLMChain(llm=llm, prompt=prompt_template)

    template = """You are a play critic from the New York Times. Given the synopsis of play, it is your job to write a review for that play.

        Play Synopsis:
        {synopsis}
        Review from a New York Times play critic of the above play:"""
    prompt_template = PromptTemplate(input_variables=["synopsis"], template=template)
    review_chain = LLMChain(llm=llm, prompt=prompt_template)

    overall_chain = SimpleSequentialChain(
        chains=[synopsis_chain, review_chain],
    )
    overall_chain.run("Tragedy at sunset on the beach", callbacks=[handler])

    handler.langfuse.flush()

    trace_id = handler.get_trace_id()

    trace = api_wrapper.get_trace(trace_id)

    assert len(trace["observations"]) == 5


@pytest.mark.skip(reason="inference cost")
def test_callback_retriever():
    api_wrapper = LangfuseAPI(os.environ.get("LF_PK"), os.environ.get("LF_SK"), os.environ.get("HOST"))
    handler = CallbackHandler(os.environ.get("LF_PK"), os.environ.get("LF_SK"), os.environ.get("HOST"), debug=True)

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
    handler.langfuse.flush()

    trace_id = handler.get_trace_id()

    trace = api_wrapper.get_trace(trace_id)

    assert len(trace["observations"]) == 5


@pytest.mark.skip(reason="inference cost")
def test_callback_retriever_with_sources():
    api_wrapper = LangfuseAPI(os.environ.get("LF_PK"), os.environ.get("LF_SK"), os.environ.get("HOST"))
    handler = CallbackHandler(os.environ.get("LF_PK"), os.environ.get("LF_SK"), os.environ.get("HOST"), debug=True)

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
    handler.langfuse.flush()

    trace_id = handler.get_trace_id()

    trace = api_wrapper.get_trace(trace_id)

    assert len(trace["observations"]) == 5


@pytest.mark.skip(reason="inference cost")
def test_callback_simple_openai():
    api_wrapper = LangfuseAPI(os.environ.get("LF_PK"), os.environ.get("LF_SK"), os.environ.get("HOST"))
    handler = CallbackHandler(os.environ.get("LF_PK"), os.environ.get("LF_SK"), os.environ.get("HOST"), debug=True)

    llm = OpenAI(openai_api_key=os.environ.get("OPENAI_API_KEY"))

    text = "What would be a good company name for a company that makes colorful socks?"

    llm.predict(text, callbacks=[handler])

    handler.langfuse.flush()

    trace_id = handler.get_trace_id()

    trace = api_wrapper.get_trace(trace_id)

    assert len(trace["observations"]) == 2


@pytest.mark.skip(reason="inference cost")
def test_callback_simple_openai_streaming():
    api_wrapper = LangfuseAPI(os.environ.get("LF_PK"), os.environ.get("LF_SK"), os.environ.get("HOST"))
    handler = CallbackHandler(os.environ.get("LF_PK"), os.environ.get("LF_SK"), os.environ.get("HOST"), debug=True)

    llm = OpenAI(openai_api_key=os.environ.get("OPENAI_API_KEY"), streaming=True)

    text = "What would be a good company name for a company that makes laptops?"

    llm.predict(text, callbacks=[handler])

    handler.langfuse.flush()

    trace_id = handler.get_trace_id()

    trace = api_wrapper.get_trace(trace_id)

    generation = trace["observations"][1]

    assert generation["promptTokens"] is not None
    assert generation["completionTokens"] is not None
    assert generation["totalTokens"] is not None

    assert len(trace["observations"]) == 2


@pytest.mark.skip(reason="inference cost")
def test_callback_simple_llm_chat():
    api_wrapper = LangfuseAPI(os.environ.get("LF_PK"), os.environ.get("LF_SK"), os.environ.get("HOST"))
    handler = CallbackHandler(os.environ.get("LF_PK"), os.environ.get("LF_SK"), os.environ.get("HOST"), debug=True)

    llm = OpenAI(openai_api_key=os.environ.get("OPENAI_API_KEY"))

    tools = load_tools(["serpapi", "llm-math"], llm=llm)

    agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

    agent.run(
        "Who is Leo DiCaprio's girlfriend? What is her current age raised to the 0.43 power?", callbacks=[handler]
    )

    handler.langfuse.flush()

    trace_id = handler.get_trace_id()

    trace = api_wrapper.get_trace(trace_id)

    assert len(trace["observations"]) > 1


@pytest.mark.skip(reason="inference cost")
def test_callback_huggingface_hub():
    api_wrapper = LangfuseAPI(os.environ.get("LF_PK"), os.environ.get("LF_SK"), os.environ.get("HOST"))
    handler = CallbackHandler(os.environ.get("LF_PK"), os.environ.get("LF_SK"), os.environ.get("HOST"), debug=True)

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
