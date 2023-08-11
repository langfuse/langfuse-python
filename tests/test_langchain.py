import os
from langchain.llms import OpenAI
from langchain.chains import LLMChain, SimpleSequentialChain, RetrievalQA
from langchain.prompts import PromptTemplate
from langfuse.callback import CallbackHandler
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.agents import AgentType, initialize_agent, load_tools

from tests.api_wrapper import LangfuseAPI


def test_callback_simple_chain():
    api_wrapper = LangfuseAPI(os.environ.get("LF-PK"), os.environ.get("LF-SK"), os.environ.get("HOST"))
    handler = CallbackHandler(os.environ.get("LF-PK"), os.environ.get("LF-SK"), os.environ.get("HOST"))

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


def test_callback_sequential_chain():
    api_wrapper = LangfuseAPI(os.environ.get("LF-PK"), os.environ.get("LF-SK"), os.environ.get("HOST"))
    handler = CallbackHandler(os.environ.get("LF-PK"), os.environ.get("LF-SK"), os.environ.get("HOST"))

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


def test_callback_retriever():
    api_wrapper = LangfuseAPI(os.environ.get("LF-PK"), os.environ.get("LF-SK"), os.environ.get("HOST"))
    handler = CallbackHandler(os.environ.get("LF-PK"), os.environ.get("LF-SK"), os.environ.get("HOST"))

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


def test_callback_simple_llm():
    api_wrapper = LangfuseAPI(os.environ.get("LF-PK"), os.environ.get("LF-SK"), os.environ.get("HOST"))
    handler = CallbackHandler(os.environ.get("LF-PK"), os.environ.get("LF-SK"), os.environ.get("HOST"))

    llm = OpenAI(openai_api_key=os.environ.get("OPENAI_API_KEY"))

    text = "What would be a good company name for a company that makes colorful socks?"

    llm_result = llm.predict(text, callbacks=[handler])
    print(llm_result)

    handler.langfuse.flush()

    trace_id = handler.get_trace_id()

    trace = api_wrapper.get_trace(trace_id)

    assert len(trace["observations"]) == 2


def test_callback_simple_llm_chat():
    api_wrapper = LangfuseAPI(os.environ.get("LF-PK"), os.environ.get("LF-SK"), os.environ.get("HOST"))
    handler = CallbackHandler(os.environ.get("LF-PK"), os.environ.get("LF-SK"), os.environ.get("HOST"))

    llm = OpenAI(openai_api_key=os.environ.get("OPENAI_API_KEY"))

    tools = load_tools(["serpapi", "llm-math"], llm=llm)

    agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

    agent.run("Who is Leo DiCaprio's girlfriend? What is her current age raised to the 0.43 power?", callbacks=[handler])

    handler.langfuse.flush()

    trace_id = handler.get_trace_id()

    trace = api_wrapper.get_trace(trace_id)

    assert len(trace["observations"]) > 1
