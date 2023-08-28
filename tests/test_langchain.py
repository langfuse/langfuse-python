import os
from langchain import Anthropic, HuggingFaceHub
from langchain.llms import OpenAI
from langchain.chains import LLMChain, SimpleSequentialChain, RetrievalQA
from langchain.prompts import PromptTemplate
import pytest
from langfuse.callback import CallbackHandler
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.agents import AgentType, initialize_agent, load_tools, Tool, ZeroShotAgent, AgentExecutor
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.document_loaders import WebBaseLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.schema import Document
from langchain.chains.query_constructor.base import AttributeInfo
from pydantic import BaseModel, Field
from langchain.chains.openai_functions import create_openai_fn_chain
from langchain.prompts import ChatPromptTemplate
import asyncio
from langchain.chains.router import MultiPromptChain
from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
from langchain.chains.router.multi_prompt_prompt import MULTI_PROMPT_ROUTER_TEMPLATE
from langchain.chains import TransformChain
from langchain import SerpAPIWrapper
from langchain import LLMMathChain

from typing import Any, Dict, List, Optional

from pydantic import Extra

from langchain.schema.language_model import BaseLanguageModel
from langchain.callbacks.manager import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)
from langchain.chains.base import Chain
from langchain.prompts.base import BasePromptTemplate
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.agents import AgentType, initialize_agent, load_tools
from langfuse.client import Langfuse
from langfuse.model import CreateTrace
from langchain.docstore.document import Document
from langchain.chat_models import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.chains import ConversationalRetrievalChain

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
def test_callback_generated_from_trace():
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

    assert handler.get_trace_id() == trace_id
    assert len(trace["observations"]) == 2
    assert trace["id"] == trace_id


@pytest.mark.skip(reason="inference cost")
def test_callback_generated_from_trace_anthropic():
    api_wrapper = LangfuseAPI(os.environ.get("LF_PK"), os.environ.get("LF_SK"), os.environ.get("HOST"))
    langfuse = Langfuse(os.environ.get("LF_PK"), os.environ.get("LF_SK"), os.environ.get("HOST"), debug=True)

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


@pytest.mark.skip(reason="inference cost")
def test_callback_from_trace_simple_chain():
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

    trace_id = handler.get_trace_id()

    trace = api_wrapper.get_trace(trace_id)

    assert len(trace["observations"]) == 2
    assert handler.get_trace_id() == trace_id
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

    handler.flush()

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


@pytest.mark.skip(reason="inference cost")
def test_stuffed_chain():
    with open("./static/state_of_the_union_short.txt", encoding="utf-8") as f:
        api_wrapper = LangfuseAPI(os.environ.get("LF_PK"), os.environ.get("LF_SK"), os.environ.get("HOST"))
        handler = CallbackHandler(os.environ.get("LF_PK"), os.environ.get("LF_SK"), os.environ.get("HOST"), debug=True)

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
    handler.flush()

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
    handler.flush()

    trace_id = handler.get_trace_id()

    trace = api_wrapper.get_trace(trace_id)

    assert len(trace["observations"]) == 5


@pytest.mark.skip(reason="inference cost")
def test_callback_retriever_conversational():
    api_wrapper = LangfuseAPI(os.environ.get("LF_PK"), os.environ.get("LF_SK"), os.environ.get("HOST"))
    handler = CallbackHandler(os.environ.get("LF_PK"), os.environ.get("LF_SK"), os.environ.get("HOST"), debug=True)

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


@pytest.mark.skip(reason="inference cost")
def test_callback_simple_openai():
    api_wrapper = LangfuseAPI(os.environ.get("LF_PK"), os.environ.get("LF_SK"), os.environ.get("HOST"))
    handler = CallbackHandler(os.environ.get("LF_PK"), os.environ.get("LF_SK"), os.environ.get("HOST"), debug=True)

    llm = OpenAI(openai_api_key=os.environ.get("OPENAI_API_KEY"))

    text = "What would be a good company name for a company that makes colorful socks?"

    llm.predict(text, callbacks=[handler])

    handler.flush()

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

    handler.flush()

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

    handler.flush()

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


@pytest.mark.skip(reason="inference cost")
def test_callback_chat_batch_messages():
    api_wrapper = LangfuseAPI(os.environ.get("LF_PK"), os.environ.get("LF_SK"), os.environ.get("HOST"))
    handler = CallbackHandler(os.environ.get("LF_PK"), os.environ.get("LF_SK"), os.environ.get("HOST"), debug=True)

    chat = ChatOpenAI()

    batch_messages = [
        [
            SystemMessage(content="You are a helpful assistant that translates English to French."),
            HumanMessage(content="I love programming."),
        ],
        [
            SystemMessage(content="You are a helpful assistant that translates English to French."),
            HumanMessage(content="I love artificial intelligence."),
        ],
    ]
    chat.generate(batch_messages, callbacks=[handler])
    handler.langfuse.flush()

    trace_id = handler.get_trace_id()

    trace = api_wrapper.get_trace(trace_id)

    assert len(trace["observations"]) == 3


@pytest.mark.skip(reason="inference cost")
def test_callback_chat_simple_chain():
    api_wrapper = LangfuseAPI(os.environ.get("LF_PK"), os.environ.get("LF_SK"), os.environ.get("HOST"))
    handler = CallbackHandler(os.environ.get("LF_PK"), os.environ.get("LF_SK"), os.environ.get("HOST"), debug=True)

    chat = ChatOpenAI()
    chat_prompt = PromptTemplate(
        input_variables=["input_language", "output_language", "text"],
        template="""
You are a helpful assistant that translates {input_language} to {output_language}.
Text: {text}
        """,
    )

    chain = LLMChain(llm=chat, prompt=chat_prompt)
    chain.run(input_language="English", output_language="French", text="I love programming.", callbacks=[handler])

    handler.langfuse.flush()

    trace_id = handler.get_trace_id()

    trace = api_wrapper.get_trace(trace_id)

    assert len(trace["observations"]) == 2


@pytest.mark.skip(reason="inference cost")
def test_callback_simple_openai_chat_streaming():
    api_wrapper = LangfuseAPI(os.environ.get("LF_PK"), os.environ.get("LF_SK"), os.environ.get("HOST"))
    handler = CallbackHandler(os.environ.get("LF_PK"), os.environ.get("LF_SK"), os.environ.get("HOST"), debug=True)

    chat = ChatOpenAI(streaming=True, callbacks=[StreamingStdOutCallbackHandler()], temperature=0.6)
    chat([HumanMessage(content="Write me a song about sparkling water.")], callbacks=[handler])

    handler.langfuse.flush()

    trace_id = handler.get_trace_id()

    trace = api_wrapper.get_trace(trace_id)

    assert len(trace["observations"]) == 2


@pytest.mark.skip(reason="inference cost")
def test_callback_multi_query_retriever():
    handler = CallbackHandler(os.environ.get("LF_PK"), os.environ.get("LF_SK"), os.environ.get("HOST"), debug=True)

    # Load blog post
    loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
    data = loader.load()

    # Split
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    splits = text_splitter.split_documents(data)

    # VectorDB
    embedding = OpenAIEmbeddings()
    vectordb = Chroma.from_documents(documents=splits, embedding=embedding)

    question = "What are the approaches to Task Decomposition?"
    llm = ChatOpenAI(temperature=0)
    retriever_from_llm = MultiQueryRetriever.from_llm(retriever=vectordb.as_retriever(), llm=llm)

    retriever_from_llm.get_relevant_documents(query=question, callbacks=[handler])

    handler.langfuse.flush()


@pytest.mark.skip(reason="inference cost")
def test_callback_chroma_self_querying():
    handler = CallbackHandler(os.environ.get("LF_PK"), os.environ.get("LF_SK"), os.environ.get("HOST"), debug=True)

    embeddings = OpenAIEmbeddings()

    docs = [
        Document(
            page_content="A bunch of scientists bring back dinosaurs and mayhem breaks loose",
            metadata={"year": 1993, "rating": 7.7, "genre": "science fiction"},
        ),
        Document(
            page_content="Leo DiCaprio gets lost in a dream within a dream within a dream within a ...",
            metadata={"year": 2010, "director": "Christopher Nolan", "rating": 8.2},
        ),
        Document(
            page_content="""A psychologist / detective gets lost in a series of dreams within dreams within dreams
             and Inception reused the idea""",
            metadata={"year": 2006, "director": "Satoshi Kon", "rating": 8.6},
        ),
        Document(
            page_content="A bunch of normal-sized women are supremely wholesome and some men pine after them",
            metadata={"year": 2019, "director": "Greta Gerwig", "rating": 8.3},
        ),
        Document(
            page_content="Toys come alive and have a blast doing so",
            metadata={"year": 1995, "genre": "animated"},
        ),
        Document(
            page_content="Three men walk into the Zone, three men walk out of the Zone",
            metadata={
                "year": 1979,
                "rating": 9.9,
                "director": "Andrei Tarkovsky",
                "genre": "science fiction",
                "rating": 9.9,
            },
        ),
    ]
    vectorstore = Chroma.from_documents(docs, embeddings)

    metadata_field_info = [
        AttributeInfo(
            name="genre",
            description="The genre of the movie",
            type="string or list[string]",
        ),
        AttributeInfo(
            name="year",
            description="The year the movie was released",
            type="integer",
        ),
        AttributeInfo(
            name="director",
            description="The name of the movie director",
            type="string",
        ),
        AttributeInfo(name="rating", description="A 1-10 rating for the movie", type="float"),
    ]
    document_content_description = "Brief summary of a movie"
    llm = OpenAI(temperature=0)
    retriever = SelfQueryRetriever.from_llm(
        llm, vectorstore, document_content_description, metadata_field_info, verbose=True
    )

    # This example specifies a query and composite filter
    retriever.get_relevant_documents(
        "What's a movie after 1990 but before 2005 that's all about toys, and preferably is animated",
        callbacks=[handler],
    )

    handler.langfuse.flush()


# error case
@pytest.mark.skip(reason="inference cost")
def test_callback_openai_functions_python():
    handler = CallbackHandler(os.environ.get("LF_PK"), os.environ.get("LF_SK"), os.environ.get("HOST"), debug=True)

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

    chain = create_openai_fn_chain([record_person, record_dog], llm, prompt, verbose=True)
    chain.run(
        "I can't find my dog Henry anywhere, he's a small brown beagle. Could you send a message about him?",
        callbacks=[handler],
    )
    handler.langfuse.flush()


# error case
# @pytest.mark.skip(reason="inference cost")
# @pytest.mark.asyncio
async def test_callback_async_chain():
    handler_serially = CallbackHandler(
        os.environ.get("LF_PK"), os.environ.get("LF_SK"), os.environ.get("HOST"), debug=True
    )
    handler_concurrently = CallbackHandler(
        os.environ.get("LF_PK"), os.environ.get("LF_SK"), os.environ.get("HOST"), debug=True
    )

    def generate_serially():
        llm = OpenAI(temperature=0.9)
        prompt = PromptTemplate(
            input_variables=["product"],
            template="What is a good name for a company that makes {product}?",
        )
        chain = LLMChain(llm=llm, prompt=prompt)
        for _ in range(5):
            chain.run(product="toothpaste", callbacks=[handler_serially])

    async def async_generate(chain):
        await chain.arun(product="toothpaste", callbacks=[handler_concurrently])

    async def generate_concurrently():
        llm = OpenAI(temperature=0.9)
        prompt = PromptTemplate(
            input_variables=["product"],
            template="What is a good name for a company that makes {product}?",
        )
        chain = LLMChain(llm=llm, prompt=prompt)
        tasks = [async_generate(chain) for _ in range(5)]
        await asyncio.gather(*tasks)

    # If running this outside of Jupyter, use asyncio.run(generate_concurrently())
    await generate_concurrently()

    generate_serially()

    handler_serially.langfuse.flush()
    handler_concurrently.langfuse.flush()


@pytest.mark.skip(reason="inference cost")
def test_callback_custom_chain():
    handler = CallbackHandler(os.environ.get("LF_PK"), os.environ.get("LF_SK"), os.environ.get("HOST"), debug=True)

    class MyCustomChain(Chain):
        """
        An example of a custom chain.
        """

        prompt: BasePromptTemplate
        """Prompt object to use."""
        llm: BaseLanguageModel
        output_key: str = "text"

        class Config:
            """Configuration for this pydantic object."""

            extra = Extra.forbid
            arbitrary_types_allowed = True

        @property
        def input_keys(self) -> List[str]:
            """Will be whatever keys the prompt expects.

            :meta private:
            """
            return self.prompt.input_variables

        @property
        def output_keys(self) -> List[str]:
            """Will always return text key.

            :meta private:
            """
            return [self.output_key]

        def _call(
            self,
            inputs: Dict[str, Any],
            run_manager: Optional[CallbackManagerForChainRun] = None,
        ) -> Dict[str, str]:
            prompt_value = self.prompt.format_prompt(**inputs)

            response = self.llm.generate_prompt(
                [prompt_value], callbacks=run_manager.get_child() if run_manager else None
            )

            if run_manager:
                run_manager.on_text("Log something about this run")

            return {self.output_key: response.generations[0][0].text}

        async def _acall(
            self,
            inputs: Dict[str, Any],
            run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
        ) -> Dict[str, str]:
            prompt_value = self.prompt.format_prompt(**inputs)

            response = await self.llm.agenerate_prompt(
                [prompt_value], callbacks=run_manager.get_child() if run_manager else None
            )

            if run_manager:
                await run_manager.on_text("Log something about this run")

            return {self.output_key: response.generations[0][0].text}

        @property
        def _chain_type(self) -> str:
            return "my_custom_chain"

    chain = MyCustomChain(
        prompt=PromptTemplate.from_template("tell us a joke about {topic}"),
        llm=ChatOpenAI(),
    )

    chain.run({"topic": "callbacks"}, callbacks=[handler])

    handler.langfuse.flush()


@pytest.mark.skip(reason="inference cost")
def test_callback_memory_chain():
    handler = CallbackHandler(os.environ.get("LF_PK"), os.environ.get("LF_SK"), os.environ.get("HOST"), debug=True)
    chat = ChatOpenAI()
    conversation = ConversationChain(llm=chat, memory=ConversationBufferMemory())

    conversation.run("Answer briefly. What are the first 3 colors of a rainbow?", callbacks=[handler])
    # -> The first three colors of a rainbow are red, orange, and yellow.
    conversation.run("And the next 4?", callbacks=[handler])
    # -> The next four colors of a rainbow are green, blue, indigo, and violet.

    handler.langfuse.flush()


# doesn't work, need to debug TODO
@pytest.mark.skip(reason="inference cost")
def test_callback_llm_router_chain():
    handler = CallbackHandler(os.environ.get("LF_PK"), os.environ.get("LF_SK"), os.environ.get("HOST"), debug=True)
    llm = OpenAI(temperature=0.3)

    physics_template = """You are a very smart physics professor. \
    You are great at answering questions about physics in a concise and easy to understand manner. \
    When you don't know the answer to a question you admit that you don't know.

    Here is a question:
    {input}"""

    math_template = """You are a very good mathematician. You are great at answering math questions. \
    You are so good because you are able to break down hard problems into their component parts, \
    answer the component parts, and then put them together to answer the broader question.

    Here is a question:
    {input}"""

    prompt_infos = [
        {
            "name": "physics",
            "description": "Good for answering questions about physics",
            "prompt_template": physics_template,
        },
        {
            "name": "math",
            "description": "Good for answering math questions",
            "prompt_template": math_template,
        },
    ]

    default_chain = ConversationChain(llm=llm, output_key="text")
    destination_chains = {}

    for p_info in prompt_infos:
        name = p_info["name"]
        prompt_template = p_info["prompt_template"]
        prompt = PromptTemplate(template=prompt_template, input_variables=["input"])
        chain = LLMChain(llm=llm, prompt=prompt)
        destination_chains[name] = chain

    destinations = [f"{p['name']}: {p['description']}" for p in prompt_infos]
    destinations_str = "\n".join(destinations)
    router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(destinations=destinations_str)
    router_prompt = PromptTemplate(
        template=router_template,
        input_variables=["input"],
        output_parser=RouterOutputParser(),
    )
    router_chain = LLMRouterChain.from_llm(llm, router_prompt)
    chain = MultiPromptChain(
        router_chain=router_chain,
        destination_chains=destination_chains,
        default_chain=default_chain,
        verbose=True,
    )
    input_list = [
        {"input": "What is black body radiation?"},
        {
            "input": "What is the first prime number greater than 40 such that one plus the prime number is divisible by 3?"
        },
    ]

    chain.apply(input_list, callbacks=[handler])

    handler.langfuse.flush()


@pytest.mark.skip(reason="inference cost")
def test_callback_simple_transformation_chain():
    handler = CallbackHandler(os.environ.get("LF_PK"), os.environ.get("LF_SK"), os.environ.get("HOST"), debug=True)

    with open("./static/state_of_the_union.txt") as f:
        state_of_the_union = f.read()

    def transform_func(inputs: dict) -> dict:
        text = inputs["text"]
        shortened_text = "\n\n".join(text.split("\n\n")[:3])
        return {"output_text": shortened_text}

    transform_chain = TransformChain(
        input_variables=["text"], output_variables=["output_text"], transform=transform_func
    )

    template = """Summarize this text:

    {output_text}

    Summary:"""
    prompt = PromptTemplate(input_variables=["output_text"], template=template)
    llm_chain = LLMChain(llm=OpenAI(), prompt=prompt)

    sequential_chain = SimpleSequentialChain(chains=[transform_chain, llm_chain])

    sequential_chain.run(state_of_the_union, callbacks=[handler])

    handler.langfuse.flush()


# confusing token logging and error with calculation
@pytest.mark.skip(reason="inference cost")
def test_callback_openai_multifunctions():
    handler = CallbackHandler(os.environ.get("LF_PK"), os.environ.get("LF_SK"), os.environ.get("HOST"), debug=True)

    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")
    search = SerpAPIWrapper()
    llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)
    tools = [
        Tool(
            name="Search",
            func=search.run,
            description="""useful for when you need to answer questions about current events.
            You should ask targeted questions""",
        ),
        Tool(
            name="Calculator",
            func=llm_math_chain.run,
            description="useful for when you need to answer questions about math",
        ),
    ]
    mrkl = initialize_agent(tools, llm, agent=AgentType.OPENAI_MULTI_FUNCTIONS, verbose=True)
    mrkl.run("What is the weather in LA and SF?", callbacks=[handler])
    mrkl.run("Who is Leo DiCaprio's girlfriend? What is her current age raised to the 0.43 power?", callbacks=[handler])

    handler.langfuse.flush()


@pytest.mark.skip(reason="inference cost")
def test_callback_agent_with_memory():
    handler = CallbackHandler(os.environ.get("LF_PK"), os.environ.get("LF_SK"), os.environ.get("HOST"), debug=True)
    search = SerpAPIWrapper()
    tools = [
        Tool(
            name="Search",
            func=search.run,
            description="""useful for when you need to answer questions about current events.
            You should ask targeted questions""",
        ),
    ]
    prefix = """Have a conversation with a human, answering the following questions as best you can.
    You have access to the following tools:"""
    suffix = """Begin!"

    {chat_history}
    Question: {input}
    {agent_scratchpad}"""

    prompt = ZeroShotAgent.create_prompt(
        tools,
        prefix=prefix,
        suffix=suffix,
        input_variables=["input", "chat_history", "agent_scratchpad"],
    )
    memory = ConversationBufferMemory(memory_key="chat_history")

    llm_chain = LLMChain(llm=OpenAI(temperature=0), prompt=prompt)
    agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True)
    agent_chain = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True, memory=memory)

    agent_chain.run(input="How many people live in canada?", callbacks=[handler])
    agent_chain.run(input="what is their national anthem called?", callbacks=[handler])

    handler.langfuse.flush()


@pytest.mark.skip(reason="inference cost")
def test_callback_csv_loader():
    handler = CallbackHandler(os.environ.get("LF_PK"), os.environ.get("LF_SK"), os.environ.get("HOST"), debug=True)

    loader = CSVLoader(file_path="./static/mlb_teams_2012.csv")
    data = loader.load()

    template = """What is the Payroll of the Dodgers based on the following information?:

    {output_text}"""
    prompt = PromptTemplate(input_variables=["output_text"], template=template)
    llm_chain = LLMChain(llm=OpenAI(), prompt=prompt)

    llm_chain.run(data, callbacks=[handler])

    handler.langfuse.flush()
