import os
import random
import string
import time
from time import sleep
from typing import Any, Dict, List, Literal, Mapping, Optional

import pytest
from langchain.chains import (
    ConversationChain,
    LLMChain,
    SimpleSequentialChain,
)
from langchain.chains.openai_functions import create_openai_fn_chain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.schema import HumanMessage, SystemMessage
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.base import RunnableLambda
from langchain_core.tools import StructuredTool, tool
from langchain_openai import ChatOpenAI, OpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from pydantic.v1 import BaseModel, Field

from langfuse._client.client import Langfuse
from langfuse.langchain import CallbackHandler
from langfuse.langchain.CallbackHandler import LANGSMITH_TAG_HIDDEN
from tests.utils import create_uuid, encode_file_to_base64, get_api


def test_callback_generated_from_trace_chain():
    langfuse = Langfuse()

    with langfuse.start_as_current_span(name="parent") as span:
        trace_id = span.trace_id
        handler = CallbackHandler()

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

    assert len(trace.observations) == 3

    langchain_span = list(
        filter(
            lambda o: o.type == "CHAIN" and o.name == "LLMChain",
            trace.observations,
        )
    )[0]

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
    langfuse = Langfuse()

    trace_id = create_uuid()

    with langfuse.start_as_current_span(name="parent") as span:
        trace_id = span.trace_id
        handler = CallbackHandler()
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

    assert trace.id == trace_id

    assert len(trace.observations) == 2

    langchain_generation_span = list(
        filter(
            lambda o: o.type == "GENERATION" and o.name == "ChatOpenAI",
            trace.observations,
        )
    )[0]

    assert langchain_generation_span.usage_details["input"] > 0
    assert langchain_generation_span.usage_details["output"] > 0
    assert langchain_generation_span.usage_details["total"] > 0
    assert langchain_generation_span.input is not None
    assert langchain_generation_span.input != ""
    assert langchain_generation_span.output is not None
    assert langchain_generation_span.output != ""


def test_callback_generated_from_lcel_chain():
    langfuse = Langfuse()

    with langfuse.start_as_current_span(name="parent") as span:
        trace_id = span.trace_id
        handler = CallbackHandler()
        prompt = ChatPromptTemplate.from_template("tell me a short joke about {topic}")
        model = ChatOpenAI(temperature=0)
        chain = prompt | model

        chain.invoke(
            {"topic": "ice cream"},
            config={
                "callbacks": [handler],
            },
        )

    langfuse.flush()

    trace = get_api().trace.get(trace_id)

    assert trace.input is None
    assert trace.output is None

    assert trace.id == trace_id

    assert len(trace.observations) > 0

    langchain_generation_span = list(
        filter(
            lambda o: o.type == "GENERATION" and o.name == "ChatOpenAI",
            trace.observations,
        )
    )[0]

    assert langchain_generation_span.usage_details["input"] > 1
    assert langchain_generation_span.usage_details["output"] > 0
    assert langchain_generation_span.usage_details["total"] > 0
    assert langchain_generation_span.input is not None
    assert langchain_generation_span.input != ""
    assert langchain_generation_span.output is not None
    assert langchain_generation_span.output != ""


def test_basic_chat_openai():
    # Create a unique name for this test
    test_name = f"Test Basic Chat {create_uuid()}"

    # Initialize handler
    handler = CallbackHandler()
    chat = ChatOpenAI(temperature=0)

    # Prepare messages
    messages = [
        SystemMessage(
            content="You are a helpful assistant that translates English to French."
        ),
        HumanMessage(
            content="Translate this sentence from English to French. I love programming."
        ),
    ]

    # Run the chat with trace metadata
    chat.invoke(messages, config={"callbacks": [handler], "run_name": test_name})

    # Ensure data is flushed to API
    sleep(2)

    # Retrieve trace by name
    traces = get_api().trace.list(name=test_name)
    assert len(traces.data) > 0
    trace = get_api().trace.get(traces.data[0].id)

    # Assertions
    assert trace.name == test_name
    assert len(trace.observations) > 0

    # Get the generation
    generations = [obs for obs in trace.observations if obs.type == "GENERATION"]
    assert len(generations) > 0

    generation = generations[0]
    assert generation.input is not None
    assert generation.output is not None


def test_callback_retriever_conversational_with_memory():
    langfuse = Langfuse()

    with langfuse.start_as_current_span(
        name="retriever_conversational_with_memory_test"
    ) as span:
        trace_id = span.trace_id
        handler = CallbackHandler()

        llm = OpenAI(openai_api_key=os.environ.get("OPENAI_API_KEY"))
        conversation = ConversationChain(
            llm=llm,
            verbose=True,
            memory=ConversationBufferMemory(),
            callbacks=[handler],
        )
        conversation.predict(input="Hi there!", callbacks=[handler])

    handler.client.flush()

    trace = get_api().trace.get(trace_id)

    # Add 1 to account for the wrapping span
    assert len(trace.observations) == 3

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


def test_callback_simple_openai():
    langfuse = Langfuse()

    with langfuse.start_as_current_span(name="simple_openai_test") as span:
        trace_id = span.trace_id

        # Create a unique name for this test
        test_name = f"Test Simple OpenAI {create_uuid()}"

        # Initialize components
        handler = CallbackHandler()
        llm = OpenAI()
        text = (
            "What would be a good company name for a company that makes colorful socks?"
        )

        # Run the LLM
        llm.invoke(text, config={"callbacks": [handler], "run_name": test_name})

        # Ensure data is flushed to API
    handler.client.flush()
    sleep(2)

    # Retrieve trace
    trace = get_api().trace.get(trace_id)

    # Assertions - add 1 for the wrapping span
    assert len(trace.observations) > 1

    # Check generation details
    generations = [obs for obs in trace.observations if obs.type == "GENERATION"]
    assert len(generations) > 0

    generation = generations[0]
    assert generation.input is not None
    assert generation.input != ""
    assert generation.output is not None
    assert generation.output != ""


def test_callback_multiple_invocations_on_different_traces():
    langfuse = Langfuse()

    with langfuse.start_as_current_span(name="multiple_invocations_test") as span:
        trace_id = span.trace_id

        # Create unique names for each test
        test_name_1 = f"Test Multiple Invocations 1 {create_uuid()}"
        test_name_2 = f"Test Multiple Invocations 2 {create_uuid()}"

        # Setup components
        llm = OpenAI()
        text = (
            "What would be a good company name for a company that makes colorful socks?"
        )

        # First invocation
        handler1 = CallbackHandler()
        llm.invoke(text, config={"callbacks": [handler1], "run_name": test_name_1})

        # Second invocation with new handler
        handler2 = CallbackHandler()
        llm.invoke(text, config={"callbacks": [handler2], "run_name": test_name_2})

    handler1.client.flush()

    # Ensure data is flushed to API
    sleep(2)

    # Retrieve trace
    trace = get_api().trace.get(trace_id)

    # Add 1 to account for the wrapping span
    assert len(trace.observations) > 2

    # Check generations
    generations = [obs for obs in trace.observations if obs.type == "GENERATION"]
    assert len(generations) > 1

    for generation in generations:
        assert generation.input is not None
        assert generation.input != ""
        assert generation.output is not None
        assert generation.output != ""


def test_callback_openai_functions_python():
    langfuse = Langfuse()

    with langfuse.start_as_current_span(name="openai_functions_python_test") as span:
        trace_id = span.trace_id
        handler = CallbackHandler()

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
            return f"Recording person {name} of age {age} with favorite food {fav_food.food}!"

        def record_dog(name: str, color: str, fav_food: OptionalFavFood) -> str:
            """Record some basic identifying information about a dog.

            Args:
                name: The dog's name.
                color: The dog's color.
                fav_food: An OptionalFavFood object that either contains the dog's favorite food or a null value.
                Food should be null if it's not known.
            """
            return (
                f"Recording dog {name} of color {color} with favorite food {fav_food}!"
            )

        chain = create_openai_fn_chain(
            [record_person, record_dog], llm, prompt, callbacks=[handler]
        )
        chain.run(
            "I can't find my dog Henry anywhere, he's a small brown beagle. Could you send a message about him?",
            callbacks=[handler],
        )

    handler.client.flush()

    trace = get_api().trace.get(trace_id)

    # Add 1 to account for the wrapping span
    assert len(trace.observations) == 3

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
    langfuse = Langfuse()

    with langfuse.start_as_current_span(name="agent_executor_chain_test") as span:
        trace_id = span.trace_id
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

        callback = CallbackHandler()
        llm = OpenAI(temperature=0)

        @tool
        def get_word_length(word: str) -> int:
            """Returns the length of a word."""
            return len(word)

        tools = [get_word_length]
        agent = create_react_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(
            agent=agent, tools=tools, handle_parsing_errors=True
        )

        agent_executor.invoke(
            {"input": "what is the length of the word LangFuse?"},
            config={"callbacks": [callback]},
        )

    callback.client.flush()

    trace = get_api().trace.get(trace_id)

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


def test_unimplemented_model():
    langfuse = Langfuse()

    with langfuse.start_as_current_span(name="unimplemented_model_test") as span:
        trace_id = span.trace_id
        callback = CallbackHandler()

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
                return "This is a great text, which i can take characters from "[
                    : self.n
                ]

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
        prompt_template = PromptTemplate(
            input_variables=["synopsis"], template=template
        )
        custom_llm_chain = LLMChain(llm=custom_llm, prompt=prompt_template)

        sequential_chain = SimpleSequentialChain(
            chains=[custom_llm_chain, synopsis_chain]
        )
        sequential_chain.run("This is a foobar thing", callbacks=[callback])

    callback.client.flush()

    trace = get_api().trace.get(trace_id)

    # Add 1 to account for the wrapping span
    assert len(trace.observations) == 6

    custom_generation = list(
        filter(
            lambda x: x.type == "GENERATION" and x.name == "CustomLLM",
            trace.observations,
        )
    )[0]

    assert custom_generation.output == "This is a"
    assert custom_generation.model is None


def test_openai_instruct_usage():
    langfuse = Langfuse()

    with langfuse.start_as_current_span(name="openai_instruct_usage_test") as span:
        trace_id = span.trace_id
        from langchain_core.output_parsers.string import StrOutputParser
        from langchain_core.runnables import Runnable
        from langchain_openai import OpenAI

        lf_handler = CallbackHandler()

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

    lf_handler.client.flush()

    observations = get_api().trace.get(trace_id).observations

    # Add 1 to account for the wrapping span
    assert len(observations) == 3

    for observation in observations:
        if observation.type == "GENERATION":
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


def test_link_langfuse_prompts_invoke():
    langfuse = Langfuse()
    trace_name = "test_link_langfuse_prompts_invoke"

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

    with langfuse.start_as_current_span(name=trace_name) as span:
        trace_id = span.trace_id
        chain.invoke(
            {"animal": "dog"},
            config={
                "callbacks": [langfuse_handler],
                "run_name": trace_name,
            },
        )

    langfuse_handler.client.flush()
    sleep(2)

    trace = get_api().trace.get(trace_id=trace_id)

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


def test_link_langfuse_prompts_stream():
    langfuse = Langfuse()
    trace_name = "test_link_langfuse_prompts_stream"

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

    with langfuse.start_as_current_span(name=trace_name) as span:
        trace_id = span.trace_id
        stream = chain.stream(
            {"animal": "dog"},
            config={
                "callbacks": [langfuse_handler],
                "run_name": trace_name,
            },
        )

        output = ""
        for chunk in stream:
            output += chunk

    langfuse_handler.client.flush()
    sleep(2)

    trace = get_api().trace.get(trace_id=trace_id)

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
    langfuse_handler = CallbackHandler()

    with langfuse.start_as_current_span(name=trace_name) as span:
        trace_id = span.trace_id
        chain.batch(
            [{"animal": "dog"}, {"animal": "cat"}, {"animal": "elephant"}],
            config={
                "callbacks": [langfuse_handler],
                "run_name": trace_name,
            },
        )

    langfuse_handler.client.flush()

    traces = get_api().trace.list(name=trace_name).data

    assert len(traces) == 1

    trace = get_api().trace.get(trace_id=trace_id)

    observations = trace.observations

    generations = sorted(
        list(filter(lambda x: x.type == "GENERATION", observations)),
        key=lambda x: x.start_time,
    )

    assert len(generations) == 6

    assert generations[0].prompt_name == joke_prompt_name
    assert generations[1].prompt_name == joke_prompt_name
    assert generations[2].prompt_name == joke_prompt_name
    assert generations[3].prompt_name == explain_prompt_name
    assert generations[4].prompt_name == explain_prompt_name
    assert generations[5].prompt_name == explain_prompt_name

    assert generations[0].prompt_version == langfuse_joke_prompt.version
    assert generations[1].prompt_version == langfuse_joke_prompt.version
    assert generations[2].prompt_version == langfuse_joke_prompt.version
    assert generations[3].prompt_version == langfuse_explain_prompt.version
    assert generations[4].prompt_version == langfuse_explain_prompt.version
    assert generations[5].prompt_version == langfuse_explain_prompt.version


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

    with handler.client.start_as_current_span(
        name="test_callback_openai_functions_with_tools"
    ) as span:
        trace_id = span.trace_id
        llm.bind_tools([address_tool, weather_tool]).invoke(messages)

    handler.client.flush()

    trace = get_api().trace.get(trace_id=trace_id)

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


@pytest.mark.skip(reason="Flaky test")
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
    langfuse = Langfuse()

    with langfuse.start_as_current_span(name="test_langfuse_overhead"):
        test_chain.invoke(inputs, config={"callbacks": [handler]})

    duration_with_langfuse = (time.monotonic() - start) * 1000

    overhead = duration_with_langfuse - duration_without_langfuse
    print(f"Langfuse overhead: {overhead}ms")

    assert (
        overhead < 100
    ), f"Langfuse tracing overhead of {overhead}ms exceeds threshold"

    langfuse.flush()

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

    with handler.client.start_as_current_span(name="test_multimodal") as span:
        trace_id = span.trace_id
        model.invoke([message], config={"callbacks": [handler]})

    handler.client.flush()

    trace = get_api().trace.get(trace_id=trace_id)

    assert len(trace.observations) == 2
    # Filter for the observation with type GENERATION
    generation_observation = next(
        (obs for obs in trace.observations if obs.type == "GENERATION"), None
    )

    assert generation_observation is not None

    assert (
        "@@@langfuseMedia:type=image/jpeg|id="
        in generation_observation.input[0]["content"][1]["image_url"]["url"]
    )


def test_langgraph():
    # Define the tools for the agent to use
    @tool
    def search(query: str):
        """Call to surf the web."""
        # This is a placeholder, but don't tell the LLM that...
        if "sf" in query.lower() or "san francisco" in query.lower():
            return "It's 60 degrees and foggy."
        return "It's 90 degrees and sunny."

    tools = [search]
    tool_node = ToolNode(tools)
    model = ChatOpenAI(model="gpt-4o-mini").bind_tools(tools)

    # Define the function that determines whether to continue or not
    def should_continue(state: MessagesState) -> Literal["tools", END]:
        messages = state["messages"]
        last_message = messages[-1]
        # If the LLM makes a tool call, then we route to the "tools" node
        if last_message.tool_calls:
            return "tools"
        # Otherwise, we stop (reply to the user)
        return END

    # Define the function that calls the model
    def call_model(state: MessagesState):
        messages = state["messages"]
        response = model.invoke(messages)
        # We return a list, because this will get added to the existing list
        return {"messages": [response]}

    # Define a new graph
    workflow = StateGraph(MessagesState)

    # Define the two nodes we will cycle between
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", tool_node)

    # Set the entrypoint as `agent`
    # This means that this node is the first one called
    workflow.add_edge(START, "agent")

    # We now add a conditional edge
    workflow.add_conditional_edges(
        # First, we define the start node. We use `agent`.
        # This means these are the edges taken after the `agent` node is called.
        "agent",
        # Next, we pass in the function that will determine which node is called next.
        should_continue,
    )

    # We now add a normal edge from `tools` to `agent`.
    # This means that after `tools` is called, `agent` node is called next.
    workflow.add_edge("tools", "agent")

    # Initialize memory to persist state between graph runs
    checkpointer = MemorySaver()

    # Finally, we compile it!
    # This compiles it into a LangChain Runnable,
    # meaning you can use it as you would any other runnable.
    # Note that we're (optionally) passing the memory when compiling the graph
    app = workflow.compile(checkpointer=checkpointer)

    handler = CallbackHandler()

    # Use the Runnable
    with handler.client.start_as_current_span(name="test_langgraph") as span:
        trace_id = span.trace_id
        final_state = app.invoke(
            {"messages": [HumanMessage(content="what is the weather in sf")]},
            config={"configurable": {"thread_id": 42}, "callbacks": [handler]},
        )
    print(final_state["messages"][-1].content)
    handler.client.flush()

    trace = get_api().trace.get(trace_id=trace_id)

    hidden_count = 0

    for observation in trace.observations:
        if LANGSMITH_TAG_HIDDEN in observation.metadata.get("tags", []):
            hidden_count += 1
            assert observation.level == "DEBUG"

        else:
            assert observation.level == "DEFAULT"

    assert hidden_count > 0


@pytest.mark.skip(reason="Flaky test")
def test_cached_token_usage():
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "This is a test prompt to reproduce the issue. "
                    "The prompt needs 1024 tokens to enable cache." * 100
                ),
            ),
            ("user", "Reply to this message {test_param}."),
        ]
    )
    chat = ChatOpenAI(model="gpt-4o-mini")
    chain = prompt | chat
    handler = CallbackHandler()
    config = {"callbacks": [handler]}

    chain.invoke({"test_param": "in a funny way"}, config)
    chain.invoke({"test_param": "in a funny way"}, config)
    sleep(1)

    # invoke again to force cached token usage
    chain.invoke({"test_param": "in a funny way"}, config)

    handler.client.flush()

    trace = get_api().trace.get(handler.get_trace_id())

    generation = next((o for o in trace.observations if o.type == "GENERATION"))

    assert generation.usage_details["input_cache_read"] > 0
    assert (
        generation.usage_details["input"]
        + generation.usage_details["input_cache_read"]
        + generation.usage_details["output"]
        == generation.usage_details["total"]
    )

    assert generation.cost_details["input_cache_read"] > 0
    assert (
        abs(
            generation.cost_details["input"]
            + generation.cost_details["input_cache_read"]
            + generation.cost_details["output"]
            - generation.cost_details["total"]
        )
        < 0.0001
    )
