import importlib
from contextvars import copy_context
from unittest.mock import patch
from uuid import uuid4

import pytest
from langchain.messages import HumanMessage
from langchain_core.messages import AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.outputs import ChatGeneration, ChatResult, Generation, LLMResult
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAI
from opentelemetry import context as otel_context

from langfuse._client.attributes import LangfuseOtelSpanAttributes
from langfuse.langchain import CallbackHandler

callback_handler_module = importlib.import_module("langfuse.langchain.CallbackHandler")


def _assert_parent_child(parent_span, child_span) -> None:
    assert child_span.parent is not None
    assert child_span.parent.span_id == parent_span.context.span_id


def test_chat_model_callback_exports_generation_span(
    langfuse_memory_client, get_span, json_attr
):
    response = ChatResult(
        generations=[
            ChatGeneration(message=AIMessage(content="bonjour"), text="bonjour")
        ],
        llm_output={
            "token_usage": {
                "prompt_tokens": 4,
                "completion_tokens": 2,
                "total_tokens": 6,
            },
            "model_name": "gpt-4o-mini",
        },
    )

    with patch.object(ChatOpenAI, "_generate", return_value=response):
        handler = CallbackHandler()

        with langfuse_memory_client.start_as_current_observation(name="parent"):
            ChatOpenAI(api_key="test", temperature=0).invoke(
                [HumanMessage(content="hello")],
                config={"callbacks": [handler]},
            )

    langfuse_memory_client.flush()
    parent_span = get_span("parent")
    generation_span = get_span("ChatOpenAI")

    _assert_parent_child(parent_span, generation_span)
    assert (
        generation_span.attributes[LangfuseOtelSpanAttributes.OBSERVATION_TYPE]
        == "generation"
    )
    assert json_attr(generation_span, LangfuseOtelSpanAttributes.OBSERVATION_INPUT) == [
        {"role": "user", "content": "hello"}
    ]
    assert json_attr(
        generation_span, LangfuseOtelSpanAttributes.OBSERVATION_OUTPUT
    ) == {
        "role": "assistant",
        "content": "bonjour",
    }
    assert (
        generation_span.attributes[LangfuseOtelSpanAttributes.OBSERVATION_MODEL]
        == "gpt-4o-mini"
    )
    assert json_attr(
        generation_span, LangfuseOtelSpanAttributes.OBSERVATION_USAGE_DETAILS
    ) == {
        "prompt_tokens": 4,
        "completion_tokens": 2,
        "total_tokens": 6,
    }


def test_llm_callback_exports_generation_span(langfuse_memory_client, get_span):
    response = LLMResult(
        generations=[[Generation(text="sockzilla")]],
        llm_output={
            "token_usage": {
                "prompt_tokens": 7,
                "completion_tokens": 3,
                "total_tokens": 10,
            },
            "model_name": "gpt-4o-mini-instruct",
        },
    )

    with patch.object(OpenAI, "_generate", return_value=response):
        handler = CallbackHandler()

        with langfuse_memory_client.start_as_current_observation(name="parent"):
            OpenAI(api_key="test", temperature=0).invoke(
                "name a sock company",
                config={"callbacks": [handler], "run_name": "sock-name"},
            )

    langfuse_memory_client.flush()
    span = get_span("sock-name")

    assert span.attributes[LangfuseOtelSpanAttributes.OBSERVATION_OUTPUT] == "sockzilla"
    assert (
        span.attributes[LangfuseOtelSpanAttributes.OBSERVATION_MODEL]
        == "gpt-4o-mini-instruct"
    )


def test_lcel_chain_exports_intermediate_chain_spans(
    langfuse_memory_client, get_span, find_spans
):
    response = ChatResult(
        generations=[
            ChatGeneration(
                message=AIMessage(content="knock knock"),
                text="knock knock",
            )
        ],
        llm_output={
            "token_usage": {
                "prompt_tokens": 4,
                "completion_tokens": 2,
                "total_tokens": 6,
            },
            "model_name": "gpt-4o-mini",
        },
    )

    with patch.object(ChatOpenAI, "_generate", return_value=response):
        handler = CallbackHandler()
        prompt = ChatPromptTemplate.from_template("tell me a joke about {topic}")
        chain = prompt | ChatOpenAI(api_key="test", temperature=0) | StrOutputParser()

        with langfuse_memory_client.start_as_current_observation(name="parent"):
            result = chain.invoke({"topic": "otters"}, config={"callbacks": [handler]})

    assert result == "knock knock"

    langfuse_memory_client.flush()
    sequence_span = get_span("RunnableSequence")
    prompt_span = get_span("ChatPromptTemplate")
    generation_span = get_span("ChatOpenAI")
    parser_span = get_span("StrOutputParser")

    _assert_parent_child(sequence_span, prompt_span)
    _assert_parent_child(sequence_span, generation_span)
    _assert_parent_child(sequence_span, parser_span)
    assert len(find_spans("ChatOpenAI")) == 1


def test_chat_model_error_marks_generation_error(langfuse_memory_client, get_span):
    with patch.object(ChatOpenAI, "_generate", side_effect=RuntimeError("boom")):
        handler = CallbackHandler()

        with langfuse_memory_client.start_as_current_observation(name="parent"):
            with pytest.raises(RuntimeError, match="boom"):
                ChatOpenAI(api_key="test", temperature=0).invoke(
                    [HumanMessage(content="hello")],
                    config={"callbacks": [handler]},
                )

    langfuse_memory_client.flush()
    span = get_span("ChatOpenAI")

    assert span.attributes[LangfuseOtelSpanAttributes.OBSERVATION_LEVEL] == "ERROR"
    assert (
        "boom" in span.attributes[LangfuseOtelSpanAttributes.OBSERVATION_STATUS_MESSAGE]
    )


def test_root_chain_metadata_propagates_trace_name(
    langfuse_memory_client, get_span, find_spans
):
    response = ChatResult(
        generations=[
            ChatGeneration(
                message=AIMessage(content="knock knock"),
                text="knock knock",
            )
        ],
        llm_output={
            "token_usage": {
                "prompt_tokens": 4,
                "completion_tokens": 2,
                "total_tokens": 6,
            },
            "model_name": "gpt-4o-mini",
        },
    )

    with patch.object(ChatOpenAI, "_generate", return_value=response):
        handler = CallbackHandler()
        prompt = ChatPromptTemplate.from_template("tell me a joke about {topic}")
        chain = prompt | ChatOpenAI(api_key="test", temperature=0) | StrOutputParser()

        result = chain.invoke(
            {"topic": "otters"},
            config={
                "callbacks": [handler],
                "metadata": {"langfuse_trace_name": "langchain-trace-name"},
            },
        )

    assert result == "knock knock"

    langfuse_memory_client.flush()
    root_span = get_span("RunnableSequence")
    generation_span = get_span("ChatOpenAI")

    assert (
        root_span.attributes[LangfuseOtelSpanAttributes.TRACE_NAME]
        == "langchain-trace-name"
    )
    assert (
        generation_span.attributes[LangfuseOtelSpanAttributes.TRACE_NAME]
        == "langchain-trace-name"
    )
    assert (
        f"{LangfuseOtelSpanAttributes.OBSERVATION_METADATA}.langfuse_trace_name"
        not in root_span.attributes
    )
    assert len(find_spans("ChatOpenAI")) == 1


def test_root_chain_exports_when_end_runs_in_copied_context(
    langfuse_memory_client, get_span
):
    handler = CallbackHandler()
    run_id = uuid4()

    handler.on_chain_start(
        {"id": ["RunnableSequence"]},
        {"topic": "otters"},
        run_id=run_id,
        metadata={"langfuse_trace_name": "async-root-trace"},
    )

    copy_context().run(
        handler.on_chain_end,
        {"output": "knock knock"},
        run_id=run_id,
    )

    langfuse_memory_client.flush()
    root_span = get_span("RunnableSequence")

    assert root_span.attributes[LangfuseOtelSpanAttributes.TRACE_NAME] == (
        "async-root-trace"
    )


def test_control_flow_errors_use_default_level_and_keep_status_message(
    langfuse_memory_client, get_span, monkeypatch
):
    class DummyControlFlowError(RuntimeError):
        pass

    monkeypatch.setattr(
        callback_handler_module,
        "CONTROL_FLOW_EXCEPTION_TYPES",
        {DummyControlFlowError},
    )

    handler = CallbackHandler()

    tool_run_id = uuid4()
    retriever_run_id = uuid4()
    llm_run_id = uuid4()
    chain_run_id = uuid4()

    handler.on_tool_start(
        {"name": "human_approval"},
        "{}",
        run_id=tool_run_id,
    )
    handler.on_tool_error(
        DummyControlFlowError("tool interrupt"),
        run_id=tool_run_id,
    )

    handler.on_retriever_start(
        {"name": "knowledge_base"},
        "approval policy",
        run_id=retriever_run_id,
    )
    handler.on_retriever_error(
        DummyControlFlowError("retriever bubble-up"),
        run_id=retriever_run_id,
    )

    handler.on_llm_start(
        {"name": "TestLLM"},
        ["need approval"],
        run_id=llm_run_id,
        invocation_params={},
    )
    handler.on_llm_error(
        DummyControlFlowError("llm bubble-up"),
        run_id=llm_run_id,
    )

    handler.on_chain_start(
        {"name": "LangGraph"},
        {"messages": ["need approval"]},
        run_id=chain_run_id,
    )
    handler.on_chain_error(
        DummyControlFlowError("graph interrupt"),
        run_id=chain_run_id,
    )

    handler._langfuse_client.flush()

    for span_name, message in [
        ("human_approval", "tool interrupt"),
        ("knowledge_base", "retriever bubble-up"),
        ("TestLLM", "llm bubble-up"),
        ("LangGraph", "graph interrupt"),
    ]:
        span = get_span(span_name)
        assert (
            span.attributes[LangfuseOtelSpanAttributes.OBSERVATION_LEVEL]
            == "DEFAULT"
        )
        assert (
            span.attributes[LangfuseOtelSpanAttributes.OBSERVATION_STATUS_MESSAGE]
            == message
        )


def test_control_flow_resume_uses_thread_keyed_explicit_resume_context(
    memory_exporter, langfuse_memory_client, monkeypatch
):
    class DummyControlFlowError(RuntimeError):
        pass

    Command = pytest.importorskip("langgraph.types").Command

    context_token = otel_context.attach(otel_context.Context())
    monkeypatch.setattr(
        callback_handler_module,
        "CONTROL_FLOW_EXCEPTION_TYPES",
        {DummyControlFlowError},
    )

    try:
        handler = CallbackHandler()

        thread_one_interrupt_run_id = uuid4()
        thread_two_interrupt_run_id = uuid4()
        thread_one_fresh_run_id = uuid4()
        thread_two_resume_run_id = uuid4()
        thread_one_resume_run_id = uuid4()

        handler.on_chain_start(
            {"name": "LangGraph"},
            {"messages": ["need approval"]},
            run_id=thread_one_interrupt_run_id,
            metadata={"thread_id": "thread-1"},
        )
        handler.on_chain_error(
            DummyControlFlowError("graph interrupt 1"),
            run_id=thread_one_interrupt_run_id,
        )

        handler.on_chain_start(
            {"name": "LangGraph"},
            {"messages": ["need approval"]},
            run_id=thread_two_interrupt_run_id,
            metadata={"thread_id": "thread-2"},
        )
        handler.on_chain_error(
            DummyControlFlowError("graph interrupt 2"),
            run_id=thread_two_interrupt_run_id,
        )

        handler.on_chain_start(
            {"name": "LangGraph"},
            {"messages": ["fresh invocation"]},
            run_id=thread_one_fresh_run_id,
            metadata={"thread_id": "thread-1"},
        )
        handler.on_chain_end(
            {"messages": ["completed"]},
            run_id=thread_one_fresh_run_id,
        )

        handler.on_chain_start(
            {"name": "LangGraph"},
            Command(resume={"approved": True}),
            run_id=thread_two_resume_run_id,
            metadata={"thread_id": "thread-2"},
        )
        handler.on_chain_end(
            {"messages": ["approved"]},
            run_id=thread_two_resume_run_id,
        )

        handler.on_chain_start(
            {"name": "LangGraph"},
            Command(resume={"approved": True}),
            run_id=thread_one_resume_run_id,
            metadata={"thread_id": "thread-1"},
        )
        handler.on_chain_end(
            {"messages": ["approved"]},
            run_id=thread_one_resume_run_id,
        )

        handler._langfuse_client.flush()

        root_spans = [
            span
            for span in memory_exporter.get_finished_spans()
            if span.name == "LangGraph"
        ]

        assert len(root_spans) == 5
        spans_by_trace_id = {}
        for span in root_spans:
            spans_by_trace_id.setdefault(span.context.trace_id, []).append(span)

        assert sorted(len(spans) for spans in spans_by_trace_id.values()) == [1, 2, 2]

        resumed_trace_spans = [
            spans for spans in spans_by_trace_id.values() if len(spans) == 2
        ]
        assert len(resumed_trace_spans) == 2

        for spans in resumed_trace_spans:
            initial_span = next(span for span in spans if span.parent is None)
            resumed_span = next(span for span in spans if span.parent is not None)
            assert resumed_span.parent.span_id == initial_span.context.span_id

        fresh_trace_spans = next(
            spans for spans in spans_by_trace_id.values() if len(spans) == 1
        )
        assert fresh_trace_spans[0].parent is None
    finally:
        otel_context.detach(context_token)
