from datetime import datetime
from uuid import uuid4


from langfuse import Langfuse
from langfuse.model import CreateEvent, CreateGeneration, CreateSpan, CreateTrace, InitialGeneration, InitialScore, InitialSpan, Usage

from langfuse.task_manager import TaskStatus
from tests.api_wrapper import LangfuseAPI

host = "http://localhost:3000/"


def create_uuid():
    return str(uuid4())


def test_flush():
    # set up the consumer with more requests than a single batch will allow
    langfuse = Langfuse("pk-lf-1234567890", "sk-lf-1234567890", host)

    for i in range(2):
        langfuse.trace(
            CreateTrace(
                name=str(i),
            )
        )

    langfuse.flush()
    # Make sure that the client queue is empty after flushing
    assert langfuse.task_manager.queue.empty()


def test_shutdown():
    langfuse = Langfuse("pk-lf-1234567890", "sk-lf-1234567890", host)

    for i in range(2):
        langfuse.trace(
            CreateTrace(
                name=str(i),
            )
        )

    langfuse.shutdown()
    # we expect two things after shutdown:
    # 1. client queue is empty
    # 2. consumer thread has stopped
    assert langfuse.task_manager.queue.empty()
    assert not langfuse.task_manager.consumer_thread.is_alive()


def test_create_score():
    langfuse = Langfuse("pk-lf-1234567890", "sk-lf-1234567890", host)
    trace = langfuse.trace(
        CreateTrace(
            name="this-is-so-great-new",
            user_id="test",
            metadata="test",
        )
    )
    langfuse.flush()
    assert langfuse.task_manager.queue.qsize() == 0
    assert all(v.status == TaskStatus.SUCCESS for v in langfuse.task_manager.result_mapping.values()), "Not all tasks succeeded"

    trace = langfuse.score(
        InitialScore(
            traceId=trace.id,
            name="this-is-so-great-new",
            value=1,
            user_id="test",
            metadata="test",
        )
    )

    trace.generation(CreateGeneration(name="yet another child", metadata="test"))

    langfuse.flush()
    assert langfuse.task_manager.queue.qsize() == 0
    assert all(v.status == TaskStatus.SUCCESS for v in langfuse.task_manager.result_mapping.values()), "Not all tasks succeeded"


def test_create_trace():
    langfuse = Langfuse("pk-lf-1234567890", "sk-lf-1234567890", host)
    api_wrapper = LangfuseAPI("pk-lf-1234567890", "sk-lf-1234567890", host)
    trace_name = create_uuid()

    trace = langfuse.trace(
        CreateTrace(
            name=trace_name,
            user_id="test",
            metadata={"key": "value"},
        )
    )

    langfuse.flush()

    trace = api_wrapper.get_trace(trace.id)
    print(trace)
    assert trace["name"] == trace_name
    assert trace["userId"] == "test"
    assert trace["metadata"] == {"key": "value"}
    assert True if not trace["externalId"] else False


def test_create_generation():
    langfuse = Langfuse("pk-lf-1234567890", "sk-lf-1234567890", host)
    api_wrapper = LangfuseAPI("pk-lf-1234567890", "sk-lf-1234567890", host)

    timestamp = datetime.now()
    langfuse.generation(
        InitialGeneration(
            name="query-generation",
            startTime=timestamp,
            endTime=timestamp,
            model="gpt-3.5-turbo",
            modelParameters={"maxTokens": "1000", "temperature": "0.9"},
            prompt=[
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": "Please generate the start of a company documentation that contains the answer to the questinon: Write a summary of the Q3 OKR goals",
                },
            ],
            completion="This document entails the OKR goals for ACME",
            usage=Usage(promptTokens=50, completionTokens=49),
            metadata={"interface": "whatsapp"},
        )
    )

    langfuse.flush()

    assert len(langfuse.task_manager.result_mapping) == 2

    trace_id = langfuse.get_trace_id()

    trace = api_wrapper.get_trace(trace_id)
    print(trace)

    assert trace["name"] == "query-generation"
    assert trace["userId"] is None
    assert trace["metadata"] is None
    assert trace["externalId"] is None

    assert len(trace["observations"]) == 1

    generation = trace["observations"][0]

    assert generation["name"] == "query-generation"
    assert generation["startTime"] is not None
    assert generation["startTime"] is not None
    assert generation["endTime"] is not None
    assert generation["model"] == "gpt-3.5-turbo"
    assert generation["modelParameters"] == {"maxTokens": "1000", "temperature": "0.9"}
    assert generation["input"] == [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": "Please generate the start of a company documentation that contains the answer to the questinon: Write a summary of the Q3 OKR goals",
        },
    ]
    assert generation["output"] == {"completion": "This document entails the OKR goals for ACME"}


def test_create_span():
    langfuse = Langfuse("pk-lf-1234567890", "sk-lf-1234567890", host)
    api_wrapper = LangfuseAPI("pk-lf-1234567890", "sk-lf-1234567890", host)

    timestamp = datetime.now()
    langfuse.span(
        InitialSpan(
            name="span",
            startTime=timestamp,
            endTime=timestamp,
            input={"key": "value"},
            output={"key": "value"},
            metadata={"interface": "whatsapp"},
        )
    )

    langfuse.flush()

    assert len(langfuse.task_manager.result_mapping) == 2

    trace_id = langfuse.get_trace_id()

    trace = api_wrapper.get_trace(trace_id)

    assert trace["name"] == "span"
    assert trace["userId"] is None
    assert trace["metadata"] is None
    assert trace["externalId"] is None

    assert len(trace["observations"]) == 1

    generation = trace["observations"][0]

    assert generation["name"] == "span"
    assert generation["startTime"] is not None
    assert generation["startTime"] is not None
    assert generation["endTime"] is not None
    assert generation["input"] == {"key": "value"}
    assert generation["output"] == {"key": "value"}


def test_score_trace():
    langfuse = Langfuse("pk-lf-1234567890", "sk-lf-1234567890", host)
    api_wrapper = LangfuseAPI("pk-lf-1234567890", "sk-lf-1234567890", host)

    trace_name = create_uuid()

    trace = langfuse.trace(CreateTrace(name=trace_name))

    langfuse.score(
        InitialScore(
            traceId=langfuse.get_trace_id(),
            name="valuation",
            value=1,
            comment="This is a comment",
        )
    )

    langfuse.flush()

    assert len(langfuse.task_manager.result_mapping) == 2

    trace_id = langfuse.get_trace_id()

    trace = api_wrapper.get_trace(trace_id)

    assert trace["name"] == trace_name

    assert len(trace["scores"]) == 1

    score = trace["scores"][0]

    assert score["name"] == "valuation"
    assert score["value"] == 1
    assert score["comment"] == "This is a comment"
    assert score["observationId"] is None


def test_score_span():
    langfuse = Langfuse("pk-lf-1234567890", "sk-lf-1234567890", host)
    api_wrapper = LangfuseAPI("pk-lf-1234567890", "sk-lf-1234567890", host)

    spanId = create_uuid()
    timestamp = datetime.now()
    langfuse.span(
        InitialSpan(
            id=spanId,
            name="span",
            startTime=timestamp,
            endTime=timestamp,
            input={"key": "value"},
            output={"key": "value"},
            metadata={"interface": "whatsapp"},
        )
    )

    langfuse.score(
        InitialScore(
            traceId=langfuse.get_trace_id(),
            observationId=spanId,
            name="valuation",
            value=1,
            comment="This is a comment",
        )
    )

    langfuse.flush()

    assert len(langfuse.task_manager.result_mapping) == 3

    trace_id = langfuse.get_trace_id()

    trace = api_wrapper.get_trace(trace_id)

    assert len(trace["scores"]) == 1
    assert len(trace["observations"]) == 1

    score = trace["scores"][0]

    assert score["name"] == "valuation"
    assert score["value"] == 1
    assert score["comment"] == "This is a comment"
    assert score["observationId"] == spanId


def test_create_trace_and_span():
    langfuse = Langfuse("pk-lf-1234567890", "sk-lf-1234567890", host)
    api_wrapper = LangfuseAPI("pk-lf-1234567890", "sk-lf-1234567890", host)

    trace_name = create_uuid()
    spanId = create_uuid()

    trace = langfuse.trace(CreateTrace(name=trace_name))
    trace.span(CreateSpan(id=spanId, name="span"))

    langfuse.flush()

    trace = api_wrapper.get_trace(trace.id)

    assert trace["name"] == trace_name
    assert len(trace["observations"]) == 1

    span = trace["observations"][0]
    assert span["name"] == "span"
    assert span["traceId"] == trace["id"]


def test_create_trace_and_generation():
    langfuse = Langfuse("pk-lf-1234567890", "sk-lf-1234567890", host)
    api_wrapper = LangfuseAPI("pk-lf-1234567890", "sk-lf-1234567890", host)

    trace_name = create_uuid()
    generationId = create_uuid()

    trace = langfuse.trace(CreateTrace(name=trace_name))
    trace.generation(CreateGeneration(id=generationId, name="generation"))

    langfuse.flush()

    trace = api_wrapper.get_trace(trace.id)

    assert trace["name"] == trace_name
    assert len(trace["observations"]) == 1

    span = trace["observations"][0]
    assert span["name"] == "generation"
    assert span["traceId"] == trace["id"]


def test_create_trace_and_event():
    langfuse = Langfuse("pk-lf-1234567890", "sk-lf-1234567890", host)
    api_wrapper = LangfuseAPI("pk-lf-1234567890", "sk-lf-1234567890", host)

    trace_name = create_uuid()
    eventId = create_uuid()

    trace = langfuse.trace(CreateTrace(name=trace_name))
    trace.event(CreateEvent(id=eventId, name="event"))

    langfuse.flush()

    trace = api_wrapper.get_trace(trace.id)

    assert trace["name"] == trace_name
    assert len(trace["observations"]) == 1

    span = trace["observations"][0]
    assert span["name"] == "event"
    assert span["traceId"] == trace["id"]
