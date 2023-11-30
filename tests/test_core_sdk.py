from asyncio import gather
from datetime import datetime
import logging

import pytest

from langfuse import Langfuse
from langfuse.model import (
    CreateEvent,
    CreateGeneration,
    CreateSpan,
    CreateTrace,
    InitialGeneration,
    InitialScore,
    InitialSpan,
    UpdateGeneration,
    UpdateSpan,
    Usage,
)

from tests.api_wrapper import LangfuseAPI
from tests.utils import create_uuid, get_api


@pytest.mark.asyncio
async def test_concurrency():
    start = datetime.now()

    async def update_generation(i, langfuse: Langfuse):
        trace = langfuse.trace(CreateTrace(name=str(i)))
        generation = trace.generation(InitialGeneration(name=str(i)))
        generation.update(UpdateGeneration(metadata={"count": str(i)}))

    langfuse = Langfuse(debug=True, threads=5)

    await gather(*(update_generation(i, langfuse) for i in range(100)))

    langfuse.flush()
    diff = datetime.now() - start
    print(diff)

    api = get_api()
    for i in range(100):
        observation = api.observations.get_many(name=str(i)).data[0]
        assert observation.name == str(i)
        assert observation.metadata == {"count": str(i)}


def test_flush():
    # set up the consumer with more requests than a single batch will allow
    langfuse = Langfuse(debug=True)

    for i in range(2):
        langfuse.trace(
            CreateTrace(
                name=str(i),
            )
        )

    langfuse.flush()
    # Make sure that the client queue is empty after flushing
    assert langfuse.task_manager._queue.empty()


def test_shutdown():
    langfuse = Langfuse(debug=False)

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
    assert langfuse.task_manager._queue.empty()


def test_create_score():
    langfuse = Langfuse(debug=True)
    api_wrapper = LangfuseAPI()

    trace = langfuse.trace(
        CreateTrace(
            name="this-is-so-great-new",
            user_id="test",
            metadata="test",
        )
    )
    logging.info("FLUSH")
    langfuse.flush()
    assert langfuse.task_manager._queue.qsize() == 0

    score_id = create_uuid()

    langfuse.score(
        InitialScore(
            id=score_id,
            traceId=trace.id,
            name="this-is-a-score",
            value=1,
            user_id="test",
            metadata="test",
        )
    )

    trace.generation(CreateGeneration(name="yet another child", metadata="test"))

    langfuse.flush()

    assert langfuse.task_manager._queue.qsize() == 0

    trace = api_wrapper.get_trace(trace.id)

    assert trace["scores"][0]["id"] == score_id


def test_create_trace():
    langfuse = Langfuse(debug=False)
    api_wrapper = LangfuseAPI()
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

    assert trace["name"] == trace_name
    assert trace["userId"] == "test"
    assert trace["metadata"] == {"key": "value"}
    assert True if not trace["externalId"] else False


def test_create_generation():
    langfuse = Langfuse(debug=True)
    api_wrapper = LangfuseAPI()

    timestamp = datetime.now()
    generation_id = create_uuid()
    langfuse.generation(
        InitialGeneration(
            id=generation_id,
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

    trace_id = langfuse.get_trace_id()

    trace = api_wrapper.get_trace(trace_id)

    assert trace["name"] == "query-generation"
    assert trace["userId"] is None
    assert trace["metadata"] is None
    assert trace["externalId"] is None

    assert len(trace["observations"]) == 1

    generation = trace["observations"][0]

    assert generation["id"] == generation_id
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
    assert generation["output"] == "This document entails the OKR goals for ACME"


def test_create_generation_complex():
    langfuse = Langfuse(debug=False)
    api_wrapper = LangfuseAPI()

    generation_id = create_uuid()
    langfuse.generation(
        InitialGeneration(
            id=generation_id,
            name="query-generation",
            prompt=[
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": "Please generate the start of a company documentation that contains the answer to the questinon: Write a summary of the Q3 OKR goals",
                },
            ],
            completion=[{"foo": "bar"}],
            usage=Usage(promptTokens=50, completionTokens=49),
            metadata=[{"tags": ["yo"]}],
        )
    )

    langfuse.flush()

    trace_id = langfuse.get_trace_id()

    trace = api_wrapper.get_trace(trace_id)

    assert trace["name"] == "query-generation"
    assert trace["userId"] is None
    assert trace["metadata"] is None
    assert trace["externalId"] is None

    assert len(trace["observations"]) == 1

    generation = trace["observations"][0]

    assert generation["id"] == generation_id
    assert generation["name"] == "query-generation"
    assert generation["input"] == [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": "Please generate the start of a company documentation that contains the answer to the questinon: Write a summary of the Q3 OKR goals",
        },
    ]
    assert generation["output"] == [{"foo": "bar"}]
    assert generation["metadata"] == [{"tags": ["yo"]}]


def test_create_span():
    langfuse = Langfuse(debug=True)
    api_wrapper = LangfuseAPI()

    timestamp = datetime.now()
    span_id = create_uuid()
    langfuse.span(
        InitialSpan(
            id=span_id,
            name="span",
            startTime=timestamp,
            endTime=timestamp,
            input={"key": "value"},
            output={"key": "value"},
            metadata={"interface": "whatsapp"},
        )
    )

    langfuse.flush()

    trace_id = langfuse.get_trace_id()

    trace = api_wrapper.get_trace(trace_id)

    assert trace["name"] == "span"
    assert trace["userId"] is None
    assert trace["metadata"] is None
    assert trace["externalId"] is None

    assert len(trace["observations"]) == 1

    span = trace["observations"][0]

    assert span["id"] == span_id
    assert span["name"] == "span"
    assert span["startTime"] is not None
    assert span["startTime"] is not None
    assert span["endTime"] is not None
    assert span["input"] == {"key": "value"}
    assert span["output"] == {"key": "value"}


def test_score_trace():
    langfuse = Langfuse(debug=False)
    api_wrapper = LangfuseAPI()

    trace_name = create_uuid()

    trace = langfuse.trace(CreateTrace(name=trace_name))

    langfuse.score(
        InitialScore(
            traceId=langfuse.get_trace_id(),
            name="valuation",
            value=0.5,
            comment="This is a comment",
        )
    )

    langfuse.flush()

    trace_id = langfuse.get_trace_id()

    trace = api_wrapper.get_trace(trace_id)

    assert trace["name"] == trace_name

    assert len(trace["scores"]) == 1

    score = trace["scores"][0]

    assert score["name"] == "valuation"
    assert score["value"] == 0.5
    assert score["comment"] == "This is a comment"
    assert score["observationId"] is None


def test_score_span():
    langfuse = Langfuse(debug=False)
    api_wrapper = LangfuseAPI()

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
    langfuse = Langfuse(debug=False)
    api_wrapper = LangfuseAPI()

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
    langfuse = Langfuse(debug=False)
    api_wrapper = LangfuseAPI()

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


def test_create_generation_and_trace():
    langfuse = Langfuse(debug=False)
    api_wrapper = LangfuseAPI()

    trace_name = create_uuid()
    trace_id = create_uuid()

    langfuse.generation(CreateGeneration(traceId=trace_id, name="generation"))
    langfuse.trace(CreateTrace(id=trace_id, name=trace_name))

    langfuse.flush()

    trace = api_wrapper.get_trace(trace_id)

    assert trace["name"] == trace_name
    assert len(trace["observations"]) == 1

    span = trace["observations"][0]
    assert span["name"] == "generation"
    assert span["traceId"] == trace["id"]


def test_update_generation():
    langfuse = Langfuse(debug=True)
    api = get_api()

    generation = langfuse.generation(InitialGeneration(name="generation"))
    generation.update(UpdateGeneration(metadata={"dict": "value"}))

    langfuse.flush()

    trace = api.trace.get(generation.trace_id)

    assert trace.name == "generation"
    assert len(trace.observations) == 1
    retrieved_generation = trace.observations[0]
    assert retrieved_generation.name == "generation"
    assert retrieved_generation.trace_id == generation.trace_id
    assert retrieved_generation.metadata == {"dict": "value"}


def test_update_span():
    langfuse = Langfuse(debug=False)
    api = get_api()

    span = langfuse.span(InitialSpan(name="span"))
    span.update(UpdateSpan(metadata={"dict": "value"}))

    langfuse.flush()

    trace = api.trace.get(span.trace_id)

    assert trace.name == "span"
    assert len(trace.observations) == 1

    retrieved_span = trace.observations[0]
    assert retrieved_span.name == "span"
    assert retrieved_span.trace_id == span.trace_id
    assert retrieved_span.metadata == {"dict": "value"}


def test_create_trace_and_event():
    langfuse = Langfuse(debug=True)
    api_wrapper = LangfuseAPI()

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


def test_create_span_and_generation():
    api = get_api()

    langfuse = Langfuse(debug=False)

    span = langfuse.span(InitialSpan(name="span"))
    langfuse.generation(InitialGeneration(traceId=span.trace_id, name="generation"))

    langfuse.flush()

    trace = api.trace.get(span.trace_id)

    assert trace.name == "span"
    assert len(trace.observations) == 2

    span = trace.observations[0]
    assert span.trace_id == trace.id

    span = trace.observations[1]
    assert span.trace_id == trace.id


def test_create_trace_with_id_and_generation():
    langfuse = Langfuse(debug=False)
    api_wrapper = LangfuseAPI()

    trace_name = create_uuid()
    trace_id = create_uuid()

    trace = langfuse.trace(CreateTrace(id=trace_id, name=trace_name))
    trace.generation(CreateGeneration(name="generation"))

    langfuse.flush()

    trace = api_wrapper.get_trace(trace_id)

    assert trace["name"] == trace_name
    assert trace["id"] == trace_id
    assert len(trace["observations"]) == 1

    span = trace["observations"][0]
    assert span["name"] == "generation"
    assert span["traceId"] == trace["id"]


def test_end_generation():
    langfuse = Langfuse()
    api_wrapper = LangfuseAPI()

    timestamp = datetime.now()
    generation = langfuse.generation(
        InitialGeneration(
            name="query-generation",
            startTime=timestamp,
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

    generation.end()

    langfuse.flush()

    trace_id = langfuse.get_trace_id()

    trace = api_wrapper.get_trace(trace_id)

    span = trace["observations"][0]
    assert span["endTime"] is not None


def test_end_span():
    langfuse = Langfuse()
    api_wrapper = LangfuseAPI()

    timestamp = datetime.now()
    span = langfuse.span(
        InitialSpan(
            name="span",
            startTime=timestamp,
            input={"key": "value"},
            output={"key": "value"},
            metadata={"interface": "whatsapp"},
        )
    )

    span.end()

    langfuse.flush()

    trace_id = langfuse.get_trace_id()

    trace = api_wrapper.get_trace(trace_id)

    span = trace["observations"][0]
    assert span["endTime"] is not None


def test_get_generations():
    langfuse = Langfuse(debug=False)

    timestamp = datetime.now()

    langfuse.generation(
        InitialGeneration(
            name=create_uuid(),
            startTime=timestamp,
            endTime=timestamp,
        )
    )

    generation_name = create_uuid()

    langfuse.generation(
        InitialGeneration(
            name=generation_name,
            startTime=timestamp,
            endTime=timestamp,
            prompt="great-prompt",
            completion="great-completion",
        )
    )

    langfuse.flush()
    generations = langfuse.get_generations(name=generation_name, limit=10, page=1)
    assert len(generations.data) == 1
    assert generations.data[0].name == generation_name
    assert generations.data[0].input == "great-prompt"
    assert generations.data[0].output == "great-completion"


def test_sth():
    from langfuse import Langfuse

    from langfuse.model import CreateSpan, UpdateSpan
    from datetime import datetime

    langfuse = Langfuse(debug=True)
    start_time = datetime.now()
    span = CreateSpan(name="test")
    span = langfuse.span(span)
    span.update(UpdateSpan(start_time=start_time, end_time=datetime.now()))


def test_get_generations_by_user():
    langfuse = Langfuse(debug=False)

    timestamp = datetime.now()

    user_id = create_uuid()
    generation_name = create_uuid()
    trace = langfuse.trace(CreateTrace(name="test-user", userId=user_id))

    trace.generation(
        CreateGeneration(
            name=generation_name,
            startTime=timestamp,
            endTime=timestamp,
            prompt="great-prompt",
            completion="great-completion",
        )
    )

    langfuse.generation(
        InitialGeneration(
            startTime=timestamp,
            endTime=timestamp,
        )
    )

    langfuse.flush()
    generations = langfuse.get_generations(limit=10, page=1, user_id=user_id)

    assert len(generations.data) == 1
    assert generations.data[0].name == generation_name
    assert generations.data[0].input == "great-prompt"
    assert generations.data[0].output == "great-completion"
