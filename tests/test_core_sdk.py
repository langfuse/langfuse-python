import os
import time
from asyncio import gather
from datetime import datetime, timedelta, timezone
from time import sleep

import pytest

from langfuse import Langfuse
from langfuse.client import (
    FetchObservationResponse,
    FetchObservationsResponse,
    FetchSessionsResponse,
    FetchTraceResponse,
    FetchTracesResponse,
)
from langfuse.utils import _get_timestamp
from tests.api_wrapper import LangfuseAPI
from tests.utils import (
    CompletionUsage,
    LlmUsage,
    LlmUsageWithCost,
    create_uuid,
    get_api,
)


@pytest.mark.asyncio
async def test_concurrency():
    start = _get_timestamp()

    async def update_generation(i, langfuse: Langfuse):
        trace = langfuse.trace(name=str(i))
        generation = trace.generation(name=str(i))
        generation.update(metadata={"count": str(i)})

    langfuse = Langfuse(debug=False, threads=5)
    print("start")
    await gather(*(update_generation(i, langfuse) for i in range(100)))
    print("flush")
    langfuse.flush()
    diff = _get_timestamp() - start
    print(diff)

    api = get_api()
    for i in range(100):
        observation = api.observations.get_many(name=str(i)).data[0]
        assert observation.name == str(i)
        assert observation.metadata == {"count": i}


def test_flush():
    # set up the consumer with more requests than a single batch will allow
    langfuse = Langfuse(debug=False)

    for i in range(2):
        langfuse.trace(
            name=str(i),
        )

    langfuse.flush()
    # Make sure that the client queue is empty after flushing
    assert langfuse.task_manager._ingestion_queue.empty()


def test_shutdown():
    langfuse = Langfuse(debug=False)

    for i in range(2):
        langfuse.trace(
            name=str(i),
        )

    langfuse.shutdown()
    # we expect two things after shutdown:
    # 1. client queue is empty
    # 2. consumer thread has stopped
    assert langfuse.task_manager._ingestion_queue.empty()


def test_invalid_score_data_does_not_raise_exception():
    langfuse = Langfuse(debug=False)

    trace = langfuse.trace(
        name="this-is-so-great-new",
        user_id="test",
        metadata="test",
    )

    langfuse.flush()
    assert langfuse.task_manager._ingestion_queue.qsize() == 0

    score_id = create_uuid()

    langfuse.score(
        id=score_id,
        trace_id=trace.id,
        name="this-is-a-score",
        value=-1,
        data_type="BOOLEAN",
    )

    langfuse.flush()
    assert langfuse.task_manager._ingestion_queue.qsize() == 0


def test_create_numeric_score():
    langfuse = Langfuse(debug=False)
    api_wrapper = LangfuseAPI()

    trace = langfuse.trace(
        name="this-is-so-great-new",
        user_id="test",
        metadata="test",
    )

    langfuse.flush()
    assert langfuse.task_manager._ingestion_queue.qsize() == 0

    score_id = create_uuid()

    langfuse.score(
        id=score_id,
        trace_id=trace.id,
        name="this-is-a-score",
        value=1,
    )

    trace.generation(name="yet another child", metadata="test")

    langfuse.flush()

    assert langfuse.task_manager._ingestion_queue.qsize() == 0

    trace = api_wrapper.get_trace(trace.id)

    assert trace["scores"][0]["id"] == score_id
    assert trace["scores"][0]["value"] == 1
    assert trace["scores"][0]["dataType"] == "NUMERIC"
    assert trace["scores"][0]["stringValue"] is None


def test_create_boolean_score():
    langfuse = Langfuse(debug=False)
    api_wrapper = LangfuseAPI()

    trace = langfuse.trace(
        name="this-is-so-great-new",
        user_id="test",
        metadata="test",
    )

    langfuse.flush()
    assert langfuse.task_manager._ingestion_queue.qsize() == 0

    score_id = create_uuid()

    langfuse.score(
        id=score_id,
        trace_id=trace.id,
        name="this-is-a-score",
        value=1,
        data_type="BOOLEAN",
    )

    trace.generation(name="yet another child", metadata="test")

    langfuse.flush()

    assert langfuse.task_manager._ingestion_queue.qsize() == 0

    trace = api_wrapper.get_trace(trace.id)

    assert trace["scores"][0]["id"] == score_id
    assert trace["scores"][0]["dataType"] == "BOOLEAN"
    assert trace["scores"][0]["value"] == 1
    assert trace["scores"][0]["stringValue"] == "True"


def test_create_categorical_score():
    langfuse = Langfuse(debug=False)
    api_wrapper = LangfuseAPI()

    trace = langfuse.trace(
        name="this-is-so-great-new",
        user_id="test",
        metadata="test",
    )

    langfuse.flush()
    assert langfuse.task_manager._ingestion_queue.qsize() == 0

    score_id = create_uuid()

    langfuse.score(
        id=score_id,
        trace_id=trace.id,
        name="this-is-a-score",
        value="high score",
    )

    trace.generation(name="yet another child", metadata="test")

    langfuse.flush()

    assert langfuse.task_manager._ingestion_queue.qsize() == 0

    trace = api_wrapper.get_trace(trace.id)

    assert trace["scores"][0]["id"] == score_id
    assert trace["scores"][0]["dataType"] == "CATEGORICAL"
    assert trace["scores"][0]["value"] == 0
    assert trace["scores"][0]["stringValue"] == "high score"


def test_create_trace():
    langfuse = Langfuse(debug=False)
    trace_name = create_uuid()

    trace = langfuse.trace(
        name=trace_name,
        user_id="test",
        metadata={"key": "value"},
        tags=["tag1", "tag2"],
        public=True,
    )

    langfuse.flush()
    sleep(2)

    trace = LangfuseAPI().get_trace(trace.id)

    assert trace["name"] == trace_name
    assert trace["userId"] == "test"
    assert trace["metadata"] == {"key": "value"}
    assert trace["tags"] == ["tag1", "tag2"]
    assert trace["public"] is True
    assert True if not trace["externalId"] else False


def test_create_update_trace():
    langfuse = Langfuse(debug=True, flush_at=1)

    trace_name = create_uuid()

    trace = langfuse.trace(
        name=trace_name,
        user_id="test",
        metadata={"key": "value"},
        public=True,
    )
    trace.update(metadata={"key2": "value2"}, public=False)

    langfuse.flush()

    trace = get_api().trace.get(trace.id)

    assert trace.name == trace_name
    assert trace.user_id == "test"
    assert trace.metadata == {"key": "value", "key2": "value2"}
    assert trace.public is False


def test_create_generation():
    langfuse = Langfuse(debug=True)

    timestamp = _get_timestamp()
    generation_id = create_uuid()
    langfuse.generation(
        id=generation_id,
        name="query-generation",
        start_time=timestamp,
        end_time=timestamp,
        model="gpt-3.5-turbo-0125",
        model_parameters={
            "max_tokens": "1000",
            "temperature": "0.9",
            "stop": ["user-1", "user-2"],
        },
        input=[
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": "Please generate the start of a company documentation that contains the answer to the questinon: Write a summary of the Q3 OKR goals",
            },
        ],
        output="This document entails the OKR goals for ACME",
        usage=LlmUsage(promptTokens=50, completionTokens=49),
        metadata={"interface": "whatsapp"},
        level="DEBUG",
    )

    langfuse.flush()

    trace_id = langfuse.get_trace_id()

    trace = get_api().trace.get(trace_id)

    assert trace.name == "query-generation"
    assert trace.user_id is None
    assert trace.metadata == {}

    assert len(trace.observations) == 1

    generation = trace.observations[0]

    assert generation.id == generation_id
    assert generation.name == "query-generation"
    assert generation.start_time is not None
    assert generation.end_time is not None
    assert generation.model == "gpt-3.5-turbo-0125"
    assert generation.model_parameters == {
        "max_tokens": "1000",
        "temperature": "0.9",
        "stop": ["user-1", "user-2"],
    }
    assert generation.input == [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": "Please generate the start of a company documentation that contains the answer to the questinon: Write a summary of the Q3 OKR goals",
        },
    ]
    assert generation.output == "This document entails the OKR goals for ACME"
    assert generation.level == "DEBUG"


@pytest.mark.parametrize(
    "usage, expected_usage, expected_input_cost, expected_output_cost, expected_total_cost",
    [
        (
            CompletionUsage(prompt_tokens=51, completion_tokens=0, total_tokens=100),
            "TOKENS",
            None,
            None,
            None,
        ),
        (
            LlmUsage(promptTokens=51, completionTokens=0, totalTokens=100),
            "TOKENS",
            None,
            None,
            None,
        ),
        (
            {
                "input": 51,
                "output": 0,
                "total": 100,
                "unit": "TOKENS",
                "input_cost": 100,
                "output_cost": 200,
                "total_cost": 300,
            },
            "TOKENS",
            100,
            200,
            300,
        ),
        (
            {
                "input": 51,
                "output": 0,
                "total": 100,
                "unit": "CHARACTERS",
                "input_cost": 100,
                "output_cost": 200,
                "total_cost": 300,
            },
            "CHARACTERS",
            100,
            200,
            300,
        ),
        (
            LlmUsageWithCost(
                promptTokens=51,
                completionTokens=0,
                totalTokens=100,
                inputCost=100,
                outputCost=200,
                totalCost=300,
            ),
            "TOKENS",
            100,
            200,
            300,
        ),
    ],
)
def test_create_generation_complex(
    usage,
    expected_usage,
    expected_input_cost,
    expected_output_cost,
    expected_total_cost,
):
    langfuse = Langfuse(debug=False)

    generation_id = create_uuid()
    langfuse.generation(
        id=generation_id,
        name="query-generation",
        input=[
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": "Please generate the start of a company documentation that contains the answer to the questinon: Write a summary of the Q3 OKR goals",
            },
        ],
        output=[{"foo": "bar"}],
        usage=usage,
        metadata=[{"tags": ["yo"]}],
    )

    langfuse.flush()

    trace_id = langfuse.get_trace_id()

    trace = get_api().trace.get(trace_id)

    assert trace.name == "query-generation"
    assert trace.user_id is None
    assert trace.metadata == {}

    assert len(trace.observations) == 1

    generation = trace.observations[0]

    assert generation.id == generation_id
    assert generation.name == "query-generation"
    assert generation.input == [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": "Please generate the start of a company documentation that contains the answer to the questinon: Write a summary of the Q3 OKR goals",
        },
    ]
    assert generation.output == [{"foo": "bar"}]
    assert generation.metadata["metadata"] == [{"tags": ["yo"]}]
    assert generation.start_time is not None
    assert generation.usage_details == {"input": 51, "output": 0, "total": 100}
    assert generation.cost_details == (
        {
            "input": expected_input_cost,
            "output": expected_output_cost,
            "total": expected_total_cost,
        }
        if any([expected_input_cost, expected_output_cost, expected_total_cost])
        else {}
    )


def test_create_span():
    langfuse = Langfuse(debug=False)

    timestamp = _get_timestamp()
    span_id = create_uuid()
    langfuse.span(
        id=span_id,
        name="span",
        start_time=timestamp,
        end_time=timestamp,
        input={"key": "value"},
        output={"key": "value"},
        metadata={"interface": "whatsapp"},
    )

    langfuse.flush()

    trace_id = langfuse.get_trace_id()

    trace = get_api().trace.get(trace_id)

    assert trace.name == "span"
    assert trace.user_id is None
    assert trace.metadata == {}

    assert len(trace.observations) == 1

    span = trace.observations[0]

    assert span.id == span_id
    assert span.name == "span"
    assert span.start_time is not None
    assert span.end_time is not None
    assert span.input == {"key": "value"}
    assert span.output == {"key": "value"}
    assert span.start_time is not None


def test_score_trace():
    langfuse = Langfuse(debug=False)
    api_wrapper = LangfuseAPI()

    trace_name = create_uuid()

    trace = langfuse.trace(name=trace_name)

    langfuse.score(
        trace_id=langfuse.get_trace_id(),
        name="valuation",
        value=0.5,
        comment="This is a comment",
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
    assert score["dataType"] == "NUMERIC"


def test_score_trace_nested_trace():
    langfuse = Langfuse(debug=False)

    trace_name = create_uuid()

    trace = langfuse.trace(name=trace_name)

    trace.score(
        name="valuation",
        value=0.5,
        comment="This is a comment",
    )

    langfuse.flush()

    trace_id = langfuse.get_trace_id()

    trace = get_api().trace.get(trace_id)

    assert trace.name == trace_name

    assert len(trace.scores) == 1

    score = trace.scores[0]

    assert score.name == "valuation"
    assert score.value == 0.5
    assert score.comment == "This is a comment"
    assert score.observation_id is None
    assert score.data_type == "NUMERIC"


def test_score_trace_nested_observation():
    langfuse = Langfuse(debug=False)

    trace_name = create_uuid()

    trace = langfuse.trace(name=trace_name)
    span = trace.span(name="span")

    span.score(
        name="valuation",
        value=0.5,
        comment="This is a comment",
    )

    langfuse.flush()

    trace_id = langfuse.get_trace_id()

    trace = get_api().trace.get(trace_id)

    assert trace.name == trace_name

    assert len(trace.scores) == 1

    score = trace.scores[0]

    assert score.name == "valuation"
    assert score.value == 0.5
    assert score.comment == "This is a comment"
    assert score.observation_id == span.id
    assert score.data_type == "NUMERIC"


def test_score_span():
    langfuse = Langfuse(debug=False)
    api_wrapper = LangfuseAPI()

    spanId = create_uuid()
    timestamp = _get_timestamp()
    langfuse.span(
        id=spanId,
        name="span",
        start_time=timestamp,
        end_time=timestamp,
        input={"key": "value"},
        output={"key": "value"},
        metadata={"interface": "whatsapp"},
    )

    langfuse.score(
        trace_id=langfuse.get_trace_id(),
        observation_id=spanId,
        name="valuation",
        value=1,
        comment="This is a comment",
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
    assert score["dataType"] == "NUMERIC"


def test_create_trace_and_span():
    langfuse = Langfuse(debug=False)

    trace_name = create_uuid()
    spanId = create_uuid()

    trace = langfuse.trace(name=trace_name)
    trace.span(id=spanId, name="span")

    langfuse.flush()

    trace = get_api().trace.get(trace.id)

    assert trace.name == trace_name
    assert len(trace.observations) == 1

    span = trace.observations[0]
    assert span.name == "span"
    assert span.trace_id == trace.id
    assert span.start_time is not None


def test_create_trace_and_generation():
    langfuse = Langfuse(debug=False)

    trace_name = create_uuid()
    generationId = create_uuid()

    trace = langfuse.trace(
        name=trace_name, input={"key": "value"}, session_id="test-session-id"
    )
    trace.generation(
        id=generationId,
        name="generation",
        start_time=datetime.now(),
        end_time=datetime.now(),
    )

    langfuse.flush()

    dbTrace = get_api().trace.get(trace.id)
    getTrace = langfuse.get_trace(trace.id)

    assert dbTrace.name == trace_name
    assert len(dbTrace.observations) == 1
    assert getTrace.name == trace_name
    assert len(getTrace.observations) == 1
    assert getTrace.session_id == "test-session-id"

    generation = getTrace.observations[0]
    assert generation.name == "generation"
    assert generation.trace_id == getTrace.id
    assert generation.start_time is not None
    assert getTrace.input == {"key": "value"}


def backwards_compatibility_sessionId():
    langfuse = Langfuse(debug=False)

    trace = langfuse.trace(name="test", sessionId="test-sessionId")

    langfuse.flush()

    trace = get_api().trace.get(trace.id)

    assert trace.name == "test"
    assert trace.session_id == "test-sessionId"


def test_create_trace_with_manual_timestamp():
    langfuse = Langfuse(debug=False)
    api_wrapper = LangfuseAPI()

    trace_name = create_uuid()
    trace_id = create_uuid()
    timestamp = _get_timestamp()

    langfuse.trace(id=trace_id, name=trace_name, timestamp=timestamp)

    langfuse.flush()

    trace = api_wrapper.get_trace(trace_id)

    assert trace["name"] == trace_name
    assert trace["id"] == trace_id
    assert str(trace["timestamp"]).find(timestamp.isoformat()[0:23]) != -1


def test_create_generation_and_trace():
    langfuse = Langfuse(debug=False)
    api_wrapper = LangfuseAPI()

    trace_name = create_uuid()
    trace_id = create_uuid()

    langfuse.generation(trace_id=trace_id, name="generation")
    langfuse.trace(id=trace_id, name=trace_name)

    langfuse.flush()
    sleep(2)

    trace = api_wrapper.get_trace(trace_id)

    assert trace["name"] == trace_name
    assert len(trace["observations"]) == 1

    span = trace["observations"][0]
    assert span["name"] == "generation"
    assert span["traceId"] == trace["id"]


def test_create_span_and_get_observation():
    langfuse = Langfuse(debug=False)

    span_id = create_uuid()
    langfuse.span(id=span_id, name="span")
    langfuse.flush()

    sleep(2)
    observation = langfuse.get_observation(span_id)
    assert observation.name == "span"
    assert observation.id == span_id


def test_update_generation():
    langfuse = Langfuse(debug=False)

    start = _get_timestamp()

    generation = langfuse.generation(name="generation")
    generation.update(start_time=start, metadata={"dict": "value"})

    langfuse.flush()

    trace = get_api().trace.get(generation.trace_id)

    assert trace.name == "generation"
    assert len(trace.observations) == 1
    retrieved_generation = trace.observations[0]
    assert retrieved_generation.name == "generation"
    assert retrieved_generation.trace_id == generation.trace_id
    assert retrieved_generation.metadata == {"dict": "value"}
    assert start.replace(
        microsecond=0, tzinfo=timezone.utc
    ) == retrieved_generation.start_time.replace(microsecond=0)


def test_update_span():
    langfuse = Langfuse(debug=False)

    span = langfuse.span(name="span")
    span.update(metadata={"dict": "value"})

    langfuse.flush()

    trace = get_api().trace.get(span.trace_id)

    assert trace.name == "span"
    assert len(trace.observations) == 1

    retrieved_span = trace.observations[0]
    assert retrieved_span.name == "span"
    assert retrieved_span.trace_id == span.trace_id
    assert retrieved_span.metadata == {"dict": "value"}


def test_create_event():
    langfuse = Langfuse(debug=False)

    event = langfuse.event(name="event")

    langfuse.flush()

    observation = get_api().observations.get(event.id)

    assert observation.type == "EVENT"
    assert observation.name == "event"


def test_create_trace_and_event():
    langfuse = Langfuse(debug=False)

    trace_name = create_uuid()
    eventId = create_uuid()

    trace = langfuse.trace(name=trace_name)
    trace.event(id=eventId, name="event")

    langfuse.flush()

    trace = get_api().trace.get(trace.id)

    assert trace.name == trace_name
    assert len(trace.observations) == 1

    span = trace.observations[0]
    assert span.name == "event"
    assert span.trace_id == trace.id
    assert span.start_time is not None


def test_create_span_and_generation():
    langfuse = Langfuse(debug=False)

    span = langfuse.span(name="span")
    langfuse.generation(trace_id=span.trace_id, name="generation")

    langfuse.flush()

    trace = get_api().trace.get(span.trace_id)

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

    trace = langfuse.trace(id=trace_id, name=trace_name)
    trace.generation(name="generation")

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

    timestamp = _get_timestamp()
    generation = langfuse.generation(
        name="query-generation",
        start_time=timestamp,
        model="gpt-3.5-turbo",
        model_parameters={"max_tokens": "1000", "temperature": "0.9"},
        input=[
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": "Please generate the start of a company documentation that contains the answer to the questinon: Write a summary of the Q3 OKR goals",
            },
        ],
        output="This document entails the OKR goals for ACME",
        metadata={"interface": "whatsapp"},
    )

    generation.end()

    langfuse.flush()

    trace_id = langfuse.get_trace_id()

    trace = api_wrapper.get_trace(trace_id)

    span = trace["observations"][0]
    assert span["endTime"] is not None


def test_end_generation_with_data():
    langfuse = Langfuse()
    trace = langfuse.trace()

    generation = trace.generation(
        name="query-generation",
    )

    generation.end(
        name="test_generation_end",
        metadata={"dict": "value"},
        level="ERROR",
        status_message="Generation ended",
        version="1.0",
        completion_start_time=datetime(2023, 1, 1, 12, 3, tzinfo=timezone.utc),
        model="test-model",
        model_parameters={"param1": "value1", "param2": "value2"},
        input=[{"test_input_key": "test_input_value"}],
        output={"test_output_key": "test_output_value"},
        usage={
            "input": 100,
            "output": 200,
            "total": 500,
            "unit": "CHARACTERS",
            "input_cost": 111,
            "output_cost": 222,
            "total_cost": 444,
        },
    )

    langfuse.flush()

    fetched_trace = get_api().trace.get(trace.id)

    generation = fetched_trace.observations[0]
    assert generation.completion_start_time == datetime(
        2023, 1, 1, 12, 3, tzinfo=timezone.utc
    )
    assert generation.name == "test_generation_end"
    assert generation.metadata == {"dict": "value"}
    assert generation.level == "ERROR"
    assert generation.status_message == "Generation ended"
    assert generation.version == "1.0"
    assert generation.model == "test-model"
    assert generation.model_parameters == {"param1": "value1", "param2": "value2"}
    assert generation.input == [{"test_input_key": "test_input_value"}]
    assert generation.output == {"test_output_key": "test_output_value"}
    assert generation.usage.input == 100
    assert generation.usage.output == 200
    assert generation.usage.total == 500
    assert generation.calculated_input_cost == 111
    assert generation.calculated_output_cost == 222
    assert generation.calculated_total_cost == 444


def test_end_generation_with_openai_token_format():
    langfuse = Langfuse()

    generation = langfuse.generation(
        name="query-generation",
    )

    generation.end(
        usage={
            "prompt_tokens": 100,
            "completion_tokens": 200,
            "total_tokens": 500,
            "input_cost": 111,
            "output_cost": 222,
            "total_cost": 444,
        },
    )

    langfuse.flush()

    trace_id = langfuse.get_trace_id()

    trace = get_api().trace.get(trace_id)
    print(trace.observations[0])

    generation = trace.observations[0]
    assert generation.end_time is not None
    assert generation.usage.input == 100
    assert generation.usage.output == 200
    assert generation.usage.total == 500
    assert generation.usage.unit == "TOKENS"
    assert generation.calculated_input_cost == 111
    assert generation.calculated_output_cost == 222
    assert generation.calculated_total_cost == 444


def test_end_span():
    langfuse = Langfuse()
    api_wrapper = LangfuseAPI()

    timestamp = _get_timestamp()
    span = langfuse.span(
        name="span",
        start_time=timestamp,
        input={"key": "value"},
        output={"key": "value"},
        metadata={"interface": "whatsapp"},
    )

    span.end()

    langfuse.flush()

    trace_id = langfuse.get_trace_id()

    trace = api_wrapper.get_trace(trace_id)

    span = trace["observations"][0]
    assert span["endTime"] is not None


def test_end_span_with_data():
    langfuse = Langfuse()

    timestamp = _get_timestamp()
    span = langfuse.span(
        name="span",
        start_time=timestamp,
        input={"key": "value"},
        output={"key": "value"},
        metadata={"interface": "whatsapp"},
    )

    span.end(metadata={"dict": "value"})

    langfuse.flush()

    trace_id = langfuse.get_trace_id()

    trace = get_api().trace.get(trace_id)

    span = trace.observations[0]
    assert span.end_time is not None
    assert span.metadata == {"dict": "value", "interface": "whatsapp"}


def test_get_generations():
    langfuse = Langfuse(debug=False)

    timestamp = _get_timestamp()

    langfuse.generation(
        name=create_uuid(),
        start_time=timestamp,
        end_time=timestamp,
    )

    generation_name = create_uuid()

    langfuse.generation(
        name=generation_name,
        start_time=timestamp,
        end_time=timestamp,
        input="great-prompt",
        output="great-completion",
    )

    langfuse.flush()

    sleep(1)
    generations = langfuse.get_generations(name=generation_name, limit=10, page=1)

    assert len(generations.data) == 1
    assert generations.data[0].name == generation_name
    assert generations.data[0].input == "great-prompt"
    assert generations.data[0].output == "great-completion"


def test_get_generations_by_user():
    langfuse = Langfuse(debug=False)

    timestamp = _get_timestamp()

    user_id = create_uuid()
    generation_name = create_uuid()
    trace = langfuse.trace(name="test-user", user_id=user_id)

    trace.generation(
        name=generation_name,
        start_time=timestamp,
        end_time=timestamp,
        input="great-prompt",
        output="great-completion",
    )

    langfuse.generation(
        start_time=timestamp,
        end_time=timestamp,
    )

    langfuse.flush()
    sleep(1)

    generations = langfuse.get_generations(limit=10, page=1, user_id=user_id)

    assert len(generations.data) == 1
    assert generations.data[0].name == generation_name
    assert generations.data[0].input == "great-prompt"
    assert generations.data[0].output == "great-completion"


def test_kwargs():
    langfuse = Langfuse()

    timestamp = _get_timestamp()

    dict = {
        "start_time": timestamp,
        "input": {"key": "value"},
        "output": {"key": "value"},
        "metadata": {"interface": "whatsapp"},
    }

    span = langfuse.span(
        name="span",
        **dict,
    )

    langfuse.flush()

    observation = get_api().observations.get(span.id)
    assert observation.start_time is not None
    assert observation.input == {"key": "value"}
    assert observation.output == {"key": "value"}
    assert observation.metadata == {"interface": "whatsapp"}


def test_timezone_awareness():
    os.environ["TZ"] = "US/Pacific"
    time.tzset()

    utc_now = datetime.now(timezone.utc)
    assert utc_now.tzinfo is not None

    langfuse = Langfuse(debug=False)

    trace = langfuse.trace(name="test")
    span = trace.span(name="span")
    span.end()
    generation = trace.generation(name="generation")
    generation.end()
    trace.event(name="event")

    langfuse.flush()

    trace = get_api().trace.get(trace.id)

    assert len(trace.observations) == 3
    for observation in trace.observations:
        delta = observation.start_time - utc_now
        assert delta.seconds < 5

        if observation.type != "EVENT":
            delta = observation.end_time - utc_now
            assert delta.seconds < 5

    os.environ["TZ"] = "UTC"
    time.tzset()


def test_timezone_awareness_setting_timestamps():
    os.environ["TZ"] = "US/Pacific"
    time.tzset()

    now = datetime.now()
    utc_now = datetime.now(timezone.utc)
    assert utc_now.tzinfo is not None

    print(now)
    print(utc_now)

    langfuse = Langfuse(debug=False)

    trace = langfuse.trace(name="test")
    trace.span(name="span", start_time=now, end_time=now)
    trace.generation(name="generation", start_time=now, end_time=now)
    trace.event(name="event", start_time=now)

    langfuse.flush()

    trace = get_api().trace.get(trace.id)

    assert len(trace.observations) == 3
    for observation in trace.observations:
        delta = utc_now - observation.start_time
        assert delta.seconds < 5

        if observation.type != "EVENT":
            delta = utc_now - observation.end_time
            assert delta.seconds < 5


def test_get_trace_by_session_id():
    langfuse = Langfuse(debug=False)

    # Create a trace with a session_id
    trace_name = create_uuid()
    session_id = create_uuid()
    trace = langfuse.trace(name=trace_name, session_id=session_id)

    # create a trace without a session_id
    langfuse.trace(name=create_uuid())

    langfuse.flush()

    # Retrieve the trace using the session_id
    traces = get_api().trace.list(session_id=session_id)

    # Verify that the trace was retrieved correctly
    assert len(traces.data) == 1
    retrieved_trace = traces.data[0]
    assert retrieved_trace.name == trace_name
    assert retrieved_trace.session_id == session_id
    assert retrieved_trace.id == trace.id


def test_fetch_trace():
    langfuse = Langfuse()

    # Create a trace
    name = create_uuid()
    trace = langfuse.trace(name=name)
    langfuse.flush()

    # Fetch the trace
    sleep(1)
    response = langfuse.fetch_trace(trace.id)

    # Assert the structure of the response
    assert isinstance(response, FetchTraceResponse)
    assert hasattr(response, "data")
    assert response.data.id == trace.id
    assert response.data.name == name


def test_fetch_traces():
    langfuse = Langfuse()

    # unique name
    name = create_uuid()

    # Create 3 traces with different timestamps
    now = datetime.now()
    trace_params = [
        {"id": create_uuid(), "timestamp": now - timedelta(seconds=10)},
        {"id": create_uuid(), "timestamp": now - timedelta(seconds=5)},
        {"id": create_uuid(), "timestamp": now},
    ]

    for trace_param in trace_params:
        langfuse.trace(
            id=trace_param["id"],
            name=name,
            session_id="session-1",
            input={"key": "value"},
            output="output-value",
            timestamp=trace_param["timestamp"],
        )
    langfuse.flush()
    sleep(1)

    all_traces = langfuse.fetch_traces(limit=10, name=name)
    assert len(all_traces.data) == 3
    assert all_traces.meta.total_items == 3

    # Assert the structure of the response
    assert isinstance(all_traces, FetchTracesResponse)
    assert hasattr(all_traces, "data")
    assert hasattr(all_traces, "meta")
    assert isinstance(all_traces.data, list)
    assert all_traces.data[0].name == name
    assert all_traces.data[0].session_id == "session-1"

    # Fetch traces with a time range that should only include the middle trace
    from_timestamp = now - timedelta(seconds=7.5)
    to_timestamp = now - timedelta(seconds=2.5)
    response = langfuse.fetch_traces(
        limit=10, name=name, from_timestamp=from_timestamp, to_timestamp=to_timestamp
    )
    assert len(response.data) == 1
    assert response.meta.total_items == 1
    fetched_trace = response.data[0]
    assert fetched_trace.name == name
    assert fetched_trace.session_id == "session-1"
    assert fetched_trace.input == '{"key":"value"}'
    assert fetched_trace.output == "output-value"
    # compare timestamps without microseconds and in UTC
    assert fetched_trace.timestamp.replace(microsecond=0) == trace_params[1][
        "timestamp"
    ].replace(microsecond=0).astimezone(timezone.utc)

    # Fetch with pagination
    paginated_response = langfuse.fetch_traces(limit=1, page=2, name=name)
    assert len(paginated_response.data) == 1
    assert paginated_response.meta.total_items == 3
    assert paginated_response.meta.total_pages == 3


def test_fetch_observation():
    langfuse = Langfuse()

    # Create a trace and a generation
    name = create_uuid()
    trace = langfuse.trace(name=name)
    generation = trace.generation(name=name)
    langfuse.flush()
    sleep(1)

    # Fetch the observation
    response = langfuse.fetch_observation(generation.id)

    # Assert the structure of the response
    assert isinstance(response, FetchObservationResponse)
    assert hasattr(response, "data")
    assert response.data.id == generation.id
    assert response.data.name == name
    assert response.data.type == "GENERATION"


def test_fetch_observations():
    langfuse = Langfuse()

    # Create a trace with multiple generations
    name = create_uuid()
    trace = langfuse.trace(name=name)
    gen1 = trace.generation(name=name)
    gen2 = trace.generation(name=name)
    langfuse.flush()
    sleep(1)

    # Fetch observations
    response = langfuse.fetch_observations(limit=10, name=name)

    # Assert the structure of the response
    assert isinstance(response, FetchObservationsResponse)
    assert hasattr(response, "data")
    assert hasattr(response, "meta")
    assert isinstance(response.data, list)
    assert len(response.data) == 2
    assert response.meta.total_items == 2
    assert response.data[0].id in [gen1.id, gen2.id]

    # fetch only one
    response = langfuse.fetch_observations(limit=1, page=2, name=name)
    assert len(response.data) == 1
    assert response.meta.total_items == 2
    assert response.meta.total_pages == 2


def test_fetch_trace_not_found():
    langfuse = Langfuse()

    # Attempt to fetch a non-existent trace
    with pytest.raises(Exception):
        langfuse.fetch_trace(create_uuid())


def test_fetch_observation_not_found():
    langfuse = Langfuse()

    # Attempt to fetch a non-existent observation
    with pytest.raises(Exception):
        langfuse.fetch_observation(create_uuid())


def test_fetch_traces_empty():
    langfuse = Langfuse()

    # Fetch traces with a filter that should return no results
    response = langfuse.fetch_traces(name=create_uuid())

    assert isinstance(response, FetchTracesResponse)
    assert len(response.data) == 0
    assert response.meta.total_items == 0


def test_fetch_observations_empty():
    langfuse = Langfuse()

    # Fetch observations with a filter that should return no results
    response = langfuse.fetch_observations(name=create_uuid())

    assert isinstance(response, FetchObservationsResponse)
    assert len(response.data) == 0
    assert response.meta.total_items == 0


def test_fetch_sessions():
    langfuse = Langfuse()

    # unique name
    name = create_uuid()
    session1 = create_uuid()
    session2 = create_uuid()
    session3 = create_uuid()

    # Create multiple traces
    langfuse.trace(name=name, session_id=session1)
    langfuse.trace(name=name, session_id=session2)
    langfuse.trace(name=name, session_id=session3)
    langfuse.flush()

    # Fetch traces
    sleep(3)
    response = langfuse.fetch_sessions()

    # Assert the structure of the response, cannot check for the exact number of sessions as the table is not cleared between tests
    assert isinstance(response, FetchSessionsResponse)
    assert hasattr(response, "data")
    assert hasattr(response, "meta")
    assert isinstance(response.data, list)

    # fetch only one, cannot check for the exact number of sessions as the table is not cleared between tests
    response = langfuse.fetch_sessions(limit=1, page=2)
    assert len(response.data) == 1


def test_create_trace_sampling_zero():
    langfuse = Langfuse(debug=True, sample_rate=0)
    api_wrapper = LangfuseAPI()
    trace_name = create_uuid()

    trace = langfuse.trace(
        name=trace_name,
        user_id="test",
        metadata={"key": "value"},
        tags=["tag1", "tag2"],
        public=True,
    )

    trace.generation(name="generation")
    trace.score(name="score", value=0.5)

    langfuse.flush()

    fetched_trace = api_wrapper.get_trace(trace.id)
    assert fetched_trace == {
        "error": "LangfuseNotFoundError",
        "message": f"Trace {trace.id} not found within authorized project",
    }


def test_mask_function():
    def mask_func(data):
        if isinstance(data, dict):
            return {k: "MASKED" for k in data}
        elif isinstance(data, str):
            return "MASKED"
        return data

    langfuse = Langfuse(debug=True, mask=mask_func)
    api_wrapper = LangfuseAPI()

    trace = langfuse.trace(name="test_trace", input={"sensitive": "data"})
    trace.update(output={"more": "sensitive"})

    gen = trace.generation(name="test_gen", input={"prompt": "secret"})
    gen.update(output="new_confidential")

    span = trace.span(name="test_span", input={"data": "private"})
    span.update(output="new_classified")

    langfuse.flush()

    fetched_trace = api_wrapper.get_trace(trace.id)
    assert fetched_trace["input"] == {"sensitive": "MASKED"}
    assert fetched_trace["output"] == {"more": "MASKED"}

    fetched_gen = [
        o for o in fetched_trace["observations"] if o["type"] == "GENERATION"
    ][0]
    assert fetched_gen["input"] == {"prompt": "MASKED"}
    assert fetched_gen["output"] == "MASKED"

    fetched_span = [o for o in fetched_trace["observations"] if o["type"] == "SPAN"][0]
    assert fetched_span["input"] == {"data": "MASKED"}
    assert fetched_span["output"] == "MASKED"

    def faulty_mask_func(data):
        raise Exception("Masking error")

    langfuse = Langfuse(debug=True, mask=faulty_mask_func)

    trace = langfuse.trace(name="test_trace", input={"sensitive": "data"})
    trace.update(output={"more": "sensitive"})
    langfuse.flush()

    fetched_trace = api_wrapper.get_trace(trace.id)
    assert fetched_trace["input"] == "<fully masked due to failed mask function>"
    assert fetched_trace["output"] == "<fully masked due to failed mask function>"


def test_get_project_id():
    langfuse = Langfuse(debug=False)
    res = langfuse._get_project_id()
    assert res is not None
    assert res == "7a88fb47-b4e2-43b8-a06c-a5ce950dc53a"


def test_generate_trace_id():
    langfuse = Langfuse(debug=False)
    trace_id = create_uuid()

    langfuse.trace(id=trace_id, name="test_trace")
    langfuse.flush()

    trace_url = langfuse.get_trace_url()
    assert (
        trace_url
        == f"http://localhost:3000/project/7a88fb47-b4e2-43b8-a06c-a5ce950dc53a/traces/{trace_id}"
    )
