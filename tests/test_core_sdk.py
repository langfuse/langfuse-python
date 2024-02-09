import os
import time
from asyncio import gather
from datetime import datetime, timezone

from langfuse.utils import _get_timestamp


import pytest

from langfuse import Langfuse
from tests.api_wrapper import LangfuseAPI
from tests.utils import LlmUsage, LlmUsageWithCost, create_uuid, get_api


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
        assert observation.metadata == {"count": str(i)}


def test_flush():
    # set up the consumer with more requests than a single batch will allow
    langfuse = Langfuse(debug=False)

    for i in range(2):
        langfuse.trace(
            name=str(i),
        )

    langfuse.flush()
    # Make sure that the client queue is empty after flushing
    assert langfuse.task_manager._queue.empty()


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
    assert langfuse.task_manager._queue.empty()


def test_create_score():
    langfuse = Langfuse(debug=False)
    api_wrapper = LangfuseAPI()

    trace = langfuse.trace(
        name="this-is-so-great-new",
        user_id="test",
        metadata="test",
    )

    langfuse.flush()
    assert langfuse.task_manager._queue.qsize() == 0

    score_id = create_uuid()

    langfuse.score(
        id=score_id,
        trace_id=trace.id,
        name="this-is-a-score",
        value=1,
    )

    trace.generation(name="yet another child", metadata="test")

    langfuse.flush()

    assert langfuse.task_manager._queue.qsize() == 0

    trace = api_wrapper.get_trace(trace.id)

    assert trace["scores"][0]["id"] == score_id


def test_create_trace():
    langfuse = Langfuse(debug=False)
    api_wrapper = LangfuseAPI()
    trace_name = create_uuid()

    trace = langfuse.trace(
        name=trace_name,
        user_id="test",
        metadata={"key": "value"},
        tags=["tag1", "tag2"],
    )

    langfuse.flush()

    trace = api_wrapper.get_trace(trace.id)

    assert trace["name"] == trace_name
    assert trace["userId"] == "test"
    assert trace["metadata"] == {"key": "value"}
    assert trace["tags"] == ["tag1", "tag2"]
    assert True if not trace["externalId"] else False


def test_create_update_trace():
    langfuse = Langfuse(debug=False)
    api = get_api()
    trace_name = create_uuid()

    trace = langfuse.trace(
        name=trace_name,
        user_id="test",
        metadata={"key": "value"},
    )
    trace.update(metadata={"key": "value2"})

    langfuse.flush()

    trace = api.trace.get(trace.id)

    assert trace.name == trace_name
    assert trace.user_id == "test"
    assert trace.metadata == {"key": "value2"}


def test_create_generation():
    langfuse = Langfuse(debug=False)
    api = get_api()

    timestamp = _get_timestamp()
    generation_id = create_uuid()

    langfuse.generation(
        id=generation_id,
        name="query-generation",
        start_time=timestamp,
        end_time=timestamp,
        model="gpt-3.5-turbo",
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

    trace = api.trace.get(trace_id)

    assert trace.name == "query-generation"
    assert trace.user_id is None
    assert trace.metadata is None

    assert len(trace.observations) == 1

    generation = trace.observations[0]

    assert generation.id == generation_id
    assert generation.name == "query-generation"
    assert generation.start_time is not None
    assert generation.end_time is not None
    assert generation.model == "gpt-3.5-turbo"
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
            LlmUsage(promptTokens=51, completionTokens=0, totalTokens=100),
            "TOKENS",
            None,
            None,
            None,
        ),
        (LlmUsage(promptTokens=51, totalTokens=100), "TOKENS", None, None, None),
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
            {"input": 51, "total": 100},
            None,
            None,
            None,
            None,
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
        (
            LlmUsageWithCost(
                promptTokens=51,
                completionTokens=0,
                totalTokens=100,
                inputCost=0.0021,
                outputCost=0.00000000000021,
                totalCost=None,
            ),
            "TOKENS",
            0.0021,
            0.00000000000021,
            None,
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
    langfuse = Langfuse(debug=True)
    api = get_api()

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

    trace = api.trace.get(trace_id)

    assert trace.name == "query-generation"
    assert trace.user_id is None
    assert trace.metadata is None

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
    assert generation.metadata == [{"tags": ["yo"]}]
    assert generation.start_time is not None
    assert generation.usage.input == 51
    assert generation.usage.output == 0
    assert generation.usage.total == 100
    assert generation.calculated_input_cost == expected_input_cost
    assert generation.calculated_output_cost == expected_output_cost
    assert generation.calculated_total_cost == expected_total_cost
    assert generation.usage.unit == expected_usage


def test_create_span():
    langfuse = Langfuse(debug=False)
    api = get_api()

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

    trace = api.trace.get(trace_id)

    assert trace.name == "span"
    assert trace.user_id is None
    assert trace.metadata is None

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


def test_score_trace_nested_trace():
    langfuse = Langfuse(debug=False)
    api = get_api()

    trace_name = create_uuid()

    trace = langfuse.trace(name=trace_name)

    trace.score(
        name="valuation",
        value=0.5,
        comment="This is a comment",
    )

    langfuse.flush()

    trace_id = langfuse.get_trace_id()

    trace = api.trace.get(trace_id)

    assert trace.name == trace_name

    assert len(trace.scores) == 1

    score = trace.scores[0]

    assert score.name == "valuation"
    assert score.value == 0.5
    assert score.comment == "This is a comment"
    assert score.observation_id is None


def test_score_trace_nested_observation():
    langfuse = Langfuse(debug=False)
    api = get_api()

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

    trace = api.trace.get(trace_id)

    assert trace.name == trace_name

    assert len(trace.scores) == 1

    score = trace.scores[0]

    assert score.name == "valuation"
    assert score.value == 0.5
    assert score.comment == "This is a comment"
    assert score.observation_id == span.id


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


def test_create_trace_and_span():
    langfuse = Langfuse(debug=False)
    api = get_api()

    trace_name = create_uuid()
    spanId = create_uuid()

    trace = langfuse.trace(name=trace_name)
    trace.span(id=spanId, name="span")

    langfuse.flush()

    trace = api.trace.get(trace.id)

    assert trace.name == trace_name
    assert len(trace.observations) == 1

    span = trace.observations[0]
    assert span.name == "span"
    assert span.trace_id == trace.id
    assert span.start_time is not None


def test_create_trace_and_generation():
    langfuse = Langfuse(debug=False)
    api = get_api()

    trace_name = create_uuid()
    generationId = create_uuid()

    trace = langfuse.trace(name=trace_name, input={"key": "value"}, sessionId="test")
    trace.generation(
        id=generationId,
        name="generation",
        start_time=datetime.now(),
        end_time=datetime.now(),
    )

    langfuse.flush()

    getTrace = langfuse.get_trace(trace.id)
    dbTrace = api.trace.get(trace.id)

    assert dbTrace.name == trace_name
    assert len(dbTrace.observations) == 1
    assert getTrace.name == trace_name
    assert len(getTrace.observations) == 1

    generation = getTrace.observations[0]
    assert generation.name == "generation"
    assert generation.trace_id == getTrace.id
    assert generation.start_time is not None
    assert getTrace.input == {"key": "value"}


def test_create_generation_and_trace():
    langfuse = Langfuse(debug=False)
    api_wrapper = LangfuseAPI()

    trace_name = create_uuid()
    trace_id = create_uuid()

    langfuse.generation(trace_id=trace_id, name="generation")
    langfuse.trace(id=trace_id, name=trace_name)

    langfuse.flush()

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

    observation = langfuse.get_observation(span_id)
    assert observation.name == "span"
    assert observation.id == span_id


def test_update_generation():
    langfuse = Langfuse(debug=False)
    api = get_api()
    start = datetime.utcnow()

    generation = langfuse.generation(name="generation")
    generation.update(start_time=start, metadata={"dict": "value"})

    langfuse.flush()

    trace = api.trace.get(generation.trace_id)

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
    api = get_api()

    span = langfuse.span(name="span")
    span.update(metadata={"dict": "value"})

    langfuse.flush()

    trace = api.trace.get(span.trace_id)

    assert trace.name == "span"
    assert len(trace.observations) == 1

    retrieved_span = trace.observations[0]
    assert retrieved_span.name == "span"
    assert retrieved_span.trace_id == span.trace_id
    assert retrieved_span.metadata == {"dict": "value"}


def test_create_event():
    langfuse = Langfuse(debug=False)
    api = get_api()

    event = langfuse.event(name="event")

    langfuse.flush()

    observation = api.observations.get(event.id)

    assert observation.type == "EVENT"
    assert observation.name == "event"


def test_create_trace_and_event():
    langfuse = Langfuse(debug=False)
    api = get_api()

    trace_name = create_uuid()
    eventId = create_uuid()

    trace = langfuse.trace(name=trace_name)
    trace.event(id=eventId, name="event")

    langfuse.flush()

    trace = api.trace.get(trace.id)

    assert trace.name == trace_name
    assert len(trace.observations) == 1

    span = trace.observations[0]
    assert span.name == "event"
    assert span.trace_id == trace.id
    assert span.start_time is not None


def test_create_span_and_generation():
    api = get_api()

    langfuse = Langfuse(debug=False)

    span = langfuse.span(name="span")
    langfuse.generation(trace_id=span.trace_id, name="generation")

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
    api = get_api()

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
        usage=LlmUsage(promptTokens=50, completionTokens=49),
        metadata={"interface": "whatsapp"},
    )

    generation.end(metadata={"dict": "value"})

    langfuse.flush()

    trace_id = langfuse.get_trace_id()

    trace = api.trace.get(trace_id)

    generation = trace.observations[0]
    assert generation.end_time is not None
    assert generation.metadata == {"dict": "value", "interface": "whatsapp"}


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
    api = get_api()

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

    trace = api.trace.get(trace_id)

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
    generations = langfuse.get_generations(limit=10, page=1, user_id=user_id)

    assert len(generations.data) == 1
    assert generations.data[0].name == generation_name
    assert generations.data[0].input == "great-prompt"
    assert generations.data[0].output == "great-completion"


def test_kwargs():
    langfuse = Langfuse()
    api = get_api()

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

    observation = api.observations.get(span.id)
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
    api = get_api()

    trace = langfuse.trace(name="test")
    span = trace.span(name="span")
    span.end()
    generation = trace.generation(name="generation")
    generation.end()
    trace.event(name="event")

    langfuse.flush()

    trace = api.trace.get(trace.id)

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
    api = get_api()

    trace = langfuse.trace(name="test")
    trace.span(name="span", start_time=now, end_time=now)
    trace.generation(name="generation", start_time=now, end_time=now)
    trace.event(name="event", start_time=now)

    langfuse.flush()

    trace = api.trace.get(trace.id)

    assert len(trace.observations) == 3
    for observation in trace.observations:
        delta = utc_now - observation.start_time
        assert delta.seconds < 5

        if observation.type != "EVENT":
            delta = utc_now - observation.end_time
            assert delta.seconds < 5
