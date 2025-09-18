import os
import time
from asyncio import gather
from datetime import datetime, timezone
from time import sleep

import pytest

from langfuse import Langfuse
from langfuse._client.resource_manager import LangfuseResourceManager
from langfuse._utils import _get_timestamp
from tests.api_wrapper import LangfuseAPI
from tests.utils import (
    create_uuid,
    get_api,
)


@pytest.mark.asyncio
async def test_concurrency():
    _get_timestamp()

    async def update_generation(i, langfuse: Langfuse):
        # Create a new trace with a generation
        with langfuse.start_as_current_span(name=f"parent-{i}") as parent_span:
            # Set trace name
            parent_span.update_trace(name=str(i))

            # Create generation as a child
            generation = langfuse.start_generation(name=str(i))

            # Update generation with metadata
            generation.update(metadata={"count": str(i)})

            # End the generation
            generation.end()

    # Create Langfuse client
    langfuse = Langfuse()

    # Run concurrent operations
    await gather(*(update_generation(i, langfuse) for i in range(100)))

    langfuse.flush()

    # Allow time for all operations to be processed
    sleep(10)

    # Verify that all spans were created properly
    api = get_api()
    for i in range(100):
        # Find the observations with the expected name
        observations = api.observations.get_many(name=str(i)).data

        # Find generation observations (there should be at least one)
        generation_obs = [obs for obs in observations if obs.type == "GENERATION"]
        assert len(generation_obs) > 0

        # Verify metadata
        observation = generation_obs[0]
        assert observation.name == str(i)
        assert observation.metadata["count"] == i


def test_flush():
    # Initialize Langfuse client with debug disabled
    langfuse = Langfuse()

    trace_ids = []
    for i in range(2):
        # Create spans and set the trace name using update_trace
        with langfuse.start_as_current_span(name="span-" + str(i)) as span:
            span.update_trace(name=str(i))
            # Store the trace ID for later verification
            trace_ids.append(langfuse.get_current_trace_id())

    # Flush all pending spans to the Langfuse API
    langfuse.flush()

    # Allow time for API to process
    sleep(2)

    # Verify traces were sent by checking they exist in the API
    api = get_api()
    for i, trace_id in enumerate(trace_ids):
        trace = api.trace.get(trace_id)
        assert trace.name == str(i)


def test_invalid_score_data_does_not_raise_exception():
    langfuse = Langfuse()

    # Create a span and set trace properties
    with langfuse.start_as_current_span(name="test-span") as span:
        span.update_trace(
            name="this-is-so-great-new",
            user_id="test",
            metadata="test",
        )
        # Get trace ID for later use
        trace_id = span.trace_id

    # Ensure data is sent
    langfuse.flush()

    # Create a score with invalid data (negative value for a BOOLEAN)
    score_id = create_uuid()
    langfuse.create_score(
        score_id=score_id,
        trace_id=trace_id,
        name="this-is-a-score",
        value=-1,
        data_type="BOOLEAN",
    )

    # Verify the operation didn't crash
    langfuse.flush()
    # We can't assert queue size in OTEL implementation, but we can verify it completes without exception


def test_create_numeric_score():
    langfuse = Langfuse()
    api_wrapper = LangfuseAPI()

    # Create a span and set trace properties
    with langfuse.start_as_current_span(name="test-span") as span:
        span.update_trace(
            name="this-is-so-great-new",
            user_id="test",
            metadata="test",
        )
        # Get trace ID for later use
        trace_id = span.trace_id

    # Ensure data is sent
    langfuse.flush()
    sleep(2)

    # Create a numeric score
    score_id = create_uuid()
    langfuse.create_score(
        score_id=score_id,
        trace_id=trace_id,
        name="this-is-a-score",
        value=1,
    )

    # Create a generation in the same trace
    generation = langfuse.start_generation(
        name="yet another child", metadata="test", trace_context={"trace_id": trace_id}
    )
    generation.end()

    # Ensure data is sent
    langfuse.flush()
    sleep(2)

    # Retrieve and verify
    trace = api_wrapper.get_trace(trace_id)

    assert trace["scores"][0]["id"] == score_id
    assert trace["scores"][0]["value"] == 1
    assert trace["scores"][0]["dataType"] == "NUMERIC"
    assert trace["scores"][0]["stringValue"] is None


def test_create_boolean_score():
    langfuse = Langfuse()
    api_wrapper = LangfuseAPI()

    # Create a span and set trace properties
    with langfuse.start_as_current_span(name="test-span") as span:
        span.update_trace(
            name="this-is-so-great-new",
            user_id="test",
            metadata="test",
        )
        # Get trace ID for later use
        trace_id = span.trace_id

    # Ensure data is sent
    langfuse.flush()
    sleep(2)

    # Create a boolean score
    score_id = create_uuid()
    langfuse.create_score(
        score_id=score_id,
        trace_id=trace_id,
        name="this-is-a-score",
        value=1,
        data_type="BOOLEAN",
    )

    # Create a generation in the same trace
    generation = langfuse.start_generation(
        name="yet another child", metadata="test", trace_context={"trace_id": trace_id}
    )
    generation.end()

    # Ensure data is sent
    langfuse.flush()
    sleep(2)

    # Retrieve and verify
    trace = api_wrapper.get_trace(trace_id)

    assert trace["scores"][0]["id"] == score_id
    assert trace["scores"][0]["dataType"] == "BOOLEAN"
    assert trace["scores"][0]["value"] == 1
    assert trace["scores"][0]["stringValue"] == "True"


def test_create_categorical_score():
    langfuse = Langfuse()
    api_wrapper = LangfuseAPI()

    # Create a span and set trace properties
    with langfuse.start_as_current_span(name="test-span") as span:
        span.update_trace(
            name="this-is-so-great-new",
            user_id="test",
            metadata="test",
        )
        # Get trace ID for later use
        trace_id = span.trace_id

    # Ensure data is sent
    langfuse.flush()
    sleep(2)

    # Create a categorical score
    score_id = create_uuid()
    langfuse.create_score(
        score_id=score_id,
        trace_id=trace_id,
        name="this-is-a-score",
        value="high score",
    )

    # Create a generation in the same trace
    generation = langfuse.start_generation(
        name="yet another child", metadata="test", trace_context={"trace_id": trace_id}
    )
    generation.end()

    # Ensure data is sent
    langfuse.flush()
    sleep(2)

    # Retrieve and verify
    trace = api_wrapper.get_trace(trace_id)

    assert trace["scores"][0]["id"] == score_id
    assert trace["scores"][0]["dataType"] == "CATEGORICAL"
    assert trace["scores"][0]["value"] == 0
    assert trace["scores"][0]["stringValue"] == "high score"


def test_create_trace():
    langfuse = Langfuse()
    trace_name = create_uuid()

    # Create a span and update the trace properties
    with langfuse.start_as_current_span(name="test-span") as span:
        span.update_trace(
            name=trace_name,
            user_id="test",
            metadata={"key": "value"},
            tags=["tag1", "tag2"],
            public=True,
        )
        # Get trace ID for later verification
        trace_id = langfuse.get_current_trace_id()

    # Ensure data is sent to the API
    langfuse.flush()
    sleep(2)

    # Retrieve the trace from the API
    trace = LangfuseAPI().get_trace(trace_id)

    # Verify all trace properties
    assert trace["name"] == trace_name
    assert trace["userId"] == "test"
    assert trace["metadata"]["key"] == "value"
    assert trace["tags"] == ["tag1", "tag2"]
    assert trace["public"] is True
    assert True if not trace["externalId"] else False


def test_create_update_trace():
    langfuse = Langfuse()

    trace_name = create_uuid()

    # Create initial span with trace properties
    with langfuse.start_as_current_span(name="test-span") as span:
        span.update_trace(
            name=trace_name,
            user_id="test",
            metadata={"key": "value"},
            public=True,
        )
        # Get trace ID for later reference
        trace_id = span.trace_id

        # Allow a small delay before updating
        sleep(1)

        # Update trace properties
        span.update_trace(metadata={"key2": "value2"}, public=False)

    # Ensure data is sent to the API
    langfuse.flush()
    sleep(2)

    assert isinstance(trace_id, str)
    # Retrieve and verify trace
    trace = get_api().trace.get(trace_id)

    assert trace.name == trace_name
    assert trace.user_id == "test"
    assert trace.metadata["key"] == "value"
    assert trace.metadata["key2"] == "value2"
    assert trace.public is False


def test_create_update_current_trace():
    langfuse = Langfuse()

    trace_name = create_uuid()

    # Create initial span with trace properties using update_current_trace
    with langfuse.start_as_current_span(name="test-span-current") as span:
        langfuse.update_current_trace(
            name=trace_name,
            user_id="test",
            metadata={"key": "value"},
            public=True,
            input="test_input"
        )
        # Get trace ID for later reference
        trace_id = span.trace_id

        # Allow a small delay before updating
        sleep(1)

        # Update trace properties using update_current_trace
        langfuse.update_current_trace(metadata={"key2": "value2"}, public=False, version="1.0")

    # Ensure data is sent to the API
    langfuse.flush()
    sleep(2)

    assert isinstance(trace_id, str)
    # Retrieve and verify trace
    trace = get_api().trace.get(trace_id)

    # The 2nd update to the trace must not erase previously set attributes
    assert trace.name == trace_name
    assert trace.user_id == "test"
    assert trace.metadata["key"] == "value"
    assert trace.metadata["key2"] == "value2"
    assert trace.public is False
    assert trace.version == "1.0"
    assert trace.input == "test_input"


def test_create_generation():
    langfuse = Langfuse()

    # Create a generation using OTEL approach
    generation = langfuse.start_generation(
        name="query-generation",
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
        usage_details={"input": 50, "output": 49, "total": 99},
        metadata={"interface": "whatsapp"},
        level="DEBUG",
    )

    # Get IDs for verification
    trace_id = generation.trace_id

    # End the generation
    generation.end()

    # Flush to ensure all data is sent
    langfuse.flush()
    sleep(2)

    # Retrieve the trace from the API
    trace = get_api().trace.get(trace_id)

    # Verify trace details
    assert trace.name == "query-generation"
    assert trace.user_id is None

    assert len(trace.observations) == 1

    # Verify generation details
    generation_api = trace.observations[0]

    assert generation_api.name == "query-generation"
    assert generation_api.start_time is not None
    assert generation_api.end_time is not None
    assert generation_api.model == "gpt-3.5-turbo-0125"
    assert generation_api.model_parameters == {
        "max_tokens": "1000",
        "temperature": "0.9",
        "stop": '["user-1","user-2"]',
    }
    assert generation_api.input == [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": "Please generate the start of a company documentation that contains the answer to the questinon: Write a summary of the Q3 OKR goals",
        },
    ]
    assert generation_api.output == "This document entails the OKR goals for ACME"
    assert generation_api.level == "DEBUG"


@pytest.mark.parametrize(
    "usage, expected_usage, expected_input_cost, expected_output_cost, expected_total_cost",
    [
        (
            {
                "input": 51,
                "output": 0,
                "total": 100,
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
            },
            "CHARACTERS",
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
    langfuse = Langfuse()

    generation = langfuse.start_generation(
        name="query-generation",
        input=[
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": "Please generate the start of a company documentation that contains the answer to the questinon: Write a summary of the Q3 OKR goals",
            },
        ],
        output=[{"foo": "bar"}],
        usage_details=usage,
        metadata={"tags": ["yo"]},
    ).end()

    langfuse.flush()
    trace_id = generation.trace_id
    trace = get_api().trace.get(trace_id)

    assert trace.name == "query-generation"
    assert trace.user_id is None

    assert len(trace.observations) == 1

    generation_api = trace.observations[0]

    assert generation_api.id == generation.id
    assert generation_api.name == "query-generation"
    assert generation_api.input == [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": "Please generate the start of a company documentation that contains the answer to the questinon: Write a summary of the Q3 OKR goals",
        },
    ]
    assert generation_api.output == [{"foo": "bar"}]

    # Check if metadata exists and has tags before asserting
    if (
        hasattr(generation_api, "metadata")
        and generation_api.metadata is not None
        and "tags" in generation_api.metadata
    ):
        assert generation_api.metadata["tags"] == ["yo"]

    assert generation_api.start_time is not None
    assert generation_api.usage_details == {"input": 51, "output": 0, "total": 100}


def test_create_span():
    langfuse = Langfuse()

    # Create span using OTEL-based client
    span = langfuse.start_span(
        name="span",
        input={"key": "value"},
        output={"key": "value"},
        metadata={"interface": "whatsapp"},
    )

    # Get IDs for verification
    span_id = span.id
    trace_id = span.trace_id

    # End the span
    span.end()

    # Ensure all data is sent
    langfuse.flush()
    sleep(2)

    # Retrieve from API
    trace = get_api().trace.get(trace_id)

    # Verify trace details
    assert trace.name == "span"
    assert trace.user_id is None

    assert len(trace.observations) == 1

    # Verify span details
    span_api = trace.observations[0]

    assert span_api.id == span_id
    assert span_api.name == "span"
    assert span_api.start_time is not None
    assert span_api.end_time is not None
    assert span_api.input == {"key": "value"}
    assert span_api.output == {"key": "value"}
    assert span_api.start_time is not None


def test_score_trace():
    langfuse = Langfuse()
    api_wrapper = LangfuseAPI()

    trace_name = create_uuid()

    # Create a span and set trace name
    with langfuse.start_as_current_span(name="test-span") as span:
        span.update_trace(name=trace_name)

        # Get trace ID for later verification
        trace_id = langfuse.get_current_trace_id()

        # Create score for the trace
        langfuse.score_current_trace(
            name="valuation",
            value=0.5,
            comment="This is a comment",
        )

    # Ensure data is sent
    langfuse.flush()
    sleep(2)

    # Retrieve and verify
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
    langfuse = Langfuse()

    trace_name = create_uuid()

    # Create a trace with span
    with langfuse.start_as_current_span(name="test-span") as span:
        # Set trace name
        span.update_trace(name=trace_name)

        # Score using the span's method for scoring the trace
        span.score_trace(
            name="valuation",
            value=0.5,
            comment="This is a comment",
        )

        # Get trace ID for verification
        trace_id = span.trace_id

    # Ensure data is sent
    langfuse.flush()
    sleep(2)

    # Retrieve and verify
    trace = get_api().trace.get(trace_id)

    assert trace.name == trace_name
    assert len(trace.scores) == 1

    score = trace.scores[0]

    assert score.name == "valuation"
    assert score.value == 0.5
    assert score.comment == "This is a comment"
    assert score.observation_id is None  # API returns this field name
    assert score.data_type == "NUMERIC"


def test_score_trace_nested_observation():
    langfuse = Langfuse()

    trace_name = create_uuid()

    # Create a parent span and set trace name
    with langfuse.start_as_current_span(name="parent-span") as parent_span:
        parent_span.update_trace(name=trace_name)

        # Create a child span
        child_span = langfuse.start_span(name="span")

        # Score the child span
        child_span.score(
            name="valuation",
            value=0.5,
            comment="This is a comment",
        )

        # Get IDs for verification
        child_span_id = child_span.id
        trace_id = parent_span.trace_id

        # End the child span
        child_span.end()

    # Ensure data is sent
    langfuse.flush()
    sleep(2)

    # Retrieve and verify
    trace = get_api().trace.get(trace_id)

    assert trace.name == trace_name
    assert len(trace.scores) == 1

    score = trace.scores[0]

    assert score.name == "valuation"
    assert score.value == 0.5
    assert score.comment == "This is a comment"
    assert score.observation_id == child_span_id  # API returns this field name
    assert score.data_type == "NUMERIC"


def test_score_span():
    langfuse = Langfuse()
    api_wrapper = LangfuseAPI()

    # Create a span
    span = langfuse.start_span(
        name="span",
        input={"key": "value"},
        output={"key": "value"},
        metadata={"interface": "whatsapp"},
    )

    # Get IDs for verification
    span_id = span.id
    trace_id = span.trace_id

    # Score the span
    langfuse.create_score(
        trace_id=trace_id,
        observation_id=span_id,  # API parameter name
        name="valuation",
        value=1,
        comment="This is a comment",
    )

    # End the span
    span.end()

    # Ensure data is sent
    langfuse.flush()
    sleep(3)

    # Retrieve and verify
    trace = api_wrapper.get_trace(trace_id)

    assert len(trace["scores"]) == 1
    assert len(trace["observations"]) == 1

    score = trace["scores"][0]

    assert score["name"] == "valuation"
    assert score["value"] == 1
    assert score["comment"] == "This is a comment"
    assert score["observationId"] == span_id
    assert score["dataType"] == "NUMERIC"


def test_create_trace_and_span():
    langfuse = Langfuse()

    trace_name = create_uuid()

    # Create parent span and set trace name
    with langfuse.start_as_current_span(name=trace_name) as parent_span:
        parent_span.update_trace(name=trace_name)

        # Create a child span
        child_span = parent_span.start_span(name="span")

        # Get trace ID for verification
        trace_id = parent_span.trace_id

        # End the child span
        child_span.end()

    # Ensure data is sent
    langfuse.flush()
    sleep(2)

    # Retrieve and verify
    trace = get_api().trace.get(trace_id)

    assert trace.name == trace_name
    assert len(trace.observations) == 2  # Parent span and child span

    # Find the child span
    child_spans = [obs for obs in trace.observations if obs.name == "span"]
    assert len(child_spans) == 1

    span = child_spans[0]
    assert span.name == "span"
    assert span.trace_id == trace_id
    assert span.start_time is not None


def test_create_trace_and_generation():
    langfuse = Langfuse()

    trace_name = create_uuid()

    # Create parent span and set trace properties
    with langfuse.start_as_current_span(name=trace_name) as parent_span:
        parent_span.update_trace(
            name=trace_name, input={"key": "value"}, session_id="test-session-id"
        )

        # Create a generation as child
        generation = parent_span.start_generation(name="generation")

        # Get IDs for verification
        trace_id = parent_span.trace_id

        # End the generation
        generation.end()

    # Ensure data is sent
    langfuse.flush()
    sleep(2)

    # Retrieve traces in two ways
    dbTrace = get_api().trace.get(trace_id)
    getTrace = get_api().trace.get(
        trace_id
    )  # Using API as direct getTrace not available

    # Verify trace details
    assert dbTrace.name == trace_name
    assert len(dbTrace.observations) == 2  # Parent span and generation
    assert getTrace.name == trace_name
    assert len(getTrace.observations) == 2
    assert getTrace.session_id == "test-session-id"

    # Find the generation
    generations = [obs for obs in getTrace.observations if obs.name == "generation"]
    assert len(generations) == 1

    generation = generations[0]
    assert generation.name == "generation"
    assert generation.trace_id == trace_id
    assert generation.start_time is not None
    assert getTrace.input == {"key": "value"}


def test_create_generation_and_trace():
    langfuse = Langfuse()
    api_wrapper = LangfuseAPI()

    trace_name = create_uuid()

    # Create trace with a generation
    trace_context = {"trace_id": langfuse.create_trace_id()}

    # Create a generation with this context
    generation = langfuse.start_generation(
        name="generation", trace_context=trace_context
    )

    # Get trace ID for verification
    trace_id = generation.trace_id

    # End the generation
    generation.end()

    sleep(0.1)

    # Update trace properties in a separate span
    with langfuse.start_as_current_span(
        name="trace-update", trace_context={"trace_id": trace_id}
    ) as span:
        span.update_trace(name=trace_name)

    # Ensure data is sent
    langfuse.flush()
    sleep(2)

    # Retrieve and verify
    trace = api_wrapper.get_trace(trace_id)

    assert trace["name"] == trace_name

    # We should have 2 observations (the generation and the span for updating trace)
    assert len(trace["observations"]) == 2

    # Find the generation
    generations = [obs for obs in trace["observations"] if obs["name"] == "generation"]
    assert len(generations) == 1

    generation_obs = generations[0]
    assert generation_obs["name"] == "generation"
    assert generation_obs["traceId"] == trace["id"]


def test_create_span_and_get_observation():
    langfuse = Langfuse()

    # Create span
    span = langfuse.start_span(name="span")

    # Get ID for verification
    span_id = span.id

    # End span
    span.end()

    # Flush and wait
    langfuse.flush()
    sleep(2)

    # Use API to fetch the observation by ID
    observation = get_api().observations.get(span_id)

    # Verify observation properties
    assert observation.name == "span"
    assert observation.id == span_id


def test_update_generation():
    langfuse = Langfuse()

    # Create a generation
    generation = langfuse.start_generation(name="generation")

    # Update generation with metadata
    generation.update(metadata={"dict": "value"})

    # Get ID for verification
    trace_id = generation.trace_id

    # End the generation
    generation.end()

    # Ensure data is sent
    langfuse.flush()
    sleep(2)

    # Retrieve and verify
    trace = get_api().trace.get(trace_id)

    # Verify trace properties
    assert trace.name == "generation"
    assert len(trace.observations) == 1

    # Verify generation updates
    retrieved_generation = trace.observations[0]
    assert retrieved_generation.name == "generation"
    assert retrieved_generation.trace_id == trace_id
    assert retrieved_generation.metadata["dict"] == "value"

    # Note: With OTEL, we can't verify exact start times from manually set timestamps,
    # as they are managed internally by the OTEL SDK


def test_update_span():
    langfuse = Langfuse()

    # Create a span
    span = langfuse.start_span(name="span")

    # Update the span with metadata
    span.update(metadata={"dict": "value"})

    # Get ID for verification
    trace_id = span.trace_id

    # End the span
    span.end()

    # Ensure data is sent
    langfuse.flush()
    sleep(2)

    # Retrieve and verify
    trace = get_api().trace.get(trace_id)

    # Verify trace properties
    assert trace.name == "span"
    assert len(trace.observations) == 1

    # Verify span updates
    retrieved_span = trace.observations[0]
    assert retrieved_span.name == "span"
    assert retrieved_span.trace_id == trace_id
    assert retrieved_span.metadata["dict"] == "value"


def test_create_span_and_generation():
    langfuse = Langfuse()

    # Create initial span
    span = langfuse.start_span(name="span")
    sleep(0.1)
    # Get trace ID for later use
    trace_id = span.trace_id
    # End the span
    span.end()

    # Create generation in the same trace
    generation = langfuse.start_generation(
        name="generation", trace_context={"trace_id": trace_id}
    )
    # End the generation
    generation.end()

    # Ensure data is sent
    langfuse.flush()
    sleep(2)

    # Retrieve and verify
    trace = get_api().trace.get(trace_id)

    # Verify trace details
    assert len(trace.observations) == 2

    # Find span and generation
    spans = [obs for obs in trace.observations if obs.name == "span"]
    generations = [obs for obs in trace.observations if obs.name == "generation"]

    assert len(spans) == 1
    assert len(generations) == 1

    # Verify both observations belong to the same trace
    span_obs = spans[0]
    gen_obs = generations[0]

    assert span_obs.trace_id == trace_id
    assert gen_obs.trace_id == trace_id


def test_create_trace_with_id_and_generation():
    langfuse = Langfuse()
    api_wrapper = LangfuseAPI()

    trace_name = create_uuid()

    # Create a trace ID using the utility method
    trace_id = langfuse.create_trace_id()

    # Create a span in this trace using the trace context
    with langfuse.start_as_current_span(
        name="parent-span", trace_context={"trace_id": trace_id}
    ) as parent_span:
        # Set trace name
        parent_span.update_trace(name=trace_name)

        # Create a generation in the same trace
        generation = parent_span.start_generation(name="generation")

        # End the generation
        generation.end()

    # Ensure data is sent
    langfuse.flush()
    sleep(2)

    # Retrieve and verify
    trace = api_wrapper.get_trace(trace_id)

    # Verify trace properties
    assert trace["name"] == trace_name
    assert trace["id"] == trace_id
    assert len(trace["observations"]) == 2  # Parent span and generation

    # Find the generation
    generations = [obs for obs in trace["observations"] if obs["name"] == "generation"]
    assert len(generations) == 1

    gen = generations[0]
    assert gen["name"] == "generation"
    assert gen["traceId"] == trace["id"]


def test_end_generation():
    langfuse = Langfuse()
    api_wrapper = LangfuseAPI()

    # Create a generation
    generation = langfuse.start_generation(
        name="query-generation",
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

    # Get trace ID for verification
    trace_id = generation.trace_id

    # Explicitly end the generation
    generation.end()

    # Ensure data is sent
    langfuse.flush()
    sleep(2)

    # Retrieve and verify
    trace = api_wrapper.get_trace(trace_id)

    # Find generation by name
    generations = [
        obs for obs in trace["observations"] if obs["name"] == "query-generation"
    ]
    assert len(generations) == 1

    gen = generations[0]
    assert gen["endTime"] is not None


def test_end_generation_with_data():
    langfuse = Langfuse()

    # Create a parent span to set trace properties
    with langfuse.start_as_current_span(name="parent-span") as parent_span:
        # Get trace ID
        trace_id = parent_span.trace_id

        # Create generation
        generation = langfuse.start_generation(
            name="query-generation",
        )

        # End generation with detailed properties
        generation.update(
            metadata={"dict": "value"},
            level="ERROR",
            status_message="Generation ended",
            version="1.0",
            completion_start_time=datetime(2023, 1, 1, 12, 3, tzinfo=timezone.utc),
            model="test-model",
            model_parameters={"param1": "value1", "param2": "value2"},
            input=[{"test_input_key": "test_input_value"}],
            output={"test_output_key": "test_output_value"},
            usage_details={
                "input": 100,
                "output": 200,
                "total": 500,
            },
            cost_details={
                "input": 111,
                "output": 222,
                "total": 444,
            },
        )

        # End the generation
        generation.end()

    # Ensure data is sent
    langfuse.flush()
    sleep(2)

    # Retrieve and verify
    fetched_trace = get_api().trace.get(trace_id)

    # Find generation by name
    generations = [
        obs for obs in fetched_trace.observations if obs.name == "query-generation"
    ]
    assert len(generations) == 1

    generation = generations[0]

    # Verify properties were updated
    assert generation.completion_start_time == datetime(
        2023, 1, 1, 12, 3, tzinfo=timezone.utc
    )
    assert generation.name == "query-generation"
    assert generation.metadata["dict"] == "value"
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

    # Create a generation
    generation = langfuse.start_generation(
        name="query-generation",
    )

    # Get trace ID for verification
    trace_id = generation.trace_id

    # Update with OpenAI-style token format
    generation.update(
        usage_details={
            "prompt_tokens": 100,  # OpenAI format
            "completion_tokens": 200,  # OpenAI format
            "total_tokens": 500,  # OpenAI format
        },
        cost_details={
            "input": 111,
            "output": 222,
            "total": 444,
        },
    )

    # End the generation
    generation.end()

    # Ensure data is sent
    langfuse.flush()
    sleep(2)

    # Retrieve and verify
    trace = get_api().trace.get(trace_id)

    # Find generation
    generations = [obs for obs in trace.observations if obs.name == "query-generation"]
    assert len(generations) == 1

    generation_api = generations[0]

    # Verify properties were converted correctly
    assert generation_api.end_time is not None
    assert generation_api.usage.input == 100  # prompt_tokens mapped to input
    assert generation_api.usage.output == 200  # completion_tokens mapped to output
    assert generation_api.usage.total == 500
    assert generation_api.usage.unit == "TOKENS"  # Default unit for OpenAI format
    assert generation_api.calculated_input_cost == 111
    assert generation_api.calculated_output_cost == 222
    assert generation_api.calculated_total_cost == 444


def test_end_span():
    langfuse = Langfuse()
    api_wrapper = LangfuseAPI()

    # Create a span
    span = langfuse.start_span(
        name="span",
        input={"key": "value"},
        output={"key": "value"},
        metadata={"interface": "whatsapp"},
    )

    # Get trace ID for verification
    trace_id = span.trace_id

    # Explicitly end the span
    span.end()

    # Ensure data is sent
    langfuse.flush()
    sleep(2)

    # Retrieve and verify
    trace = api_wrapper.get_trace(trace_id)

    # Find span
    spans = [obs for obs in trace["observations"] if obs["name"] == "span"]
    assert len(spans) == 1

    span_api = spans[0]

    # Verify end time was set
    assert span_api["endTime"] is not None


def test_end_span_with_data():
    langfuse = Langfuse()

    # Create a span
    span = langfuse.start_span(
        name="span",
        input={"key": "value"},
        output={"key": "value"},
        metadata={"interface": "whatsapp"},
    )

    # Get trace ID for verification
    trace_id = span.trace_id

    # Update span with metadata then end it
    span.update(metadata={"dict": "value"})
    span.end()

    # Ensure data is sent
    langfuse.flush()
    sleep(2)

    # Retrieve and verify
    trace = get_api().trace.get(trace_id)

    # Find span
    spans = [obs for obs in trace.observations if obs.name == "span"]
    assert len(spans) == 1

    span_api = spans[0]

    # Verify end time and metadata were updated
    assert span_api.end_time is not None
    assert span_api.metadata["dict"] == "value"
    assert span_api.metadata["interface"] == "whatsapp"


def test_get_generations():
    langfuse = Langfuse()

    # Create a first generation with random name
    generation1 = langfuse.start_generation(
        name=create_uuid(),
    )
    generation1.end()

    # Create a second generation with specific name and content
    generation_name = create_uuid()

    generation2 = langfuse.start_generation(
        name=generation_name,
        input="great-prompt",
        output="great-completion",
    )
    generation2.end()

    # Ensure data is sent
    langfuse.flush()
    sleep(3)

    # Fetch generations using API
    generations = get_api().observations.get_many(name=generation_name)

    # Verify fetched generation matches what we created
    assert len(generations.data) == 1
    assert generations.data[0].name == generation_name
    assert generations.data[0].input == "great-prompt"
    assert generations.data[0].output == "great-completion"


def test_get_generations_by_user():
    langfuse = Langfuse()

    # Generate unique IDs for test
    user_id = create_uuid()
    generation_name = create_uuid()

    # Create a trace with user ID and a generation as its child
    with langfuse.start_as_current_span(name="test-user") as parent_span:
        # Set user ID on the trace
        parent_span.update_trace(name="test-user", user_id=user_id)

        # Create a generation within the trace
        generation = parent_span.start_generation(
            name=generation_name,
            input="great-prompt",
            output="great-completion",
        )
        generation.end()

    # Create another generation that doesn't have this user ID
    other_gen = langfuse.start_generation(name="other-generation")
    other_gen.end()

    # Ensure data is sent
    langfuse.flush()
    sleep(3)

    # Fetch generations by user ID using the API
    generations = get_api().observations.get_many(user_id=user_id, type="GENERATION")

    # Verify fetched generation matches what we created
    assert len(generations.data) == 1
    assert generations.data[0].name == generation_name
    assert generations.data[0].input == "great-prompt"
    assert generations.data[0].output == "great-completion"


def test_kwargs():
    langfuse = Langfuse()

    # Create kwargs dict with valid parameters for start_span
    kwargs_dict = {
        "input": {"key": "value"},
        "output": {"key": "value"},
        "metadata": {"interface": "whatsapp"},
    }

    # Create span with specific kwargs instead of using **kwargs_dict
    span = langfuse.start_span(
        name="span",
        input=kwargs_dict["input"],
        output=kwargs_dict["output"],
        metadata=kwargs_dict["metadata"],
    )

    # Get ID for verification
    span_id = span.id

    # End span
    span.end()

    # Ensure data is sent
    langfuse.flush()
    sleep(2)

    # Retrieve and verify
    observation = get_api().observations.get(span_id)

    # Verify kwargs were properly set as attributes
    assert observation.start_time is not None
    assert observation.input == {"key": "value"}
    assert observation.output == {"key": "value"}
    assert observation.metadata["interface"] == "whatsapp"


@pytest.mark.skip("Flaky")
def test_timezone_awareness():
    os.environ["TZ"] = "US/Pacific"
    time.tzset()

    # Get current time in UTC for comparison
    utc_now = datetime.now(timezone.utc)
    assert utc_now.tzinfo is not None

    # Create Langfuse client
    langfuse = Langfuse()

    # Create a trace with various observation types
    with langfuse.start_as_current_span(name="test") as parent_span:
        # Set the trace name
        parent_span.update_trace(name="test")

        # Get trace ID for verification
        trace_id = parent_span.trace_id

        # Create a span
        span = parent_span.start_span(name="span")
        span.end()

        # Create a generation
        generation = parent_span.start_generation(name="generation")
        generation.end()

        # In OTEL-based client, "events" are just spans with minimal duration
        event_span = parent_span.start_span(name="event")
        event_span.end()

    # Ensure data is sent
    langfuse.flush()
    sleep(2)

    # Retrieve and verify
    trace = get_api().trace.get(trace_id)

    # Verify timestamps are in UTC regardless of local timezone
    assert (
        len(trace.observations) == 4
    )  # Parent span, child span, generation, and event
    for observation in trace.observations:
        # Check that start_time is within 5 seconds of current time
        delta = observation.start_time - utc_now
        assert delta.seconds < 5

        # Check end_time for all observations (in OTEL client, all spans have end time)
        delta = observation.end_time - utc_now
        assert delta.seconds < 5

    # Reset timezone
    os.environ["TZ"] = "UTC"
    time.tzset()


def test_timezone_awareness_setting_timestamps():
    # Note: In the OTEL-based client, timestamps are handled by the OTEL SDK
    # and we can't directly set custom timestamps for spans. Instead, we'll
    # verify that timestamps are properly converted to UTC regardless of local timezone.

    os.environ["TZ"] = "US/Pacific"
    time.tzset()

    # Get current time in various formats
    utc_now = datetime.now(timezone.utc)  # UTC time
    assert utc_now.tzinfo is not None

    # Create client
    langfuse = Langfuse()

    # Create a trace with different observation types
    with langfuse.start_as_current_span(name="test") as parent_span:
        # Set trace name
        parent_span.update_trace(name="test")

        # Get trace ID for verification
        trace_id = parent_span.trace_id

        # Create span
        span = parent_span.start_span(name="span")
        span.end()

        # Create generation
        generation = parent_span.start_generation(name="generation")
        generation.end()

        # Create event-like span
        event_span = parent_span.start_span(name="event")
        event_span.end()

    # Ensure data is sent
    langfuse.flush()
    sleep(2)

    # Retrieve and verify
    trace = get_api().trace.get(trace_id)

    # Verify timestamps are in UTC regardless of local timezone
    assert (
        len(trace.observations) == 4
    )  # Parent span, child span, generation, and event
    for observation in trace.observations:
        # Check that start_time is within 5 seconds of current time
        delta = abs((utc_now - observation.start_time).total_seconds())
        assert delta < 5

        # Check that end_time is within 5 seconds of current time
        delta = abs((utc_now - observation.end_time).total_seconds())
        assert delta < 5

    # Reset timezone
    os.environ["TZ"] = "UTC"
    time.tzset()


def test_get_trace_by_session_id():
    langfuse = Langfuse()

    # Create unique IDs for test
    trace_name = create_uuid()
    session_id = create_uuid()

    # Create a trace with a session_id
    with langfuse.start_as_current_span(name="test-span") as span:
        span.update_trace(name=trace_name, session_id=session_id)
        # Get trace ID for verification
        trace_id = span.trace_id

    # Create another trace without a session_id
    with langfuse.start_as_current_span(name=create_uuid()):
        pass

    # Ensure data is sent
    langfuse.flush()
    sleep(2)

    # Retrieve the trace using the session_id
    traces = get_api().trace.list(session_id=session_id)

    # Verify that the trace was retrieved correctly
    assert len(traces.data) == 1
    retrieved_trace = traces.data[0]
    assert retrieved_trace.name == trace_name
    assert retrieved_trace.session_id == session_id
    assert retrieved_trace.id == trace_id


def test_fetch_trace():
    langfuse = Langfuse()

    # Create a trace
    name = create_uuid()

    # Create a span and set trace properties
    with langfuse.start_as_current_span(name="test-span") as span:
        span.update_trace(name=name)
        # Get trace ID for verification
        trace_id = span.trace_id

    # Ensure data is sent
    langfuse.flush()
    sleep(2)

    # Fetch the trace using the get_api client
    # Note: In the OTEL-based client, we use the API client directly
    trace = get_api().trace.get(trace_id)

    # Verify trace properties
    assert trace.id == trace_id
    assert trace.name == name


def test_fetch_traces():
    langfuse = Langfuse()

    # Use a unique name for this test
    name = create_uuid()

    # Create 3 traces with different properties, but same name
    trace_ids = []

    # First trace
    with langfuse.start_as_current_span(name="test1") as span:
        span.update_trace(
            name=name,
            session_id="session-1",
            input={"key": "value"},
            output="output-value",
        )
        trace_ids.append(span.trace_id)

    sleep(1)  # Ensure traces have different timestamps

    # Second trace
    with langfuse.start_as_current_span(name="test2") as span:
        span.update_trace(
            name=name,
            session_id="session-1",
            input={"key": "value"},
            output="output-value",
        )
        trace_ids.append(span.trace_id)

    sleep(1)  # Ensure traces have different timestamps

    # Third trace
    with langfuse.start_as_current_span(name="test3") as span:
        span.update_trace(
            name=name,
            session_id="session-1",
            input={"key": "value"},
            output="output-value",
        )
        trace_ids.append(span.trace_id)

    # Ensure data is sent
    langfuse.flush()
    sleep(3)

    # Fetch all traces with the same name
    # Note: Using session_id in the query is causing a server error,
    # but we keep the session_id in the trace data to ensure it's being stored correctly
    all_traces = get_api().trace.list(name=name, limit=10)

    # Verify we got all traces
    assert len(all_traces.data) == 3
    assert all_traces.meta.total_items == 3

    # Verify trace properties
    for trace in all_traces.data:
        assert trace.name == name
        assert trace.session_id == "session-1"
        assert trace.input == {"key": "value"}
        assert trace.output == "output-value"

    # Test pagination by fetching just one trace
    paginated_response = get_api().trace.list(name=name, limit=1, page=2)
    assert len(paginated_response.data) == 1
    assert paginated_response.meta.total_items == 3
    assert paginated_response.meta.total_pages == 3


def test_get_observation():
    langfuse = Langfuse()

    # Create a trace and a generation
    name = create_uuid()

    # Create a span and set trace properties
    with langfuse.start_as_current_span(name="parent-span") as parent_span:
        parent_span.update_trace(name=name)

        # Create a generation as child
        generation = parent_span.start_generation(name=name)

        # Get IDs for verification
        generation_id = generation.id

        # End the generation
        generation.end()

    # Ensure data is sent
    langfuse.flush()
    sleep(2)

    # Fetch the observation using the API
    observation = get_api().observations.get(generation_id)

    # Verify observation properties
    assert observation.id == generation_id
    assert observation.name == name
    assert observation.type == "GENERATION"


def test_get_observations():
    langfuse = Langfuse()

    # Create a trace with multiple generations
    name = create_uuid()

    # Create a span and set trace properties
    with langfuse.start_as_current_span(name="parent-span") as parent_span:
        parent_span.update_trace(name=name)

        # Create first generation
        gen1 = parent_span.start_generation(name=name)
        gen1_id = gen1.id
        gen1.end()

        # Create second generation
        gen2 = parent_span.start_generation(name=name)
        gen2_id = gen2.id
        gen2.end()

    # Ensure data is sent
    langfuse.flush()
    sleep(2)

    # Fetch observations using the API
    observations = get_api().observations.get_many(name=name, limit=10)

    # Verify fetched observations
    assert len(observations.data) == 2

    # Filter for just the generations
    generations = [obs for obs in observations.data if obs.type == "GENERATION"]
    assert len(generations) == 2

    # Verify the generation IDs match what we created
    gen_ids = [gen.id for gen in generations]
    assert gen1_id in gen_ids
    assert gen2_id in gen_ids

    # Test pagination
    paginated_response = get_api().observations.get_many(name=name, limit=1, page=2)
    assert len(paginated_response.data) == 1
    assert paginated_response.meta.total_items == 2  # Parent span + 2 generations
    assert paginated_response.meta.total_pages == 2


def test_get_trace_not_found():
    # Attempt to fetch a non-existent trace using the API
    with pytest.raises(Exception):
        get_api().trace.get(create_uuid())


def test_get_observation_not_found():
    # Attempt to fetch a non-existent observation using the API
    with pytest.raises(Exception):
        get_api().observations.get(create_uuid())


def test_get_traces_empty():
    # Fetch traces with a filter that should return no results
    response = get_api().trace.list(name=create_uuid())

    assert len(response.data) == 0
    assert response.meta.total_items == 0


def test_get_observations_empty():
    # Fetch observations with a filter that should return no results
    response = get_api().observations.get_many(name=create_uuid())

    assert len(response.data) == 0
    assert response.meta.total_items == 0


def test_get_sessions():
    langfuse = Langfuse()

    # unique name
    name = create_uuid()
    session1 = create_uuid()
    session2 = create_uuid()
    session3 = create_uuid()

    # Create multiple traces with different session IDs
    # Create first trace
    with langfuse.start_as_current_span(name=name) as span1:
        span1.update_trace(name=name, session_id=session1)

    # Create second trace
    with langfuse.start_as_current_span(name=name) as span2:
        span2.update_trace(name=name, session_id=session2)

    # Create third trace
    with langfuse.start_as_current_span(name=name) as span3:
        span3.update_trace(name=name, session_id=session3)

    langfuse.flush()

    # Fetch sessions
    sleep(3)
    response = get_api().sessions.list()

    # Assert the structure of the response, cannot check for the exact number of sessions as the table is not cleared between tests
    assert hasattr(response, "data")
    assert hasattr(response, "meta")
    assert isinstance(response.data, list)

    # fetch only one, cannot check for the exact number of sessions as the table is not cleared between tests
    response = get_api().sessions.list(limit=1, page=2)
    assert len(response.data) == 1


@pytest.mark.skip(
    "Flaky in concurrent environment as the global tracer provider is already configured"
)
def test_create_trace_sampling_zero():
    langfuse = Langfuse(sample_rate=0)
    api_wrapper = LangfuseAPI()
    trace_name = create_uuid()

    # Create a span with trace properties - with sample_rate=0, this will not be sent to the API
    with langfuse.start_as_current_span(name="test-span") as span:
        span.update_trace(
            name=trace_name,
            user_id="test",
            metadata={"key": "value"},
            tags=["tag1", "tag2"],
            public=True,
        )
        # Get trace ID for verification
        trace_id = span.trace_id

        # Add a score and a child generation
        langfuse.score_current_trace(name="score", value=0.5)
        generation = span.start_generation(name="generation")
        generation.end()

    # Ensure data is sent, but should be dropped due to sampling
    langfuse.flush()
    sleep(2)

    # Try to fetch the trace - should fail as it wasn't sent to the API
    fetched_trace = api_wrapper.get_trace(trace_id)
    assert fetched_trace == {
        "error": "LangfuseNotFoundError",
        "message": f"Trace {trace_id} not found within authorized project",
    }


def test_mask_function():
    LangfuseResourceManager.reset()

    def mask_func(data):
        if isinstance(data, dict):
            if "should_raise" in data:
                raise
            return {k: "MASKED" for k in data}
        elif isinstance(data, str):
            return "MASKED"
        return data

    langfuse = Langfuse(mask=mask_func)
    api_wrapper = LangfuseAPI()

    # Create a root span with trace properties
    with langfuse.start_as_current_span(name="test-span") as root_span:
        root_span.update_trace(name="test_trace", input={"sensitive": "data"})
        # Get trace ID for later use
        trace_id = root_span.trace_id
        # Add output to the trace
        root_span.update_trace(output={"more": "sensitive"})

        # Create a generation as child
        gen = root_span.start_generation(name="test_gen", input={"prompt": "secret"})
        gen.update(output="new_confidential")
        gen.end()

        # Create a span as child
        sub_span = root_span.start_span(name="test_span", input={"data": "private"})
        sub_span.update(output="new_classified")
        sub_span.end()

    # Ensure data is sent
    langfuse.flush()
    sleep(2)

    # Retrieve and verify
    fetched_trace = api_wrapper.get_trace(trace_id)
    assert fetched_trace["input"] == {"sensitive": "MASKED"}
    assert fetched_trace["output"] == {"more": "MASKED"}

    fetched_gen = [
        o for o in fetched_trace["observations"] if o["type"] == "GENERATION"
    ][0]
    assert fetched_gen["input"] == {"prompt": "MASKED"}
    assert fetched_gen["output"] == "MASKED"

    fetched_span = [
        o
        for o in fetched_trace["observations"]
        if o["type"] == "SPAN" and o["name"] == "test_span"
    ][0]
    assert fetched_span["input"] == {"data": "MASKED"}
    assert fetched_span["output"] == "MASKED"

    # Create a root span with trace properties
    with langfuse.start_as_current_span(name="test-span") as root_span:
        root_span.update_trace(name="test_trace", input={"should_raise": "data"})
        # Get trace ID for later use
        trace_id = root_span.trace_id
        # Add output to the trace
        root_span.update_trace(output={"should_raise": "sensitive"})

    # Ensure data is sent
    langfuse.flush()
    sleep(2)

    # Retrieve and verify
    fetched_trace = api_wrapper.get_trace(trace_id)
    assert fetched_trace["input"] == "<fully masked due to failed mask function>"
    assert fetched_trace["output"] == "<fully masked due to failed mask function>"


def test_get_project_id():
    langfuse = Langfuse()
    res = langfuse._get_project_id()
    assert res is not None
    assert res == "7a88fb47-b4e2-43b8-a06c-a5ce950dc53a"


def test_generate_trace_id():
    langfuse = Langfuse()
    trace_id = langfuse.create_trace_id()

    # Create a trace with the specific ID using trace_context
    with langfuse.start_as_current_span(
        name="test-span", trace_context={"trace_id": trace_id}
    ) as span:
        span.update_trace(name="test_trace")

    langfuse.flush()

    # Test the trace URL generation
    project_id = langfuse._get_project_id()
    trace_url = langfuse.get_trace_url(trace_id=trace_id)
    assert trace_url == f"http://localhost:3000/project/{project_id}/traces/{trace_id}"


def test_start_as_current_observation_types():
    """Test creating different observation types using start_as_current_observation."""
    langfuse = Langfuse()

    observation_types = [
        "span",
        "generation",
        "agent",
        "tool",
        "chain",
        "retriever",
        "evaluator",
        "embedding",
        "guardrail",
    ]

    with langfuse.start_as_current_span(name="parent") as parent_span:
        parent_span.update_trace(name="observation-types-test")
        trace_id = parent_span.trace_id

        for obs_type in observation_types:
            with parent_span.start_as_current_observation(
                name=f"test-{obs_type}", as_type=obs_type
            ):
                pass

    langfuse.flush()
    sleep(2)

    api = get_api()
    trace = api.trace.get(trace_id)

    # Check we have all expected observation types
    found_types = {obs.type for obs in trace.observations}
    expected_types = {obs_type.upper() for obs_type in observation_types} | {
        "SPAN"
    }  # includes parent span
    assert expected_types.issubset(found_types), (
        f"Missing types: {expected_types - found_types}"
    )

    # Verify each specific observation exists
    for obs_type in observation_types:
        observations = [
            obs
            for obs in trace.observations
            if obs.name == f"test-{obs_type}" and obs.type == obs_type.upper()
        ]
        assert len(observations) == 1, f"Expected one {obs_type.upper()} observation"


def test_that_generation_like_properties_are_actually_created():
    """Test that generation-like observation types properly support generation properties."""
    from langfuse._client.constants import (
        ObservationTypeGenerationLike,
        get_observation_types_list,
    )

    langfuse = Langfuse()
    generation_like_types = get_observation_types_list(ObservationTypeGenerationLike)

    test_model = "test-model"
    test_completion_start_time = datetime.now(timezone.utc)
    test_model_parameters = {"temperature": "0.7", "max_tokens": "100"}
    test_usage_details = {"prompt_tokens": 10, "completion_tokens": 20}
    test_cost_details = {"input": 0.01, "output": 0.02, "total": 0.03}

    with langfuse.start_as_current_span(name="parent") as parent_span:
        parent_span.update_trace(name="generation-properties-test")
        trace_id = parent_span.trace_id

        for obs_type in generation_like_types:
            with parent_span.start_as_current_observation(
                name=f"test-{obs_type}",
                as_type=obs_type,
                model=test_model,
                completion_start_time=test_completion_start_time,
                model_parameters=test_model_parameters,
                usage_details=test_usage_details,
                cost_details=test_cost_details,
            ) as obs:
                # Verify the properties are accessible on the observation object
                if hasattr(obs, "model"):
                    assert obs.model == test_model, (
                        f"{obs_type} should have model property"
                    )
                if hasattr(obs, "completion_start_time"):
                    assert obs.completion_start_time == test_completion_start_time, (
                        f"{obs_type} should have completion_start_time property"
                    )
                if hasattr(obs, "model_parameters"):
                    assert obs.model_parameters == test_model_parameters, (
                        f"{obs_type} should have model_parameters property"
                    )
                if hasattr(obs, "usage_details"):
                    assert obs.usage_details == test_usage_details, (
                        f"{obs_type} should have usage_details property"
                    )
                if hasattr(obs, "cost_details"):
                    assert obs.cost_details == test_cost_details, (
                        f"{obs_type} should have cost_details property"
                    )

    langfuse.flush()

    api = get_api()
    trace = api.trace.get(trace_id)

    # Verify that the properties are persisted in the API for generation-like types
    for obs_type in generation_like_types:
        observations = [
            obs
            for obs in trace.observations
            if obs.name == f"test-{obs_type}" and obs.type == obs_type.upper()
        ]
        assert len(observations) == 1, (
            f"Expected one {obs_type.upper()} observation, but found {len(observations)}"
        )

        obs = observations[0]

        assert obs.model == test_model, f"{obs_type} should have model property"
        assert obs.model_parameters == test_model_parameters, (
            f"{obs_type} should have model_parameters property"
        )

        # usage_details
        assert hasattr(obs, "usage_details"), f"{obs_type} should have usage_details"
        assert obs.usage_details == dict(test_usage_details, total=30), (
            f"{obs_type} should persist usage_details"
        )  # API adds total

        assert obs.cost_details == test_cost_details, (
            f"{obs_type} should persist cost_details"
        )

        # completion_start_time, because of time skew not asserting time
        assert obs.completion_start_time is not None, (
            f"{obs_type} should persist completion_start_time property"
        )
