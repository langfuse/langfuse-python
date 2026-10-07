import json
import os
import time
from asyncio import gather
from datetime import datetime, timedelta, timezone
from time import sleep

import pytest

from langfuse import Langfuse, propagate_attributes
from langfuse._client.resource_manager import LangfuseResourceManager
from langfuse._utils import _get_timestamp
from tests.support.utils import (
    create_uuid,
    get_api,
    get_observations,
    get_root_observation,
    get_scores,
    user_metadata,
    wait_for_observations,
    wait_for_root_observation,
    wait_for_scores,
)


@pytest.mark.asyncio
async def test_concurrency():
    _get_timestamp()

    async def update_generation(i, langfuse: Langfuse):
        # Create a new trace with a generation
        with langfuse.start_as_current_observation(name=f"parent-{i}"):
            with propagate_attributes(trace_name=str(i)):
                # Create generation as a child
                generation = langfuse.start_observation(
                    as_type="generation", name=str(i)
                )

                # Update generation with metadata
                generation.update(metadata={"count": str(i)})

                # End the generation
                generation.end()

                return generation.trace_id

    # Create Langfuse client
    langfuse = Langfuse()

    # Run concurrent operations
    trace_ids = await gather(*(update_generation(i, langfuse) for i in range(100)))

    langfuse.flush()

    # Verify that all spans were created properly
    for i, trace_id in enumerate(trace_ids):
        observations = wait_for_observations(trace_id, min_count=2)

        generation_obs = [obs for obs in observations if obs.type == "GENERATION"]
        assert len(generation_obs) == 1

        # Verify metadata
        observation = generation_obs[0]
        assert observation.name == str(i)
        assert user_metadata(observation)["count"] == i
        assert get_root_observation(observations).trace_name == str(i)


def test_flush():
    # Initialize Langfuse client with debug disabled
    langfuse = Langfuse()

    trace_ids = []
    for i in range(2):
        # Create spans and set the trace name using propagate_attributes
        with langfuse.start_as_current_observation(name="span-" + str(i)):
            with propagate_attributes(trace_name=str(i)):
                # Store the trace ID for later verification
                trace_ids.append(langfuse.get_current_trace_id())

    # Flush all pending spans to the Langfuse API
    langfuse.flush()

    # Verify traces were sent by checking they exist in the API
    for i, trace_id in enumerate(trace_ids):
        root = wait_for_root_observation(trace_id)
        assert root.trace_name == str(i)


def test_invalid_score_data_does_not_raise_exception():
    langfuse = Langfuse()

    # Create a span and set trace properties
    with langfuse.start_as_current_observation(name="test-span") as span:
        with propagate_attributes(
            trace_name="this-is-so-great-new",
            user_id="test",
            metadata={"test": "test"},
        ):
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


def test_create_session_score():
    langfuse = Langfuse()

    session_id = "my-session"

    # Create a span and set trace properties
    with langfuse.start_as_current_observation(name="test-span"):
        with propagate_attributes(
            trace_name="this-is-so-great-new",
            user_id="test",
            metadata={"test": "test"},
            session_id=session_id,
        ):
            pass

    # Ensure data is sent
    langfuse.flush()
    sleep(2)

    # Create a numeric score
    score_id = create_uuid()

    langfuse.create_score(
        score_id=score_id,
        session_id=session_id,
        name="this-is-a-score",
        value=1,
    )

    # Ensure data is sent
    langfuse.flush()

    # Retrieve and verify
    scores = wait_for_scores(session_id=session_id, id=score_id)

    assert len(scores) == 1
    score = scores[0]
    assert score.value == 1
    assert score.data_type == "NUMERIC"
    assert score.subject.kind == "session"
    assert score.subject.id == session_id


def test_create_numeric_score():
    langfuse = Langfuse()

    # Create a span and set trace properties
    with langfuse.start_as_current_observation(name="test-span") as span:
        with propagate_attributes(
            trace_name="this-is-so-great-new",
            user_id="test",
            metadata={"test": "test"},
        ):
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
    generation = langfuse.start_observation(
        as_type="generation",
        name="yet another child",
        metadata="test",
        trace_context={"trace_id": trace_id},
    )
    generation.end()

    # Ensure data is sent
    langfuse.flush()

    # Retrieve and verify
    wait_for_observations(trace_id, min_count=2)
    scores = wait_for_scores(trace_id=trace_id, name="this-is-a-score")

    assert len(scores) == 1
    score = scores[0]
    assert score.id == score_id
    assert score.value == 1
    assert score.data_type == "NUMERIC"
    assert score.subject.kind == "trace"


def test_create_boolean_score():
    langfuse = Langfuse()

    # Create a span and set trace properties
    with langfuse.start_as_current_observation(name="test-span") as span:
        with propagate_attributes(
            trace_name="this-is-so-great-new",
            user_id="test",
            metadata={"test": "test"},
        ):
            # Get trace ID for later use
            trace_id = span.trace_id

    # Ensure data is sent
    langfuse.flush()
    wait_for_root_observation(trace_id)

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
    generation = langfuse.start_observation(
        as_type="generation",
        name="yet another child",
        metadata="test",
        trace_context={"trace_id": trace_id},
    )
    generation.end()

    # Ensure data is sent
    langfuse.flush()

    # Retrieve and verify
    scores = wait_for_scores(trace_id=trace_id, name="this-is-a-score")

    assert len(scores) == 1, "Score not found in trace"
    created_score = scores[0]
    assert created_score.id == score_id
    assert created_score.data_type == "BOOLEAN"
    assert created_score.value is True


def test_create_categorical_score():
    langfuse = Langfuse()

    # Create a span and set trace properties
    with langfuse.start_as_current_observation(name="test-span") as span:
        with propagate_attributes(
            trace_name="this-is-so-great-new",
            user_id="test",
            metadata={"test": "test"},
        ):
            # Get trace ID for later use
            trace_id = span.trace_id

    # Ensure data is sent
    langfuse.flush()
    wait_for_root_observation(trace_id)

    # Create a categorical score
    score_id = create_uuid()
    langfuse.create_score(
        score_id=score_id,
        trace_id=trace_id,
        name="this-is-a-score",
        value="high score",
    )

    # Create a generation in the same trace
    generation = langfuse.start_observation(
        as_type="generation",
        name="yet another child",
        metadata="test",
        trace_context={"trace_id": trace_id},
    )
    generation.end()

    # Ensure data is sent
    langfuse.flush()

    # Retrieve and verify
    scores = wait_for_scores(trace_id=trace_id, name="this-is-a-score")

    assert len(scores) == 1, "Score not found in trace"
    created_score = scores[0]
    assert created_score.id == score_id
    assert created_score.data_type == "CATEGORICAL"
    assert created_score.value == "high score"


def test_create_text_score():
    langfuse = Langfuse()

    # Create a span and set trace properties
    with langfuse.start_as_current_observation(name="test-span") as span:
        with propagate_attributes(
            trace_name="this-is-so-great-new",
            user_id="test",
            metadata={"test": "test"},
        ):
            # Get trace ID for later use
            trace_id = span.trace_id

    # Ensure data is sent
    langfuse.flush()
    sleep(2)

    # Create a text score
    score_id = create_uuid()
    langfuse.create_score(
        score_id=score_id,
        trace_id=trace_id,
        name="this-is-a-score",
        value="This is a detailed text evaluation of the output quality.",
        data_type="TEXT",
    )

    # Create a generation in the same trace
    generation = langfuse.start_observation(
        as_type="generation",
        name="yet another child",
        metadata="test",
        trace_context={"trace_id": trace_id},
    )
    generation.end()

    # Ensure data is sent
    langfuse.flush()

    # Retrieve and verify
    scores = wait_for_scores(trace_id=trace_id, name="this-is-a-score")

    assert len(scores) == 1, "Score not found in trace"
    created_score = scores[0]
    assert created_score.id == score_id
    assert created_score.data_type == "TEXT"
    assert (
        created_score.value
        == "This is a detailed text evaluation of the output quality."
    )


def test_create_score_with_custom_timestamp():
    langfuse = Langfuse()

    # Create a span and set trace properties
    with langfuse.start_as_current_observation(name="test-span") as span:
        with propagate_attributes(
            trace_name="test-custom-timestamp",
            user_id="test",
            metadata={"test": "test"},
        ):
            # Get trace ID for later use
            trace_id = span.trace_id

    # Ensure data is sent
    langfuse.flush()
    wait_for_root_observation(trace_id)

    custom_timestamp = datetime.now(timezone.utc) - timedelta(hours=1)
    score_id = create_uuid()
    langfuse.create_score(
        score_id=score_id,
        trace_id=trace_id,
        name="custom-timestamp-score",
        value=0.85,
        data_type="NUMERIC",
        timestamp=custom_timestamp,
    )

    # Ensure data is sent
    langfuse.flush()

    # Retrieve and verify
    scores = wait_for_scores(trace_id=trace_id, name="custom-timestamp-score")

    assert len(scores) == 1, "Score not found in trace"
    created_score = scores[0]
    assert created_score.id == score_id
    assert created_score.data_type == "NUMERIC"
    assert created_score.value == 0.85

    # Verify timestamp is close to our custom timestamp
    response_timestamp = created_score.timestamp

    # Check that the timestamps are within 1 second of each other
    # (allowing for some processing time and rounding)
    time_diff = abs((response_timestamp - custom_timestamp).total_seconds())
    assert time_diff < 1, (
        f"Timestamp difference too large: {time_diff}s. Expected < 1s. Custom: {custom_timestamp}, Response: {response_timestamp}"
    )


def test_create_trace():
    langfuse = Langfuse()
    trace_name = create_uuid()

    # Create a span and set the trace properties using propagate_attributes
    with langfuse.start_as_current_observation(name="test-span") as span:
        with propagate_attributes(
            trace_name=trace_name,
            user_id="test",
            metadata={"key": "value"},
            tags=["tag1", "tag2"],
        ):
            span.set_trace_as_public()
            # Get trace ID for later verification
            trace_id = langfuse.get_current_trace_id()

    # Ensure data is sent to the API
    langfuse.flush()

    # Retrieve the trace from the API
    root = wait_for_root_observation(trace_id)

    # Verify all trace properties
    assert root.trace_name == trace_name
    assert root.user_id == "test"
    assert user_metadata(root) == {"key": "value"}
    assert root.tags == ["tag1", "tag2"]
    assert root.public is True


def test_create_update_trace():
    langfuse = Langfuse()

    trace_name = create_uuid()

    # Create initial span with trace properties
    with langfuse.start_as_current_observation(name="test-span") as span:
        with propagate_attributes(
            trace_name=trace_name,
            user_id="test",
            metadata={"key": "value"},
        ):
            span.set_trace_as_public()
            # Get trace ID for later reference
            trace_id = span.trace_id

            # Allow a small delay before updating
            sleep(1)

            # Update trace properties with additional metadata
            with propagate_attributes(metadata={"key2": "value2"}):
                pass  # Metadata update only, set_trace_as_public is one-way

    # Ensure data is sent to the API
    langfuse.flush()

    assert isinstance(trace_id, str)
    # Retrieve and verify trace
    root = wait_for_root_observation(trace_id)

    assert root.trace_name == trace_name
    assert root.user_id == "test"
    assert user_metadata(root) == {"key": "value", "key2": "value2"}
    assert root.public is True


def test_create_update_current_trace():
    langfuse = Langfuse()

    trace_name = create_uuid()

    # Create initial span with trace properties using propagate_attributes
    with langfuse.start_as_current_observation(name="test-span-current") as span:
        with propagate_attributes(
            trace_name=trace_name,
            user_id="test",
            metadata={"key": "value"},
        ):
            langfuse.update_current_span(input="test_input")
            langfuse.set_current_trace_as_public()
            # Get trace ID for later reference
            trace_id = span.trace_id

            # Allow a small delay before updating
            sleep(1)

            # Update trace properties with additional metadata and version
            with propagate_attributes(metadata={"key2": "value2"}, version="1.0"):
                pass  # Metadata update only, publish is one-way

    # Ensure data is sent to the API
    langfuse.flush()

    assert isinstance(trace_id, str)
    # Retrieve and verify trace
    root = wait_for_root_observation(trace_id)

    # The 2nd update to the trace must not erase previously set attributes
    assert root.trace_name == trace_name
    assert root.user_id == "test"
    assert user_metadata(root) == {"key": "value", "key2": "value2"}
    assert root.public is True
    assert root.version == "1.0"
    assert root.input == "test_input"


def test_create_generation():
    langfuse = Langfuse()

    # Create a generation using OTEL approach
    generation = langfuse.start_observation(
        as_type="generation",
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

    # Retrieve the trace from the API
    observations = wait_for_observations(trace_id)

    assert len(observations) == 1

    # Verify generation details
    generation_api = observations[0]

    # Verify trace details
    assert generation_api.is_root_observation is True
    assert generation_api.trace_name == "query-generation"
    assert generation_api.user_id is None

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

    generation = langfuse.start_observation(
        as_type="generation",
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
    observations = wait_for_observations(trace_id)

    assert len(observations) == 1

    generation_api = observations[0]

    assert generation_api.trace_name == "query-generation"
    assert generation_api.user_id is None
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

    assert user_metadata(generation_api) == {"tags": ["yo"]}

    assert generation_api.start_time is not None
    assert generation_api.usage_details == {"input": 51, "output": 0, "total": 100}


def test_create_span():
    langfuse = Langfuse()

    # Create span using OTEL-based client
    span = langfuse.start_observation(
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

    # Retrieve from API
    observations = wait_for_observations(trace_id)

    assert len(observations) == 1

    # Verify span details
    span_api = observations[0]

    # Verify trace details
    assert span_api.trace_name == "span"
    assert span_api.user_id is None

    assert span_api.id == span_id
    assert span_api.name == "span"
    assert span_api.start_time is not None
    assert span_api.end_time is not None
    assert span_api.input == {"key": "value"}
    assert span_api.output == {"key": "value"}
    assert span_api.start_time is not None


def test_score_trace():
    langfuse = Langfuse()

    trace_name = create_uuid()

    # Create a span and set trace name
    with langfuse.start_as_current_observation(name="test-span"):
        with propagate_attributes(trace_name=trace_name):
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

    # Retrieve and verify
    assert wait_for_root_observation(trace_id).trace_name == trace_name

    scores = wait_for_scores(trace_id=trace_id, name="valuation")
    assert len(scores) == 1
    score = scores[0]
    assert score.value == 0.5
    assert score.comment == "This is a comment"
    assert score.subject.kind == "trace"
    assert score.data_type == "NUMERIC"


def test_score_trace_nested_trace():
    langfuse = Langfuse()

    trace_name = create_uuid()

    # Create a trace with span
    with langfuse.start_as_current_observation(name="test-span") as span:
        with propagate_attributes(trace_name=trace_name):
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

    # Retrieve and verify
    assert wait_for_root_observation(trace_id).trace_name == trace_name

    scores = wait_for_scores(trace_id=trace_id, name="valuation")
    assert len(scores) == 1
    score = scores[0]
    assert score.value == 0.5
    assert score.comment == "This is a comment"
    assert score.subject.kind == "trace"
    assert score.data_type == "NUMERIC"


def test_score_trace_nested_observation():
    langfuse = Langfuse()

    trace_name = create_uuid()

    # Create a parent span and set trace name
    with langfuse.start_as_current_observation(name="parent-span") as parent_span:
        with propagate_attributes(trace_name=trace_name):
            # Create a child span
            child_span = langfuse.start_observation(name="span")

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

    # Retrieve and verify
    assert wait_for_root_observation(trace_id).trace_name == trace_name

    scores = wait_for_scores(trace_id=trace_id, name="valuation")
    assert len(scores) == 1
    score = scores[0]
    assert score.value == 0.5
    assert score.comment == "This is a comment"
    assert score.subject.kind == "observation"
    assert score.subject.id == child_span_id
    assert score.subject.trace_id == trace_id
    assert score.data_type == "NUMERIC"


def test_score_span():
    langfuse = Langfuse()

    # Create a span
    span = langfuse.start_observation(
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

    # Retrieve and verify
    assert len(wait_for_observations(trace_id)) == 1

    scores = wait_for_scores(trace_id=trace_id, observation_id=span_id)
    assert len(scores) == 1
    score = scores[0]
    assert score.name == "valuation"
    assert score.value == 1
    assert score.comment == "This is a comment"
    assert score.subject.kind == "observation"
    assert score.subject.id == span_id
    assert score.data_type == "NUMERIC"


def test_create_trace_and_span():
    langfuse = Langfuse()

    trace_name = create_uuid()

    # Create parent span and set trace name
    with langfuse.start_as_current_observation(name=trace_name) as parent_span:
        with propagate_attributes(trace_name=trace_name):
            # Create a child span
            child_span = parent_span.start_observation(name="span")

            # Get trace ID for verification
            trace_id = parent_span.trace_id

            # End the child span
            child_span.end()

    # Ensure data is sent
    langfuse.flush()

    # Retrieve and verify
    observations = wait_for_observations(trace_id, min_count=2)

    assert get_root_observation(observations).trace_name == trace_name
    assert len(observations) == 2  # Parent span and child span

    # Find the child span
    child_spans = [obs for obs in observations if obs.name == "span"]
    assert len(child_spans) == 1

    span = child_spans[0]
    assert span.name == "span"
    assert span.trace_id == trace_id
    assert span.start_time is not None


def test_create_trace_and_generation():
    langfuse = Langfuse()

    trace_name = create_uuid()

    # Create parent span and set trace properties
    with langfuse.start_as_current_observation(name=trace_name) as parent_span:
        with propagate_attributes(trace_name=trace_name, session_id="test-session-id"):
            parent_span.update(input={"key": "value"})

            # Create a generation as child
            generation = parent_span.start_observation(
                as_type="generation", name="generation"
            )

            # Get IDs for verification
            trace_id = parent_span.trace_id

            # End the generation
            generation.end()

    # Ensure data is sent
    langfuse.flush()

    # Retrieve and verify
    observations = wait_for_observations(trace_id, min_count=2)
    root = get_root_observation(observations)

    # Verify trace details
    assert root.trace_name == trace_name
    assert len(observations) == 2  # Parent span and generation
    assert root.session_id == "test-session-id"

    # Find the generation
    generations = [obs for obs in observations if obs.name == "generation"]
    assert len(generations) == 1

    generation = generations[0]
    assert generation.name == "generation"
    assert generation.trace_id == trace_id
    assert generation.session_id == "test-session-id"
    assert generation.start_time is not None
    assert root.input == {"key": "value"}


def test_create_generation_and_trace():
    langfuse = Langfuse()

    trace_name = create_uuid()

    # Create trace with a generation
    trace_context = {"trace_id": langfuse.create_trace_id()}

    # Create a generation with this context
    generation = langfuse.start_observation(
        as_type="generation",
        name="generation",
        trace_context=trace_context,
    )

    # Get trace ID for verification
    trace_id = generation.trace_id

    # End the generation
    generation.end()

    sleep(0.1)

    # Update trace properties in a separate span
    with langfuse.start_as_current_observation(
        name="trace-update", trace_context={"trace_id": trace_id}
    ):
        with propagate_attributes(trace_name=trace_name):
            pass

    # Ensure data is sent
    langfuse.flush()

    # Retrieve and verify
    observations = wait_for_observations(trace_id, min_count=2)

    # We should have 2 observations (the generation and the span for updating trace)
    assert len(observations) == 2

    trace_update_spans = [obs for obs in observations if obs.name == "trace-update"]
    assert len(trace_update_spans) == 1
    assert trace_update_spans[0].trace_name == trace_name

    # Find the generation
    generations = [obs for obs in observations if obs.name == "generation"]
    assert len(generations) == 1

    generation_obs = generations[0]
    assert generation_obs.name == "generation"
    assert generation_obs.trace_id == trace_id


def test_create_span_and_get_observation():
    langfuse = Langfuse()

    # Create span
    span = langfuse.start_observation(name="span")

    # Get ID for verification
    span_id = span.id

    # End span
    span.end()

    # Flush and wait
    langfuse.flush()

    observations = wait_for_observations(span.trace_id)

    # Verify observation properties
    assert len(observations) == 1
    observation = observations[0]
    assert observation.name == "span"
    assert observation.id == span_id


def test_update_generation():
    langfuse = Langfuse()

    # Create a generation
    generation = langfuse.start_observation(as_type="generation", name="generation")

    # Update generation with metadata
    generation.update(metadata={"dict": "value"})

    # Get ID for verification
    trace_id = generation.trace_id

    # End the generation
    generation.end()

    # Ensure data is sent
    langfuse.flush()

    # Retrieve and verify
    observations = wait_for_observations(trace_id)

    # Verify trace properties
    assert len(observations) == 1

    # Verify generation updates
    retrieved_generation = observations[0]
    assert retrieved_generation.trace_name == "generation"
    assert retrieved_generation.name == "generation"
    assert retrieved_generation.trace_id == trace_id
    assert user_metadata(retrieved_generation) == {"dict": "value"}

    # Note: With OTEL, we can't verify exact start times from manually set timestamps,
    # as they are managed internally by the OTEL SDK


def test_update_span():
    langfuse = Langfuse()

    # Create a span
    span = langfuse.start_observation(name="span")

    # Update the span with metadata
    span.update(metadata={"dict": "value"})

    # Get ID for verification
    trace_id = span.trace_id

    # End the span
    span.end()

    # Ensure data is sent
    langfuse.flush()

    # Retrieve and verify
    observations = wait_for_observations(trace_id)

    # Verify trace properties
    assert len(observations) == 1

    # Verify span updates
    retrieved_span = observations[0]
    assert retrieved_span.trace_name == "span"
    assert retrieved_span.name == "span"
    assert retrieved_span.trace_id == trace_id
    assert user_metadata(retrieved_span) == {"dict": "value"}


def test_create_span_and_generation():
    langfuse = Langfuse()

    # Create initial span
    span = langfuse.start_observation(name="span")
    sleep(0.1)
    # Get trace ID for later use
    trace_id = span.trace_id
    # End the span
    span.end()

    # Create generation in the same trace
    generation = langfuse.start_observation(
        as_type="generation",
        name="generation",
        trace_context={"trace_id": trace_id},
    )
    # End the generation
    generation.end()

    # Ensure data is sent
    langfuse.flush()

    # Retrieve and verify
    observations = wait_for_observations(trace_id, min_count=2)

    # Verify trace details
    assert len(observations) == 2

    # Find span and generation
    spans = [obs for obs in observations if obs.name == "span"]
    generations = [obs for obs in observations if obs.name == "generation"]

    assert len(spans) == 1
    assert len(generations) == 1

    # Verify both observations belong to the same trace
    span_obs = spans[0]
    gen_obs = generations[0]

    assert span_obs.trace_id == trace_id
    assert gen_obs.trace_id == trace_id


def test_create_trace_with_id_and_generation():
    langfuse = Langfuse()

    trace_name = create_uuid()

    # Create a trace ID using the utility method
    trace_id = langfuse.create_trace_id()

    # Create a span in this trace using the trace context
    with langfuse.start_as_current_observation(
        name="parent-span", trace_context={"trace_id": trace_id}
    ):
        with propagate_attributes(trace_name=trace_name):
            # Create a generation in the same trace
            generation = langfuse.start_observation(
                as_type="generation", name="generation"
            )

            # End the generation
            generation.end()

    # Ensure data is sent
    langfuse.flush()

    # Retrieve and verify
    observations = wait_for_observations(trace_id, min_count=2)

    # Verify trace properties
    root = get_root_observation(observations)
    assert root.trace_name == trace_name
    assert root.trace_id == trace_id
    assert len(observations) == 2  # Parent span and generation

    # Find the generation
    generations = [obs for obs in observations if obs.name == "generation"]
    assert len(generations) == 1

    gen = generations[0]
    assert gen.name == "generation"
    assert gen.trace_id == trace_id


def test_end_generation():
    langfuse = Langfuse()

    # Create a generation
    generation = langfuse.start_observation(
        as_type="generation",
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

    # Retrieve and verify
    observations = wait_for_observations(trace_id)

    # Find generation by name
    generations = [obs for obs in observations if obs.name == "query-generation"]
    assert len(generations) == 1

    gen = generations[0]
    assert gen.end_time is not None


def test_end_generation_with_data():
    langfuse = Langfuse()

    # Create a parent span to set trace properties
    with langfuse.start_as_current_observation(name="parent-span") as parent_span:
        # Get trace ID
        trace_id = parent_span.trace_id

        # Create generation
        generation = langfuse.start_observation(
            as_type="generation",
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

    # Retrieve and verify
    observations = wait_for_observations(trace_id, min_count=2)

    # Find generation by name
    generations = [obs for obs in observations if obs.name == "query-generation"]
    assert len(generations) == 1

    generation = generations[0]

    # Verify properties were updated
    assert generation.completion_start_time == datetime(
        2023, 1, 1, 12, 3, tzinfo=timezone.utc
    )
    assert generation.name == "query-generation"
    assert user_metadata(generation) == {"dict": "value"}
    assert generation.level == "ERROR"
    assert generation.status_message == "Generation ended"
    assert generation.version == "1.0"
    assert generation.model == "test-model"
    assert generation.model_parameters == {"param1": "value1", "param2": "value2"}
    assert generation.input == [{"test_input_key": "test_input_value"}]
    assert generation.output == {"test_output_key": "test_output_value"}
    assert generation.usage_details == {"input": 100, "output": 200, "total": 500}
    assert generation.cost_details == {"input": 111, "output": 222, "total": 444}
    assert generation.total_cost == 444


def test_end_generation_with_openai_token_format():
    langfuse = Langfuse()

    # Create a generation
    generation = langfuse.start_observation(
        as_type="generation",
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

    # Retrieve and verify
    observations = wait_for_observations(trace_id)

    # Find generation
    generations = [obs for obs in observations if obs.name == "query-generation"]
    assert len(generations) == 1

    generation_api = generations[0]

    # Verify properties were converted correctly
    assert generation_api.end_time is not None
    # OpenAI-style keys are mapped to input/output/total
    assert generation_api.usage_details == {"input": 100, "output": 200, "total": 500}
    assert generation_api.cost_details == {"input": 111, "output": 222, "total": 444}
    assert generation_api.total_cost == 444


def test_end_span():
    langfuse = Langfuse()

    # Create a span
    span = langfuse.start_observation(
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

    # Retrieve and verify
    observations = wait_for_observations(trace_id)

    # Find span
    spans = [obs for obs in observations if obs.name == "span"]
    assert len(spans) == 1

    span_api = spans[0]

    # Verify end time was set
    assert span_api.end_time is not None


def test_end_span_with_data():
    langfuse = Langfuse()

    # Create a span
    span = langfuse.start_observation(
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

    # Retrieve and verify
    observations = wait_for_observations(trace_id)

    # Find span
    spans = [obs for obs in observations if obs.name == "span"]
    assert len(spans) == 1

    span_api = spans[0]

    # Verify end time and metadata were updated
    assert span_api.end_time is not None
    assert user_metadata(span_api) == {"dict": "value", "interface": "whatsapp"}


def test_get_generations():
    langfuse = Langfuse()

    # Create a first generation with random name
    generation1 = langfuse.start_observation(
        as_type="generation",
        name=create_uuid(),
    )
    generation1.end()

    # Create a second generation with specific name and content
    generation_name = create_uuid()

    generation2 = langfuse.start_observation(
        as_type="generation",
        name=generation_name,
        input="great-prompt",
        output="great-completion",
    )
    generation2.end()

    # Ensure data is sent
    langfuse.flush()

    # Fetch generations using API
    generations = wait_for_observations(name=generation_name)

    # Verify fetched generation matches what we created
    assert len(generations) == 1
    assert generations[0].name == generation_name
    assert generations[0].input == "great-prompt"
    assert generations[0].output == "great-completion"


def test_get_generations_by_user():
    langfuse = Langfuse()

    # Generate unique IDs for test
    user_id = create_uuid()
    generation_name = create_uuid()

    # Create a trace with user ID and a generation as its child
    with langfuse.start_as_current_observation(name="test-user"):
        with propagate_attributes(trace_name="test-user", user_id=user_id):
            # Create a generation within the trace
            generation = langfuse.start_observation(
                as_type="generation",
                name=generation_name,
                input="great-prompt",
                output="great-completion",
            )
            generation.end()

    # Create another generation that doesn't have this user ID
    other_gen = langfuse.start_observation(
        as_type="generation", name="other-generation"
    )
    other_gen.end()

    # Ensure data is sent
    langfuse.flush()

    # Fetch generations by user ID using the API
    generations = wait_for_observations(user_id=user_id, type="GENERATION")

    # Verify fetched generation matches what we created
    assert len(generations) == 1
    assert generations[0].name == generation_name
    assert generations[0].input == "great-prompt"
    assert generations[0].output == "great-completion"


def test_kwargs():
    langfuse = Langfuse()

    # Create kwargs dict with valid parameters for start_observation
    kwargs_dict = {
        "input": {"key": "value"},
        "output": {"key": "value"},
        "metadata": {"interface": "whatsapp"},
    }

    # Create span with specific kwargs instead of using **kwargs_dict
    span = langfuse.start_observation(
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

    # Retrieve and verify
    observations = wait_for_observations(span.trace_id)
    assert [observation.id for observation in observations] == [span_id]
    observation = observations[0]

    # Verify kwargs were properly set as attributes
    assert observation.start_time is not None
    assert observation.input == {"key": "value"}
    assert observation.output == {"key": "value"}
    assert user_metadata(observation) == {"interface": "whatsapp"}


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
    with langfuse.start_as_current_observation(name="test") as parent_span:
        with propagate_attributes(trace_name="test"):
            # Get trace ID for verification
            trace_id = parent_span.trace_id

            # Create a span
            span = parent_span.start_observation(name="span")
            span.end()

            # Create a generation
            generation = parent_span.start_observation(
                as_type="generation", name="generation"
            )
            generation.end()

        # In OTEL-based client, "events" are just spans with minimal duration
        event_span = parent_span.start_observation(name="event")
        event_span.end()

    # Ensure data is sent
    langfuse.flush()

    # Retrieve and verify
    observations = wait_for_observations(trace_id, min_count=4)

    # Verify timestamps are in UTC regardless of local timezone
    assert len(observations) == 4  # Parent span, child span, generation, and event
    for observation in observations:
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
    with langfuse.start_as_current_observation(name="test") as parent_span:
        with propagate_attributes(trace_name="test"):
            # Get trace ID for verification
            trace_id = parent_span.trace_id

            # Create span
            span = parent_span.start_observation(name="span")
            span.end()

            # Create generation
            generation = parent_span.start_observation(
                as_type="generation", name="generation"
            )
            generation.end()

            # Create event-like span
            event_span = parent_span.start_observation(name="event")
            event_span.end()

    # Ensure data is sent
    langfuse.flush()

    # Retrieve and verify
    observations = wait_for_observations(trace_id, min_count=4)

    # Verify timestamps are in UTC regardless of local timezone
    assert len(observations) == 4  # Parent span, child span, generation, and event
    for observation in observations:
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
    with langfuse.start_as_current_observation(name="test-span") as span:
        with propagate_attributes(trace_name=trace_name, session_id=session_id):
            # Get trace ID for verification
            trace_id = span.trace_id

    # Create another trace without a session_id
    with langfuse.start_as_current_observation(name=create_uuid()):
        pass

    # Ensure data is sent
    langfuse.flush()

    # Retrieve the trace's observations using the session_id
    observations = wait_for_observations(session_id=session_id)

    # Verify that the trace was retrieved correctly
    assert len(observations) == 1
    retrieved_root = observations[0]
    assert retrieved_root.is_root_observation is True
    assert retrieved_root.trace_name == trace_name
    assert retrieved_root.session_id == session_id
    assert retrieved_root.trace_id == trace_id


def test_fetch_trace():
    langfuse = Langfuse()

    # Create a trace
    name = create_uuid()

    # Create a span and set trace properties
    with langfuse.start_as_current_observation(name="test-span") as span:
        with propagate_attributes(trace_name=name):
            # Get trace ID for verification
            trace_id = span.trace_id

    # Ensure data is sent
    langfuse.flush()

    root = wait_for_root_observation(trace_id)

    # Verify trace properties
    assert root.trace_id == trace_id
    assert root.trace_name == name


def test_fetch_traces():
    langfuse = Langfuse()

    # Use a unique name for this test
    name = create_uuid()

    # Create 3 traces with different properties, but same name
    trace_ids = []

    # First trace
    with langfuse.start_as_current_observation(name="test1") as span:
        with propagate_attributes(trace_name=name, session_id="session-1"):
            span.update(input={"key": "value"}, output="output-value")
            trace_ids.append(span.trace_id)

    sleep(1)  # Ensure traces have different timestamps

    # Second trace
    with langfuse.start_as_current_observation(name="test2") as span:
        with propagate_attributes(trace_name=name, session_id="session-1"):
            span.update(input={"key": "value"}, output="output-value")
            trace_ids.append(span.trace_id)

    sleep(1)  # Ensure traces have different timestamps

    # Third trace
    with langfuse.start_as_current_observation(name="test3") as span:
        with propagate_attributes(trace_name=name, session_id="session-1"):
            span.update(input={"key": "value"}, output="output-value")
            trace_ids.append(span.trace_id)

    # Ensure data is sent
    langfuse.flush()

    expected_trace_ids = set(trace_ids)
    api = get_api(retry=False)
    trace_name_filter = json.dumps(
        [{"type": "string", "column": "traceName", "operator": "=", "value": name}]
    )

    # Fetch the root observations of all traces with the same name.
    roots = wait_for_observations(
        filter=trace_name_filter,
        is_result_ready=lambda observations: (
            {o.trace_id for o in observations} == expected_trace_ids
        ),
    )

    # Verify we got all traces
    assert len(roots) == 3

    # Verify trace properties
    for root in roots:
        assert root.is_root_observation is True
        assert root.trace_name == name
        assert root.session_id == "session-1"
        assert root.input == {"key": "value"}
        assert root.output == "output-value"

    # Test cursor pagination by walking pages of one item and confirming they
    # collectively cover the created traces.
    paginated_ids = []
    cursor = None
    for _ in range(3):
        page = api.observations.get_many(
            filter=trace_name_filter, limit=1, cursor=cursor
        )
        assert len(page.data) == 1
        paginated_ids.append(page.data[0].trace_id)
        cursor = page.meta.cursor

    assert set(paginated_ids) == expected_trace_ids
    assert len(paginated_ids) == 3
    if cursor is not None:
        assert (
            api.observations.get_many(
                filter=trace_name_filter, limit=1, cursor=cursor
            ).data
            == []
        )


def test_get_observation():
    langfuse = Langfuse()

    # Create a trace and a generation
    name = create_uuid()

    # Create a span and set trace properties
    with langfuse.start_as_current_observation(name="parent-span") as parent_span:
        with propagate_attributes(trace_name=name):
            # Create a generation as child
            generation = parent_span.start_observation(as_type="generation", name=name)

            # Get IDs for verification
            generation_id = generation.id

            # End the generation
            generation.end()

    # Ensure data is sent
    langfuse.flush()

    # Fetch the observation using the API
    observations = wait_for_observations(parent_span.trace_id, min_count=2)
    matching = [o for o in observations if o.id == generation_id]
    assert len(matching) == 1
    observation = matching[0]

    # Verify observation properties
    assert observation.id == generation_id
    assert observation.name == name
    assert observation.type == "GENERATION"


def test_get_observations():
    langfuse = Langfuse()

    # Create a trace with multiple generations
    name = create_uuid()

    # Create a span and set trace properties
    with langfuse.start_as_current_observation(name="parent-span"):
        with propagate_attributes(trace_name=name):
            # Create first generation
            gen1 = langfuse.start_observation(as_type="generation", name=name)
            gen1_id = gen1.id
            gen1.end()

            # Create second generation
            gen2 = langfuse.start_observation(as_type="generation", name=name)
            gen2_id = gen2.id
            gen2.end()

    # Ensure data is sent
    langfuse.flush()
    api = get_api(retry=False)

    # Fetch observations using the API
    expected_generation_ids = {gen1_id, gen2_id}
    observations = wait_for_observations(
        name=name,
        is_result_ready=lambda observations: expected_generation_ids.issubset(
            {obs.id for obs in observations}
        ),
    )

    # Verify fetched observations
    assert len(observations) == 2

    # Filter for just the generations
    generations = [obs for obs in observations if obs.type == "GENERATION"]
    assert len(generations) == 2

    # Verify the generation IDs match what we created
    gen_ids = [gen.id for gen in generations]
    assert gen1_id in gen_ids
    assert gen2_id in gen_ids

    # Test cursor pagination by confirming both created generations can be
    # reached across separate pages.
    first_page = api.observations.get_many(name=name, limit=1)
    assert len(first_page.data) == 1
    assert first_page.meta.cursor is not None
    second_page = api.observations.get_many(
        name=name, limit=1, cursor=first_page.meta.cursor
    )
    assert len(second_page.data) == 1

    assert {first_page.data[0].id, second_page.data[0].id} == expected_generation_ids


def test_get_observations_for_unknown_trace_is_empty():
    response = get_api(retry=False).observations.get_many(trace_id=create_uuid())

    assert response.data == []
    assert response.meta.cursor is None


def test_get_observations_empty():
    # Fetch observations with a filter that should return no results
    response = get_api(retry=False).observations.get_many(name=create_uuid())

    assert response.data == []
    assert response.meta.cursor is None


@pytest.mark.skip(
    "Flaky in concurrent environment as the global tracer provider is already configured"
)
def test_create_trace_sampling_zero():
    langfuse = Langfuse(sample_rate=0)
    trace_name = create_uuid()

    # Create a span with trace properties - with sample_rate=0, this will not be sent to the API
    with langfuse.start_as_current_observation(name="test-span") as span:
        with propagate_attributes(
            trace_name=trace_name,
            user_id="test",
            metadata={"key": "value"},
            tags=["tag1", "tag2"],
        ):
            span.set_trace_as_public()
            # Get trace ID for verification
            trace_id = span.trace_id

            # Add a score and a child generation
            langfuse.score_current_trace(name="score", value=0.5)
            generation = span.start_observation(as_type="generation", name="generation")
            generation.end()

    # Ensure data is sent, but should be dropped due to sampling
    langfuse.flush()
    sleep(2)

    # The trace's observations must not exist as they were never sent to the API
    assert get_observations(trace_id=trace_id) == []
    assert get_scores(trace_id=trace_id) == []


def test_mask_function(request):
    LangfuseResourceManager.reset()
    request.addfinalizer(LangfuseResourceManager.reset)

    def mask_func(data):
        if isinstance(data, dict):
            if "should_raise" in data:
                raise
            return {k: "MASKED" for k in data}
        elif isinstance(data, str):
            return "MASKED"
        return data

    langfuse = Langfuse(mask=mask_func)

    # Create a root span with trace properties
    with langfuse.start_as_current_observation(name="test-span") as root_span:
        with propagate_attributes(trace_name="test_trace"):
            root_span.update(input={"sensitive": "data"})
            # Get trace ID for later use
            trace_id = root_span.trace_id
            # Add output to the trace
            root_span.update(output={"more": "sensitive"})

            # Create a generation as child
            gen = root_span.start_observation(
                as_type="generation",
                name="test_gen",
                input={"prompt": "secret"},
            )
            gen.update(output="new_confidential")
            gen.end()

            # Create a span as child
            sub_span = root_span.start_observation(
                name="test_span", input={"data": "private"}
            )
            sub_span.update(output="new_classified")
            sub_span.end()

    # Ensure data is sent
    langfuse.flush()

    # Retrieve and verify
    observations = wait_for_observations(trace_id, min_count=3)
    fetched_root = get_root_observation(observations)
    assert fetched_root.input == {"sensitive": "MASKED"}
    assert fetched_root.output == {"more": "MASKED"}

    fetched_gen = [o for o in observations if o.type == "GENERATION"][0]
    assert fetched_gen.input == {"prompt": "MASKED"}
    assert fetched_gen.output == "MASKED"

    fetched_span = [
        o for o in observations if o.type == "SPAN" and o.name == "test_span"
    ][0]
    assert fetched_span.input == {"data": "MASKED"}
    assert fetched_span.output == "MASKED"

    # Create a root span with trace properties
    with langfuse.start_as_current_observation(name="test-span") as root_span:
        with propagate_attributes(trace_name="test_trace"):
            root_span.update(input={"should_raise": "data"})
            # Get trace ID for later use
            trace_id = root_span.trace_id
            # Add output to the trace
            root_span.update(output={"should_raise": "sensitive"})

    # Ensure data is sent
    langfuse.flush()

    # Retrieve and verify
    fetched_root = wait_for_root_observation(trace_id)
    assert fetched_root.input == "<fully masked due to failed mask function>"
    assert fetched_root.output == "<fully masked due to failed mask function>"


def test_get_project_id():
    langfuse = Langfuse()
    res = langfuse._get_project_id()
    assert res is not None
    assert res == "7a88fb47-b4e2-43b8-a06c-a5ce950dc53a"


def test_generate_trace_id():
    langfuse = Langfuse()
    trace_id = langfuse.create_trace_id()

    # Create a trace with the specific ID using trace_context
    with langfuse.start_as_current_observation(
        name="test-span", trace_context={"trace_id": trace_id}
    ):
        with propagate_attributes(trace_name="test_trace"):
            pass

    langfuse.flush()

    # Test the trace URL generation
    project_id = langfuse._get_project_id()
    trace_url = langfuse.get_trace_url(trace_id=trace_id)
    assert trace_url == f"http://localhost:3000/project/{project_id}/traces/{trace_id}"


def test_generate_trace_url_client_disabled():
    langfuse = Langfuse(tracing_enabled=False)

    with langfuse.start_as_current_observation(
        name="test-span",
    ):
        # The trace URL should be None because the client is disabled
        trace_url = langfuse.get_trace_url()
        assert trace_url is None

    langfuse.flush()


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

    with langfuse.start_as_current_observation(name="parent") as parent_span:
        with propagate_attributes(trace_name="observation-types-test"):
            trace_id = parent_span.trace_id

            for obs_type in observation_types:
                with parent_span.start_as_current_observation(
                    name=f"test-{obs_type}", as_type=obs_type
                ):
                    pass

    langfuse.flush()

    observations = wait_for_observations(trace_id, min_count=len(observation_types) + 1)

    # Check we have all expected observation types
    found_types = {obs.type for obs in observations}
    expected_types = {obs_type.upper() for obs_type in observation_types} | {
        "SPAN"
    }  # includes parent span
    assert expected_types.issubset(found_types), (
        f"Missing types: {expected_types - found_types}"
    )

    # Verify each specific observation exists
    for obs_type in observation_types:
        matching = [
            obs
            for obs in observations
            if obs.name == f"test-{obs_type}" and obs.type == obs_type.upper()
        ]
        assert len(matching) == 1, f"Expected one {obs_type.upper()} observation"


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

    with langfuse.start_as_current_observation(name="parent") as parent_span:
        with propagate_attributes(trace_name="generation-properties-test"):
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
                        assert (
                            obs.completion_start_time == test_completion_start_time
                        ), f"{obs_type} should have completion_start_time property"
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

    observations = wait_for_observations(
        trace_id, min_count=len(generation_like_types) + 1
    )

    # Verify that the properties are persisted in the API for generation-like types
    for obs_type in generation_like_types:
        matching = [
            obs
            for obs in observations
            if obs.name == f"test-{obs_type}" and obs.type == obs_type.upper()
        ]
        assert len(matching) == 1, (
            f"Expected one {obs_type.upper()} observation, but found {len(matching)}"
        )

        obs = matching[0]

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
