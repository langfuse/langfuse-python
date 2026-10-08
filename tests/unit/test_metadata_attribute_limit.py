"""Metadata must stay within the OTel span attribute limit.

The OTel SDK evicts the oldest attribute once a span holds more than
``SpanLimits.max_span_attributes`` attributes. Langfuse core attributes are
written first, so unbounded metadata used to evict them. Metadata also leaves
room for the reserved observation attributes that later updates may write.
"""

import logging

import pytest

from langfuse import propagate_attributes
from langfuse._client.attributes import LangfuseOtelSpanAttributes
from langfuse._client.span import _RESERVED_OBSERVATION_ATTRIBUTE_KEYS

METADATA_PREFIX = LangfuseOtelSpanAttributes.OBSERVATION_METADATA + "."


def _metadata_keys(span):
    return [key for key in span.attributes if key.startswith(METADATA_PREFIX)]


def _missing_reserved_keys(span):
    return [
        key
        for key in _RESERVED_OBSERVATION_ATTRIBUTE_KEYS
        if key not in span.attributes
    ]


def _limit_warnings(caplog):
    return [
        record
        for record in caplog.records
        if record.name == "langfuse"
        and record.levelno == logging.WARNING
        and "attribute limit" in record.getMessage()
    ]


@pytest.fixture
def small_limit_client(monkeypatch, request):
    monkeypatch.setenv("OTEL_SPAN_ATTRIBUTE_COUNT_LIMIT", "40")
    return request.getfixturevalue("langfuse_memory_client")


def test_metadata_over_limit_at_start_keeps_core_attributes(
    langfuse_memory_client, get_span, caplog
):
    metadata = {f"key_{i}": f"value_{i}" for i in range(130)}

    generation = langfuse_memory_client.start_observation(
        name="big-generation",
        as_type="generation",
        input="the input",
        model="gpt-4o",
        version="v1",
        metadata=metadata,
    )
    generation.end()
    langfuse_memory_client.flush()

    span = get_span("big-generation")
    attributes = span.attributes

    assert len(attributes) <= 128
    assert span.dropped_attributes == 0
    assert attributes[LangfuseOtelSpanAttributes.OBSERVATION_TYPE] == "generation"
    assert attributes[LangfuseOtelSpanAttributes.OBSERVATION_INPUT] == "the input"
    assert attributes[LangfuseOtelSpanAttributes.OBSERVATION_MODEL] == "gpt-4o"
    assert attributes[LangfuseOtelSpanAttributes.VERSION] == "v1"

    # Kept metadata keys are the first ones in insertion order.
    kept = _metadata_keys(span)
    assert kept == [f"{METADATA_PREFIX}key_{i}" for i in range(len(kept))]
    assert 0 < len(kept) < 130

    warnings = _limit_warnings(caplog)
    assert len(warnings) == 1
    message = warnings[0].getMessage()
    assert str(130 - len(kept)) in message
    assert "128" in message
    assert "big-generation" in message
    assert f"key_{len(kept)}" in message


def test_update_over_limit_keeps_earlier_attributes(
    langfuse_memory_client, get_span, caplog
):
    span_wrapper = langfuse_memory_client.start_observation(
        name="updated-span",
        input="the input",
        metadata={f"first_{i}": i for i in range(100)},
    )
    assert _limit_warnings(caplog) == []

    span_wrapper.update(
        output="the output", metadata={f"second_{i}": i for i in range(50)}
    )
    span_wrapper.end()
    langfuse_memory_client.flush()

    span = get_span("updated-span")
    attributes = span.attributes

    assert len(attributes) + len(_missing_reserved_keys(span)) == 128
    assert span.dropped_attributes == 0
    assert attributes[LangfuseOtelSpanAttributes.OBSERVATION_INPUT] == "the input"
    assert attributes[LangfuseOtelSpanAttributes.OBSERVATION_OUTPUT] == "the output"
    for i in range(100):
        assert attributes[f"{METADATA_PREFIX}first_{i}"] == i

    second = [key for key in _metadata_keys(span) if ".second_" in key]
    assert second == [f"{METADATA_PREFIX}second_{i}" for i in range(len(second))]
    assert len(second) < 50

    assert len(_limit_warnings(caplog)) == 1


def test_update_current_span_over_limit_keeps_earlier_attributes(
    langfuse_memory_client, get_span, caplog
):
    with langfuse_memory_client.start_as_current_observation(
        name="current-span", input="the input"
    ):
        langfuse_memory_client.update_current_span(
            metadata={f"key_{i}": i for i in range(200)}
        )
    langfuse_memory_client.flush()

    span = get_span("current-span")
    assert span.dropped_attributes == 0
    assert span.attributes[LangfuseOtelSpanAttributes.OBSERVATION_INPUT] == "the input"
    assert len(_limit_warnings(caplog)) == 1


def test_overwriting_existing_metadata_keys_at_limit_is_allowed(
    small_limit_client, get_span, caplog
):
    span_wrapper = small_limit_client.start_observation(
        name="overwrite-span", metadata={f"key_{i}": "old" for i in range(30)}
    )
    span_at_start = span_wrapper._otel_span
    assert (
        len(span_at_start.attributes) + len(_missing_reserved_keys(span_at_start)) == 40
    )
    kept = _metadata_keys(span_at_start)
    caplog.clear()

    span_wrapper.update(metadata={key[len(METADATA_PREFIX) :]: "new" for key in kept})
    span_wrapper.end()
    small_limit_client.flush()

    span = get_span("overwrite-span")
    assert span.dropped_attributes == 0
    assert len(span.attributes) == len(span_at_start.attributes)
    for key in kept:
        assert span.attributes[key] == "new"
    assert _limit_warnings(caplog) == []


def test_custom_smaller_span_limit_is_respected(small_limit_client, get_span, caplog):
    span_wrapper = small_limit_client.start_observation(
        name="small-limit-span",
        input="the input",
        metadata={f"key_{i}": i for i in range(30)},
    )
    span_wrapper.end()
    small_limit_client.flush()

    span = get_span("small-limit-span")
    assert len(span.attributes) + len(_missing_reserved_keys(span)) == 40
    assert span.dropped_attributes == 0
    assert span.attributes[LangfuseOtelSpanAttributes.OBSERVATION_INPUT] == "the input"
    assert span.attributes[LangfuseOtelSpanAttributes.OBSERVATION_TYPE] == "span"

    warnings = _limit_warnings(caplog)
    assert len(warnings) == 1
    assert "40" in warnings[0].getMessage()


def test_propagated_trace_attributes_count_toward_limit(
    small_limit_client, get_span, caplog
):
    with propagate_attributes(
        user_id="user-1",
        session_id="session-1",
        metadata={f"trace_{i}": str(i) for i in range(5)},
    ):
        span_wrapper = small_limit_client.start_observation(
            name="propagated-span",
            input="the input",
            metadata={f"key_{i}": i for i in range(30)},
        )
        span_wrapper.end()
    small_limit_client.flush()

    span = get_span("propagated-span")
    attributes = span.attributes
    assert len(attributes) + len(_missing_reserved_keys(span)) == 40
    assert span.dropped_attributes == 0
    assert attributes[LangfuseOtelSpanAttributes.TRACE_USER_ID] == "user-1"
    assert attributes[LangfuseOtelSpanAttributes.TRACE_SESSION_ID] == "session-1"
    for i in range(5):
        assert attributes[f"{LangfuseOtelSpanAttributes.TRACE_METADATA}.trace_{i}"]
    assert attributes[LangfuseOtelSpanAttributes.OBSERVATION_INPUT] == "the input"
    assert len(_limit_warnings(caplog)) == 1


def test_metadata_under_limit_is_unchanged(langfuse_memory_client, get_span, caplog):
    metadata = {f"key_{i}": i for i in range(50)}
    span_wrapper = langfuse_memory_client.start_observation(
        name="small-span", input="the input", metadata=metadata
    )
    span_wrapper.update(metadata={"extra": "value"})
    span_wrapper.end()
    langfuse_memory_client.flush()

    span = get_span("small-span")
    assert span.dropped_attributes == 0
    assert len(_metadata_keys(span)) == 51
    for i in range(50):
        assert span.attributes[f"{METADATA_PREFIX}key_{i}"] == i
    assert span.attributes[f"{METADATA_PREFIX}extra"] == "value"
    assert _limit_warnings(caplog) == []


def test_later_updates_do_not_evict_attributes_after_metadata_cap(
    langfuse_memory_client, get_span, caplog
):
    generation = langfuse_memory_client.start_observation(
        name="capped-generation",
        as_type="generation",
        input="the input",
        model="gpt-4o",
        metadata={f"key_{i}": i for i in range(130)},
    )
    generation.update(output="the output")
    generation.update(
        usage_details={"input": 10, "output": 20},
        cost_details={"input": 0.1, "output": 0.2},
    )
    generation.end()
    langfuse_memory_client.flush()

    span = get_span("capped-generation")
    attributes = span.attributes

    assert span.dropped_attributes == 0
    assert attributes[LangfuseOtelSpanAttributes.IS_APP_ROOT] is True
    assert attributes[LangfuseOtelSpanAttributes.OBSERVATION_INPUT] == "the input"
    assert attributes[LangfuseOtelSpanAttributes.OBSERVATION_MODEL] == "gpt-4o"
    assert attributes[LangfuseOtelSpanAttributes.OBSERVATION_OUTPUT] == "the output"
    assert LangfuseOtelSpanAttributes.OBSERVATION_USAGE_DETAILS in attributes
    assert LangfuseOtelSpanAttributes.OBSERVATION_COST_DETAILS in attributes
    assert len(attributes) + len(_missing_reserved_keys(span)) == 128
    assert len(_limit_warnings(caplog)) == 1


def test_reserved_keys_already_written_cost_no_extra_slot(small_limit_client, get_span):
    metadata = {f"key_{i}": i for i in range(50)}
    reserved_values = {
        "input": "the input",
        "output": "the output",
        "version": "v1",
        "level": "WARNING",
        "status_message": "careful",
    }

    small_limit_client.start_observation(name="metadata-only", metadata=metadata).end()
    small_limit_client.start_observation(
        name="reserved-in-same-write", metadata=metadata, **reserved_values
    ).end()
    small_limit_client.start_observation(
        name="reserved-already-on-span", **reserved_values
    ).update(metadata=metadata).end()
    small_limit_client.flush()

    kept_counts = {
        name: len(_metadata_keys(get_span(name)))
        for name in (
            "metadata-only",
            "reserved-in-same-write",
            "reserved-already-on-span",
        )
    }
    assert 0 < kept_counts["metadata-only"] < 50
    assert len(set(kept_counts.values())) == 1, kept_counts
