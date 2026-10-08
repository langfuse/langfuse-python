"""Langfuse writes must stay within the OTel span attribute limit.

The Python OTel SDK evicts the oldest attribute once a span holds more than
``SpanLimits.max_span_attributes`` attributes, which used to drop Langfuse core
attributes written first. The SDK now drops the newest attributes instead, like
OTel JS: excess new metadata keys go first, then any other new keys that still
do not fit. Metadata also leaves room for the reserved observation attributes
that later updates may write.
"""

import logging
from datetime import datetime
from types import SimpleNamespace

import pytest

from langfuse import propagate_attributes
from langfuse._client.attributes import LangfuseOtelSpanAttributes
from langfuse._client.span import (
    _RESERVED_OBSERVATION_ATTRIBUTE_KEYS,
    _RESERVED_SPAN_ATTRIBUTE_KEYS,
    _drop_attributes_over_span_limit,
    _reserved_attribute_keys,
)
from langfuse.api import DatasetItem, DatasetStatus

METADATA_PREFIX = LangfuseOtelSpanAttributes.OBSERVATION_METADATA + "."


def _metadata_keys(span):
    return [key for key in span.attributes if key.startswith(METADATA_PREFIX)]


def _missing_reserved_keys(span):
    observation_type = span.attributes.get(LangfuseOtelSpanAttributes.OBSERVATION_TYPE)
    return [
        key
        for key in _reserved_attribute_keys(observation_type)
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


@pytest.fixture(autouse=True)
def default_attribute_limit(monkeypatch):
    # Pin the OTel default so a custom limit in the environment cannot break
    # the assertions below that expect 128.
    monkeypatch.delenv("OTEL_ATTRIBUTE_COUNT_LIMIT", raising=False)
    monkeypatch.setenv("OTEL_SPAN_ATTRIBUTE_COUNT_LIMIT", "128")


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
        name="overwrite-span", metadata={f"key_{i}": "old" for i in range(40)}
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
        metadata={f"key_{i}": i for i in range(40)},
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


def test_reserved_keys_depend_on_observation_type(langfuse_memory_client, get_span):
    assert len(_RESERVED_SPAN_ATTRIBUTE_KEYS) == 7
    assert len(_RESERVED_OBSERVATION_ATTRIBUTE_KEYS) == 14
    for observation_type in ("generation", "embedding", None):
        assert (
            _reserved_attribute_keys(observation_type)
            == _RESERVED_OBSERVATION_ATTRIBUTE_KEYS
        )
    for observation_type in (
        "span",
        "agent",
        "tool",
        "chain",
        "retriever",
        "evaluator",
        "guardrail",
        "event",
    ):
        assert _reserved_attribute_keys(observation_type) == (
            _RESERVED_SPAN_ATTRIBUTE_KEYS
        )

    metadata = {f"key_{i}": i for i in range(130)}
    for as_type in ("span", "generation"):
        langfuse_memory_client.start_observation(
            name=f"typed-{as_type}",
            as_type=as_type,
            input="the input",
            metadata=metadata,
        ).end()
    langfuse_memory_client.flush()

    span = get_span("typed-span")
    generation = get_span("typed-generation")
    span_count = len(_metadata_keys(span))
    generation_count = len(_metadata_keys(generation))
    # The span reserves 5 open slots (level, status message, version, output,
    # plus environment when unset) instead of 12 for the generation.
    assert span_count - generation_count == 7
    assert span.dropped_attributes == 0
    assert generation.dropped_attributes == 0
    assert len(span.attributes) + len(_missing_reserved_keys(span)) == 128
    assert len(generation.attributes) + len(_missing_reserved_keys(generation)) == 128


def test_generation_keeps_room_for_model_usage_and_output(
    small_limit_client, get_span, caplog
):
    generation = small_limit_client.start_observation(
        name="roomy-generation",
        as_type="generation",
        input="the input",
        metadata={f"key_{i}": i for i in range(40)},
    )
    caplog.clear()
    generation.update(
        output="the output",
        model="gpt-4o",
        usage_details={"input": 1},
        cost_details={"input": 0.1},
        model_parameters={"temperature": 0},
    )
    generation.end()
    small_limit_client.flush()

    span = get_span("roomy-generation")
    attributes = span.attributes
    assert span.dropped_attributes == 0
    assert attributes[LangfuseOtelSpanAttributes.OBSERVATION_INPUT] == "the input"
    assert attributes[LangfuseOtelSpanAttributes.OBSERVATION_OUTPUT] == "the output"
    assert attributes[LangfuseOtelSpanAttributes.OBSERVATION_MODEL] == "gpt-4o"
    assert LangfuseOtelSpanAttributes.OBSERVATION_USAGE_DETAILS in attributes
    assert LangfuseOtelSpanAttributes.OBSERVATION_COST_DETAILS in attributes
    assert LangfuseOtelSpanAttributes.OBSERVATION_MODEL_PARAMETERS in attributes
    assert _limit_warnings(caplog) == []


def test_warning_message_format(small_limit_client, caplog):
    small_limit_client.start_observation(
        name="warned-span", metadata={f"key_{i}": i for i in range(40)}
    ).end()

    warnings = _limit_warnings(caplog)
    assert len(warnings) == 1
    message = warnings[0].getMessage()
    assert message.startswith("Dropped ")
    assert (
        " metadata key(s) from observation 'warned-span' to stay within the span "
        "attribute limit of 40 (SpanLimits.max_span_attributes / "
        "OTEL_SPAN_ATTRIBUTE_COUNT_LIMIT). Dropped keys include: "
    ) in message
    dropped_keys = message.split("Dropped keys include: ")[1].split(", ")
    assert len(dropped_keys) == 5
    assert all(key.startswith("key_") for key in dropped_keys)


def test_new_attributes_beyond_capacity_drop_newest_instead_of_evicting(
    small_limit_client, get_span, caplog
):
    with small_limit_client.start_as_current_observation(
        name="full-generation",
        as_type="generation",
        input="the input",
        model="gpt-4o",
        metadata={f"key_{i}": i for i in range(40)},
    ) as generation:
        generation.update(output="the output", usage_details={"input": 1})
        otel_span = generation._otel_span
        free_slots = 40 - len(otel_span.attributes)
        assert 0 < free_slots < 20
        attributes_before = dict(otel_span.attributes)
        caplog.clear()

        with propagate_attributes(
            metadata={f"trace_{i:02d}": str(i) for i in range(20)}
        ):
            pass
    small_limit_client.flush()

    span = get_span("full-generation")
    attributes = span.attributes

    assert span.dropped_attributes == 0
    assert len(attributes) == 40
    for key, value in attributes_before.items():
        assert attributes[key] == value
    assert attributes[LangfuseOtelSpanAttributes.IS_APP_ROOT] is True
    assert attributes[LangfuseOtelSpanAttributes.OBSERVATION_INPUT] == "the input"
    assert attributes[LangfuseOtelSpanAttributes.OBSERVATION_MODEL] == "gpt-4o"
    assert attributes[LangfuseOtelSpanAttributes.OBSERVATION_OUTPUT] == "the output"

    trace_prefix = LangfuseOtelSpanAttributes.TRACE_METADATA + "."
    kept_trace_keys = [key for key in attributes if key.startswith(trace_prefix)]
    assert kept_trace_keys == [
        f"{trace_prefix}trace_{i:02d}" for i in range(free_slots)
    ]

    warnings = _limit_warnings(caplog)
    assert len(warnings) == 1
    message = warnings[0].getMessage()
    assert f"0 metadata key(s) and {20 - free_slots} other attribute(s)" in message
    assert f"{trace_prefix}trace_{free_slots:02d}" in message


def _fake_span(limit, existing):
    return SimpleNamespace(
        name="fake",
        attributes=dict(existing),
        _limits=SimpleNamespace(max_span_attributes=limit),
    )


def test_hard_guard_keeps_overwrites_and_drops_new_tail(caplog):
    span = _fake_span(5, {f"a_{i}": i for i in range(4)})

    result = _drop_attributes_over_span_limit(
        span,
        {
            "a_0": "overwritten",
            "b_0": 0,
            f"{METADATA_PREFIX}m": 1,
            "b_1": 1,
            "b_2": 2,
        },
    )

    assert result == {"a_0": "overwritten", "b_0": 0}
    message = _limit_warnings(caplog)[0].getMessage()
    assert "1 metadata key(s) and 2 other attribute(s)" in message
    assert "Dropped keys include: m, b_1, b_2" in message


def test_hard_guard_never_raises():
    class BrokenAttributes:
        def __contains__(self, key):
            raise RuntimeError("boom")

        def __len__(self):
            return 1

    span = SimpleNamespace(
        attributes=BrokenAttributes(),
        _limits=SimpleNamespace(max_span_attributes=1),
    )
    attributes = {"key": "value"}

    assert _drop_attributes_over_span_limit(span, attributes) is attributes


def test_experiment_run_keys_survive_metadata_truncation(
    langfuse_memory_client, get_span
):
    item = DatasetItem(
        id="item-1",
        status=DatasetStatus.ACTIVE,
        input="question",
        expected_output="answer",
        metadata={
            "experiment_name": "user-value",
            **{f"item_{i}": str(i) for i in range(150)},
        },
        source_trace_id=None,
        source_observation_id=None,
        dataset_id="dataset-1",
        dataset_name="Dataset",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        media_references=[],
    )

    langfuse_memory_client.run_experiment(
        name="big-metadata-experiment",
        data=[item],
        task=lambda **kwargs: "answer",
        max_concurrency=1,
    )
    langfuse_memory_client.flush()

    task_span = get_span("experiment-item-task")
    attributes = task_span.attributes
    assert task_span.dropped_attributes == 0
    assert 0 < len(_metadata_keys(task_span)) < 151
    assert attributes[LangfuseOtelSpanAttributes.OBSERVATION_INPUT] == "question"
    # The run value wins over the user key of the same name.
    assert attributes[f"{METADATA_PREFIX}experiment_name"] == "big-metadata-experiment"
    assert attributes[f"{METADATA_PREFIX}experiment_run_name"].startswith(
        "big-metadata-experiment"
    )
    assert attributes[f"{METADATA_PREFIX}dataset_id"] == "dataset-1"
    assert attributes[f"{METADATA_PREFIX}dataset_item_id"] == "item-1"
    assert f"{METADATA_PREFIX}item_149" not in attributes

    item_run = get_span("experiment-item-run")
    assert item_run.dropped_attributes == 0
    assert LangfuseOtelSpanAttributes.EXPERIMENT_ID in item_run.attributes
