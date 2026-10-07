"""Span attribute management for Langfuse OpenTelemetry integration.

This module defines constants and functions for managing OpenTelemetry span attributes
used by Langfuse. It provides a structured approach to creating and manipulating
attributes for different span types (trace, span, generation) while ensuring consistency.

The module includes:
- Attribute name constants organized by category
- Functions to create attribute dictionaries for different entity types
- Utilities for serializing and processing attribute values
"""

import json
from datetime import datetime
from typing import Any, Dict, Literal, Optional, Tuple, TypeVar, Union

from langfuse._client.constants import (
    ObservationTypeGenerationLike,
    ObservationTypeSpanLike,
)
from langfuse._utils.serializer import EventSerializer
from langfuse.api import MapValue
from langfuse.logger import langfuse_logger
from langfuse.model import PromptClient
from langfuse.types import SpanLevel

MAX_OBSERVATION_METADATA_KEYS = 128
"""Maximum number of top-level keys allowed in observation metadata."""


class ObservationMetadataKeyLimitError(ValueError):
    """Raised when observation metadata has more than 128 top-level keys."""


_FAILED_TO_SERIALIZE = "<failed to serialize>"
_T = TypeVar("_T")
_COMPACT_SEPARATORS = (",", ":")


class LangfuseOtelSpanAttributes:
    # Langfuse-Trace attributes
    TRACE_NAME = "langfuse.trace.name"
    TRACE_USER_ID = "user.id"
    TRACE_SESSION_ID = "session.id"
    TRACE_TAGS = "langfuse.trace.tags"
    TRACE_PUBLIC = "langfuse.trace.public"
    TRACE_METADATA = "langfuse.trace.metadata"
    TRACE_INPUT = "langfuse.trace.input"
    TRACE_OUTPUT = "langfuse.trace.output"

    # Langfuse-observation attributes
    OBSERVATION_TYPE = "langfuse.observation.type"
    OBSERVATION_METADATA = "langfuse.observation.metadata"
    OBSERVATION_LEVEL = "langfuse.observation.level"
    OBSERVATION_STATUS_MESSAGE = "langfuse.observation.status_message"
    OBSERVATION_INPUT = "langfuse.observation.input"
    OBSERVATION_OUTPUT = "langfuse.observation.output"

    # Langfuse-observation of type Generation attributes
    OBSERVATION_COMPLETION_START_TIME = "langfuse.observation.completion_start_time"
    OBSERVATION_MODEL = "langfuse.observation.model.name"
    OBSERVATION_MODEL_PARAMETERS = "langfuse.observation.model.parameters"
    OBSERVATION_USAGE_DETAILS = "langfuse.observation.usage_details"
    OBSERVATION_COST_DETAILS = "langfuse.observation.cost_details"
    OBSERVATION_PROMPT_NAME = "langfuse.observation.prompt.name"
    OBSERVATION_PROMPT_VERSION = "langfuse.observation.prompt.version"

    # General
    ENVIRONMENT = "langfuse.environment"
    RELEASE = "langfuse.release"
    VERSION = "langfuse.version"

    # Internal
    AS_ROOT = "langfuse.internal.as_root"
    IS_APP_ROOT = "langfuse.internal.is_app_root"

    # Experiments
    EXPERIMENT_ID = "langfuse.experiment.id"
    EXPERIMENT_NAME = "langfuse.experiment.name"
    EXPERIMENT_DESCRIPTION = "langfuse.experiment.description"
    EXPERIMENT_METADATA = "langfuse.experiment.metadata"
    EXPERIMENT_DATASET_ID = "langfuse.experiment.dataset.id"
    EXPERIMENT_ITEM_ID = "langfuse.experiment.item.id"
    EXPERIMENT_ITEM_EXPECTED_OUTPUT = "langfuse.experiment.item.expected_output"
    EXPERIMENT_ITEM_METADATA = "langfuse.experiment.item.metadata"
    EXPERIMENT_ITEM_ROOT_OBSERVATION_ID = "langfuse.experiment.item.root_observation_id"


def create_trace_attributes(
    *,
    input: Optional[Any] = None,
    output: Optional[Any] = None,
    public: Optional[bool] = None,
) -> dict:
    attributes = {
        LangfuseOtelSpanAttributes.TRACE_INPUT: _serialize(input),
        LangfuseOtelSpanAttributes.TRACE_OUTPUT: _serialize(output),
        LangfuseOtelSpanAttributes.TRACE_PUBLIC: public,
    }

    return {k: v for k, v in attributes.items() if v is not None}


def create_span_attributes(
    *,
    metadata: Optional[Any] = None,
    input: Optional[Any] = None,
    output: Optional[Any] = None,
    level: Optional[SpanLevel] = None,
    status_message: Optional[str] = None,
    version: Optional[str] = None,
    observation_type: Optional[
        Union[ObservationTypeSpanLike, Literal["event"]]
    ] = "span",
) -> dict:
    attributes = {
        LangfuseOtelSpanAttributes.OBSERVATION_TYPE: observation_type,
        LangfuseOtelSpanAttributes.OBSERVATION_LEVEL: level,
        LangfuseOtelSpanAttributes.OBSERVATION_STATUS_MESSAGE: status_message,
        LangfuseOtelSpanAttributes.VERSION: version,
        LangfuseOtelSpanAttributes.OBSERVATION_INPUT: _serialize(input),
        LangfuseOtelSpanAttributes.OBSERVATION_OUTPUT: _serialize(output),
        LangfuseOtelSpanAttributes.OBSERVATION_METADATA: merge_observation_metadata(
            None, serialize_observation_metadata(metadata)
        )[0],
    }

    return {k: v for k, v in attributes.items() if v is not None}


def create_generation_attributes(
    *,
    name: Optional[str] = None,
    completion_start_time: Optional[datetime] = None,
    metadata: Optional[Any] = None,
    level: Optional[SpanLevel] = None,
    status_message: Optional[str] = None,
    version: Optional[str] = None,
    model: Optional[str] = None,
    model_parameters: Optional[Dict[str, MapValue]] = None,
    input: Optional[Any] = None,
    output: Optional[Any] = None,
    usage_details: Optional[Dict[str, int]] = None,
    cost_details: Optional[Dict[str, float]] = None,
    prompt: Optional[PromptClient] = None,
    observation_type: Optional[ObservationTypeGenerationLike] = "generation",
) -> dict:
    attributes = {
        LangfuseOtelSpanAttributes.OBSERVATION_TYPE: observation_type,
        LangfuseOtelSpanAttributes.OBSERVATION_LEVEL: level,
        LangfuseOtelSpanAttributes.OBSERVATION_STATUS_MESSAGE: status_message,
        LangfuseOtelSpanAttributes.VERSION: version,
        LangfuseOtelSpanAttributes.OBSERVATION_INPUT: _serialize(input),
        LangfuseOtelSpanAttributes.OBSERVATION_OUTPUT: _serialize(output),
        LangfuseOtelSpanAttributes.OBSERVATION_MODEL: model,
        LangfuseOtelSpanAttributes.OBSERVATION_PROMPT_NAME: prompt.name
        if prompt and not prompt.is_fallback
        else None,
        LangfuseOtelSpanAttributes.OBSERVATION_PROMPT_VERSION: prompt.version
        if prompt and not prompt.is_fallback
        else None,
        LangfuseOtelSpanAttributes.OBSERVATION_USAGE_DETAILS: _serialize(usage_details),
        LangfuseOtelSpanAttributes.OBSERVATION_COST_DETAILS: _serialize(cost_details),
        LangfuseOtelSpanAttributes.OBSERVATION_COMPLETION_START_TIME: _serialize(
            completion_start_time
        ),
        LangfuseOtelSpanAttributes.OBSERVATION_MODEL_PARAMETERS: _serialize(
            model_parameters
        ),
        LangfuseOtelSpanAttributes.OBSERVATION_METADATA: merge_observation_metadata(
            None, serialize_observation_metadata(metadata)
        )[0],
    }

    return {k: v for k, v in attributes.items() if v is not None}


def _serialize(obj: Any) -> Optional[str]:
    if obj is None or isinstance(obj, str):
        return obj

    return json.dumps(obj, cls=EventSerializer)


def _flatten_and_serialize_metadata_values(
    metadata: Optional[Dict[str, Any]],
) -> Optional[Dict[str, str]]:
    if metadata is None:
        return None

    flattened_metadata: Dict[str, str] = {}

    def flatten_value(path: str, value: Any) -> None:
        if isinstance(value, dict):
            for nested_key, nested_value in value.items():
                flatten_value(f"{path}.{nested_key}", nested_value)

            return

        serialized_value = _serialize(value)

        if serialized_value is not None:
            flattened_metadata[path] = serialized_value

    for key, value in metadata.items():
        flatten_value(str(key), value)

    return flattened_metadata


def serialize_observation_metadata(
    metadata: Any,
) -> Union[None, str, Dict[str, str]]:
    """Serialize observation metadata for merge_observation_metadata.

    Values of dict metadata are serialized one by one, so a value that fails to
    serialize becomes `"<failed to serialize>"` instead of dropping all metadata.
    Keys with `None` values are skipped.

    Returns:
        None if there is no metadata, the serialized value for non-dict
        metadata, or the serialized values keyed by top-level key.
    """
    if metadata is None:
        return None

    if not isinstance(metadata, dict):
        return _serialize_metadata_value(metadata, top_level=True)

    return {
        str(key): _serialize_metadata_value(value)
        for key, value in metadata.items()
        if value is not None
    }


def merge_observation_metadata(
    previous: Optional[Dict[str, str]],
    serialized: Union[None, str, Dict[str, str]],
) -> Tuple[Optional[str], Optional[Dict[str, str]]]:
    """Merge serialized metadata into earlier metadata of the same observation.

    Observation metadata is written as one JSON object to the
    `langfuse.observation.metadata` attribute. Top-level keys of `serialized`
    overwrite keys in `previous`. Non-dict metadata replaces earlier metadata.

    Args:
        previous: Serialized values from earlier updates, keyed by top-level key
        serialized: Output of serialize_observation_metadata for this update

    Returns:
        The attribute value to write (None if there is nothing to write) and the
        serialized values to keep for the next update (None if there are none).

    Raises:
        ObservationMetadataKeyLimitError: If the merged metadata has more than
            MAX_OBSERVATION_METADATA_KEYS top-level keys.
    """
    if serialized is None:
        return None, previous

    if isinstance(serialized, str):
        return serialized, None

    merged = {**(previous or {}), **serialized}

    if len(merged) > MAX_OBSERVATION_METADATA_KEYS:
        message = (
            f"Observation metadata has {len(merged)} keys, which exceeds the "
            f"maximum of {MAX_OBSERVATION_METADATA_KEYS}."
        )
        raise ObservationMetadataKeyLimitError(message)

    if not merged:
        return None, None

    attribute_value = (
        "{"
        + ",".join(f"{json.dumps(key)}:{value}" for key, value in merged.items())
        + "}"
    )

    return attribute_value, merged


def drop_metadata_over_key_limit(metadata: _T) -> Optional[_T]:
    """Drop dict metadata with more than MAX_OBSERVATION_METADATA_KEYS keys.

    For integrations: the SDK raises on too many metadata keys, but raising inside
    instrumentation would break or lose the user's call. Integrations drop the
    metadata with a warning instead. Keys with `None` values are not counted.
    """
    if not isinstance(metadata, dict):
        return metadata

    key_count = sum(1 for value in metadata.values() if value is not None)

    if key_count > MAX_OBSERVATION_METADATA_KEYS:
        langfuse_logger.warning(
            "Dropping observation metadata: it has %s keys, which exceeds the "
            "maximum of %s.",
            key_count,
            MAX_OBSERVATION_METADATA_KEYS,
        )
        return None

    return metadata


def _serialize_metadata_value(value: Any, *, top_level: bool = False) -> str:
    if top_level and isinstance(value, str):
        return value

    try:
        return json.dumps(value, cls=EventSerializer, separators=_COMPACT_SEPARATORS)
    except Exception:
        return _FAILED_TO_SERIALIZE if top_level else json.dumps(_FAILED_TO_SERIALIZE)
