import json
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from langfuse.model import PromptClient

from ..serializer import EventSerializer
from ..types import MapValue, SpanLevel


class LangfuseSpanAttributes:
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
    OBSERVATION_NAME = "langfuse.observation.name"
    OBSERVATION_TYPE = "langfuse.observation.type"
    OBSERVATION_METADATA = "langfuse.observation.metadata"
    OBSERVATION_LEVEL = "langfuse.observation.level"
    OBSERVATION_STATUS_MESSAGE = "langfuse.observation.status_message"
    OBSERVATION_INPUT = "langfuse.observation.input"
    OBSERVATION_OUTPUT = "langfuse.observation.output"

    # Langfuse-observation of type Generation attributes
    OBSERVATION_COMPLETION_START_TIME = "langfuse.observation.completion_start_time"
    OBSERVATION_MODEL = "gen_ai.response.model"
    OBSERVATION_MODEL_PARAMETERS = "langfuse.observation.model_parameters"
    OBSERVATION_USAGE_DETAILS = "gen_ai.usage"
    OBSERVATION_COST_DETAILS = "langfuse.observation.cost_details"
    OBSERVATION_PROMPT_NAME = "langfuse.observation.prompt.name"
    OBSERVATION_PROMPT_VERSION = "langfuse.observation.prompt.version"

    # General
    ENVIRONMENT = "langfuse.environment"
    RELEASE = "langfuse.release"
    VERSION = "langfuse.version"

    # Internal
    AS_ROOT = "langfuse.internal.as_root"


def create_trace_attributes(
    *,
    name: Optional[str] = None,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    version: Optional[str] = None,
    release: Optional[str] = None,
    input: Optional[Any] = None,
    output: Optional[Any] = None,
    metadata: Optional[Any] = None,
    tags: Optional[List[str]] = None,
    public: Optional[bool] = None,
):
    attributes = {
        LangfuseSpanAttributes.TRACE_NAME: name,
        LangfuseSpanAttributes.TRACE_USER_ID: user_id,
        LangfuseSpanAttributes.TRACE_SESSION_ID: session_id,
        LangfuseSpanAttributes.VERSION: version,
        LangfuseSpanAttributes.RELEASE: release,
        LangfuseSpanAttributes.TRACE_INPUT: _serialize(input),
        LangfuseSpanAttributes.TRACE_OUTPUT: _serialize(output),
        LangfuseSpanAttributes.TRACE_METADATA: _flatten_and_serialize_metadata(
            metadata, "trace"
        ),
        LangfuseSpanAttributes.TRACE_TAGS: tags,
        LangfuseSpanAttributes.TRACE_PUBLIC: public,
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
):
    attributes = {
        LangfuseSpanAttributes.OBSERVATION_TYPE: "span",
        LangfuseSpanAttributes.OBSERVATION_LEVEL: level,
        LangfuseSpanAttributes.OBSERVATION_STATUS_MESSAGE: status_message,
        LangfuseSpanAttributes.VERSION: version,
        LangfuseSpanAttributes.OBSERVATION_INPUT: _serialize(input),
        LangfuseSpanAttributes.OBSERVATION_OUTPUT: _serialize(output),
        LangfuseSpanAttributes.OBSERVATION_METADATA: _flatten_and_serialize_metadata(
            metadata, "observation"
        ),
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
):
    attributes = {
        LangfuseSpanAttributes.OBSERVATION_TYPE: "generation",
        LangfuseSpanAttributes.OBSERVATION_NAME: name,
        LangfuseSpanAttributes.OBSERVATION_LEVEL: level,
        LangfuseSpanAttributes.OBSERVATION_STATUS_MESSAGE: status_message,
        LangfuseSpanAttributes.VERSION: version,
        LangfuseSpanAttributes.OBSERVATION_INPUT: _serialize(input),
        LangfuseSpanAttributes.OBSERVATION_OUTPUT: _serialize(output),
        LangfuseSpanAttributes.OBSERVATION_METADATA: _flatten_and_serialize_metadata(
            metadata, "observation"
        ),
        LangfuseSpanAttributes.OBSERVATION_MODEL: model,
        LangfuseSpanAttributes.OBSERVATION_PROMPT_NAME: prompt and prompt.name,
        LangfuseSpanAttributes.OBSERVATION_PROMPT_VERSION: prompt and prompt.version,
        LangfuseSpanAttributes.OBSERVATION_USAGE_DETAILS: _serialize(usage_details),
        LangfuseSpanAttributes.OBSERVATION_COST_DETAILS: _serialize(cost_details),
        LangfuseSpanAttributes.OBSERVATION_COMPLETION_START_TIME: _serialize(
            completion_start_time
        ),
        LangfuseSpanAttributes.OBSERVATION_MODEL_PARAMETERS: _serialize(
            model_parameters
        ),
    }

    return {k: v for k, v in attributes.items() if v is not None}


def _serialize(obj):
    return json.dumps(obj, cls=EventSerializer) if obj is not None else None


def _flatten_and_serialize_metadata(
    metadata: Any, type: Literal["observation", "trace"]
):
    prefix = (
        LangfuseSpanAttributes.OBSERVATION_METADATA
        if type == "observation"
        else LangfuseSpanAttributes.TRACE_METADATA
    )

    metadata_attributes = {}

    if not isinstance(metadata, dict):
        metadata_attributes[prefix] = _serialize(metadata)
    else:
        for key, value in metadata.items():
            metadata_attributes[f"{prefix}.{key}"] = _serialize(value)

    return metadata_attributes
