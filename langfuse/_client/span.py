"""OTEL span wrapper for Langfuse.

This module defines custom span classes that extend OpenTelemetry spans with
Langfuse-specific functionality. These wrapper classes provide methods for
creating, updating, and scoring various types of spans used in AI application tracing.

Classes:
- LangfuseObservationWrapper: Abstract base class for all Langfuse spans
- LangfuseSpan: Implementation for general-purpose spans
- LangfuseGeneration: Specialized span implementation for LLM generations

All span classes provide methods for media processing, attribute management,
and scoring integration specific to Langfuse's observability platform.
"""

from datetime import datetime
from time import time_ns
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    Type,
    Union,
    cast,
    overload,
)

from opentelemetry import trace as otel_trace_api
from opentelemetry.trace.status import Status, StatusCode
from opentelemetry.util._decorator import _AgnosticContextManager

from langfuse.model import PromptClient

if TYPE_CHECKING:
    from langfuse._client.client import Langfuse


from langfuse._client.attributes import (
    LangfuseOtelSpanAttributes,
    create_generation_attributes,
    create_span_attributes,
    create_trace_attributes,
)
from langfuse._client.constants import (
    MASK_FALLBACK_VALUE,
    ObservationTypeGenerationLike,
    ObservationTypeLiteral,
    ObservationTypeLiteralNoEvent,
    ObservationTypeSpanLike,
    get_observation_types_list,
)
from langfuse._client.utils import get_string_span_attribute
from langfuse.api import MapValue, ScoreDataType
from langfuse.logger import langfuse_logger
from langfuse.types import SpanLevel

# Factory mapping for observation classes
# Note: "event" is handled separately due to special instantiation logic
# Populated after class definitions
_OBSERVATION_CLASS_MAP: Dict[str, Type["LangfuseObservationWrapper"]] = {}


def _is_observation_metadata_key(key: str) -> bool:
    prefix = LangfuseOtelSpanAttributes.OBSERVATION_METADATA
    return key == prefix or key.startswith(prefix + ".")


# Attributes the SDK may write in later updates of an observation. Metadata
# leaves room for them, so that a later update (e.g. the output) still fits
# into the span. Span-like observations and events only use the shared keys;
# generation-like observations, and writes whose observation type is unknown,
# also reserve the model keys. Mirrors langfuse-js.
_RESERVED_SPAN_ATTRIBUTE_KEYS: Tuple[str, ...] = (
    LangfuseOtelSpanAttributes.OBSERVATION_TYPE,
    LangfuseOtelSpanAttributes.OBSERVATION_LEVEL,
    LangfuseOtelSpanAttributes.OBSERVATION_STATUS_MESSAGE,
    LangfuseOtelSpanAttributes.VERSION,
    LangfuseOtelSpanAttributes.ENVIRONMENT,
    LangfuseOtelSpanAttributes.OBSERVATION_INPUT,
    LangfuseOtelSpanAttributes.OBSERVATION_OUTPUT,
)
_RESERVED_OBSERVATION_ATTRIBUTE_KEYS: Tuple[str, ...] = (
    *_RESERVED_SPAN_ATTRIBUTE_KEYS,
    LangfuseOtelSpanAttributes.OBSERVATION_MODEL,
    LangfuseOtelSpanAttributes.OBSERVATION_USAGE_DETAILS,
    LangfuseOtelSpanAttributes.OBSERVATION_COST_DETAILS,
    LangfuseOtelSpanAttributes.OBSERVATION_COMPLETION_START_TIME,
    LangfuseOtelSpanAttributes.OBSERVATION_MODEL_PARAMETERS,
    LangfuseOtelSpanAttributes.OBSERVATION_PROMPT_NAME,
    LangfuseOtelSpanAttributes.OBSERVATION_PROMPT_VERSION,
)


def _reserved_attribute_keys(observation_type: Optional[str]) -> Tuple[str, ...]:
    if (
        observation_type in get_observation_types_list(ObservationTypeSpanLike)
        or observation_type == "event"
    ):
        return _RESERVED_SPAN_ATTRIBUTE_KEYS

    return _RESERVED_OBSERVATION_ATTRIBUTE_KEYS


def _drop_attributes_over_span_limit(
    span: otel_trace_api.Span,
    attributes: Dict[str, Any],
    reserved_keys: Tuple[str, ...] = _RESERVED_OBSERVATION_ATTRIBUTE_KEYS,
) -> Dict[str, Any]:
    """Drop new attributes that would exceed the span's attribute limit.

    Once a span is full, the Python OTel SDK evicts the oldest attribute, which
    would drop Langfuse core attributes (input, model, ...) written earlier.
    OTel JS drops the newest attribute instead; this helper gives Python the
    same semantics so existing attributes are never evicted.

    1. Metadata budget: counts the attributes already on the span, the new
       keys, and the reserved keys not yet on the span, which later updates
       may still write. Excess new metadata keys are dropped from the tail.
       ``reserved_keys`` defaults to the full generation set, the safe choice
       when the observation type is unknown.
    2. Hard guard: if the remaining new keys still exceed the physical
       capacity (limit minus attributes already on the span), the excess new
       keys are dropped from the tail. Reserved slots do not apply here, so
       reserved attributes such as the output still land.

    Overwrites of existing keys are always kept, as they need no extra slot.
    At most one warning is logged per write. Spans without SDK limits (e.g.
    non-recording spans) are left untouched. Never raises.
    """
    try:
        max_attributes = getattr(
            getattr(span, "_limits", None), "max_span_attributes", None
        )
        if not isinstance(max_attributes, int):
            return attributes

        existing = getattr(span, "attributes", None) or {}
        new_keys = [
            key
            for key, value in attributes.items()
            if value is not None and key not in existing
        ]
        new_key_set = set(new_keys)
        reserved_count = sum(
            1 for key in reserved_keys if key not in existing and key not in new_key_set
        )
        if len(existing) + reserved_count + len(new_keys) <= max_attributes:
            return attributes

        new_metadata_keys = [
            key for key in new_keys if _is_observation_metadata_key(key)
        ]
        free_metadata_slots = max(
            0,
            max_attributes
            - len(existing)
            - reserved_count
            - (len(new_keys) - len(new_metadata_keys)),
        )
        dropped_metadata = new_metadata_keys[free_metadata_slots:]
        dropped_metadata_set = set(dropped_metadata)

        remaining_new_keys = [
            key for key in new_keys if key not in dropped_metadata_set
        ]
        capacity = max(0, max_attributes - len(existing))
        dropped_other = remaining_new_keys[capacity:]
        # The metadata budget is never looser than the hard guard, so only
        # non-metadata keys can be dropped here. Count any metadata defensively.
        dropped_metadata += [
            key for key in dropped_other if _is_observation_metadata_key(key)
        ]
        dropped_other = [
            key for key in dropped_other if not _is_observation_metadata_key(key)
        ]

        if not dropped_metadata and not dropped_other:
            return attributes

        _warn_dropped_attributes(
            span_name=getattr(span, "name", None),
            max_attributes=max_attributes,
            dropped_metadata=dropped_metadata,
            dropped_other=dropped_other,
        )

        dropped_keys = set(dropped_metadata) | set(dropped_other)
        return {
            key: value for key, value in attributes.items() if key not in dropped_keys
        }
    except Exception as e:
        langfuse_logger.debug("Failed to apply span attribute limit: %s", e)
        return attributes


def _warn_dropped_attributes(
    *,
    span_name: Optional[str],
    max_attributes: int,
    dropped_metadata: List[str],
    dropped_other: List[str],
) -> None:
    metadata_prefix = LangfuseOtelSpanAttributes.OBSERVATION_METADATA + "."
    dropped_keys = ", ".join(
        (
            [key.removeprefix(metadata_prefix) for key in dropped_metadata]
            + dropped_other
        )[:5]
    )
    if not dropped_other:
        langfuse_logger.warning(
            "Dropped %s metadata key(s) from observation '%s' to stay within the "
            "span attribute limit of %s (SpanLimits.max_span_attributes / "
            "OTEL_SPAN_ATTRIBUTE_COUNT_LIMIT). Dropped keys include: %s",
            len(dropped_metadata),
            span_name,
            max_attributes,
            dropped_keys,
        )
        return

    langfuse_logger.warning(
        "Dropped %s metadata key(s) and %s other attribute(s) from observation "
        "'%s' to stay within the span attribute limit of %s "
        "(SpanLimits.max_span_attributes / OTEL_SPAN_ATTRIBUTE_COUNT_LIMIT). "
        "Dropped keys include: %s",
        len(dropped_metadata),
        len(dropped_other),
        span_name,
        max_attributes,
        dropped_keys,
    )


def _set_span_attributes_within_limit(
    span: otel_trace_api.Span,
    attributes: Dict[str, Any],
    reserved_keys: Tuple[str, ...] = _RESERVED_OBSERVATION_ATTRIBUTE_KEYS,
) -> None:
    """Set attributes on a span without evicting existing attributes."""
    span.set_attributes(
        _drop_attributes_over_span_limit(span, attributes, reserved_keys)
    )


class LangfuseObservationWrapper:
    """Abstract base class for all Langfuse span types.

    This class provides common functionality for all Langfuse span types, including
    media processing, attribute management, and scoring. It wraps an OpenTelemetry
    span and extends it with Langfuse-specific features.

    Attributes:
        _otel_span: The underlying OpenTelemetry span
        _langfuse_client: Reference to the parent Langfuse client
        trace_id: The trace ID for this span
        observation_id: The observation ID (span ID) for this span
    """

    def __init__(
        self,
        *,
        otel_span: otel_trace_api.Span,
        langfuse_client: "Langfuse",
        as_type: ObservationTypeLiteral,
        input: Optional[Any] = None,
        output: Optional[Any] = None,
        metadata: Optional[Any] = None,
        environment: Optional[str] = None,
        release: Optional[str] = None,
        version: Optional[str] = None,
        level: Optional[SpanLevel] = None,
        status_message: Optional[str] = None,
        completion_start_time: Optional[datetime] = None,
        model: Optional[str] = None,
        model_parameters: Optional[Dict[str, MapValue]] = None,
        usage_details: Optional[Dict[str, int]] = None,
        cost_details: Optional[Dict[str, float]] = None,
        prompt: Optional[PromptClient] = None,
    ):
        """Initialize a new Langfuse span wrapper.

        Args:
            otel_span: The OpenTelemetry span to wrap
            langfuse_client: Reference to the parent Langfuse client
            as_type: The type of span ("span" or "generation")
            input: Input data for the span (any JSON-serializable object)
            output: Output data from the span (any JSON-serializable object)
            metadata: Additional metadata to associate with the span
            environment: Local tracing environment fallback. If the underlying
                OpenTelemetry span already has `langfuse.environment` from
                propagated context or baggage, that propagated value takes
                precedence over this fallback.
            release: Release identifier for the application
            version: Version identifier for the code or component
            level: Importance level of the span (info, warning, error)
            status_message: Optional status message for the span
            completion_start_time: When the model started generating the response
            model: Name/identifier of the AI model used (e.g., "gpt-4")
            model_parameters: Parameters used for the model (e.g., temperature, max_tokens)
            usage_details: Token usage information (e.g., prompt_tokens, completion_tokens)
            cost_details: Cost information for the model call
            prompt: Associated prompt template from Langfuse prompt management
        """
        self._otel_span = otel_span
        reserved_keys = _reserved_attribute_keys(as_type)
        _set_span_attributes_within_limit(
            self._otel_span,
            {LangfuseOtelSpanAttributes.OBSERVATION_TYPE: as_type},
            reserved_keys,
        )
        self._langfuse_client = langfuse_client
        self._observation_type = as_type

        self.trace_id = self._langfuse_client._get_otel_trace_id(otel_span)
        self.id = self._langfuse_client._get_otel_span_id(otel_span)

        existing_environment = get_string_span_attribute(
            self._otel_span, LangfuseOtelSpanAttributes.ENVIRONMENT
        )
        self._environment = (
            existing_environment or environment or self._langfuse_client._environment
        )
        if self._environment is not None:
            _set_span_attributes_within_limit(
                self._otel_span,
                {LangfuseOtelSpanAttributes.ENVIRONMENT: self._environment},
                reserved_keys,
            )

        self._release = release or self._langfuse_client._release
        if self._release is not None:
            _set_span_attributes_within_limit(
                self._otel_span,
                {LangfuseOtelSpanAttributes.RELEASE: self._release},
                reserved_keys,
            )

        # Handle media only if span is sampled
        if self._otel_span.is_recording():
            media_processed_input = self._process_media_and_apply_mask(
                data=input, field="input", span=self._otel_span
            )
            media_processed_output = self._process_media_and_apply_mask(
                data=output, field="output", span=self._otel_span
            )
            media_processed_metadata = self._process_media_and_apply_mask(
                data=metadata, field="metadata", span=self._otel_span
            )

            attributes = {}

            if as_type in get_observation_types_list(ObservationTypeGenerationLike):
                attributes = create_generation_attributes(
                    input=media_processed_input,
                    output=media_processed_output,
                    metadata=media_processed_metadata,
                    version=version,
                    level=level,
                    status_message=status_message,
                    completion_start_time=completion_start_time,
                    model=model,
                    model_parameters=model_parameters,
                    usage_details=usage_details,
                    cost_details=cost_details,
                    prompt=prompt,
                    observation_type=cast(
                        ObservationTypeGenerationLike,
                        as_type,
                    ),
                )

            else:
                # For span-like types and events
                attributes = create_span_attributes(
                    input=media_processed_input,
                    output=media_processed_output,
                    metadata=media_processed_metadata,
                    version=version,
                    level=level,
                    status_message=status_message,
                    observation_type=cast(
                        Optional[Union[ObservationTypeSpanLike, Literal["event"]]],
                        as_type
                        if as_type
                        in get_observation_types_list(ObservationTypeSpanLike)
                        or as_type == "event"
                        else None,
                    ),
                )

            # We don't want to overwrite the observation type, and already set it
            attributes.pop(LangfuseOtelSpanAttributes.OBSERVATION_TYPE, None)

            _set_span_attributes_within_limit(
                self._otel_span,
                {k: v for k, v in attributes.items() if v is not None},
                reserved_keys,
            )
            # Set OTEL span status if level is ERROR
            self._set_otel_span_status_if_error(
                level=level, status_message=status_message
            )

    def end(self, *, end_time: Optional[int] = None) -> "LangfuseObservationWrapper":
        """End the span, marking it as completed.

        This method ends the wrapped OpenTelemetry span, marking the end of the
        operation being traced. After this method is called, the span is considered
        complete and can no longer be modified.

        Args:
            end_time: Optional explicit end time in nanoseconds since epoch
        """
        self._otel_span.end(end_time=end_time)

        return self

    def set_trace_as_public(self) -> "LangfuseObservationWrapper":
        """Make this trace publicly accessible via its URL.

        When a trace is published, anyone with the trace link can view the full trace
        without needing to be logged in to Langfuse. This action cannot be undone
        programmatically - once any span in a trace is published, the entire trace
        becomes public.

        Returns:
            The span instance for method chaining.
        """
        if not self._otel_span.is_recording():
            return self

        attributes = create_trace_attributes(public=True)

        _set_span_attributes_within_limit(
            self._otel_span,
            attributes,
            _reserved_attribute_keys(self._observation_type),
        )

        return self

    @overload
    def score(
        self,
        *,
        name: str,
        value: float,
        score_id: Optional[str] = None,
        data_type: Optional[
            Literal[ScoreDataType.NUMERIC, ScoreDataType.BOOLEAN]
        ] = None,
        comment: Optional[str] = None,
        config_id: Optional[str] = None,
        timestamp: Optional[datetime] = None,
        metadata: Optional[Any] = None,
    ) -> None: ...

    @overload
    def score(
        self,
        *,
        name: str,
        value: str,
        score_id: Optional[str] = None,
        data_type: Optional[
            Literal[
                ScoreDataType.CATEGORICAL, ScoreDataType.TEXT, ScoreDataType.CORRECTION
            ]
        ] = ScoreDataType.CATEGORICAL,
        comment: Optional[str] = None,
        config_id: Optional[str] = None,
        timestamp: Optional[datetime] = None,
        metadata: Optional[Any] = None,
    ) -> None: ...

    def score(
        self,
        *,
        name: str,
        value: Union[float, str],
        score_id: Optional[str] = None,
        data_type: Optional[ScoreDataType] = None,
        comment: Optional[str] = None,
        config_id: Optional[str] = None,
        timestamp: Optional[datetime] = None,
        metadata: Optional[Any] = None,
    ) -> None:
        """Create a score for this specific span.

        This method creates a score associated with this specific span (observation).
        Scores can represent any kind of evaluation, feedback, or quality metric.
        The score uses this span wrapper's resolved environment. That means a
        request-scoped environment propagated via ``propagate_attributes`` takes
        precedence over the client-level environment when this score is created.

        Args:
            name: Name of the score (e.g., "relevance", "accuracy")
            value: Score value (numeric for NUMERIC/BOOLEAN, string for CATEGORICAL/TEXT/CORRECTION)
            score_id: Optional custom ID for the score (auto-generated if not provided)
            data_type: Type of score (NUMERIC, BOOLEAN, CATEGORICAL, TEXT, or CORRECTION)
            comment: Optional comment or explanation for the score
            config_id: Optional ID of a score config defined in Langfuse
            timestamp: Optional timestamp for the score (defaults to current UTC time)
            metadata: Optional metadata to be attached to the score

        Example:
            ```python
            with langfuse.start_as_current_observation(name="process-query") as span:
                # Do work
                result = process_data()

                # Score the span
                span.score(
                    name="accuracy",
                    value=0.95,
                    data_type="NUMERIC",
                    comment="High accuracy result"
                )
            ```
        """
        self._langfuse_client.create_score(
            name=name,
            value=cast(str, value),
            trace_id=self.trace_id,
            observation_id=self.id,
            score_id=score_id,
            data_type=cast(Literal["CATEGORICAL", "TEXT", "CORRECTION"], data_type),
            comment=comment,
            config_id=config_id,
            timestamp=timestamp,
            metadata=metadata,
            environment=self._environment,
        )

    @overload
    def score_trace(
        self,
        *,
        name: str,
        value: float,
        score_id: Optional[str] = None,
        data_type: Optional[
            Literal[ScoreDataType.NUMERIC, ScoreDataType.BOOLEAN]
        ] = None,
        comment: Optional[str] = None,
        config_id: Optional[str] = None,
        timestamp: Optional[datetime] = None,
        metadata: Optional[Any] = None,
    ) -> None: ...

    @overload
    def score_trace(
        self,
        *,
        name: str,
        value: str,
        score_id: Optional[str] = None,
        data_type: Optional[
            Literal[
                ScoreDataType.CATEGORICAL, ScoreDataType.TEXT, ScoreDataType.CORRECTION
            ]
        ] = ScoreDataType.CATEGORICAL,
        comment: Optional[str] = None,
        config_id: Optional[str] = None,
        timestamp: Optional[datetime] = None,
        metadata: Optional[Any] = None,
    ) -> None: ...

    def score_trace(
        self,
        *,
        name: str,
        value: Union[float, str],
        score_id: Optional[str] = None,
        data_type: Optional[ScoreDataType] = None,
        comment: Optional[str] = None,
        config_id: Optional[str] = None,
        timestamp: Optional[datetime] = None,
        metadata: Optional[Any] = None,
    ) -> None:
        """Create a score for the entire trace that this span belongs to.

        This method creates a score associated with the entire trace that this span
        belongs to, rather than the specific span. This is useful for overall
        evaluations that apply to the complete trace. The score uses this span
        wrapper's resolved environment, including any request-scoped environment
        propagated via ``propagate_attributes``.

        Args:
            name: Name of the score (e.g., "user_satisfaction", "overall_quality")
            value: Score value (numeric for NUMERIC/BOOLEAN, string for CATEGORICAL/TEXT/CORRECTION)
            score_id: Optional custom ID for the score (auto-generated if not provided)
            data_type: Type of score (NUMERIC, BOOLEAN, CATEGORICAL, TEXT, or CORRECTION)
            comment: Optional comment or explanation for the score
            config_id: Optional ID of a score config defined in Langfuse
            timestamp: Optional timestamp for the score (defaults to current UTC time)
            metadata: Optional metadata to be attached to the score

        Example:
            ```python
            with langfuse.start_as_current_observation(name="handle-request") as span:
                # Process the complete request
                result = process_request()

                # Score the entire trace (not just this span)
                span.score_trace(
                    name="overall_quality",
                    value=0.9,
                    data_type="NUMERIC",
                    comment="Good overall experience"
                )
            ```
        """
        self._langfuse_client.create_score(
            name=name,
            value=cast(str, value),
            trace_id=self.trace_id,
            score_id=score_id,
            data_type=cast(Literal["CATEGORICAL", "TEXT", "CORRECTION"], data_type),
            comment=comment,
            config_id=config_id,
            timestamp=timestamp,
            metadata=metadata,
            environment=self._environment,
        )

    def _set_processed_span_attributes(
        self,
        *,
        span: otel_trace_api.Span,
        as_type: Optional[ObservationTypeLiteral] = None,
        input: Optional[Any] = None,
        output: Optional[Any] = None,
        metadata: Optional[Any] = None,
    ) -> None:
        """Set span attributes after processing media and applying masks.

        Internal method that processes media in the input, output, and metadata
        and applies any configured masking before setting them as span attributes.

        Args:
            span: The OpenTelemetry span to set attributes on
            as_type: The type of span ("span" or "generation")
            input: Input data to process and set
            output: Output data to process and set
            metadata: Metadata to process and set
        """
        processed_input = self._process_media_and_apply_mask(
            span=span,
            data=input,
            field="input",
        )
        processed_output = self._process_media_and_apply_mask(
            span=span,
            data=output,
            field="output",
        )
        processed_metadata = self._process_media_and_apply_mask(
            span=span,
            data=metadata,
            field="metadata",
        )

        media_processed_attributes = (
            create_generation_attributes(
                input=processed_input,
                output=processed_output,
                metadata=processed_metadata,
            )
            if as_type == "generation"
            else create_span_attributes(
                input=processed_input,
                output=processed_output,
                metadata=processed_metadata,
            )
        )

        _set_span_attributes_within_limit(
            span,
            media_processed_attributes,
            _reserved_attribute_keys(as_type or self._observation_type),
        )

    def _process_media_and_apply_mask(
        self,
        *,
        data: Optional[Any] = None,
        span: otel_trace_api.Span,
        field: Union[Literal["input"], Literal["output"], Literal["metadata"]],
    ) -> Optional[Any]:
        """Process media in an attribute and apply masking.

        Internal method that processes any media content in the data and applies
        the configured masking function to the result.

        Args:
            data: The data to process
            span: The OpenTelemetry span context
            field: Which field this data represents (input, output, or metadata)

        Returns:
            The processed and masked data
        """
        return self._mask_attribute(
            data=self._process_media_in_attribute(data=data, field=field), field=field
        )

    def _mask_attribute(
        self,
        *,
        data: Any,
        field: Optional[
            Union[Literal["input"], Literal["output"], Literal["metadata"]]
        ] = None,
    ) -> Any:
        """Apply the configured mask function to data.

        Internal method that applies the client's configured masking function to
        the provided data, with error handling and fallback.

        Args:
            data: The data to mask
            field: The attribute the data belongs to

        Returns:
            The masked data, or the original data if no mask is configured
        """
        if not self._langfuse_client._mask:
            return data

        try:
            return self._langfuse_client._mask(data=data)
        except Exception as e:
            langfuse_logger.error(
                "Masking error: Custom mask function threw exception when processing "
                "data. Using fallback masking. Error: %s",
                e,
            )

            # Dict metadata is written per key, so mask each key instead of
            # writing a plain string to the bare metadata attribute
            if field == "metadata" and isinstance(data, dict):
                return {key: MASK_FALLBACK_VALUE for key in data}

            return MASK_FALLBACK_VALUE

    def _process_media_in_attribute(
        self,
        *,
        data: Optional[Any] = None,
        field: Union[Literal["input"], Literal["output"], Literal["metadata"]],
    ) -> Optional[Any]:
        """Process any media content in the attribute data.

        Internal method that identifies and processes any media content in the
        provided data, using the client's media manager.

        Args:
            data: The data to process for media content
            span: The OpenTelemetry span context
            field: Which field this data represents (input, output, or metadata)

        Returns:
            The data with any media content processed
        """
        if self._langfuse_client._resources is not None:
            return (
                self._langfuse_client._resources._media_manager._find_and_process_media(
                    data=data,
                    field=field,
                    trace_id=self.trace_id,
                    observation_id=self.id,
                )
            )

        return data

    def _set_otel_span_status_if_error(
        self, *, level: Optional[SpanLevel] = None, status_message: Optional[str] = None
    ) -> None:
        """Set OpenTelemetry span status to ERROR if level is ERROR.

        This method sets the underlying OpenTelemetry span status to ERROR when the
        Langfuse observation level is set to ERROR, ensuring consistency between
        Langfuse and OpenTelemetry error states.

        Args:
            level: The span level to check
            status_message: Optional status message to include as description
        """
        if level == "ERROR" and self._otel_span.is_recording():
            try:
                self._otel_span.set_status(
                    Status(StatusCode.ERROR, description=status_message)
                )
            except Exception:
                # Silently ignore any errors when setting OTEL status to avoid existing flow disruptions
                pass

    def update(
        self,
        *,
        name: Optional[str] = None,
        input: Optional[Any] = None,
        output: Optional[Any] = None,
        metadata: Optional[Any] = None,
        version: Optional[str] = None,
        level: Optional[SpanLevel] = None,
        status_message: Optional[str] = None,
        completion_start_time: Optional[datetime] = None,
        model: Optional[str] = None,
        model_parameters: Optional[Dict[str, MapValue]] = None,
        usage_details: Optional[Dict[str, int]] = None,
        cost_details: Optional[Dict[str, float]] = None,
        prompt: Optional[PromptClient] = None,
        **kwargs: Any,
    ) -> "LangfuseObservationWrapper":
        """Update this observation with new information.

        This method updates the observation with new information that becomes available
        during execution, such as outputs, metadata, or status changes.

        Args:
            name: Observation name
            input: Updated input data for the operation
            output: Output data from the operation
            metadata: Additional metadata to associate with the observation
            version: Version identifier for the code or component
            level: Importance level of the observation (info, warning, error)
            status_message: Optional status message for the observation
            completion_start_time: When the generation started (for generation types)
            model: Model identifier used (for generation types)
            model_parameters: Parameters passed to the model (for generation types)
            usage_details: Token or other usage statistics (for generation types)
            cost_details: Cost breakdown for the operation (for generation types)
            prompt: Reference to the prompt used (for generation types)
            **kwargs: Additional keyword arguments (ignored)
        """
        if not self._otel_span.is_recording():
            return self

        processed_input = self._process_media_and_apply_mask(
            data=input, field="input", span=self._otel_span
        )
        processed_output = self._process_media_and_apply_mask(
            data=output, field="output", span=self._otel_span
        )
        processed_metadata = self._process_media_and_apply_mask(
            data=metadata, field="metadata", span=self._otel_span
        )

        if name:
            self._otel_span.update_name(name)

        if self._observation_type in get_observation_types_list(
            ObservationTypeGenerationLike
        ):
            attributes = create_generation_attributes(
                input=processed_input,
                output=processed_output,
                metadata=processed_metadata,
                version=version,
                level=level,
                status_message=status_message,
                observation_type=cast(
                    ObservationTypeGenerationLike,
                    self._observation_type,
                ),
                completion_start_time=completion_start_time,
                model=model,
                model_parameters=model_parameters,
                usage_details=usage_details,
                cost_details=cost_details,
                prompt=prompt,
            )
        else:
            # For span-like types and events
            attributes = create_span_attributes(
                input=processed_input,
                output=processed_output,
                metadata=processed_metadata,
                version=version,
                level=level,
                status_message=status_message,
                observation_type=cast(
                    Optional[Union[ObservationTypeSpanLike, Literal["event"]]],
                    self._observation_type
                    if self._observation_type
                    in get_observation_types_list(ObservationTypeSpanLike)
                    or self._observation_type == "event"
                    else None,
                ),
            )

        _set_span_attributes_within_limit(
            self._otel_span,
            attributes,
            _reserved_attribute_keys(self._observation_type),
        )
        # Set OTEL span status if level is ERROR
        self._set_otel_span_status_if_error(level=level, status_message=status_message)

        return self

    @overload
    def start_observation(
        self,
        *,
        name: str,
        as_type: Literal["span"],
        input: Optional[Any] = None,
        output: Optional[Any] = None,
        metadata: Optional[Any] = None,
        version: Optional[str] = None,
        level: Optional[SpanLevel] = None,
        status_message: Optional[str] = None,
    ) -> "LangfuseSpan": ...

    @overload
    def start_observation(
        self,
        *,
        name: str,
        as_type: Literal["generation"],
        input: Optional[Any] = None,
        output: Optional[Any] = None,
        metadata: Optional[Any] = None,
        version: Optional[str] = None,
        level: Optional[SpanLevel] = None,
        status_message: Optional[str] = None,
        completion_start_time: Optional[datetime] = None,
        model: Optional[str] = None,
        model_parameters: Optional[Dict[str, MapValue]] = None,
        usage_details: Optional[Dict[str, int]] = None,
        cost_details: Optional[Dict[str, float]] = None,
        prompt: Optional[PromptClient] = None,
    ) -> "LangfuseGeneration": ...

    @overload
    def start_observation(
        self,
        *,
        name: str,
        as_type: Literal["agent"],
        input: Optional[Any] = None,
        output: Optional[Any] = None,
        metadata: Optional[Any] = None,
        version: Optional[str] = None,
        level: Optional[SpanLevel] = None,
        status_message: Optional[str] = None,
    ) -> "LangfuseAgent": ...

    @overload
    def start_observation(
        self,
        *,
        name: str,
        as_type: Literal["tool"],
        input: Optional[Any] = None,
        output: Optional[Any] = None,
        metadata: Optional[Any] = None,
        version: Optional[str] = None,
        level: Optional[SpanLevel] = None,
        status_message: Optional[str] = None,
    ) -> "LangfuseTool": ...

    @overload
    def start_observation(
        self,
        *,
        name: str,
        as_type: Literal["chain"],
        input: Optional[Any] = None,
        output: Optional[Any] = None,
        metadata: Optional[Any] = None,
        version: Optional[str] = None,
        level: Optional[SpanLevel] = None,
        status_message: Optional[str] = None,
    ) -> "LangfuseChain": ...

    @overload
    def start_observation(
        self,
        *,
        name: str,
        as_type: Literal["retriever"],
        input: Optional[Any] = None,
        output: Optional[Any] = None,
        metadata: Optional[Any] = None,
        version: Optional[str] = None,
        level: Optional[SpanLevel] = None,
        status_message: Optional[str] = None,
    ) -> "LangfuseRetriever": ...

    @overload
    def start_observation(
        self,
        *,
        name: str,
        as_type: Literal["evaluator"],
        input: Optional[Any] = None,
        output: Optional[Any] = None,
        metadata: Optional[Any] = None,
        version: Optional[str] = None,
        level: Optional[SpanLevel] = None,
        status_message: Optional[str] = None,
    ) -> "LangfuseEvaluator": ...

    @overload
    def start_observation(
        self,
        *,
        name: str,
        as_type: Literal["embedding"],
        input: Optional[Any] = None,
        output: Optional[Any] = None,
        metadata: Optional[Any] = None,
        version: Optional[str] = None,
        level: Optional[SpanLevel] = None,
        status_message: Optional[str] = None,
        completion_start_time: Optional[datetime] = None,
        model: Optional[str] = None,
        model_parameters: Optional[Dict[str, MapValue]] = None,
        usage_details: Optional[Dict[str, int]] = None,
        cost_details: Optional[Dict[str, float]] = None,
        prompt: Optional[PromptClient] = None,
    ) -> "LangfuseEmbedding": ...

    @overload
    def start_observation(
        self,
        *,
        name: str,
        as_type: Literal["guardrail"],
        input: Optional[Any] = None,
        output: Optional[Any] = None,
        metadata: Optional[Any] = None,
        version: Optional[str] = None,
        level: Optional[SpanLevel] = None,
        status_message: Optional[str] = None,
    ) -> "LangfuseGuardrail": ...

    @overload
    def start_observation(
        self,
        *,
        name: str,
        as_type: Literal["event"],
        input: Optional[Any] = None,
        output: Optional[Any] = None,
        metadata: Optional[Any] = None,
        version: Optional[str] = None,
        level: Optional[SpanLevel] = None,
        status_message: Optional[str] = None,
    ) -> "LangfuseEvent": ...

    def start_observation(
        self,
        *,
        name: str,
        as_type: ObservationTypeLiteral = "span",
        input: Optional[Any] = None,
        output: Optional[Any] = None,
        metadata: Optional[Any] = None,
        version: Optional[str] = None,
        level: Optional[SpanLevel] = None,
        status_message: Optional[str] = None,
        completion_start_time: Optional[datetime] = None,
        model: Optional[str] = None,
        model_parameters: Optional[Dict[str, MapValue]] = None,
        usage_details: Optional[Dict[str, int]] = None,
        cost_details: Optional[Dict[str, float]] = None,
        prompt: Optional[PromptClient] = None,
    ) -> Union[
        "LangfuseSpan",
        "LangfuseGeneration",
        "LangfuseAgent",
        "LangfuseTool",
        "LangfuseChain",
        "LangfuseRetriever",
        "LangfuseEvaluator",
        "LangfuseEmbedding",
        "LangfuseGuardrail",
        "LangfuseEvent",
    ]:
        """Create a new child observation of the specified type.

        This is the generic method for creating any type of child observation.
        Unlike start_as_current_observation(), this method does not set the new
        observation as the current observation in the context.

        Args:
            name: Name of the observation
            as_type: Type of observation to create
            input: Input data for the operation
            output: Output data from the operation
            metadata: Additional metadata to associate with the observation
            version: Version identifier for the code or component
            level: Importance level of the observation (info, warning, error)
            status_message: Optional status message for the observation
            completion_start_time: When the model started generating (for generation types)
            model: Name/identifier of the AI model used (for generation types)
            model_parameters: Parameters used for the model (for generation types)
            usage_details: Token usage information (for generation types)
            cost_details: Cost information (for generation types)
            prompt: Associated prompt template (for generation types)

        Returns:
            A new observation of the specified type that must be ended with .end()
        """
        if as_type == "event":
            timestamp = time_ns()
            event_span = self._langfuse_client._otel_tracer.start_span(
                name=name, start_time=timestamp
            )
            return cast(
                LangfuseEvent,
                LangfuseEvent(
                    otel_span=event_span,
                    langfuse_client=self._langfuse_client,
                    input=input,
                    output=output,
                    metadata=metadata,
                    environment=self._environment,
                    release=self._release,
                    version=version,
                    level=level,
                    status_message=status_message,
                ).end(end_time=timestamp),
            )

        observation_class = _OBSERVATION_CLASS_MAP.get(as_type)
        if not observation_class:
            langfuse_logger.warning(
                "Unknown observation type: %s, falling back to LangfuseSpan", as_type
            )
            observation_class = LangfuseSpan

        with otel_trace_api.use_span(self._otel_span):
            new_otel_span = self._langfuse_client._otel_tracer.start_span(name=name)

        common_args = {
            "otel_span": new_otel_span,
            "langfuse_client": self._langfuse_client,
            "environment": self._environment,
            "release": self._release,
            "input": input,
            "output": output,
            "metadata": metadata,
            "version": version,
            "level": level,
            "status_message": status_message,
        }

        if as_type in get_observation_types_list(ObservationTypeGenerationLike):
            common_args.update(
                {
                    "completion_start_time": completion_start_time,
                    "model": model,
                    "model_parameters": model_parameters,
                    "usage_details": usage_details,
                    "cost_details": cost_details,
                    "prompt": prompt,
                }
            )

        return observation_class(**common_args)  # type: ignore[no-any-return,return-value,arg-type]

    @overload
    def start_as_current_observation(
        self,
        *,
        name: str,
        as_type: Literal["span"],
        input: Optional[Any] = None,
        output: Optional[Any] = None,
        metadata: Optional[Any] = None,
        version: Optional[str] = None,
        level: Optional[SpanLevel] = None,
        status_message: Optional[str] = None,
    ) -> _AgnosticContextManager["LangfuseSpan"]: ...

    @overload
    def start_as_current_observation(
        self,
        *,
        name: str,
        as_type: Literal["generation"],
        input: Optional[Any] = None,
        output: Optional[Any] = None,
        metadata: Optional[Any] = None,
        version: Optional[str] = None,
        level: Optional[SpanLevel] = None,
        status_message: Optional[str] = None,
        completion_start_time: Optional[datetime] = None,
        model: Optional[str] = None,
        model_parameters: Optional[Dict[str, MapValue]] = None,
        usage_details: Optional[Dict[str, int]] = None,
        cost_details: Optional[Dict[str, float]] = None,
        prompt: Optional[PromptClient] = None,
    ) -> _AgnosticContextManager["LangfuseGeneration"]: ...

    @overload
    def start_as_current_observation(
        self,
        *,
        name: str,
        as_type: Literal["embedding"],
        input: Optional[Any] = None,
        output: Optional[Any] = None,
        metadata: Optional[Any] = None,
        version: Optional[str] = None,
        level: Optional[SpanLevel] = None,
        status_message: Optional[str] = None,
        completion_start_time: Optional[datetime] = None,
        model: Optional[str] = None,
        model_parameters: Optional[Dict[str, MapValue]] = None,
        usage_details: Optional[Dict[str, int]] = None,
        cost_details: Optional[Dict[str, float]] = None,
        prompt: Optional[PromptClient] = None,
    ) -> _AgnosticContextManager["LangfuseEmbedding"]: ...

    @overload
    def start_as_current_observation(
        self,
        *,
        name: str,
        as_type: Literal["agent"],
        input: Optional[Any] = None,
        output: Optional[Any] = None,
        metadata: Optional[Any] = None,
        version: Optional[str] = None,
        level: Optional[SpanLevel] = None,
        status_message: Optional[str] = None,
    ) -> _AgnosticContextManager["LangfuseAgent"]: ...

    @overload
    def start_as_current_observation(
        self,
        *,
        name: str,
        as_type: Literal["tool"],
        input: Optional[Any] = None,
        output: Optional[Any] = None,
        metadata: Optional[Any] = None,
        version: Optional[str] = None,
        level: Optional[SpanLevel] = None,
        status_message: Optional[str] = None,
    ) -> _AgnosticContextManager["LangfuseTool"]: ...

    @overload
    def start_as_current_observation(
        self,
        *,
        name: str,
        as_type: Literal["chain"],
        input: Optional[Any] = None,
        output: Optional[Any] = None,
        metadata: Optional[Any] = None,
        version: Optional[str] = None,
        level: Optional[SpanLevel] = None,
        status_message: Optional[str] = None,
    ) -> _AgnosticContextManager["LangfuseChain"]: ...

    @overload
    def start_as_current_observation(
        self,
        *,
        name: str,
        as_type: Literal["retriever"],
        input: Optional[Any] = None,
        output: Optional[Any] = None,
        metadata: Optional[Any] = None,
        version: Optional[str] = None,
        level: Optional[SpanLevel] = None,
        status_message: Optional[str] = None,
    ) -> _AgnosticContextManager["LangfuseRetriever"]: ...

    @overload
    def start_as_current_observation(
        self,
        *,
        name: str,
        as_type: Literal["evaluator"],
        input: Optional[Any] = None,
        output: Optional[Any] = None,
        metadata: Optional[Any] = None,
        version: Optional[str] = None,
        level: Optional[SpanLevel] = None,
        status_message: Optional[str] = None,
    ) -> _AgnosticContextManager["LangfuseEvaluator"]: ...

    @overload
    def start_as_current_observation(
        self,
        *,
        name: str,
        as_type: Literal["guardrail"],
        input: Optional[Any] = None,
        output: Optional[Any] = None,
        metadata: Optional[Any] = None,
        version: Optional[str] = None,
        level: Optional[SpanLevel] = None,
        status_message: Optional[str] = None,
    ) -> _AgnosticContextManager["LangfuseGuardrail"]: ...

    def start_as_current_observation(  # type: ignore[misc]
        self,
        *,
        name: str,
        as_type: ObservationTypeLiteralNoEvent = "span",
        input: Optional[Any] = None,
        output: Optional[Any] = None,
        metadata: Optional[Any] = None,
        version: Optional[str] = None,
        level: Optional[SpanLevel] = None,
        status_message: Optional[str] = None,
        completion_start_time: Optional[datetime] = None,
        model: Optional[str] = None,
        model_parameters: Optional[Dict[str, MapValue]] = None,
        usage_details: Optional[Dict[str, int]] = None,
        cost_details: Optional[Dict[str, float]] = None,
        prompt: Optional[PromptClient] = None,
        # TODO: or union of context managers?
    ) -> _AgnosticContextManager[
        Union[
            "LangfuseSpan",
            "LangfuseGeneration",
            "LangfuseAgent",
            "LangfuseTool",
            "LangfuseChain",
            "LangfuseRetriever",
            "LangfuseEvaluator",
            "LangfuseEmbedding",
            "LangfuseGuardrail",
        ]
    ]:
        """Create a new child observation and set it as the current observation in a context manager.

        This is the generic method for creating any type of child observation with
        context management. It delegates to the client's _create_span_with_parent_context method.

        Args:
            name: Name of the observation
            as_type: Type of observation to create
            input: Input data for the operation
            output: Output data from the operation
            metadata: Additional metadata to associate with the observation
            version: Version identifier for the code or component
            level: Importance level of the observation (info, warning, error)
            status_message: Optional status message for the observation
            completion_start_time: When the model started generating (for generation types)
            model: Name/identifier of the AI model used (for generation types)
            model_parameters: Parameters used for the model (for generation types)
            usage_details: Token usage information (for generation types)
            cost_details: Cost information (for generation types)
            prompt: Associated prompt template (for generation types)

        Returns:
            A context manager that yields a new observation of the specified type
        """
        return self._langfuse_client._create_span_with_parent_context(
            name=name,
            as_type=as_type,
            remote_parent_span=None,
            parent=self._otel_span,
            input=input,
            output=output,
            metadata=metadata,
            version=version,
            level=level,
            status_message=status_message,
            completion_start_time=completion_start_time,
            model=model,
            model_parameters=model_parameters,
            usage_details=usage_details,
            cost_details=cost_details,
            prompt=prompt,
        )

    def create_event(
        self,
        *,
        name: str,
        input: Optional[Any] = None,
        output: Optional[Any] = None,
        metadata: Optional[Any] = None,
        version: Optional[str] = None,
        level: Optional[SpanLevel] = None,
        status_message: Optional[str] = None,
    ) -> "LangfuseEvent":
        """Create a new Langfuse observation of type 'EVENT'.

        Args:
            name: Name of the span (e.g., function or operation name)
            input: Input data for the operation (can be any JSON-serializable object)
            output: Output data from the operation (can be any JSON-serializable object)
            metadata: Additional metadata to associate with the span
            version: Version identifier for the code or component
            level: Importance level of the span (info, warning, error)
            status_message: Optional status message for the span

        Returns:
            The LangfuseEvent object

        Example:
            ```python
            event = langfuse.create_event(name="process-event")
            ```
        """
        timestamp = time_ns()

        with otel_trace_api.use_span(self._otel_span):
            new_otel_span = self._langfuse_client._otel_tracer.start_span(
                name=name, start_time=timestamp
            )

        return cast(
            "LangfuseEvent",
            LangfuseEvent(
                otel_span=new_otel_span,
                langfuse_client=self._langfuse_client,
                input=input,
                output=output,
                metadata=metadata,
                environment=self._environment,
                release=self._release,
                version=version,
                level=level,
                status_message=status_message,
            ).end(end_time=timestamp),
        )


class LangfuseSpan(LangfuseObservationWrapper):
    """Standard span implementation for general operations in Langfuse.

    This class represents a general-purpose span that can be used to trace
    any operation in your application. It extends the base LangfuseObservationWrapper
    with specific methods for creating child spans, generations, and updating
    span-specific attributes. If possible, use a more specific type for
    better observability and insights.
    """

    def __init__(
        self,
        *,
        otel_span: otel_trace_api.Span,
        langfuse_client: "Langfuse",
        input: Optional[Any] = None,
        output: Optional[Any] = None,
        metadata: Optional[Any] = None,
        environment: Optional[str] = None,
        release: Optional[str] = None,
        version: Optional[str] = None,
        level: Optional[SpanLevel] = None,
        status_message: Optional[str] = None,
    ):
        """Initialize a new LangfuseSpan.

        Args:
            otel_span: The OpenTelemetry span to wrap
            langfuse_client: Reference to the parent Langfuse client
            input: Input data for the span (any JSON-serializable object)
            output: Output data from the span (any JSON-serializable object)
            metadata: Additional metadata to associate with the span
            environment: The tracing environment
            release: Release identifier for the application
            version: Version identifier for the code or component
            level: Importance level of the span (info, warning, error)
            status_message: Optional status message for the span
        """
        super().__init__(
            otel_span=otel_span,
            as_type="span",
            langfuse_client=langfuse_client,
            input=input,
            output=output,
            metadata=metadata,
            environment=environment,
            release=release,
            version=version,
            level=level,
            status_message=status_message,
        )


class LangfuseGeneration(LangfuseObservationWrapper):
    """Specialized span implementation for AI model generations in Langfuse.

    This class represents a generation span specifically designed for tracking
    AI/LLM operations. It extends the base LangfuseObservationWrapper with specialized
    attributes for model details, token usage, and costs.
    """

    def __init__(
        self,
        *,
        otel_span: otel_trace_api.Span,
        langfuse_client: "Langfuse",
        input: Optional[Any] = None,
        output: Optional[Any] = None,
        metadata: Optional[Any] = None,
        environment: Optional[str] = None,
        release: Optional[str] = None,
        version: Optional[str] = None,
        level: Optional[SpanLevel] = None,
        status_message: Optional[str] = None,
        completion_start_time: Optional[datetime] = None,
        model: Optional[str] = None,
        model_parameters: Optional[Dict[str, MapValue]] = None,
        usage_details: Optional[Dict[str, int]] = None,
        cost_details: Optional[Dict[str, float]] = None,
        prompt: Optional[PromptClient] = None,
    ):
        """Initialize a new LangfuseGeneration span.

        Args:
            otel_span: The OpenTelemetry span to wrap
            langfuse_client: Reference to the parent Langfuse client
            input: Input data for the generation (e.g., prompts)
            output: Output from the generation (e.g., completions)
            metadata: Additional metadata to associate with the generation
            environment: The tracing environment
            release: Release identifier for the application
            version: Version identifier for the model or component
            level: Importance level of the generation (info, warning, error)
            status_message: Optional status message for the generation
            completion_start_time: When the model started generating the response
            model: Name/identifier of the AI model used (e.g., "gpt-4")
            model_parameters: Parameters used for the model (e.g., temperature, max_tokens)
            usage_details: Token usage information (e.g., prompt_tokens, completion_tokens)
            cost_details: Cost information for the model call
            prompt: Associated prompt template from Langfuse prompt management
        """
        super().__init__(
            as_type="generation",
            otel_span=otel_span,
            langfuse_client=langfuse_client,
            input=input,
            output=output,
            metadata=metadata,
            environment=environment,
            release=release,
            version=version,
            level=level,
            status_message=status_message,
            completion_start_time=completion_start_time,
            model=model,
            model_parameters=model_parameters,
            usage_details=usage_details,
            cost_details=cost_details,
            prompt=prompt,
        )


class LangfuseEvent(LangfuseObservationWrapper):
    """Specialized span implementation for Langfuse Events."""

    def __init__(
        self,
        *,
        otel_span: otel_trace_api.Span,
        langfuse_client: "Langfuse",
        input: Optional[Any] = None,
        output: Optional[Any] = None,
        metadata: Optional[Any] = None,
        environment: Optional[str] = None,
        release: Optional[str] = None,
        version: Optional[str] = None,
        level: Optional[SpanLevel] = None,
        status_message: Optional[str] = None,
    ):
        """Initialize a new LangfuseEvent span.

        Args:
            otel_span: The OpenTelemetry span to wrap
            langfuse_client: Reference to the parent Langfuse client
            input: Input data for the event
            output: Output from the event
            metadata: Additional metadata to associate with the generation
            environment: The tracing environment
            release: Release identifier for the application
            version: Version identifier for the model or component
            level: Importance level of the generation (info, warning, error)
            status_message: Optional status message for the generation
        """
        super().__init__(
            otel_span=otel_span,
            as_type="event",
            langfuse_client=langfuse_client,
            input=input,
            output=output,
            metadata=metadata,
            environment=environment,
            release=release,
            version=version,
            level=level,
            status_message=status_message,
        )

    def update(
        self,
        *,
        name: Optional[str] = None,
        input: Optional[Any] = None,
        output: Optional[Any] = None,
        metadata: Optional[Any] = None,
        version: Optional[str] = None,
        level: Optional[SpanLevel] = None,
        status_message: Optional[str] = None,
        completion_start_time: Optional[datetime] = None,
        model: Optional[str] = None,
        model_parameters: Optional[Dict[str, MapValue]] = None,
        usage_details: Optional[Dict[str, int]] = None,
        cost_details: Optional[Dict[str, float]] = None,
        prompt: Optional[PromptClient] = None,
        **kwargs: Any,
    ) -> "LangfuseEvent":
        """Update is not allowed for LangfuseEvent because events cannot be updated.

        This method logs a warning and returns self without making changes.

        Returns:
            self: Returns the unchanged LangfuseEvent instance
        """
        langfuse_logger.warning(
            "Attempted to update LangfuseEvent observation. Events cannot be updated after creation."
        )
        return self


class LangfuseAgent(LangfuseObservationWrapper):
    """Agent observation for reasoning blocks that act on tools using LLM guidance."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize a new LangfuseAgent span."""
        kwargs["as_type"] = "agent"
        super().__init__(**kwargs)


class LangfuseTool(LangfuseObservationWrapper):
    """Tool observation representing external tool calls, e.g., calling a weather API."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize a new LangfuseTool span."""
        kwargs["as_type"] = "tool"
        super().__init__(**kwargs)


class LangfuseChain(LangfuseObservationWrapper):
    """Chain observation for connecting LLM application steps, e.g. passing context from retriever to LLM."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize a new LangfuseChain span."""
        kwargs["as_type"] = "chain"
        super().__init__(**kwargs)


class LangfuseRetriever(LangfuseObservationWrapper):
    """Retriever observation for data retrieval steps, e.g. vector store or database queries."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize a new LangfuseRetriever span."""
        kwargs["as_type"] = "retriever"
        super().__init__(**kwargs)


class LangfuseEmbedding(LangfuseObservationWrapper):
    """Embedding observation for LLM embedding calls, typically used before retrieval."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize a new LangfuseEmbedding span."""
        kwargs["as_type"] = "embedding"
        super().__init__(**kwargs)


class LangfuseEvaluator(LangfuseObservationWrapper):
    """Evaluator observation for assessing relevance, correctness, or helpfulness of LLM outputs."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize a new LangfuseEvaluator span."""
        kwargs["as_type"] = "evaluator"
        super().__init__(**kwargs)


class LangfuseGuardrail(LangfuseObservationWrapper):
    """Guardrail observation for protection e.g. against jailbreaks or offensive content."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize a new LangfuseGuardrail span."""
        kwargs["as_type"] = "guardrail"
        super().__init__(**kwargs)


_OBSERVATION_CLASS_MAP.update(
    {
        "span": LangfuseSpan,
        "generation": LangfuseGeneration,
        "agent": LangfuseAgent,
        "tool": LangfuseTool,
        "chain": LangfuseChain,
        "retriever": LangfuseRetriever,
        "evaluator": LangfuseEvaluator,
        "embedding": LangfuseEmbedding,
        "guardrail": LangfuseGuardrail,
    }
)
