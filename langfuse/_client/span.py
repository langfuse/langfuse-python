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
import warnings
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Type,
    Union,
    cast,
    get_args,
    overload,
)

from opentelemetry import trace as otel_trace_api
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
    ObservationTypeLiteral,
    ObservationTypeGenerationLike,
    ObservationTypeLiteralNoEvent,
)
from langfuse._client.mixins import MediaProcessingMixin, ScoringMixin, TraceUpdateMixin, AttributeMixin
from langfuse.logger import langfuse_logger
from langfuse.types import MapValue, ScoreDataType, SpanLevel

# Factory mapping for observation classes
# Note: "event" is handled separately due to special instantiation logic
# Populated after class definitions
_OBSERVATION_CLASS_MAP: Dict[str, Type["LangfuseObservationWrapper"]] = {}


class LangfuseObservationWrapper(MediaProcessingMixin, ScoringMixin, TraceUpdateMixin, AttributeMixin):
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
            environment: The tracing environment
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
        self._otel_span.set_attribute(
            LangfuseOtelSpanAttributes.OBSERVATION_TYPE, as_type
        )
        self._langfuse_client = langfuse_client
        self._observation_type = as_type

        self.trace_id = self._langfuse_client._get_otel_trace_id(otel_span)
        self.id = self._langfuse_client._get_otel_span_id(otel_span)

        self._environment = environment
        if self._environment is not None:
            self._otel_span.set_attribute(
                LangfuseOtelSpanAttributes.ENVIRONMENT, self._environment
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

            if as_type in get_args(ObservationTypeGenerationLike):
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
                        Literal[
                            "generation",
                            "agent",
                            "tool",
                            "chain",
                            "retriever",
                            "evaluator",
                            "embedding",
                        ],
                        as_type,
                    ),
                )

            else:
                # For span-like types: "span", "guardrail", "event"
                attributes = create_span_attributes(
                    input=media_processed_input,
                    output=media_processed_output,
                    metadata=media_processed_metadata,
                    version=version,
                    level=level,
                    status_message=status_message,
                    observation_type=cast(
                        Optional[Literal["span", "guardrail", "event"]],
                        as_type if as_type in ["span", "guardrail", "event"] else None,
                    ),
                )

            attributes.pop(LangfuseOtelSpanAttributes.OBSERVATION_TYPE, None)

            self._otel_span.set_attributes(
                {k: v for k, v in attributes.items() if v is not None}
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

        if self._observation_type in get_args(ObservationTypeGenerationLike):
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
            # For span-like types: "span", "guardrail", "event"
            attributes = create_span_attributes(
                input=processed_input,
                output=processed_output,
                metadata=processed_metadata,
                version=version,
                level=level,
                status_message=status_message,
                observation_type=cast(
                    Optional[Literal["span", "guardrail", "event"]],
                    self._observation_type
                    if self._observation_type in ["span", "guardrail", "event"]
                    else None,
                ),
            )

        self._otel_span.set_attributes(attributes=attributes)

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
        completion_start_time: Optional[datetime] = None,
        model: Optional[str] = None,
        model_parameters: Optional[Dict[str, MapValue]] = None,
        usage_details: Optional[Dict[str, int]] = None,
        cost_details: Optional[Dict[str, float]] = None,
        prompt: Optional[PromptClient] = None,
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
        completion_start_time: Optional[datetime] = None,
        model: Optional[str] = None,
        model_parameters: Optional[Dict[str, MapValue]] = None,
        usage_details: Optional[Dict[str, int]] = None,
        cost_details: Optional[Dict[str, float]] = None,
        prompt: Optional[PromptClient] = None,
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
        completion_start_time: Optional[datetime] = None,
        model: Optional[str] = None,
        model_parameters: Optional[Dict[str, MapValue]] = None,
        usage_details: Optional[Dict[str, int]] = None,
        cost_details: Optional[Dict[str, float]] = None,
        prompt: Optional[PromptClient] = None,
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
        completion_start_time: Optional[datetime] = None,
        model: Optional[str] = None,
        model_parameters: Optional[Dict[str, MapValue]] = None,
        usage_details: Optional[Dict[str, int]] = None,
        cost_details: Optional[Dict[str, float]] = None,
        prompt: Optional[PromptClient] = None,
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
        completion_start_time: Optional[datetime] = None,
        model: Optional[str] = None,
        model_parameters: Optional[Dict[str, MapValue]] = None,
        usage_details: Optional[Dict[str, int]] = None,
        cost_details: Optional[Dict[str, float]] = None,
        prompt: Optional[PromptClient] = None,
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
        as_type: ObservationTypeLiteral,
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
                    version=version,
                    level=level,
                    status_message=status_message,
                ).end(end_time=timestamp),
            )

        observation_class = _OBSERVATION_CLASS_MAP.get(as_type)
        if not observation_class:
            raise ValueError(f"Unknown observation type: {as_type}")

        with otel_trace_api.use_span(self._otel_span):
            new_otel_span = self._langfuse_client._otel_tracer.start_span(name=name)

        common_args = {
            "otel_span": new_otel_span,
            "langfuse_client": self._langfuse_client,
            "environment": self._environment,
            "input": input,
            "output": output,
            "metadata": metadata,
            "version": version,
            "level": level,
            "status_message": status_message,
        }

        if as_type in get_args(ObservationTypeGenerationLike):
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
        as_type: ObservationTypeGenerationLike,
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
    ) -> _AgnosticContextManager[
        Union[
            "LangfuseAgent",
            "LangfuseTool",
            "LangfuseChain",
            "LangfuseRetriever",
            "LangfuseEvaluator",
            "LangfuseEmbedding",
        ]
    ]: ...

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
        as_type: ObservationTypeLiteralNoEvent,
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
            version=version,
            level=level,
            status_message=status_message,
        )

    def start_span(
        self,
        name: str,
        input: Optional[Any] = None,
        output: Optional[Any] = None,
        metadata: Optional[Any] = None,
        version: Optional[str] = None,
        level: Optional[SpanLevel] = None,
        status_message: Optional[str] = None,
    ) -> "LangfuseSpan":
        """Create a new child span.

        This method creates a new child span with this span as the parent.
        Unlike start_as_current_span(), this method does not set the new span
        as the current span in the context.

        Args:
            name: Name of the span (e.g., function or operation name)
            input: Input data for the operation
            output: Output data from the operation
            metadata: Additional metadata to associate with the span
            version: Version identifier for the code or component
            level: Importance level of the span (info, warning, error)
            status_message: Optional status message for the span

        Returns:
            A new LangfuseSpan that must be ended with .end() when complete

        Example:
            ```python
            parent_span = langfuse.start_span(name="process-request")
            try:
                # Create a child span
                child_span = parent_span.start_span(name="validate-input")
                try:
                    # Do validation work
                    validation_result = validate(request_data)
                    child_span.update(output=validation_result)
                finally:
                    child_span.end()

                # Continue with parent span
                result = process_validated_data(validation_result)
                parent_span.update(output=result)
            finally:
                parent_span.end()
            ```
        """
        return self.start_observation(
            name=name,
            as_type="span",
            input=input,
            output=output,
            metadata=metadata,
            version=version,
            level=level,
            status_message=status_message,
        )

    def start_as_current_span(
        self,
        *,
        name: str,
        input: Optional[Any] = None,
        output: Optional[Any] = None,
        metadata: Optional[Any] = None,
        version: Optional[str] = None,
        level: Optional[SpanLevel] = None,
        status_message: Optional[str] = None,
    ) -> _AgnosticContextManager["LangfuseSpan"]:
        """[DEPRECATED] Create a new child span and set it as the current span in a context manager.

        DEPRECATED: This method is deprecated and will be removed in a future version.
        Use start_as_current_observation(as_type='span') instead.

        This method creates a new child span and sets it as the current span within
        a context manager. It should be used with a 'with' statement to automatically
        manage the span's lifecycle.

        Args:
            name: Name of the span (e.g., function or operation name)
            input: Input data for the operation
            output: Output data from the operation
            metadata: Additional metadata to associate with the span
            version: Version identifier for the code or component
            level: Importance level of the span (info, warning, error)
            status_message: Optional status message for the span

        Returns:
            A context manager that yields a new LangfuseSpan

        Example:
            ```python
            with langfuse.start_as_current_span(name="process-request") as parent_span:
                # Parent span is active here

                # Create a child span with context management
                with parent_span.start_as_current_span(name="validate-input") as child_span:
                    # Child span is active here
                    validation_result = validate(request_data)
                    child_span.update(output=validation_result)

                # Back to parent span context
                result = process_validated_data(validation_result)
                parent_span.update(output=result)
            ```
        """
        warnings.warn(
            "start_as_current_span is deprecated and will be removed in a future version. "
            "Use start_as_current_observation(as_type='span') instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.start_as_current_observation(
            name=name,
            as_type="span",
            input=input,
            output=output,
            metadata=metadata,
            version=version,
            level=level,
            status_message=status_message,
        )

    def start_generation(
        self,
        *,
        name: str,
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
    ) -> "LangfuseGeneration":
        """[DEPRECATED] Create a new child generation span.

        DEPRECATED: This method is deprecated and will be removed in a future version.
        Use start_observation(as_type='generation') instead.

        This method creates a new child generation span with this span as the parent.
        Generation spans are specialized for AI/LLM operations and include additional
        fields for model information, usage stats, and costs.

        Unlike start_as_current_generation(), this method does not set the new span
        as the current span in the context.

        Args:
            name: Name of the generation operation
            input: Input data for the model (e.g., prompts)
            output: Output from the model (e.g., completions)
            metadata: Additional metadata to associate with the generation
            version: Version identifier for the model or component
            level: Importance level of the generation (info, warning, error)
            status_message: Optional status message for the generation
            completion_start_time: When the model started generating the response
            model: Name/identifier of the AI model used (e.g., "gpt-4")
            model_parameters: Parameters used for the model (e.g., temperature, max_tokens)
            usage_details: Token usage information (e.g., prompt_tokens, completion_tokens)
            cost_details: Cost information for the model call
            prompt: Associated prompt template from Langfuse prompt management

        Returns:
            A new LangfuseGeneration that must be ended with .end() when complete

        Example:
            ```python
            span = langfuse.start_span(name="process-query")
            try:
                # Create a generation child span
                generation = span.start_generation(
                    name="generate-answer",
                    model="gpt-4",
                    input={"prompt": "Explain quantum computing"}
                )
                try:
                    # Call model API
                    response = llm.generate(...)

                    generation.update(
                        output=response.text,
                        usage_details={
                            "prompt_tokens": response.usage.prompt_tokens,
                            "completion_tokens": response.usage.completion_tokens
                        }
                    )
                finally:
                    generation.end()

                # Continue with parent span
                span.update(output={"answer": response.text, "source": "gpt-4"})
            finally:
                span.end()
            ```
        """
        warnings.warn(
            "start_generation is deprecated and will be removed in a future version. "
            "Use start_observation(as_type='generation') instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.start_observation(
            name=name,
            as_type="generation",
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

    def start_as_current_generation(
        self,
        *,
        name: str,
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
    ) -> _AgnosticContextManager["LangfuseGeneration"]:
        """[DEPRECATED] Create a new child generation span and set it as the current span in a context manager.

        DEPRECATED: This method is deprecated and will be removed in a future version.
        Use start_as_current_observation(as_type='generation') instead.

        This method creates a new child generation span and sets it as the current span
        within a context manager. Generation spans are specialized for AI/LLM operations
        and include additional fields for model information, usage stats, and costs.

        Args:
            name: Name of the generation operation
            input: Input data for the model (e.g., prompts)
            output: Output from the model (e.g., completions)
            metadata: Additional metadata to associate with the generation
            version: Version identifier for the model or component
            level: Importance level of the generation (info, warning, error)
            status_message: Optional status message for the generation
            completion_start_time: When the model started generating the response
            model: Name/identifier of the AI model used (e.g., "gpt-4")
            model_parameters: Parameters used for the model (e.g., temperature, max_tokens)
            usage_details: Token usage information (e.g., prompt_tokens, completion_tokens)
            cost_details: Cost information for the model call
            prompt: Associated prompt template from Langfuse prompt management

        Returns:
            A context manager that yields a new LangfuseGeneration

        Example:
            ```python
            with langfuse.start_as_current_span(name="process-request") as span:
                # Prepare data
                query = preprocess_user_query(user_input)

                # Create a generation span with context management
                with span.start_as_current_generation(
                    name="generate-answer",
                    model="gpt-4",
                    input={"query": query}
                ) as generation:
                    # Generation span is active here
                    response = llm.generate(query)

                    # Update with results
                    generation.update(
                        output=response.text,
                        usage_details={
                            "prompt_tokens": response.usage.prompt_tokens,
                            "completion_tokens": response.usage.completion_tokens
                        }
                    )

                # Back to parent span context
                span.update(output={"answer": response.text, "source": "gpt-4"})
            ```
        """
        warnings.warn(
            "start_as_current_generation is deprecated and will be removed in a future version. "
            "Use start_as_current_observation(as_type='generation') instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return cast(
            _AgnosticContextManager["LangfuseGeneration"],
            self.start_as_current_observation(
                name=name,
                as_type="generation",
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
            ),
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
                version=version,
                level=level,
                status_message=status_message,
            ).end(end_time=timestamp),
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
    ) -> "LangfuseObservationWrapper":
        """Update is not allowed for LangfuseEvent because events cannot be updated.

        This method logs a warning and returns self without making changes.

        Returns:
            self: Returns the unchanged LangfuseEvent instance
        """
        langfuse_logger.warning(
            "Attempted to update LangfuseEvent observation. Events cannot be updated after creation."
        )
        return self


# Factory function to create observation instances dynamically
def _create_observation_wrapper(as_type: str, **kwargs: Any) -> LangfuseObservationWrapper:
    """Create an observation wrapper instance with the specified type.
    
    This factory function replaces the redundant subclasses that only set as_type.
    
    Args:
        as_type: The observation type to create
        **kwargs: Arguments to pass to the LangfuseObservationWrapper constructor
        
    Returns:
        A LangfuseObservationWrapper instance with the appropriate as_type set
    """
    kwargs["as_type"] = as_type
    return LangfuseObservationWrapper(**kwargs)

# Type aliases that maintain the same API surface while using the factory internally
class LangfuseAgent(LangfuseObservationWrapper):
    """Agent observation for reasoning blocks that act on tools using LLM guidance."""
    
    def __new__(cls, **kwargs: Any) -> LangfuseObservationWrapper:
        return _create_observation_wrapper("agent", **kwargs)

class LangfuseTool(LangfuseObservationWrapper):
    """Tool observation representing external tool calls, e.g., calling a weather API."""
    
    def __new__(cls, **kwargs: Any) -> LangfuseObservationWrapper:
        return _create_observation_wrapper("tool", **kwargs)

class LangfuseChain(LangfuseObservationWrapper):
    """Chain observation for connecting LLM application steps, e.g. passing context from retriever to LLM."""
    
    def __new__(cls, **kwargs: Any) -> LangfuseObservationWrapper:
        return _create_observation_wrapper("chain", **kwargs)

class LangfuseRetriever(LangfuseObservationWrapper):
    """Retriever observation for data retrieval steps, e.g. vector store or database queries."""
    
    def __new__(cls, **kwargs: Any) -> LangfuseObservationWrapper:
        return _create_observation_wrapper("retriever", **kwargs)

class LangfuseEmbedding(LangfuseObservationWrapper):
    """Embedding observation for LLM embedding calls, typically used before retrieval."""
    
    def __new__(cls, **kwargs: Any) -> LangfuseObservationWrapper:
        return _create_observation_wrapper("embedding", **kwargs)

class LangfuseEvaluator(LangfuseObservationWrapper):
    """Evaluator observation for assessing relevance, correctness, or helpfulness of LLM outputs."""
    
    def __new__(cls, **kwargs: Any) -> LangfuseObservationWrapper:
        return _create_observation_wrapper("evaluator", **kwargs)

class LangfuseGuardrail(LangfuseObservationWrapper):
    """Guardrail observation for protection e.g. against jailbreaks or offensive content."""
    
    def __new__(cls, **kwargs: Any) -> LangfuseObservationWrapper:
        return _create_observation_wrapper("guardrail", **kwargs)


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
