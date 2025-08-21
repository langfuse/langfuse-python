"""Factory for creating observations without overload bloat.

This module provides a centralized factory for creating Langfuse observations,
eliminating the need for repetitive overloaded methods while maintaining
perfect type safety through clean overload delegation.
"""

from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Union, cast, get_args
from opentelemetry import trace as otel_trace_api
from opentelemetry.util._decorator import _AgnosticContextManager, _agnosticcontextmanager

if TYPE_CHECKING:
    from langfuse._client.client import Langfuse

from langfuse._client.attributes import LangfuseOtelSpanAttributes
from langfuse._client.constants import ObservationTypeGenerationLike, ObservationTypeLiteralNoEvent
from langfuse.model import PromptClient
from langfuse.types import MapValue, SpanLevel, TraceContext, ScoreDataType
from langfuse.logger import langfuse_logger


class ObservationFactory:
    """Factory for creating observations and scores with single implementation logic."""
    
    def __init__(self, client: "Langfuse"):
        """Initialize factory with reference to client.
        
        Args:
            client: The Langfuse client instance
        """
        self._client = client
    
    def create_observation(
        self,
        *,
        as_type: ObservationTypeLiteralNoEvent,
        name: str,
        trace_context: Optional[TraceContext] = None,
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
    ]:
        """Create a new observation of the specified type.
        
        This is the single implementation that handles all observation creation logic,
        eliminating the need for repetitive overloaded methods.
        
        Args:
            as_type: Type of observation to create
            name: Name of the observation
            trace_context: Optional context for connecting to an existing trace
            input: Input data for the operation
            output: Output data from the operation
            metadata: Additional metadata to associate with the observation
            version: Version identifier for the code or component
            level: Importance level of the observation
            status_message: Optional status message for the observation
            completion_start_time: When the model started generating (for generation types)
            model: Name/identifier of the AI model used (for generation types)
            model_parameters: Parameters used for the model (for generation types)
            usage_details: Token usage information (for generation types)
            cost_details: Cost information (for generation types)
            prompt: Associated prompt template (for generation types)
            
        Returns:
            An observation object of the appropriate type
        """
        # Handle trace context (creates remote parent span)
        if trace_context:
            trace_id = trace_context.get("trace_id", None)
            parent_span_id = trace_context.get("parent_span_id", None)

            if trace_id:
                remote_parent_span = self._client._create_remote_parent_span(
                    trace_id=trace_id, parent_span_id=parent_span_id
                )

                with otel_trace_api.use_span(
                    cast(otel_trace_api.Span, remote_parent_span)
                ):
                    otel_span = self._client._otel_tracer.start_span(name=name)
                    otel_span.set_attribute(LangfuseOtelSpanAttributes.AS_ROOT, True)

                    return self._create_observation_from_otel_span(
                        otel_span=otel_span,
                        as_type=as_type,
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

        # Normal span creation (no trace context)
        otel_span = self._client._otel_tracer.start_span(name=name)

        return self._create_observation_from_otel_span(
            otel_span=otel_span,
            as_type=as_type,
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
    
    def create_as_current_observation(
        self,
        *,
        as_type: ObservationTypeLiteralNoEvent,
        name: str,
        trace_context: Optional[TraceContext] = None,
        end_on_exit: Optional[bool] = None,
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
        """Create a new observation and set it as current span in a context manager.
        
        This is the single implementation for all start_as_current_observation methods.
        
        Args:
            as_type: Type of observation to create
            name: Name of the observation
            trace_context: Optional context for connecting to an existing trace
            end_on_exit: Whether to end span automatically when leaving context manager
            input: Input data for the operation
            output: Output data from the operation  
            metadata: Additional metadata to associate with the observation
            version: Version identifier for the code or component
            level: Importance level of the observation
            status_message: Optional status message for the observation
            completion_start_time: When the model started generating (for generation types)
            model: Name/identifier of the AI model used (for generation types)
            model_parameters: Parameters used for the model (for generation types)
            usage_details: Token usage information (for generation types)
            cost_details: Cost information (for generation types)
            prompt: Associated prompt template (for generation types)
            
        Returns:
            A context manager that yields an observation of the specified type
        """
        # Handle trace context case
        if trace_context:
            trace_id = trace_context.get("trace_id", None)
            parent_span_id = trace_context.get("parent_span_id", None)

            if trace_id:
                remote_parent_span = self._client._create_remote_parent_span(
                    trace_id=trace_id, parent_span_id=parent_span_id
                )

                return self._client._create_span_with_parent_context(
                    as_type=as_type,
                    name=name,
                    remote_parent_span=remote_parent_span,
                    parent=None,
                    end_on_exit=end_on_exit,
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

        # Normal context manager creation
        return self._client._start_as_current_otel_span_with_processed_media(
            as_type=as_type,
            name=name,
            end_on_exit=end_on_exit,
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
    
    def _create_observation_from_otel_span(
        self,
        *,
        otel_span: otel_trace_api.Span,
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
    ]:
        """Create the appropriate observation type from an OTEL span.
        
        This method handles the creation logic that was previously duplicated
        in _create_observation_from_otel_span in the client.
        """
        # Import here to avoid circular imports
        from langfuse._client.span import (
            LangfuseSpan, LangfuseGeneration, LangfuseAgent, LangfuseTool,
            LangfuseChain, LangfuseRetriever, LangfuseEvaluator, LangfuseEmbedding,
            LangfuseGuardrail
        )
        
        if as_type in get_args(ObservationTypeGenerationLike):
            observation_class = self._client._get_span_class(as_type)
            return observation_class(  # type: ignore[return-value,call-arg]
                otel_span=otel_span,
                langfuse_client=self._client,
                environment=self._client._environment,
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
        else:
            # For other types (e.g. span, guardrail), create appropriate class without generation properties
            observation_class = self._client._get_span_class(as_type)
            return observation_class(  # type: ignore[return-value,call-arg]
                otel_span=otel_span,
                langfuse_client=self._client,
                environment=self._client._environment,
                input=input,
                output=output,
                metadata=metadata,
                version=version,
                level=level,
                status_message=status_message,
            )
    
    def score_current_span(
        self,
        *,
        name: str,
        value: Union[float, str],
        score_id: Optional[str] = None,
        data_type: Optional[ScoreDataType] = None,
        comment: Optional[str] = None,
        config_id: Optional[str] = None,
    ) -> None:
        """Create a score for the current active span.
        
        Single implementation that handles all score_current_span logic,
        eliminating the need for repetitive overloaded methods.
        
        Args:
            name: Name of the score (e.g., "relevance", "accuracy")
            value: Score value (can be numeric for NUMERIC/BOOLEAN types or string for CATEGORICAL)
            score_id: Optional custom ID for the score (auto-generated if not provided)
            data_type: Type of score (NUMERIC, BOOLEAN, or CATEGORICAL)
            comment: Optional comment or explanation for the score
            config_id: Optional ID of a score config defined in Langfuse
        """
        current_span = self._client._get_current_otel_span()

        if current_span is not None:
            trace_id = self._client._get_otel_trace_id(current_span)
            observation_id = self._client._get_otel_span_id(current_span)

            langfuse_logger.info(
                f"Score: Creating score name='{name}' value={value} for current span ({observation_id}) in trace {trace_id}"
            )

            self._client.create_score(
                trace_id=trace_id,
                observation_id=observation_id,
                name=name,
                value=cast(str, value),
                score_id=score_id,
                data_type=cast(Literal["CATEGORICAL"], data_type),
                comment=comment,
                config_id=config_id,
            )
    
    def score_current_trace(
        self,
        *,
        name: str,
        value: Union[float, str],
        score_id: Optional[str] = None,
        data_type: Optional[ScoreDataType] = None,
        comment: Optional[str] = None,
        config_id: Optional[str] = None,
    ) -> None:
        """Create a score for the current trace.
        
        Single implementation that handles all score_current_trace logic,
        eliminating the need for repetitive overloaded methods.
        
        Args:
            name: Name of the score (e.g., "user_satisfaction", "overall_quality")
            value: Score value (can be numeric for NUMERIC/BOOLEAN types or string for CATEGORICAL)
            score_id: Optional custom ID for the score (auto-generated if not provided)
            data_type: Type of score (NUMERIC, BOOLEAN, or CATEGORICAL)
            comment: Optional comment or explanation for the score
            config_id: Optional ID of a score config defined in Langfuse
        """
        current_span = self._client._get_current_otel_span()

        if current_span is not None:
            trace_id = self._client._get_otel_trace_id(current_span)

            langfuse_logger.info(
                f"Score: Creating score name='{name}' value={value} for entire trace {trace_id}"
            )

            self._client.create_score(
                trace_id=trace_id,
                name=name,
                value=cast(str, value),
                score_id=score_id,
                data_type=cast(Literal["CATEGORICAL"], data_type),
                comment=comment,
                config_id=config_id,
            )