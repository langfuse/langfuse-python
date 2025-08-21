"""Mixin classes for LangfuseObservationWrapper functionality.

This module contains mixin classes that break down the monolithic LangfuseObservationWrapper
base class into smaller, focused components. These mixins provide reusable functionality
for media processing, scoring, attribute management, and trace updates.
"""

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Union, TYPE_CHECKING, cast, overload
from opentelemetry import trace as otel_trace_api

if TYPE_CHECKING:
    from langfuse._client.client import Langfuse

from langfuse._client.attributes import create_trace_attributes
from langfuse.logger import langfuse_logger
from langfuse.model import PromptClient
from langfuse.types import MapValue, ScoreDataType, SpanLevel


class MediaProcessingMixin:
    """Mixin providing media processing and masking functionality."""
    
    # These attributes will be provided by the main class
    _langfuse_client: "Langfuse"
    trace_id: str
    id: str
    _otel_span: otel_trace_api.Span

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
            data=self._process_media_in_attribute(data=data, field=field)
        )

    def _mask_attribute(self, *, data: Any) -> Any:
        """Apply the configured mask function to data.

        Internal method that applies the client's configured masking function to
        the provided data, with error handling and fallback.

        Args:
            data: The data to mask

        Returns:
            The masked data, or the original data if no mask is configured
        """
        if not self._langfuse_client._mask:
            return data

        try:
            return self._langfuse_client._mask(data=data)
        except Exception as e:
            langfuse_logger.error(
                f"Masking error: Custom mask function threw exception when processing data. Using fallback masking. Error: {e}"
            )

            return "<fully masked due to failed mask function>"

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


class ScoringMixin:
    """Mixin providing scoring functionality for observations and traces."""
    
    # These attributes will be provided by the main class
    _langfuse_client: "Langfuse"
    trace_id: str
    id: str

    @overload
    def score(
        self,
        *,
        name: str,
        value: float,
        score_id: Optional[str] = None,
        data_type: Optional[Literal["NUMERIC", "BOOLEAN"]] = None,
        comment: Optional[str] = None,
        config_id: Optional[str] = None,
    ) -> None: ...

    @overload
    def score(
        self,
        *,
        name: str,
        value: str,
        score_id: Optional[str] = None,
        data_type: Optional[Literal["CATEGORICAL"]] = "CATEGORICAL",
        comment: Optional[str] = None,
        config_id: Optional[str] = None,
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
    ) -> None:
        """Create a score for this specific span.

        This method creates a score associated with this specific span (observation).
        Scores can represent any kind of evaluation, feedback, or quality metric.

        Args:
            name: Name of the score (e.g., "relevance", "accuracy")
            value: Score value (numeric for NUMERIC/BOOLEAN, string for CATEGORICAL)
            score_id: Optional custom ID for the score (auto-generated if not provided)
            data_type: Type of score (NUMERIC, BOOLEAN, or CATEGORICAL)
            comment: Optional comment or explanation for the score
            config_id: Optional ID of a score config defined in Langfuse

        Example:
            ```python
            with langfuse.start_as_current_span(name="process-query") as span:
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
            data_type=cast(Literal["CATEGORICAL"], data_type),
            comment=comment,
            config_id=config_id,
        )

    @overload
    def score_trace(
        self,
        *,
        name: str,
        value: float,
        score_id: Optional[str] = None,
        data_type: Optional[Literal["NUMERIC", "BOOLEAN"]] = None,
        comment: Optional[str] = None,
        config_id: Optional[str] = None,
    ) -> None: ...

    @overload
    def score_trace(
        self,
        *,
        name: str,
        value: str,
        score_id: Optional[str] = None,
        data_type: Optional[Literal["CATEGORICAL"]] = "CATEGORICAL",
        comment: Optional[str] = None,
        config_id: Optional[str] = None,
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
    ) -> None:
        """Create a score for the entire trace that this span belongs to.

        This method creates a score associated with the entire trace that this span
        belongs to, rather than the specific span. This is useful for overall
        evaluations that apply to the complete trace.

        Args:
            name: Name of the score (e.g., "user_satisfaction", "overall_quality")
            value: Score value (numeric for NUMERIC/BOOLEAN, string for CATEGORICAL)
            score_id: Optional custom ID for the score (auto-generated if not provided)
            data_type: Type of score (NUMERIC, BOOLEAN, or CATEGORICAL)
            comment: Optional comment or explanation for the score
            config_id: Optional ID of a score config defined in Langfuse

        Example:
            ```python
            with langfuse.start_as_current_span(name="handle-request") as span:
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
            data_type=cast(Literal["CATEGORICAL"], data_type),
            comment=comment,
            config_id=config_id,
        )


class TraceUpdateMixin:
    """Mixin providing trace update functionality."""
    
    # These attributes will be provided by the main class
    _otel_span: otel_trace_api.Span

    def update_trace(
        self,
        *,
        name: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        version: Optional[str] = None,
        input: Optional[Any] = None,
        output: Optional[Any] = None,
        metadata: Optional[Any] = None,
        tags: Optional[List[str]] = None,
        public: Optional[bool] = None,
    ) -> "LangfuseObservationWrapper":
        """Update the trace that this span belongs to.

        This method updates trace-level attributes of the trace that this span
        belongs to. This is useful for adding or modifying trace-wide information
        like user ID, session ID, or tags.

        Args:
            name: Updated name for the trace
            user_id: ID of the user who initiated the trace
            session_id: Session identifier for grouping related traces
            version: Version identifier for the application or service
            input: Input data for the overall trace
            output: Output data from the overall trace
            metadata: Additional metadata to associate with the trace
            tags: List of tags to categorize the trace
            public: Whether the trace should be publicly accessible
        """
        if not self._otel_span.is_recording():
            return self

        # Use media processing from mixin
        media_processed_input = self._process_media_and_apply_mask(
            data=input, field="input", span=self._otel_span
        )
        media_processed_output = self._process_media_and_apply_mask(
            data=output, field="output", span=self._otel_span
        )
        media_processed_metadata = self._process_media_and_apply_mask(
            data=metadata, field="metadata", span=self._otel_span
        )

        attributes = create_trace_attributes(
            name=name,
            user_id=user_id,
            session_id=session_id,
            version=version,
            input=media_processed_input,
            output=media_processed_output,
            metadata=media_processed_metadata,
            tags=tags,
            public=public,
        )

        self._otel_span.set_attributes(attributes)

        return self


class AttributeMixin:
    """Mixin providing OTEL attribute management functionality."""
    
    # These attributes will be provided by the main class
    _otel_span: otel_trace_api.Span
    _observation_type: str

    def _set_span_attributes(self, attributes: Dict[str, Any]) -> None:
        """Set attributes on the underlying OTEL span.
        
        Args:
            attributes: Dictionary of attributes to set on the span
        """
        if self._otel_span.is_recording():
            self._otel_span.set_attributes(
                {k: v for k, v in attributes.items() if v is not None}
            )

    def _update_span_status(self, *, status_message: Optional[str] = None) -> None:
        """Update the span status with an optional message.
        
        Args:
            status_message: Optional status message for the span
        """
        if self._otel_span.is_recording() and status_message:
            self._otel_span.set_status(
                otel_trace_api.Status(
                    status_code=otel_trace_api.StatusCode.ERROR,
                    description=status_message
                )
            )