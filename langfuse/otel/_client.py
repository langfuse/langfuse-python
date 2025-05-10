"""Langfuse OpenTelemetry integration module.

This module implements Langfuse's core observability functionality using the OpenTelemetry (OTel) standard.
It provides structured tracing of AI/LLM application events, including spans, generations, and scoring,
with built-in support for batching, sampling, and distributed tracing.

The main class, Langfuse, provides a high-level interface for:
- Creating and managing distributed trace contexts
- Tracking AI model interactions with specialized span types (spans, generations)
- Scoring and evaluating model outputs with different metrics types
- Capturing detailed metadata, usage statistics, and cost information
- Processing and uploading media content in observations

This implementation uses OpenTelemetry's distributed tracing framework to ensure
consistency, interoperability, and compliance with observability standards.
All span and trace IDs follow the W3C Trace Context specification.
"""

import logging
import os
import re
from datetime import datetime
from hashlib import sha256
from typing import Any, Dict, List, Literal, Optional, Union, cast, overload

import httpx
from opentelemetry import trace
from opentelemetry import trace as otel_trace_api
from opentelemetry.sdk.trace.id_generator import RandomIdGenerator
from opentelemetry.util._decorator import _agnosticcontextmanager

from langfuse.api.resources.ingestion.types.score_body import ScoreBody
from langfuse.model import PromptClient
from langfuse.otel._span import LangfuseGeneration, LangfuseSpan
from langfuse.otel.attributes import (
    LangfuseSpanAttributes,
    create_generation_attributes,
    create_span_attributes,
)
from langfuse.otel.environment_variables import (
    LANGFUSE_DEBUG,
    LANGFUSE_HOST,
    LANGFUSE_PUBLIC_KEY,
    LANGFUSE_SECRET_KEY,
    LANGFUSE_TRACING_ENABLED,
    LANGFUSE_TRACING_ENVIRONMENT,
)
from langfuse.utils import _get_timestamp

from ..types import MapValue, MaskFunction, ScoreDataType, SpanLevel, TraceContext
from ._logger import langfuse_logger
from ._tracer import LangfuseTracer


class Langfuse:
    """Main client for Langfuse observability using OpenTelemetry.

    This class provides a high-level interface for creating and managing traces, spans,
    and generations in Langfuse. It implements a fully W3C-compliant distributed tracing
    system built on OpenTelemetry standards, optimized for AI/LLM application observability.

    The client features a thread-safe singleton pattern for each unique API key combination,
    ensuring consistent trace context propagation across your application. It implements
    efficient batching of spans with configurable flush settings and includes background
    thread management for media uploads and score ingestion.

    Configuration is flexible through either direct parameters or environment variables,
    with graceful fallbacks and runtime configuration updates. The client preserves backward
    compatibility with the legacy Langfuse API while adding enhanced OpenTelemetry features.

    Attributes:
        tracer: The underlying OpenTelemetry tracer used for creating spans
        api: Synchronous API client for Langfuse backend communication
        async_api: Asynchronous API client for Langfuse backend communication
        host: The Langfuse API host URL
        environment: The environment name for categorizing traces (e.g., "production", "staging")
        tracing_enabled: Boolean indicating whether tracing is currently active
        langfuse_tracer: Internal LangfuseTracer instance managing OpenTelemetry components

    Thread Safety:
        All methods are thread-safe. The client maintains context propagation using
        OpenTelemetry's context management, allowing concurrent operations across
        different threads while maintaining correct trace relationships.

    Example:
        ```python
        from langfuse.otel import Langfuse

        # Initialize the client (reads from env vars if not provided)
        langfuse = Langfuse(
            public_key="your-public-key",
            secret_key="your-secret-key",
            host="https://cloud.langfuse.com",  # Optional, default shown
            debug=False,                         # Optional, enables detailed logging
            sample_rate=1.0                      # Optional, control trace sampling
        )

        # Create a trace span
        with langfuse.start_as_current_span(name="process-query") as span:
            # Your application code here

            # Create a nested generation span for an LLM call
            with span.start_as_current_generation(
                name="generate-response",
                model="gpt-4",
                input={"query": "Tell me about AI"},
                model_parameters={"temperature": 0.7, "max_tokens": 500}
            ) as generation:
                # Generate response here
                response = "AI is a field of computer science..."

                generation.update(
                    output=response,
                    usage_details={"prompt_tokens": 10, "completion_tokens": 50},
                    cost_details={"total_cost": 0.0023}
                )

                # Score the generation (supports NUMERIC, BOOLEAN, CATEGORICAL)
                generation.score(name="relevance", value=0.95, data_type="NUMERIC")
        ```
    """

    def __init__(
        self,
        *,
        public_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        host: Optional[str] = None,
        timeout: Optional[int] = None,  # seconds
        httpx_client: Optional[httpx.Client] = None,
        debug: bool = False,
        tracing_enabled: Optional[bool] = True,
        flush_at: Optional[int] = None,
        flush_interval: Optional[float] = None,
        environment: Optional[str] = None,
        release: Optional[str] = None,
        media_upload_thread_count: Optional[int] = None,
        sample_rate: Optional[float] = None,
        mask: Optional[MaskFunction] = None,
    ):
        debug = debug if debug else (os.getenv(LANGFUSE_DEBUG, "False") == "True")

        if debug:
            logging.basicConfig(
                format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            langfuse_logger.setLevel(logging.DEBUG)

        public_key = public_key or os.environ.get(LANGFUSE_PUBLIC_KEY)
        if public_key is None:
            langfuse_logger.warning(
                "Authentication error: Langfuse client initialized without public_key. Client will be disabled. "
                "Provide a public_key parameter or set LANGFUSE_PUBLIC_KEY environment variable. "
                "See documentation: https://langfuse.com/docs/sdk/python/low-level-sdk#initialize-client"
            )
            self.tracer = otel_trace_api.NoOpTracer()
            return

        secret_key = secret_key or os.environ.get(LANGFUSE_SECRET_KEY)
        if secret_key is None:
            langfuse_logger.warning(
                "Authentication error: Langfuse client initialized without secret_key. Client will be disabled. "
                "Provide a secret_key parameter or set LANGFUSE_SECRET_KEY environment variable. "
                "See documentation: https://langfuse.com/docs/sdk/python/low-level-sdk#initialize-client"
            )
            self.tracer = otel_trace_api.NoOpTracer()
            return

        self.host = host or os.environ.get(LANGFUSE_HOST, "https://cloud.langfuse.com")
        self.environment = environment or os.environ.get(LANGFUSE_TRACING_ENVIRONMENT)

        self.tracing_enabled = (
            tracing_enabled
            and os.environ.get(LANGFUSE_TRACING_ENABLED, "True") != "False"
        )

        if not self.tracing_enabled:
            langfuse_logger.info(
                "Configuration: Langfuse tracing is explicitly disabled. No data will be sent to the Langfuse API."
            )

        self._mask = mask

        # Initialize api and tracer if requirements are met
        self.langfuse_tracer = LangfuseTracer(
            public_key=public_key,
            secret_key=secret_key,
            host=self.host,
            timeout=timeout,
            environment=environment,
            release=release,
            flush_at=flush_at,
            flush_interval=flush_interval,
            httpx_client=httpx_client,
            media_upload_thread_count=media_upload_thread_count,
            sample_rate=sample_rate,
        )

        self.tracer = (
            self.langfuse_tracer.tracer
            if self.tracing_enabled
            else otel_trace_api.NoOpTracer()
        )
        self.api = self.langfuse_tracer.api
        self.async_api = self.langfuse_tracer.async_api

    def start_span(
        self,
        *,
        trace_context: Optional[TraceContext] = None,
        name: str,
        input: Optional[Any] = None,
        output: Optional[Any] = None,
        metadata: Optional[Any] = None,
        version: Optional[str] = None,
        level: Optional[SpanLevel] = None,
        status_message: Optional[str] = None,
    ) -> LangfuseSpan:
        """Create a new span for tracing a unit of work.

        This method creates a new span but does not set it as the current span in the
        context. To create and use a span within a context, use start_as_current_span().

        Args:
            trace_context: Optional context for connecting to an existing trace
            name: Name of the span (e.g., function or operation name)
            input: Input data for the operation (can be any JSON-serializable object)
            output: Output data from the operation (can be any JSON-serializable object)
            metadata: Additional metadata to associate with the span
            version: Version identifier for the code or component
            level: Importance level of the span (info, warning, error)
            status_message: Optional status message for the span

        Returns:
            A LangfuseSpan object that must be ended with .end() when the operation completes

        Example:
            ```python
            span = langfuse.start_span(name="process-data")
            try:
                # Do work
                span.update(output="result")
            finally:
                span.end()
            ```
        """
        attributes = create_span_attributes(
            input=input,
            output=output,
            metadata=metadata,
            version=version,
            level=level,
            status_message=status_message,
        )

        if trace_context:
            trace_id = trace_context.get("trace_id", None)
            parent_span_id = trace_context.get("parent_span_id", None)

            if trace_id:
                remote_parent_span = self._create_remote_parent_span(
                    trace_id=trace_id, parent_span_id=parent_span_id
                )

                with otel_trace_api.use_span(
                    cast(otel_trace_api.Span, remote_parent_span)
                ):
                    otel_span = self.tracer.start_span(name=name, attributes=attributes)
                    otel_span.set_attribute(LangfuseSpanAttributes.AS_ROOT, True)

                    return LangfuseSpan(
                        otel_span=otel_span,
                        langfuse_client=self,
                        input=input,
                        output=output,
                        metadata=metadata,
                    )

        otel_span = self.tracer.start_span(name=name, attributes=attributes)

        return LangfuseSpan(
            otel_span=otel_span,
            langfuse_client=self,
            input=input,
            output=output,
            metadata=metadata,
        )

    def start_as_current_span(
        self,
        *,
        trace_context: Optional[TraceContext] = None,
        name: str,
        input: Optional[Any] = None,
        output: Optional[Any] = None,
        metadata: Optional[Any] = None,
        version: Optional[str] = None,
        level: Optional[SpanLevel] = None,
        status_message: Optional[str] = None,
        end_on_exit: Optional[bool] = None,
    ):
        """Create a new span and set it as the current span in a context manager.

        This method creates a new span and sets it as the current span within a context
        manager. Use this method with a 'with' statement to automatically handle span
        lifecycle within a code block.

        Args:
            trace_context: Optional context for connecting to an existing trace
            name: Name of the span (e.g., function or operation name)
            input: Input data for the operation (can be any JSON-serializable object)
            output: Output data from the operation (can be any JSON-serializable object)
            metadata: Additional metadata to associate with the span
            version: Version identifier for the code or component
            level: Importance level of the span (info, warning, error)
            status_message: Optional status message for the span
            end_on_exit (default: True): Whether to end the span automatically when leaving the context manager. If False, the span must be manually ended to avoid memory leaks.

        Returns:
            A context manager that yields a LangfuseSpan

        Example:
            ```python
            with langfuse.start_as_current_span(name="process-query") as span:
                # Do work
                result = process_data()
                span.update(output=result)

                # Create a child span automatically
                with span.start_as_current_span(name="sub-operation") as child_span:
                    # Do sub-operation work
                    child_span.update(output="sub-result")
            ```
        """
        attributes = create_span_attributes(
            input=input,
            output=output,
            metadata=metadata,
            version=version,
            level=level,
            status_message=status_message,
        )

        if trace_context:
            trace_id = trace_context.get("trace_id", None)
            parent_span_id = trace_context.get("parent_span_id", None)

            if trace_id:
                remote_parent_span = self._create_remote_parent_span(
                    trace_id=trace_id, parent_span_id=parent_span_id
                )

                return self._create_span_with_parent_context(
                    as_type="span",
                    name=name,
                    attributes=attributes,
                    remote_parent_span=remote_parent_span,
                    parent=None,
                    input=input,
                    output=output,
                    metadata=metadata,
                    end_on_exit=end_on_exit,
                )

        return self._start_as_current_otel_span_with_processed_media(
            as_type="span",
            name=name,
            attributes=attributes,
            input=input,
            output=output,
            metadata=metadata,
            end_on_exit=end_on_exit,
        )

    def start_generation(
        self,
        *,
        trace_context: Optional[TraceContext] = None,
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
    ) -> LangfuseGeneration:
        """Create a new generation span for AI model interactions.

        This method creates a specialized span for tracking AI model generations/completions.
        It includes additional fields specific to LLM operations such as model name,
        token usage, and cost details.

        Args:
            trace_context: Optional context for connecting to an existing trace
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
            A LangfuseGeneration object that must be ended with .end() when complete

        Example:
            ```python
            generation = langfuse.start_generation(
                name="answer-generation",
                model="gpt-4",
                input={"prompt": "Explain quantum computing"},
                model_parameters={"temperature": 0.7}
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
            ```
        """
        attributes = create_generation_attributes(
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

        if trace_context:
            trace_id = trace_context.get("trace_id", None)
            parent_span_id = trace_context.get("parent_span_id", None)

            if trace_id:
                remote_parent_span = self._create_remote_parent_span(
                    trace_id=trace_id, parent_span_id=parent_span_id
                )

                with otel_trace_api.use_span(
                    cast(otel_trace_api.Span, remote_parent_span)
                ):
                    otel_span = self.tracer.start_span(name=name, attributes=attributes)
                    otel_span.set_attribute(LangfuseSpanAttributes.AS_ROOT, True)

                    return LangfuseGeneration(
                        otel_span=otel_span,
                        langfuse_client=self,
                        input=input,
                        output=output,
                        metadata=metadata,
                    )

        otel_span = self.tracer.start_span(name=name, attributes=attributes)

        return LangfuseGeneration(
            otel_span=otel_span,
            langfuse_client=self,
            input=input,
            output=output,
            metadata=metadata,
        )

    def start_as_current_generation(
        self,
        *,
        trace_context: Optional[TraceContext] = None,
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
        end_on_exit: Optional[bool] = None,
    ):
        """Create a new generation span and set it as the current span in a context manager.

        This method creates a specialized span for AI model generations and sets it as the
        current span within a context manager. Use this method with a 'with' statement to
        automatically handle the generation span lifecycle within a code block.

        Args:
            trace_context: Optional context for connecting to an existing trace
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
            end_on_exit (default: True): Whether to end the span automatically when leaving the context manager. If False, the span must be manually ended to avoid memory leaks.

        Returns:
            A context manager that yields a LangfuseGeneration

        Example:
            ```python
            with langfuse.start_as_current_generation(
                name="answer-generation",
                model="gpt-4",
                input={"prompt": "Explain quantum computing"}
            ) as generation:
                # Call model API
                response = llm.generate(...)

                # Update with results
                generation.update(
                    output=response.text,
                    usage_details={
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens
                    }
                )
            ```
        """
        attributes = create_generation_attributes(
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

        if trace_context:
            trace_id = trace_context.get("trace_id", None)
            parent_span_id = trace_context.get("parent_span_id", None)

            if trace_id:
                remote_parent_span = self._create_remote_parent_span(
                    trace_id=trace_id, parent_span_id=parent_span_id
                )

                return self._create_span_with_parent_context(
                    as_type="generation",
                    name=name,
                    attributes=attributes,
                    remote_parent_span=remote_parent_span,
                    parent=None,
                    input=input,
                    output=output,
                    metadata=metadata,
                    end_on_exit=end_on_exit,
                )

        return self._start_as_current_otel_span_with_processed_media(
            as_type="generation",
            name=name,
            attributes=attributes,
            input=input,
            output=output,
            metadata=metadata,
            end_on_exit=end_on_exit,
        )

    @_agnosticcontextmanager
    def _create_span_with_parent_context(
        self,
        *,
        name,
        parent,
        remote_parent_span,
        attributes,
        as_type: Literal["generation", "span"],
        input: Optional[Any] = None,
        output: Optional[Any] = None,
        metadata: Optional[Any] = None,
        end_on_exit: Optional[bool] = None,
    ):
        parent_span = parent or cast(otel_trace_api.Span, remote_parent_span)

        with otel_trace_api.use_span(parent_span):
            with self._start_as_current_otel_span_with_processed_media(
                name=name,
                attributes=attributes,
                as_type=as_type,
                input=input,
                output=output,
                metadata=metadata,
                end_on_exit=end_on_exit,
            ) as langfuse_span:
                if remote_parent_span is not None:
                    langfuse_span._otel_span.set_attribute(
                        LangfuseSpanAttributes.AS_ROOT, True
                    )

                yield langfuse_span

    @_agnosticcontextmanager
    def _start_as_current_otel_span_with_processed_media(
        self,
        *,
        name: str,
        attributes: Dict[str, str],
        as_type: Optional[Literal["generation", "span"]] = None,
        input: Optional[Any] = None,
        output: Optional[Any] = None,
        metadata: Optional[Any] = None,
        end_on_exit: Optional[bool] = None,
    ):
        with self.tracer.start_as_current_span(
            name=name,
            attributes=attributes,
            end_on_exit=end_on_exit if end_on_exit is not None else True,
        ) as otel_span:
            yield (
                LangfuseSpan(
                    otel_span=otel_span,
                    langfuse_client=self,
                    input=input,
                    output=output,
                    metadata=metadata,
                )
                if as_type == "span"
                else LangfuseGeneration(
                    otel_span=otel_span,
                    langfuse_client=self,
                    input=input,
                    output=output,
                    metadata=metadata,
                )
            )

    def _get_current_otel_span(self) -> Optional[otel_trace_api.Span]:
        current_span = otel_trace_api.get_current_span()

        if current_span is otel_trace_api.INVALID_SPAN:
            langfuse_logger.warning(
                "Context error: No active span in current context. Operations that depend on an active span will be skipped. "
                "Ensure spans are created with start_as_current_span() or that you're operating within an active span context."
            )
            return None

        return current_span

    def update_current_generation(
        self,
        *,
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
    ) -> None:
        """Update the current active generation span with new information.

        This method updates the current generation span in the active context with
        additional information. It's useful for adding output, usage stats, or other
        details that become available during or after model generation.

        Args:
            input: Updated input data for the model
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

        Example:
            ```python
            with langfuse.start_as_current_generation(name="answer-query") as generation:
                # Initial setup and API call
                response = llm.generate(...)

                # Update with results that weren't available at creation time
                langfuse.update_current_generation(
                    output=response.text,
                    usage_details={
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens
                    }
                )
            ```
        """
        if not self.tracing_enabled:
            langfuse_logger.debug(
                "Operation skipped: update_current_generation - Tracing is disabled or client is in no-op mode."
            )
            return

        current_otel_span = self._get_current_otel_span()

        if current_otel_span is not None:
            generation = LangfuseGeneration(
                otel_span=current_otel_span, langfuse_client=self
            )

            generation.update(
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

    def update_current_span(
        self,
        *,
        input: Optional[Any] = None,
        output: Optional[Any] = None,
        metadata: Optional[Any] = None,
        version: Optional[str] = None,
        level: Optional[SpanLevel] = None,
        status_message: Optional[str] = None,
    ) -> None:
        """Update the current active span with new information.

        This method updates the current span in the active context with
        additional information. It's useful for adding outputs or metadata
        that become available during execution.

        Args:
            input: Updated input data for the operation
            output: Output data from the operation
            metadata: Additional metadata to associate with the span
            version: Version identifier for the code or component
            level: Importance level of the span (info, warning, error)
            status_message: Optional status message for the span

        Example:
            ```python
            with langfuse.start_as_current_span(name="process-data") as span:
                # Initial processing
                result = process_first_part()

                # Update with intermediate results
                langfuse.update_current_span(metadata={"intermediate_result": result})

                # Continue processing
                final_result = process_second_part(result)

                # Final update
                langfuse.update_current_span(output=final_result)
            ```
        """
        if not self.tracing_enabled:
            langfuse_logger.debug(
                "Operation skipped: update_current_span - Tracing is disabled or client is in no-op mode."
            )
            return

        current_otel_span = self._get_current_otel_span()

        if current_otel_span is not None:
            span = LangfuseSpan(otel_span=current_otel_span, langfuse_client=self)

            span.update(
                input=input,
                output=output,
                metadata=metadata,
                version=version,
                level=level,
                status_message=status_message,
            )

    def update_current_trace(
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
    ):
        """Update the current trace with additional information.

        This method updates the trace that the current span belongs to. It's useful for
        adding trace-level metadata like user ID, session ID, or tags that apply to
        the entire trace rather than just a single span.

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

        Example:
            ```python
            with langfuse.start_as_current_span(name="handle-request") as span:
                # Get user information
                user = authenticate_user(request)

                # Update trace with user context
                langfuse.update_current_trace(
                    user_id=user.id,
                    session_id=request.session_id,
                    tags=["production", "web-app"]
                )

                # Continue processing
                response = process_request(request)

                # Update span with results
                span.update(output=response)
            ```
        """
        if not self.tracing_enabled:
            langfuse_logger.debug(
                "Operation skipped: update_current_trace - Tracing is disabled or client is in no-op mode."
            )
            return

        current_otel_span = self._get_current_otel_span()

        if current_otel_span is not None:
            span = LangfuseSpan(otel_span=current_otel_span, langfuse_client=self)

            span.update_trace(
                name=name,
                user_id=user_id,
                session_id=session_id,
                version=version,
                input=input,
                output=output,
                metadata=metadata,
                tags=tags,
                public=public,
            )

    def _create_remote_parent_span(
        self, *, trace_id: str, parent_span_id: Optional[str]
    ):
        if not self._is_valid_trace_id(trace_id):
            langfuse_logger.warning(
                f"Passed trace ID '{trace_id}' is not a valid 32 lowercase hex char Langfuse trace id. Ignoring trace ID."
            )

        if parent_span_id and not self._is_valid_span_id(parent_span_id):
            langfuse_logger.warning(
                f"Passed span ID '{parent_span_id}' is not a valid 16 lowercase hex char Langfuse span id. Ignoring parent span ID."
            )

        int_trace_id = int(trace_id, 16)
        int_parent_span_id = (
            int(parent_span_id, 16)
            if parent_span_id
            else RandomIdGenerator().generate_span_id()
        )

        span_context = otel_trace_api.SpanContext(
            trace_id=int_trace_id,
            span_id=int_parent_span_id,
            trace_flags=otel_trace_api.TraceFlags(0x01),  # mark span as sampled
            is_remote=False,
        )

        return trace.NonRecordingSpan(span_context)

    def _is_valid_trace_id(self, trace_id):
        pattern = r"^[0-9a-f]{32}$"

        return bool(re.match(pattern, trace_id))

    def _is_valid_span_id(self, span_id):
        pattern = r"^[0-9a-f]{16}$"

        return bool(re.match(pattern, span_id))

    def create_observation_id(self, *, seed: Optional[str] = None) -> str:
        """Create a unique observation ID for use with Langfuse.

        This method generates a unique observation ID (span ID in OpenTelemetry terms)
        for use with various Langfuse APIs. It can either generate a random ID or
        create a deterministic ID based on a seed string.

        Observation IDs must be 16 lowercase hexadecimal characters, representing 8 bytes.
        This method ensures the generated ID meets this requirement. If you need to
        correlate an external ID with a Langfuse observation ID, use the external ID as
        the seed to get a valid, deterministic observation ID.

        Args:
            seed: Optional string to use as a seed for deterministic ID generation.
                 If provided, the same seed will always produce the same ID.
                 If not provided, a random ID will be generated.

        Returns:
            A 16-character lowercase hexadecimal string representing the observation ID.

        Example:
            ```python
            # Generate a random observation ID
            obs_id = langfuse.create_observation_id()

            # Generate a deterministic ID based on a seed
            user_obs_id = langfuse.create_observation_id(seed="user-123-feedback")

            # Correlate an external item ID with a Langfuse observation ID
            item_id = "item-789012"
            correlated_obs_id = langfuse.create_observation_id(seed=item_id)

            # Use the ID with Langfuse APIs
            langfuse.create_score(
                name="relevance",
                value=0.95,
                trace_id=trace_id,
                observation_id=obs_id
            )
            ```
        """
        if not seed:
            span_id_int = RandomIdGenerator().generate_span_id()

            return self._format_otel_span_id(span_id_int)

        return sha256(seed.encode("utf-8")).digest()[:8].hex()

    def create_trace_id(self, *, seed: Optional[str] = None) -> str:
        """Create a unique trace ID for use with Langfuse.

        This method generates a unique trace ID for use with various Langfuse APIs.
        It can either generate a random ID or create a deterministic ID based on
        a seed string.

        Trace IDs must be 32 lowercase hexadecimal characters, representing 16 bytes.
        This method ensures the generated ID meets this requirement. If you need to
        correlate an external ID with a Langfuse trace ID, use the external ID as the
        seed to get a valid, deterministic Langfuse trace ID.

        Args:
            seed: Optional string to use as a seed for deterministic ID generation.
                 If provided, the same seed will always produce the same ID.
                 If not provided, a random ID will be generated.

        Returns:
            A 32-character lowercase hexadecimal string representing the trace ID.

        Example:
            ```python
            # Generate a random trace ID
            trace_id = langfuse.create_trace_id()

            # Generate a deterministic ID based on a seed
            session_trace_id = langfuse.create_trace_id(seed="session-456")

            # Correlate an external ID with a Langfuse trace ID
            external_id = "external-system-123456"
            correlated_trace_id = langfuse.create_trace_id(seed=external_id)

            # Use the ID with trace context
            with langfuse.start_as_current_span(
                name="process-request",
                trace_context={"trace_id": trace_id}
            ) as span:
                # Operation will be part of the specific trace
                pass
            ```
        """
        if not seed:
            trace_id_int = RandomIdGenerator().generate_trace_id()

            return self._format_otel_trace_id(trace_id_int)

        return sha256(seed.encode("utf-8")).digest()[:16].hex()

    def _get_otel_trace_id(self, otel_span: otel_trace_api.Span):
        span_context = otel_span.get_span_context()

        return self._format_otel_trace_id(span_context.trace_id)

    def _get_otel_span_id(self, otel_span: otel_trace_api.Span):
        span_context = otel_span.get_span_context()

        return self._format_otel_span_id(span_context.span_id)

    def _format_otel_span_id(self, span_id_int: int) -> str:
        """Format an integer span ID to a 16-character lowercase hex string.

        Internal method to convert an OpenTelemetry integer span ID to the standard
        W3C Trace Context format (16-character lowercase hex string).

        Args:
            span_id_int: 64-bit integer representing a span ID

        Returns:
            A 16-character lowercase hexadecimal string
        """
        return format(span_id_int, "016x")

    def _format_otel_trace_id(self, trace_id_int: int) -> str:
        """Format an integer trace ID to a 32-character lowercase hex string.

        Internal method to convert an OpenTelemetry integer trace ID to the standard
        W3C Trace Context format (32-character lowercase hex string).

        Args:
            trace_id_int: 128-bit integer representing a trace ID

        Returns:
            A 32-character lowercase hexadecimal string
        """
        return format(trace_id_int, "032x")

    @overload
    def create_score(
        self,
        *,
        name: str,
        value: float,
        trace_id: str,
        observation_id: Optional[str] = None,
        score_id: Optional[str] = None,
        data_type: Optional[Literal["NUMERIC", "BOOLEAN"]] = None,
        comment: Optional[str] = None,
        config_id: Optional[str] = None,
    ) -> None: ...

    @overload
    def create_score(
        self,
        *,
        name: str,
        value: str,
        trace_id: str,
        score_id: Optional[str] = None,
        observation_id: Optional[str] = None,
        data_type: Optional[Literal["CATEGORICAL"]] = "CATEGORICAL",
        comment: Optional[str] = None,
        config_id: Optional[str] = None,
    ) -> None: ...

    def create_score(
        self,
        *,
        name: str,
        value: Union[float, str],
        trace_id: str,
        observation_id: Optional[str] = None,
        score_id: Optional[str] = None,
        data_type: Optional[ScoreDataType] = None,
        comment: Optional[str] = None,
        config_id: Optional[str] = None,
    ) -> None:
        """Create a score for a specific trace or observation.

        This method creates a score for evaluating a trace or observation. Scores can be
        used to track quality metrics, user feedback, or automated evaluations.

        Args:
            name: Name of the score (e.g., "relevance", "accuracy")
            value: Score value (can be numeric for NUMERIC/BOOLEAN types or string for CATEGORICAL)
            trace_id: ID of the trace to associate the score with
            observation_id: Optional ID of the specific observation to score
            score_id: Optional custom ID for the score (auto-generated if not provided)
            data_type: Type of score (NUMERIC, BOOLEAN, or CATEGORICAL)
            comment: Optional comment or explanation for the score
            config_id: Optional ID of a score config defined in Langfuse

        Example:
            ```python
            # Create a numeric score for accuracy
            langfuse.create_score(
                name="accuracy",
                value=0.92,
                trace_id="abcdef123456",
                data_type="NUMERIC",
                comment="High accuracy with minor irrelevant details"
            )

            # Create a categorical score for sentiment
            langfuse.create_score(
                name="sentiment",
                value="positive",
                trace_id="abcdef123456",
                observation_id="789012",
                data_type="CATEGORICAL"
            )
            ```
        """
        if not self.tracing_enabled:
            return

        score_id = score_id or self.create_observation_id()

        try:
            score_event = {
                "id": score_id,
                "trace_id": trace_id,
                "observation_id": observation_id,
                "name": name,
                "value": value,
                "data_type": data_type,
                "comment": comment,
                "config_id": config_id,
                "environment": self.environment,
            }

            new_body = ScoreBody(**score_event)

            event = {
                "id": self.create_trace_id(),
                "type": "score-create",
                "timestamp": _get_timestamp(),
                "body": new_body,
            }
            self.langfuse_tracer.add_score_task(event)

        except Exception as e:
            langfuse_logger.exception(
                f"Error creating score: Failed to process score event for trace_id={trace_id}, name={name}. Error: {e}"
            )

    @overload
    def score_current_span(
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
    def score_current_span(
        self,
        *,
        name: str,
        value: str,
        score_id: Optional[str] = None,
        data_type: Optional[Literal["CATEGORICAL"]] = "CATEGORICAL",
        comment: Optional[str] = None,
        config_id: Optional[str] = None,
    ) -> None: ...

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

        This method scores the currently active span in the context. It's a convenient
        way to score the current operation without needing to know its trace and span IDs.

        Args:
            name: Name of the score (e.g., "relevance", "accuracy")
            value: Score value (can be numeric for NUMERIC/BOOLEAN types or string for CATEGORICAL)
            score_id: Optional custom ID for the score (auto-generated if not provided)
            data_type: Type of score (NUMERIC, BOOLEAN, or CATEGORICAL)
            comment: Optional comment or explanation for the score
            config_id: Optional ID of a score config defined in Langfuse

        Example:
            ```python
            with langfuse.start_as_current_generation(name="answer-query") as generation:
                # Generate answer
                response = generate_answer(...)
                generation.update(output=response)

                # Score the generation
                langfuse.score_current_span(
                    name="relevance",
                    value=0.85,
                    data_type="NUMERIC",
                    comment="Mostly relevant but contains some tangential information"
                )
            ```
        """
        current_span = self._get_current_otel_span()

        if current_span is not None:
            trace_id = self._get_otel_trace_id(current_span)
            observation_id = self._get_otel_span_id(current_span)

            langfuse_logger.info(
                f"Score: Creating score name='{name}' value={value} for current span ({observation_id}) in trace {trace_id}"
            )

            self.create_score(
                trace_id=trace_id,
                observation_id=observation_id,
                name=name,
                value=cast(str, value),
                score_id=score_id,
                data_type=cast(Literal["CATEGORICAL"], data_type),
                comment=comment,
                config_id=config_id,
            )

    @overload
    def score_current_trace(
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
    def score_current_trace(
        self,
        *,
        name: str,
        value: str,
        score_id: Optional[str] = None,
        data_type: Optional[Literal["CATEGORICAL"]] = "CATEGORICAL",
        comment: Optional[str] = None,
        config_id: Optional[str] = None,
    ) -> None: ...

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

        This method scores the trace of the currently active span. Unlike score_current_span,
        this method associates the score with the entire trace rather than a specific span.
        It's useful for scoring overall performance or quality of the entire operation.

        Args:
            name: Name of the score (e.g., "user_satisfaction", "overall_quality")
            value: Score value (can be numeric for NUMERIC/BOOLEAN types or string for CATEGORICAL)
            score_id: Optional custom ID for the score (auto-generated if not provided)
            data_type: Type of score (NUMERIC, BOOLEAN, or CATEGORICAL)
            comment: Optional comment or explanation for the score
            config_id: Optional ID of a score config defined in Langfuse

        Example:
            ```python
            with langfuse.start_as_current_span(name="process-user-request") as span:
                # Process request
                result = process_complete_request()
                span.update(output=result)

                # Score the overall trace
                langfuse.score_current_trace(
                    name="overall_quality",
                    value=0.95,
                    data_type="NUMERIC",
                    comment="High quality end-to-end response"
                )
            ```
        """
        current_span = self._get_current_otel_span()

        if current_span is not None:
            trace_id = self._get_otel_trace_id(current_span)

            langfuse_logger.info(
                f"Score: Creating score name='{name}' value={value} for entire trace {trace_id}"
            )

            self.create_score(
                trace_id=trace_id,
                name=name,
                value=cast(str, value),
                score_id=score_id,
                data_type=cast(Literal["CATEGORICAL"], data_type),
                comment=comment,
                config_id=config_id,
            )

    def update_finished_trace(self):
        pass

    def flush(self):
        """Force flush all pending spans and events to the Langfuse API.

        This method manually flushes any pending spans, scores, and other events to the
        Langfuse API. It's useful in scenarios where you want to ensure all data is sent
        before proceeding, without waiting for the automatic flush interval.

        Example:
            ```python
            # Record some spans and scores
            with langfuse.start_as_current_span(name="operation") as span:
                # Do work...
                pass

            # Ensure all data is sent to Langfuse before proceeding
            langfuse.flush()

            # Continue with other work
            ```
        """
        self.langfuse_tracer.flush()

    def shutdown(self):
        """Shut down the Langfuse client and flush all pending data.

        This method cleanly shuts down the Langfuse client, ensuring all pending data
        is flushed to the API and all background threads are properly terminated.

        It's important to call this method when your application is shutting down to
        prevent data loss and resource leaks. For most applications, using the client
        as a context manager or relying on the automatic shutdown via atexit is sufficient.

        Example:
            ```python
            # Initialize Langfuse
            langfuse = Langfuse(public_key="...", secret_key="...")

            # Use Langfuse throughout your application
            # ...

            # When application is shutting down
            langfuse.shutdown()
            ```
        """
        self.langfuse_tracer.shutdown()

    def get_current_trace_id(self) -> Optional[str]:
        """Get the trace ID of the current active span.

        This method retrieves the trace ID from the currently active span in the context.
        It can be used to get the trace ID for referencing in logs, external systems,
        or for creating related operations.

        Returns:
            The current trace ID as a 32-character lowercase hexadecimal string,
            or None if there is no active span.

        Example:
            ```python
            with langfuse.start_as_current_span(name="process-request") as span:
                # Get the current trace ID for reference
                trace_id = langfuse.get_current_trace_id()

                # Use it for external correlation
                log.info(f"Processing request with trace_id: {trace_id}")

                # Or pass to another system
                external_system.process(data, trace_id=trace_id)
            ```
        """
        current_otel_span = self._get_current_otel_span()

        return self._get_otel_trace_id(current_otel_span) if current_otel_span else None

    def get_current_observation_id(self) -> Optional[str]:
        """Get the observation ID (span ID) of the current active span.

        This method retrieves the observation ID from the currently active span in the context.
        It can be used to get the observation ID for referencing in logs, external systems,
        or for creating scores or other related operations.

        Returns:
            The current observation ID as a 16-character lowercase hexadecimal string,
            or None if there is no active span.

        Example:
            ```python
            with langfuse.start_as_current_span(name="process-user-query") as span:
                # Get the current observation ID
                observation_id = langfuse.get_current_observation_id()

                # Store it for later reference
                cache.set(f"query_{query_id}_observation", observation_id)

                # Process the query...
            ```
        """
        current_otel_span = self._get_current_otel_span()

        return self._get_otel_span_id(current_otel_span) if current_otel_span else None

    def _get_project_id(self) -> Optional[str]:
        """Fetch and return the current project id. Persisted across requests. Returns None if no project id is found for api keys."""
        if not self.project_id:
            proj = self.api.projects.get()
            if not proj.data or not proj.data[0].id:
                return None

            self.project_id = proj.data[0].id

        return self.project_id

    def get_trace_url(self, *, trace_id: Optional[str] = None) -> Optional[str]:
        """Get the URL to view a trace in the Langfuse UI.

        This method generates a URL that links directly to a trace in the Langfuse UI.
        It's useful for providing links in logs, notifications, or debugging tools.

        Args:
            trace_id: Optional trace ID to generate a URL for. If not provided,
                     the trace ID of the current active span will be used.

        Returns:
            A URL string pointing to the trace in the Langfuse UI,
            or None if the project ID couldn't be retrieved or no trace ID is available.

        Example:
            ```python
            # Get URL for the current trace
            with langfuse.start_as_current_span(name="process-request") as span:
                trace_url = langfuse.get_trace_url()
                log.info(f"Processing trace: {trace_url}")

            # Get URL for a specific trace
            specific_trace_url = langfuse.get_trace_url(trace_id="1234567890abcdef1234567890abcdef")
            send_notification(f"Review needed for trace: {specific_trace_url}")
            ```
        """
        project_id = self._get_project_id()
        current_trace_id = self.get_current_trace_id()
        final_trace_id = trace_id or current_trace_id

        return (
            f"{self.host}/project/{project_id}/traces/{final_trace_id}"
            if project_id and final_trace_id
            else None
        )
