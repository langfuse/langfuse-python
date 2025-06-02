"""Langfuse OpenTelemetry integration module.

This module implements Langfuse's core observability functionality on top of the OpenTelemetry (OTel) standard.
"""

import logging
import os
import re
import urllib.parse
from datetime import datetime
from hashlib import sha256
from time import time_ns
from typing import Any, Dict, List, Literal, Optional, Union, cast, overload

import backoff
import httpx
from opentelemetry import trace
from opentelemetry import trace as otel_trace_api
from opentelemetry.sdk.trace.id_generator import RandomIdGenerator
from opentelemetry.util._decorator import (
    _AgnosticContextManager,
    _agnosticcontextmanager,
)

from langfuse._client.attributes import (
    LangfuseOtelSpanAttributes,
    create_generation_attributes,
    create_span_attributes,
)
from langfuse._client.datasets import DatasetClient, DatasetItemClient
from langfuse._client.environment_variables import (
    LANGFUSE_DEBUG,
    LANGFUSE_HOST,
    LANGFUSE_PUBLIC_KEY,
    LANGFUSE_SAMPLE_RATE,
    LANGFUSE_SECRET_KEY,
    LANGFUSE_TRACING_ENABLED,
    LANGFUSE_TRACING_ENVIRONMENT,
)
from langfuse._client.resource_manager import LangfuseResourceManager
from langfuse._client.span import (
    LangfuseEvent,
    LangfuseGeneration,
    LangfuseSpan,
)
from langfuse._utils import _get_timestamp
from langfuse._utils.parse_error import handle_fern_exception
from langfuse._utils.prompt_cache import PromptCache
from langfuse.api.resources.commons.errors.error import Error
from langfuse.api.resources.ingestion.types.score_body import ScoreBody
from langfuse.api.resources.prompts.types import (
    CreatePromptRequest_Chat,
    CreatePromptRequest_Text,
    Prompt_Chat,
    Prompt_Text,
)
from langfuse.logger import langfuse_logger
from langfuse.media import LangfuseMedia
from langfuse.model import (
    ChatMessageDict,
    ChatPromptClient,
    CreateDatasetItemRequest,
    CreateDatasetRequest,
    Dataset,
    DatasetItem,
    DatasetStatus,
    MapValue,
    PromptClient,
    TextPromptClient,
)
from langfuse.types import MaskFunction, ScoreDataType, SpanLevel, TraceContext


class Langfuse:
    """Main client for Langfuse tracing and platform features.

    This class provides an interface for creating and managing traces, spans,
    and generations in Langfuse as well as interacting with the Langfuse API.

    The client features a thread-safe singleton pattern for each unique public API key,
    ensuring consistent trace context propagation across your application. It implements
    efficient batching of spans with configurable flush settings and includes background
    thread management for media uploads and score ingestion.

    Configuration is flexible through either direct parameters or environment variables,
    with graceful fallbacks and runtime configuration updates.

    Attributes:
        api: Synchronous API client for Langfuse backend communication
        async_api: Asynchronous API client for Langfuse backend communication
        langfuse_tracer: Internal LangfuseTracer instance managing OpenTelemetry components

    Parameters:
        public_key (Optional[str]): Your Langfuse public API key. Can also be set via LANGFUSE_PUBLIC_KEY environment variable.
        secret_key (Optional[str]): Your Langfuse secret API key. Can also be set via LANGFUSE_SECRET_KEY environment variable.
        host (Optional[str]): The Langfuse API host URL. Defaults to "https://cloud.langfuse.com". Can also be set via LANGFUSE_HOST environment variable.
        timeout (Optional[int]): Timeout in seconds for API requests. Defaults to 30 seconds.
        httpx_client (Optional[httpx.Client]): Custom httpx client for making non-tracing HTTP requests. If not provided, a default client will be created.
        debug (bool): Enable debug logging. Defaults to False. Can also be set via LANGFUSE_DEBUG environment variable.
        tracing_enabled (Optional[bool]): Enable or disable tracing. Defaults to True. Can also be set via LANGFUSE_TRACING_ENABLED environment variable.
        flush_at (Optional[int]): Number of spans to batch before sending to the API. Defaults to 512. Can also be set via LANGFUSE_FLUSH_AT environment variable.
        flush_interval (Optional[float]): Time in seconds between batch flushes. Defaults to 5 seconds. Can also be set via LANGFUSE_FLUSH_INTERVAL environment variable.
        environment (Optional[str]): Environment name for tracing. Default is 'default'. Can also be set via LANGFUSE_TRACING_ENVIRONMENT environment variable. Can be any lowercase alphanumeric string with hyphens and underscores that does not start with 'langfuse'.
        release (Optional[str]): Release version/hash of your application. Used for grouping analytics by release.
        media_upload_thread_count (Optional[int]): Number of background threads for handling media uploads. Defaults to 1. Can also be set via LANGFUSE_MEDIA_UPLOAD_THREAD_COUNT environment variable.
        sample_rate (Optional[float]): Sampling rate for traces (0.0 to 1.0). Defaults to 1.0 (100% of traces are sampled). Can also be set via LANGFUSE_SAMPLE_RATE environment variable.
        mask (Optional[MaskFunction]): Function to mask sensitive data in traces before sending to the API.

    Example:
        ```python
        from langfuse.otel import Langfuse

        # Initialize the client (reads from env vars if not provided)
        langfuse = Langfuse(
            public_key="your-public-key",
            secret_key="your-secret-key",
            host="https://cloud.langfuse.com",  # Optional, default shown
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
        timeout: Optional[int] = None,
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
        self._host = host or os.environ.get(LANGFUSE_HOST, "https://cloud.langfuse.com")
        self._environment = environment or os.environ.get(LANGFUSE_TRACING_ENVIRONMENT)
        self._mask = mask
        self._project_id = None
        sample_rate = sample_rate or float(os.environ.get(LANGFUSE_SAMPLE_RATE, 1.0))
        if not 0.0 <= sample_rate <= 1.0:
            raise ValueError(
                f"Sample rate must be between 0.0 and 1.0, got {sample_rate}"
            )

        self._tracing_enabled = (
            tracing_enabled
            and os.environ.get(LANGFUSE_TRACING_ENABLED, "True") != "False"
        )
        if not self._tracing_enabled:
            langfuse_logger.info(
                "Configuration: Langfuse tracing is explicitly disabled. No data will be sent to the Langfuse API."
            )

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
            self._otel_tracer = otel_trace_api.NoOpTracer()
            return

        secret_key = secret_key or os.environ.get(LANGFUSE_SECRET_KEY)
        if secret_key is None:
            langfuse_logger.warning(
                "Authentication error: Langfuse client initialized without secret_key. Client will be disabled. "
                "Provide a secret_key parameter or set LANGFUSE_SECRET_KEY environment variable. "
                "See documentation: https://langfuse.com/docs/sdk/python/low-level-sdk#initialize-client"
            )
            self._otel_tracer = otel_trace_api.NoOpTracer()
            return

        # Initialize api and tracer if requirements are met
        self._resources = LangfuseResourceManager(
            public_key=public_key,
            secret_key=secret_key,
            host=self._host,
            timeout=timeout,
            environment=environment,
            release=release,
            flush_at=flush_at,
            flush_interval=flush_interval,
            httpx_client=httpx_client,
            media_upload_thread_count=media_upload_thread_count,
            sample_rate=sample_rate,
        )

        self._otel_tracer = (
            self._resources.tracer
            if self._tracing_enabled
            else otel_trace_api.NoOpTracer()
        )
        self.api = self._resources.api
        self.async_api = self._resources.async_api

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

        The created span will be the child of the current span in the context.

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
                    otel_span = self._otel_tracer.start_span(
                        name=name, attributes=attributes
                    )
                    otel_span.set_attribute(LangfuseOtelSpanAttributes.AS_ROOT, True)

                    return LangfuseSpan(
                        otel_span=otel_span,
                        langfuse_client=self,
                        input=input,
                        output=output,
                        metadata=metadata,
                        environment=self._environment,
                    )

        otel_span = self._otel_tracer.start_span(name=name, attributes=attributes)

        return LangfuseSpan(
            otel_span=otel_span,
            langfuse_client=self,
            input=input,
            output=output,
            metadata=metadata,
            environment=self._environment,
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
    ) -> _AgnosticContextManager[LangfuseSpan]:
        """Create a new span and set it as the current span in a context manager.

        This method creates a new span and sets it as the current span within a context
        manager. Use this method with a 'with' statement to automatically handle span
        lifecycle within a code block.

        The created span will be the child of the current span in the context.

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

                return cast(
                    _AgnosticContextManager[LangfuseSpan],
                    self._create_span_with_parent_context(
                        as_type="span",
                        name=name,
                        attributes=attributes,
                        remote_parent_span=remote_parent_span,
                        parent=None,
                        input=input,
                        output=output,
                        metadata=metadata,
                        end_on_exit=end_on_exit,
                    ),
                )

        return cast(
            _AgnosticContextManager[LangfuseSpan],
            self._start_as_current_otel_span_with_processed_media(
                as_type="span",
                name=name,
                attributes=attributes,
                input=input,
                output=output,
                metadata=metadata,
                end_on_exit=end_on_exit,
            ),
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
        """Create a new generation span for model generations.

        This method creates a specialized span for tracking model generations.
        It includes additional fields specific to model generations such as model name,
        token usage, and cost details.

        The created generation span will be the child of the current span in the context.

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
                    otel_span = self._otel_tracer.start_span(
                        name=name, attributes=attributes
                    )
                    otel_span.set_attribute(LangfuseOtelSpanAttributes.AS_ROOT, True)

                    return LangfuseGeneration(
                        otel_span=otel_span,
                        langfuse_client=self,
                        input=input,
                        output=output,
                        metadata=metadata,
                    )

        otel_span = self._otel_tracer.start_span(name=name, attributes=attributes)

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
    ) -> _AgnosticContextManager[LangfuseGeneration]:
        """Create a new generation span and set it as the current span in a context manager.

        This method creates a specialized span for model generations and sets it as the
        current span within a context manager. Use this method with a 'with' statement to
        automatically handle the generation span lifecycle within a code block.

        The created generation span will be the child of the current span in the context.

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

                return cast(
                    _AgnosticContextManager[LangfuseGeneration],
                    self._create_span_with_parent_context(
                        as_type="generation",
                        name=name,
                        attributes=attributes,
                        remote_parent_span=remote_parent_span,
                        parent=None,
                        input=input,
                        output=output,
                        metadata=metadata,
                        end_on_exit=end_on_exit,
                    ),
                )

        return cast(
            _AgnosticContextManager[LangfuseGeneration],
            self._start_as_current_otel_span_with_processed_media(
                as_type="generation",
                name=name,
                attributes=attributes,
                input=input,
                output=output,
                metadata=metadata,
                end_on_exit=end_on_exit,
            ),
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
                        LangfuseOtelSpanAttributes.AS_ROOT, True
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
        with self._otel_tracer.start_as_current_span(
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
                    environment=self._environment,
                )
                if as_type == "span"
                else LangfuseGeneration(
                    otel_span=otel_span,
                    langfuse_client=self,
                    input=input,
                    output=output,
                    metadata=metadata,
                    environment=self._environment,
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
        if not self._tracing_enabled:
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
        if not self._tracing_enabled:
            langfuse_logger.debug(
                "Operation skipped: update_current_span - Tracing is disabled or client is in no-op mode."
            )
            return

        current_otel_span = self._get_current_otel_span()

        if current_otel_span is not None:
            span = LangfuseSpan(
                otel_span=current_otel_span,
                langfuse_client=self,
                environment=self._environment,
            )

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

        This method updates the Langfuse trace that the current span belongs to. It's useful for
        adding trace-level metadata like user ID, session ID, or tags that apply to
        the entire Langfuse trace rather than just a single observation.

        Args:
            name: Updated name for the Langfuse trace
            user_id: ID of the user who initiated the Langfuse trace
            session_id: Session identifier for grouping related Langfuse traces
            version: Version identifier for the application or service
            input: Input data for the overall Langfuse trace
            output: Output data from the overall Langfuse trace
            metadata: Additional metadata to associate with the Langfuse trace
            tags: List of tags to categorize the Langfuse trace
            public: Whether the Langfuse trace should be publicly accessible

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
        if not self._tracing_enabled:
            langfuse_logger.debug(
                "Operation skipped: update_current_trace - Tracing is disabled or client is in no-op mode."
            )
            return

        current_otel_span = self._get_current_otel_span()

        if current_otel_span is not None:
            span = LangfuseSpan(
                otel_span=current_otel_span,
                langfuse_client=self,
                environment=self._environment,
            )

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

    def create_event(
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
    ) -> LangfuseEvent:
        """Create a new Langfuse observation of type 'EVENT'.

        The created Langfuse Event observation will be the child of the current span in the context.

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
            The Langfuse Event object

        Example:
            ```python
            event = langfuse.create_event(name="process-event")
            ```
        """
        timestamp = time_ns()
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
                    otel_span = self._otel_tracer.start_span(
                        name=name, attributes=attributes, start_time=timestamp
                    )
                    otel_span.set_attribute(LangfuseOtelSpanAttributes.AS_ROOT, True)

                    return LangfuseEvent(
                        otel_span=otel_span,
                        langfuse_client=self,
                        input=input,
                        output=output,
                        metadata=metadata,
                        environment=self._environment,
                    ).end(end_time=timestamp)

        otel_span = self._otel_tracer.start_span(
            name=name, attributes=attributes, start_time=timestamp
        )

        return LangfuseEvent(
            otel_span=otel_span,
            langfuse_client=self,
            input=input,
            output=output,
            metadata=metadata,
            environment=self._environment,
        ).end(end_time=timestamp)

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

    def _create_observation_id(self, *, seed: Optional[str] = None) -> str:
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

    @staticmethod
    def create_trace_id(*, seed: Optional[str] = None) -> str:
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
            A 32-character lowercase hexadecimal string representing the Langfuse trace ID.

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

            return Langfuse._format_otel_trace_id(trace_id_int)

        return sha256(seed.encode("utf-8")).digest()[:16].hex()

    def _get_otel_trace_id(self, otel_span: otel_trace_api.Span):
        span_context = otel_span.get_span_context()

        return self._format_otel_trace_id(span_context.trace_id)

    def _get_otel_span_id(self, otel_span: otel_trace_api.Span):
        span_context = otel_span.get_span_context()

        return self._format_otel_span_id(span_context.span_id)

    @staticmethod
    def _format_otel_span_id(span_id_int: int) -> str:
        """Format an integer span ID to a 16-character lowercase hex string.

        Internal method to convert an OpenTelemetry integer span ID to the standard
        W3C Trace Context format (16-character lowercase hex string).

        Args:
            span_id_int: 64-bit integer representing a span ID

        Returns:
            A 16-character lowercase hexadecimal string
        """
        return format(span_id_int, "016x")

    @staticmethod
    def _format_otel_trace_id(trace_id_int: int) -> str:
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
        session_id: Optional[str] = None,
        dataset_run_id: Optional[str] = None,
        trace_id: Optional[str] = None,
        observation_id: Optional[str] = None,
        score_id: Optional[str] = None,
        data_type: Optional[Literal["NUMERIC", "BOOLEAN"]] = None,
        comment: Optional[str] = None,
        config_id: Optional[str] = None,
        metadata: Optional[Any] = None,
    ) -> None: ...

    @overload
    def create_score(
        self,
        *,
        name: str,
        value: str,
        session_id: Optional[str] = None,
        dataset_run_id: Optional[str] = None,
        trace_id: Optional[str] = None,
        score_id: Optional[str] = None,
        observation_id: Optional[str] = None,
        data_type: Optional[Literal["CATEGORICAL"]] = "CATEGORICAL",
        comment: Optional[str] = None,
        config_id: Optional[str] = None,
        metadata: Optional[Any] = None,
    ) -> None: ...

    def create_score(
        self,
        *,
        name: str,
        value: Union[float, str],
        session_id: Optional[str] = None,
        dataset_run_id: Optional[str] = None,
        trace_id: Optional[str] = None,
        observation_id: Optional[str] = None,
        score_id: Optional[str] = None,
        data_type: Optional[ScoreDataType] = None,
        comment: Optional[str] = None,
        config_id: Optional[str] = None,
        metadata: Optional[Any] = None,
    ) -> None:
        """Create a score for a specific trace or observation.

        This method creates a score for evaluating a Langfuse trace or observation. Scores can be
        used to track quality metrics, user feedback, or automated evaluations.

        Args:
            name: Name of the score (e.g., "relevance", "accuracy")
            value: Score value (can be numeric for NUMERIC/BOOLEAN types or string for CATEGORICAL)
            session_id: ID of the Langfuse session to associate the score with
            dataset_run_id: ID of the Langfuse dataset run to associate the score with
            trace_id: ID of the Langfuse trace to associate the score with
            observation_id: Optional ID of the specific observation to score. Trace ID must be provided too.
            score_id: Optional custom ID for the score (auto-generated if not provided)
            data_type: Type of score (NUMERIC, BOOLEAN, or CATEGORICAL)
            comment: Optional comment or explanation for the score
            config_id: Optional ID of a score config defined in Langfuse
            metadata: Optional metadata to be attached to the score

        Example:
            ```python
            # Create a numeric score for accuracy
            langfuse.create_score(
                name="accuracy",
                value=0.92,
                trace_id="abcdef1234567890abcdef1234567890",
                data_type="NUMERIC",
                comment="High accuracy with minor irrelevant details"
            )

            # Create a categorical score for sentiment
            langfuse.create_score(
                name="sentiment",
                value="positive",
                trace_id="abcdef1234567890abcdef1234567890",
                observation_id="abcdef1234567890",
                data_type="CATEGORICAL"
            )
            ```
        """
        if not self._tracing_enabled:
            return

        score_id = score_id or self._create_observation_id()

        try:
            score_event = {
                "id": score_id,
                "session_id": session_id,
                "dataset_run_id": dataset_run_id,
                "trace_id": trace_id,
                "observation_id": observation_id,
                "name": name,
                "value": value,
                "data_type": data_type,
                "comment": comment,
                "config_id": config_id,
                "environment": self._environment,
                "metadata": metadata,
            }

            new_body = ScoreBody(**score_event)

            event = {
                "id": self.create_trace_id(),
                "type": "score-create",
                "timestamp": _get_timestamp(),
                "body": new_body,
            }
            self._resources.add_score_task(event)

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
        self._resources.flush()

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
        self._resources.shutdown()

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
        if not self._project_id:
            proj = self.api.projects.get()
            if not proj.data or not proj.data[0].id:
                return None

            self._project_id = proj.data[0].id

        return self._project_id

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
            f"{self._host}/project/{project_id}/traces/{final_trace_id}"
            if project_id and final_trace_id
            else None
        )

    def get_dataset(
        self, name: str, *, fetch_items_page_size: Optional[int] = 50
    ) -> "DatasetClient":
        """Fetch a dataset by its name.

        Args:
            name (str): The name of the dataset to fetch.
            fetch_items_page_size (Optional[int]): All items of the dataset will be fetched in chunks of this size. Defaults to 50.

        Returns:
            DatasetClient: The dataset with the given name.
        """
        try:
            langfuse_logger.debug(f"Getting datasets {name}")
            dataset = self.api.datasets.get(dataset_name=name)

            dataset_items = []
            page = 1

            while True:
                new_items = self.api.dataset_items.list(
                    dataset_name=self._url_encode(name),
                    page=page,
                    limit=fetch_items_page_size,
                )
                dataset_items.extend(new_items.data)

                if new_items.meta.total_pages <= page:
                    break

                page += 1

            items = [DatasetItemClient(i, langfuse=self) for i in dataset_items]

            return DatasetClient(dataset, items=items)

        except Error as e:
            handle_fern_exception(e)
            raise e

    def auth_check(self) -> bool:
        """Check if the provided credentials (public and secret key) are valid.

        Raises:
            Exception: If no projects were found for the provided credentials.

        Note:
            This method is blocking. It is discouraged to use it in production code.
        """
        try:
            projects = self.api.projects.get()
            langfuse_logger.debug(
                f"Auth check successful, found {len(projects.data)} projects"
            )
            if len(projects.data) == 0:
                raise Exception(
                    "Auth check failed, no project found for the keys provided."
                )
            return True

        except Error as e:
            handle_fern_exception(e)
            raise e

    def create_dataset(
        self,
        *,
        name: str,
        description: Optional[str] = None,
        metadata: Optional[Any] = None,
    ) -> Dataset:
        """Create a dataset with the given name on Langfuse.

        Args:
            name: Name of the dataset to create.
            description: Description of the dataset. Defaults to None.
            metadata: Additional metadata. Defaults to None.

        Returns:
            Dataset: The created dataset as returned by the Langfuse API.
        """
        try:
            body = CreateDatasetRequest(
                name=name, description=description, metadata=metadata
            )
            langfuse_logger.debug(f"Creating datasets {body}")

            return self.api.datasets.create(request=body)

        except Error as e:
            handle_fern_exception(e)
            raise e

    def create_dataset_item(
        self,
        *,
        dataset_name: str,
        input: Optional[Any] = None,
        expected_output: Optional[Any] = None,
        metadata: Optional[Any] = None,
        source_trace_id: Optional[str] = None,
        source_observation_id: Optional[str] = None,
        status: Optional[DatasetStatus] = None,
        id: Optional[str] = None,
    ) -> DatasetItem:
        """Create a dataset item.

        Upserts if an item with id already exists.

        Args:
            dataset_name: Name of the dataset in which the dataset item should be created.
            input: Input data. Defaults to None. Can contain any dict, list or scalar.
            expected_output: Expected output data. Defaults to None. Can contain any dict, list or scalar.
            metadata: Additional metadata. Defaults to None. Can contain any dict, list or scalar.
            source_trace_id: Id of the source trace. Defaults to None.
            source_observation_id: Id of the source observation. Defaults to None.
            status: Status of the dataset item. Defaults to ACTIVE for newly created items.
            id: Id of the dataset item. Defaults to None. Provide your own id if you want to dedupe dataset items. Id needs to be globally unique and cannot be reused across datasets.

        Returns:
            DatasetItem: The created dataset item as returned by the Langfuse API.

        Example:
            ```python
            from langfuse import Langfuse

            langfuse = Langfuse()

            # Uploading items to the Langfuse dataset named "capital_cities"
            langfuse.create_dataset_item(
                dataset_name="capital_cities",
                input={"input": {"country": "Italy"}},
                expected_output={"expected_output": "Rome"},
                metadata={"foo": "bar"}
            )
            ```
        """
        try:
            body = CreateDatasetItemRequest(
                datasetName=dataset_name,
                input=input,
                expectedOutput=expected_output,
                metadata=metadata,
                sourceTraceId=source_trace_id,
                sourceObservationId=source_observation_id,
                status=status,
                id=id,
            )
            langfuse_logger.debug(f"Creating dataset item {body}")
            return self.api.dataset_items.create(request=body)
        except Error as e:
            handle_fern_exception(e)
            raise e

    def resolve_media_references(
        self,
        *,
        obj: Any,
        resolve_with: Literal["base64_data_uri"],
        max_depth: int = 10,
        content_fetch_timeout_seconds: int = 10,
    ):
        """Replace media reference strings in an object with base64 data URIs.

        This method recursively traverses an object (up to max_depth) looking for media reference strings
        in the format "@@@langfuseMedia:...@@@". When found, it (synchronously) fetches the actual media content using
        the provided Langfuse client and replaces the reference string with a base64 data URI.

        If fetching media content fails for a reference string, a warning is logged and the reference
        string is left unchanged.

        Args:
            obj: The object to process. Can be a primitive value, array, or nested object.
                If the object has a __dict__ attribute, a dict will be returned instead of the original object type.
            resolve_with: The representation of the media content to replace the media reference string with.
                Currently only "base64_data_uri" is supported.
            max_depth: int: The maximum depth to traverse the object. Default is 10.
            content_fetch_timeout_seconds: int: The timeout in seconds for fetching media content. Default is 10.

        Returns:
            A deep copy of the input object with all media references replaced with base64 data URIs where possible.
            If the input object has a __dict__ attribute, a dict will be returned instead of the original object type.

        Example:
            obj = {
                "image": "@@@langfuseMedia:type=image/jpeg|id=123|source=bytes@@@",
                "nested": {
                    "pdf": "@@@langfuseMedia:type=application/pdf|id=456|source=bytes@@@"
                }
            }

            result = await LangfuseMedia.resolve_media_references(obj, langfuse_client)

            # Result:
            # {
            #     "image": "data:image/jpeg;base64,/9j/4AAQSkZJRg...",
            #     "nested": {
            #         "pdf": "data:application/pdf;base64,JVBERi0xLjcK..."
            #     }
            # }
        """
        return LangfuseMedia.resolve_media_references(
            langfuse_client=self,
            obj=obj,
            resolve_with=resolve_with,
            max_depth=max_depth,
            content_fetch_timeout_seconds=content_fetch_timeout_seconds,
        )

    @overload
    def get_prompt(
        self,
        name: str,
        *,
        version: Optional[int] = None,
        label: Optional[str] = None,
        type: Literal["chat"],
        cache_ttl_seconds: Optional[int] = None,
        fallback: Optional[List[ChatMessageDict]] = None,
        max_retries: Optional[int] = None,
        fetch_timeout_seconds: Optional[int] = None,
    ) -> ChatPromptClient: ...

    @overload
    def get_prompt(
        self,
        name: str,
        *,
        version: Optional[int] = None,
        label: Optional[str] = None,
        type: Literal["text"] = "text",
        cache_ttl_seconds: Optional[int] = None,
        fallback: Optional[str] = None,
        max_retries: Optional[int] = None,
        fetch_timeout_seconds: Optional[int] = None,
    ) -> TextPromptClient: ...

    def get_prompt(
        self,
        name: str,
        *,
        version: Optional[int] = None,
        label: Optional[str] = None,
        type: Literal["chat", "text"] = "text",
        cache_ttl_seconds: Optional[int] = None,
        fallback: Union[Optional[List[ChatMessageDict]], Optional[str]] = None,
        max_retries: Optional[int] = None,
        fetch_timeout_seconds: Optional[int] = None,
    ) -> PromptClient:
        """Get a prompt.

        This method attempts to fetch the requested prompt from the local cache. If the prompt is not found
        in the cache or if the cached prompt has expired, it will try to fetch the prompt from the server again
        and update the cache. If fetching the new prompt fails, and there is an expired prompt in the cache, it will
        return the expired prompt as a fallback.

        Args:
            name (str): The name of the prompt to retrieve.

        Keyword Args:
            version (Optional[int]): The version of the prompt to retrieve. If no label and version is specified, the `production` label is returned. Specify either version or label, not both.
            label: Optional[str]: The label of the prompt to retrieve. If no label and version is specified, the `production` label is returned. Specify either version or label, not both.
            cache_ttl_seconds: Optional[int]: Time-to-live in seconds for caching the prompt. Must be specified as a
            keyword argument. If not set, defaults to 60 seconds. Disables caching if set to 0.
            type: Literal["chat", "text"]: The type of the prompt to retrieve. Defaults to "text".
            fallback: Union[Optional[List[ChatMessageDict]], Optional[str]]: The prompt string to return if fetching the prompt fails. Important on the first call where no cached prompt is available. Follows Langfuse prompt formatting with double curly braces for variables. Defaults to None.
            max_retries: Optional[int]: The maximum number of retries in case of API/network errors. Defaults to 2. The maximum value is 4. Retries have an exponential backoff with a maximum delay of 10 seconds.
            fetch_timeout_seconds: Optional[int]: The timeout in milliseconds for fetching the prompt. Defaults to the default timeout set on the SDK, which is 10 seconds per default.

        Returns:
            The prompt object retrieved from the cache or directly fetched if not cached or expired of type
            - TextPromptClient, if type argument is 'text'.
            - ChatPromptClient, if type argument is 'chat'.

        Raises:
            Exception: Propagates any exceptions raised during the fetching of a new prompt, unless there is an
            expired prompt in the cache, in which case it logs a warning and returns the expired prompt.
        """
        if version is not None and label is not None:
            raise ValueError("Cannot specify both version and label at the same time.")

        if not name:
            raise ValueError("Prompt name cannot be empty.")

        cache_key = PromptCache.generate_cache_key(name, version=version, label=label)
        bounded_max_retries = self._get_bounded_max_retries(
            max_retries, default_max_retries=2, max_retries_upper_bound=4
        )

        langfuse_logger.debug(f"Getting prompt '{cache_key}'")
        cached_prompt = self._resources.prompt_cache.get(cache_key)

        if cached_prompt is None or cache_ttl_seconds == 0:
            langfuse_logger.debug(
                f"Prompt '{cache_key}' not found in cache or caching disabled."
            )
            try:
                return self._fetch_prompt_and_update_cache(
                    name,
                    version=version,
                    label=label,
                    ttl_seconds=cache_ttl_seconds,
                    max_retries=bounded_max_retries,
                    fetch_timeout_seconds=fetch_timeout_seconds,
                )
            except Exception as e:
                if fallback:
                    langfuse_logger.warning(
                        f"Returning fallback prompt for '{cache_key}' due to fetch error: {e}"
                    )

                    fallback_client_args = {
                        "name": name,
                        "prompt": fallback,
                        "type": type,
                        "version": version or 0,
                        "config": {},
                        "labels": [label] if label else [],
                        "tags": [],
                    }

                    if type == "text":
                        return TextPromptClient(
                            prompt=Prompt_Text(**fallback_client_args),
                            is_fallback=True,
                        )

                    if type == "chat":
                        return ChatPromptClient(
                            prompt=Prompt_Chat(**fallback_client_args),
                            is_fallback=True,
                        )

                raise e

        if cached_prompt.is_expired():
            langfuse_logger.debug(f"Stale prompt '{cache_key}' found in cache.")
            try:
                # refresh prompt in background thread, refresh_prompt deduplicates tasks
                langfuse_logger.debug(f"Refreshing prompt '{cache_key}' in background.")
                self._resources.prompt_cache.add_refresh_prompt_task(
                    cache_key,
                    lambda: self._fetch_prompt_and_update_cache(
                        name,
                        version=version,
                        label=label,
                        ttl_seconds=cache_ttl_seconds,
                        max_retries=bounded_max_retries,
                        fetch_timeout_seconds=fetch_timeout_seconds,
                    ),
                )
                langfuse_logger.debug(
                    f"Returning stale prompt '{cache_key}' from cache."
                )
                # return stale prompt
                return cached_prompt.value

            except Exception as e:
                langfuse_logger.warning(
                    f"Error when refreshing cached prompt '{cache_key}', returning cached version. Error: {e}"
                )
                # creation of refresh prompt task failed, return stale prompt
                return cached_prompt.value

        return cached_prompt.value

    def _fetch_prompt_and_update_cache(
        self,
        name: str,
        *,
        version: Optional[int] = None,
        label: Optional[str] = None,
        ttl_seconds: Optional[int] = None,
        max_retries: int,
        fetch_timeout_seconds,
    ) -> PromptClient:
        cache_key = PromptCache.generate_cache_key(name, version=version, label=label)
        langfuse_logger.debug(f"Fetching prompt '{cache_key}' from server...")

        try:

            @backoff.on_exception(
                backoff.constant, Exception, max_tries=max_retries, logger=None
            )
            def fetch_prompts():
                return self.api.prompts.get(
                    self._url_encode(name),
                    version=version,
                    label=label,
                    request_options={
                        "timeout_in_seconds": fetch_timeout_seconds,
                    }
                    if fetch_timeout_seconds is not None
                    else None,
                )

            prompt_response = fetch_prompts()

            if prompt_response.type == "chat":
                prompt = ChatPromptClient(prompt_response)
            else:
                prompt = TextPromptClient(prompt_response)

            self._resources.prompt_cache.set(cache_key, prompt, ttl_seconds)

            return prompt

        except Exception as e:
            langfuse_logger.error(
                f"Error while fetching prompt '{cache_key}': {str(e)}"
            )
            raise e

    def _get_bounded_max_retries(
        self,
        max_retries: Optional[int],
        *,
        default_max_retries: int = 2,
        max_retries_upper_bound: int = 4,
    ) -> int:
        if max_retries is None:
            return default_max_retries

        bounded_max_retries = min(
            max(max_retries, 0),
            max_retries_upper_bound,
        )

        return bounded_max_retries

    @overload
    def create_prompt(
        self,
        *,
        name: str,
        prompt: List[ChatMessageDict],
        labels: List[str] = [],
        tags: Optional[List[str]] = None,
        type: Optional[Literal["chat"]],
        config: Optional[Any] = None,
        commit_message: Optional[str] = None,
    ) -> ChatPromptClient: ...

    @overload
    def create_prompt(
        self,
        *,
        name: str,
        prompt: str,
        labels: List[str] = [],
        tags: Optional[List[str]] = None,
        type: Optional[Literal["text"]] = "text",
        config: Optional[Any] = None,
        commit_message: Optional[str] = None,
    ) -> TextPromptClient: ...

    def create_prompt(
        self,
        *,
        name: str,
        prompt: Union[str, List[ChatMessageDict]],
        labels: List[str] = [],
        tags: Optional[List[str]] = None,
        type: Optional[Literal["chat", "text"]] = "text",
        config: Optional[Any] = None,
        commit_message: Optional[str] = None,
    ) -> PromptClient:
        """Create a new prompt in Langfuse.

        Keyword Args:
            name : The name of the prompt to be created.
            prompt : The content of the prompt to be created.
            is_active [DEPRECATED] : A flag indicating whether the prompt is active or not. This is deprecated and will be removed in a future release. Please use the 'production' label instead.
            labels: The labels of the prompt. Defaults to None. To create a default-served prompt, add the 'production' label.
            tags: The tags of the prompt. Defaults to None. Will be applied to all versions of the prompt.
            config: Additional structured data to be saved with the prompt. Defaults to None.
            type: The type of the prompt to be created. "chat" vs. "text". Defaults to "text".
            commit_message: Optional string describing the change.

        Returns:
            TextPromptClient: The prompt if type argument is 'text'.
            ChatPromptClient: The prompt if type argument is 'chat'.
        """
        try:
            langfuse_logger.debug(f"Creating prompt {name=}, {labels=}")

            if type == "chat":
                if not isinstance(prompt, list):
                    raise ValueError(
                        "For 'chat' type, 'prompt' must be a list of chat messages with role and content attributes."
                    )
                request = CreatePromptRequest_Chat(
                    name=name,
                    prompt=cast(Any, prompt),
                    labels=labels,
                    tags=tags,
                    config=config or {},
                    commitMessage=commit_message,
                    type="chat",
                )
                server_prompt = self.api.prompts.create(request=request)

                self._resources.prompt_cache.invalidate(name)

                return ChatPromptClient(prompt=cast(Prompt_Chat, server_prompt))

            if not isinstance(prompt, str):
                raise ValueError("For 'text' type, 'prompt' must be a string.")

            request = CreatePromptRequest_Text(
                name=name,
                prompt=prompt,
                labels=labels,
                tags=tags,
                config=config or {},
                commitMessage=commit_message,
                type="text",
            )

            server_prompt = self.api.prompts.create(request=request)

            self._resources.prompt_cache.invalidate(name)

            return TextPromptClient(prompt=cast(Prompt_Text, server_prompt))

        except Error as e:
            handle_fern_exception(e)
            raise e

    def update_prompt(
        self,
        *,
        name: str,
        version: int,
        new_labels: List[str] = [],
    ):
        """Update an existing prompt version in Langfuse. The Langfuse SDK prompt cache is invalidated for all prompts witht he specified name.

        Args:
            name (str): The name of the prompt to update.
            version (int): The version number of the prompt to update.
            new_labels (List[str], optional): New labels to assign to the prompt version. Labels are unique across versions. The "latest" label is reserved and managed by Langfuse. Defaults to [].

        Returns:
            Prompt: The updated prompt from the Langfuse API.

        """
        updated_prompt = self.api.prompt_version.update(
            name=name,
            version=version,
            new_labels=new_labels,
        )
        self._resources.prompt_cache.invalidate(name)

        return updated_prompt

    def _url_encode(self, url: str) -> str:
        return urllib.parse.quote(url)
