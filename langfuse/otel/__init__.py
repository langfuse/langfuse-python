import logging
import os
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Union, cast

import httpx
from opentelemetry import trace
from opentelemetry import trace as otel_trace_api
from opentelemetry.sdk.trace.id_generator import RandomIdGenerator

from langfuse.model import PromptClient
from langfuse.otel.attributes import (
    LangfuseSpanAttributes,
    create_generation_attributes,
    create_span_attributes,
    create_trace_attributes,
)
from langfuse.otel.environment_variables import (
    LANGFUSE_DEBUG,
    LANGFUSE_HOST,
    LANGFUSE_PUBLIC_KEY,
    LANGFUSE_SECRET_KEY,
    LANGFUSE_TRACING_ENABLED,
    LANGFUSE_TRACING_ENVIRONMENT,
)

from ..types import MapValue, SpanLevel
from ._logger import langfuse_logger
from ._tracer import LangfuseTracer


class Langfuse:
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
        # sample_rate: Optional[float] = None, # TODO: Implement sampling
        # mask: Optional[MaskFunction] = None, # TODO: implement masking
        # sdk_integration: Optional[str] = "default", -> TO BE DEPRECATED
        # threads: Optional[int] = None, -> TO BE DEPRECATED
        # max_retries: Optional[int] = None, -> TO BE DEPRECATED
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
                "Langfuse client is disabled since no public_key was provided as a parameter or environment variable 'LANGFUSE_PUBLIC_KEY'. See our docs: https://langfuse.com/docs/sdk/python/low-level-sdk#initialize-client"
            )
            self.tracer = otel_trace_api.NoOpTracer()
            return

        secret_key = secret_key or os.environ.get(LANGFUSE_SECRET_KEY)
        if secret_key is None:
            langfuse_logger.warning(
                "Langfuse client is disabled since no secret_key was provided as a parameter or environment variable 'LANGFUSE_SECRET_KEY'. See our docs: https://langfuse.com/docs/sdk/python/low-level-sdk#initialize-client"
            )
            self.tracer = otel_trace_api.NoOpTracer()
            return

        host = host or os.environ.get(LANGFUSE_HOST, "https://cloud.langfuse.com")
        environment = environment or os.environ.get(LANGFUSE_TRACING_ENVIRONMENT)

        self.tracing_enabled = (
            tracing_enabled
            and os.environ.get(LANGFUSE_TRACING_ENABLED, "True") != "False"
        )

        if not self.tracing_enabled:
            langfuse_logger.info("Langfuse tracing is disabled")

        # Initialize api and tracer if requirements are met
        self.langfuse_tracer = LangfuseTracer(
            public_key=public_key,
            secret_key=secret_key,
            host=host,
            timeout=timeout,
            environment=environment,
            release=release,
            flush_at=flush_at,
            flush_interval=flush_interval,
            httpx_client=httpx_client,
            media_upload_thread_count=media_upload_thread_count,
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
        name: str,
        parent: Optional[otel_trace_api.Span] = None,
        trace_context: Optional[Dict[str, str]] = None,
        as_type: Optional[Literal["generation"]] = None,
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
    ):
        """Start a new span with the given parent span.

        Args:
            name: The name of the span
            parent_span: The parent span

        Returns:
            A new span with the parent context
        """
        attributes = (
            create_generation_attributes(
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
            if as_type == "generation"
            else create_span_attributes(
                input=input,
                output=output,
                metadata=metadata,
                version=version,
                level=level,
                status_message=status_message,
            )
        )
        remote_parent_span = None

        if trace_context:
            trace_id = trace_context.get("trace_id", None)
            parent_span_id = trace_context.get("parent_span_id", None)

            if trace_id:
                remote_parent_span = Langfuse._create_remote_parent_span(
                    trace_id=trace_id, parent_span_id=parent_span_id
                )

        if parent is not None or remote_parent_span is not None:
            with otel_trace_api.use_span(
                parent or cast(otel_trace_api.Span, remote_parent_span)
            ):
                return self.tracer.start_span(name=name, attributes=attributes)

        span = self.tracer.start_span(name=name, attributes=attributes)

        self._process_media_span_attributes(
            span=span,
            as_type=as_type,
            input=input,
            output=output,
            metadata=metadata,
        )

        return span

    def start_as_current_span(
        self,
        *,
        name: str,
        parent: Optional[otel_trace_api.Span] = None,
        trace_context: Optional[Dict[str, str]] = None,
        as_type: Optional[Literal["generation"]] = None,
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
    ):
        attributes = (
            create_generation_attributes(
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
            if as_type == "generation"
            else create_span_attributes(
                input=input,
                output=output,
                metadata=metadata,
                version=version,
                level=level,
                status_message=status_message,
            )
        )
        remote_parent_span = None

        if trace_context:
            trace_id = trace_context.get("trace_id", None)
            parent_span_id = trace_context.get("parent_span_id", None)

            if trace_id:
                remote_parent_span = Langfuse._create_remote_parent_span(
                    trace_id=trace_id, parent_span_id=parent_span_id
                )

        if parent is not None or remote_parent_span is not None:
            return self._create_span_with_parent_context(
                name=name,
                attributes=attributes,
                remote_parent_span=remote_parent_span,
                parent=parent,
            )

        return self._start_as_current_span_with_processed_media(
            name=name,
            attributes=attributes,
            as_type=as_type,
            input=input,
            output=output,
            metadata=metadata,
        )

    @contextmanager
    def _create_span_with_parent_context(
        self, *, name, parent, remote_parent_span, attributes
    ):
        parent_span = parent or cast(otel_trace_api.Span, remote_parent_span)

        with otel_trace_api.use_span(parent_span):
            with self.tracer.start_as_current_span(
                name=name, attributes=attributes
            ) as span:
                yield span

    @contextmanager
    def _start_as_current_span_with_processed_media(
        self,
        *,
        name: str,
        attributes: Dict[str, str],
        as_type: Optional[Literal["generation"]] = None,
        input: Optional[Any] = None,
        output: Optional[Any] = None,
        metadata: Optional[Any] = None,
    ):
        with self.tracer.start_as_current_span(
            name=name, attributes=attributes
        ) as span:
            self._process_media_span_attributes(
                span=span,
                as_type=as_type,
                input=input,
                output=output,
                metadata=metadata,
            )

            yield span

    def _process_media_span_attributes(
        self,
        *,
        span: otel_trace_api.Span,
        as_type: Optional[Literal["generation"]] = None,
        input: Optional[Any] = None,
        output: Optional[Any] = None,
        metadata: Optional[Any] = None,
    ):
        media_processed_input = self._process_media_attribute(
            span=span,
            data=input,
            field="input",
        )
        media_processed_output = self._process_media_attribute(
            span=span,
            data=output,
            field="output",
        )
        media_processed_metadata = self._process_media_attribute(
            span=span,
            data=metadata,
            field="metadata",
        )

        media_processed_attributes = (
            create_generation_attributes(
                input=media_processed_input,
                output=media_processed_output,
                metadata=media_processed_metadata,
            )
            if as_type == "generation"
            else create_span_attributes(
                input=media_processed_input,
                output=media_processed_output,
                metadata=media_processed_metadata,
            )
        )

        span.set_attributes(media_processed_attributes)

    def _process_media_attribute(
        self,
        *,
        data: Optional[Any] = None,
        span: otel_trace_api.Span,
        field: Union[Literal["input"], Literal["output"], Literal["metadata"]],
    ):
        span_context = span.get_span_context()
        trace_id = self._format_trace_id(span_context.trace_id)
        span_id = self._format_span_id(span_context.span_id)

        media_processed_attribute = (
            self.langfuse_tracer._media_manager._find_and_process_media(
                data=data,
                field=field,
                trace_id=trace_id,
                observation_id=span_id,
                project_id=self.langfuse_tracer.project_id,
            )
        )

        return media_processed_attribute

    @staticmethod  # TODO: reconsider marking methods as static as changing object method later is breaking change
    def get_current_span():
        return otel_trace_api.get_current_span()

    def update_current_span(
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
        if not self.tracing_enabled:
            langfuse_logger.debug("Tracing is disabled. Skipping span update.")
            return

        current_span = Langfuse.get_current_span()

        if current_span is otel_trace_api.INVALID_SPAN:
            langfuse_logger.warning(
                "Current span not found. Please verify you have started span before trying to update it."
            )

        self.update_span(
            current_span,
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
        if not self.tracing_enabled:
            langfuse_logger.debug("Tracing is disabled. Skipping trace update.")
            return

        current_span = Langfuse.get_current_span()

        if current_span is otel_trace_api.INVALID_SPAN:
            langfuse_logger.warning(
                "Current span not found. Please verify you have started span before trying to update it."
            )

        self.update_trace(
            current_span,
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

    def update_span(
        self,
        span: otel_trace_api.Span,
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
    ):
        media_processed_input = self._process_media_attribute(
            data=input, field="input", span=span
        )
        media_processed_output = self._process_media_attribute(
            data=output, field="output", span=span
        )
        media_processed_metadata = self._process_media_attribute(
            data=metadata, field="metadata", span=span
        )

        if any(
            [
                completion_start_time,
                model,
                model_parameters,
                usage_details,
                cost_details,
                prompt,
            ],
        ):
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
            )
        else:
            attributes = create_span_attributes(
                input=media_processed_input,
                output=media_processed_output,
                metadata=media_processed_metadata,
                version=version,
                level=level,
                status_message=status_message,
            )

            # We do not want to redeclare a generation as a span only because the update
            # has not generation specific attributes
            attributes.pop(LangfuseSpanAttributes.OBSERVATION_TYPE)

        span.set_attributes(attributes=attributes)

    def update_trace(
        self,
        span: otel_trace_api.Span,
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
        media_processed_input = self._process_media_attribute(
            data=input, field="input", span=span
        )
        media_processed_output = self._process_media_attribute(
            data=output, field="output", span=span
        )
        media_processed_metadata = self._process_media_attribute(
            data=metadata, field="metadata", span=span
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

        span.set_attributes(attributes)

    def _create_remote_parent_span(*, trace_id: str, parent_span_id: Optional[str]):
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

    @staticmethod
    def create_span_id() -> str:
        span_id_int = RandomIdGenerator().generate_span_id()

        return Langfuse._format_span_id(span_id_int)

    @staticmethod
    def create_trace_id() -> str:
        trace_id_int = RandomIdGenerator().generate_trace_id()

        return Langfuse._format_trace_id(trace_id_int)

    @staticmethod
    def _format_span_id(span_id_int: int) -> str:
        return format(span_id_int, "016x")

    @staticmethod
    def _format_trace_id(trace_id_int: int) -> str:
        return format(trace_id_int, "032x")

    @staticmethod
    def score_current_span():
        pass

    @staticmethod
    def score_current_trace():
        pass

    @staticmethod
    def score_span():
        pass

    @staticmethod
    def score_trace():
        pass

    @staticmethod
    def update_finished_trace():
        pass

    @staticmethod
    def update_finished_span():
        pass

    def flush(self):
        self.langfuse_tracer.flush()

    def shutdown(self):
        self.langfuse_tracer.shutdown()
