import logging
import os
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, cast

import httpx
from opentelemetry import trace
from opentelemetry import trace as otel_trace_api
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.id_generator import RandomIdGenerator

from langfuse.api.client import AsyncFernLangfuse, FernLangfuse
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
from ..version import __version__ as langfuse_version
from ._logger import logger
from .tracer import LangfuseTracer


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
        # sample_rate: Optional[float] = None, # TODO: Implement sampling
        # mask: Optional[MaskFunction] = None, # TODO: implement masking
        # sdk_integration: Optional[str] = "default", -> TO BE DEPRECATED
        # threads: Optional[int] = None, -> TO BE DEPRECATED
        # max_retries: Optional[int] = None, -> TO BE DEPRECATED
    ):
        debug = debug if debug else (os.getenv(LANGFUSE_DEBUG, "False") == "True")

        if debug:
            logger.setLevel(logging.DEBUG)

        public_key = public_key or os.environ.get(LANGFUSE_PUBLIC_KEY)
        if public_key is None:
            logger.warning(
                "Langfuse client is disabled since no public_key was provided as a parameter or environment variable 'LANGFUSE_PUBLIC_KEY'. See our docs: https://langfuse.com/docs/sdk/python/low-level-sdk#initialize-client"
            )
            self.tracer = otel_trace_api.NoOpTracer()
            return

        secret_key = secret_key or os.environ.get(LANGFUSE_SECRET_KEY)
        if secret_key is None:
            logger.warning(
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
            logger.info("Langfuse tracing is disabled")

        # Initialize api and tracer if requirements are met
        self.httpx_client = httpx_client or httpx.Client(timeout=timeout)
        self.api = FernLangfuse(
            base_url=host,
            username=public_key,
            password=secret_key,
            x_langfuse_sdk_name="python",
            x_langfuse_sdk_version=langfuse_version,
            x_langfuse_public_key=public_key,
            httpx_client=self.httpx_client,
            timeout=timeout,
        )
        self.async_api = AsyncFernLangfuse(
            base_url=host,
            username=public_key,
            password=secret_key,
            x_langfuse_sdk_name="python",
            x_langfuse_sdk_version=langfuse_version,
            x_langfuse_public_key=public_key,
            timeout=timeout,
        )

        self.tracer = (
            LangfuseTracer(
                public_key=public_key,
                secret_key=secret_key,
                host=host,
                timeout=timeout,
                environment=environment,
                release=release,
                flush_at=flush_at,
                flush_interval=flush_interval,
            ).tracer
            if self.tracing_enabled
            else otel_trace_api.NoOpTracer()
        )

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

        return self.tracer.start_span(name=name, attributes=attributes)

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

        return self.tracer.start_as_current_span(name=name, attributes=attributes)

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
            logger.debug("Tracing is disabled. Skipping span update.")
            return

        current_span = Langfuse.get_current_span()

        if current_span is otel_trace_api.INVALID_SPAN:
            logger.warning(
                "Current span not found. Please verify you have started span before trying to update it."
            )

        Langfuse.update_span(
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
            logger.debug("Tracing is disabled. Skipping trace update.")
            return

        current_span = Langfuse.get_current_span()

        if current_span is otel_trace_api.INVALID_SPAN:
            logger.warning(
                "Current span not found. Please verify you have started span before trying to update it."
            )

        Langfuse.update_trace(
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

    @staticmethod
    def update_span(
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
            attributes = create_span_attributes(
                input=input,
                output=output,
                metadata=metadata,
                version=version,
                level=level,
                status_message=status_message,
            )

            # We do not want to redeclare a generation as a span only because the update
            # has not generation specific attributes
            attributes.pop(LangfuseSpanAttributes.OBSERVATION_TYPE)

        span.set_attributes(attributes=attributes)

    @staticmethod
    def update_trace(
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
        **kwargs,
    ):
        attributes = create_trace_attributes(
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

        return format(span_id_int, "016x")

    @staticmethod
    def create_trace_id() -> str:
        trace_id_int = RandomIdGenerator().generate_trace_id()

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

    @staticmethod
    def flush():
        tracer_provider = cast(TracerProvider, otel_trace_api.get_tracer_provider())
        if isinstance(tracer_provider, otel_trace_api.ProxyTracerProvider):
            return

        tracer_provider.force_flush()

    @staticmethod
    def shutdown():
        tracer_provider = cast(TracerProvider, otel_trace_api.get_tracer_provider())
        if isinstance(tracer_provider, otel_trace_api.ProxyTracerProvider):
            return

        tracer_provider.force_flush()
