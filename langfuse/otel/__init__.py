import logging
import os
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Union, cast, overload

import httpx
from opentelemetry import trace
from opentelemetry import trace as otel_trace_api
from opentelemetry.sdk.trace.id_generator import RandomIdGenerator

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

from ..types import MapValue, MaskFunction, ScoreDataType, SpanLevel
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
        self.environment = environment or os.environ.get(LANGFUSE_TRACING_ENVIRONMENT)

        self.tracing_enabled = (
            tracing_enabled
            and os.environ.get(LANGFUSE_TRACING_ENABLED, "True") != "False"
        )

        if not self.tracing_enabled:
            langfuse_logger.info("Langfuse tracing is disabled")

        self._mask = mask

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
        trace_context: Optional[Dict[str, str]] = None,
        name: str,
        input: Optional[Any] = None,
        output: Optional[Any] = None,
        metadata: Optional[Any] = None,
        version: Optional[str] = None,
        level: Optional[SpanLevel] = None,
        status_message: Optional[str] = None,
    ):
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
                remote_parent_span = Langfuse._create_remote_parent_span(
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
        trace_context: Optional[Dict[str, str]] = None,  # TODO: improve typing
        name: str,
        input: Optional[Any] = None,
        output: Optional[Any] = None,
        metadata: Optional[Any] = None,
        version: Optional[str] = None,
        level: Optional[SpanLevel] = None,
        status_message: Optional[str] = None,
    ):
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
                remote_parent_span = Langfuse._create_remote_parent_span(
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
                )

        return self._start_as_current_otel_span_with_processed_media(
            as_type="span",
            name=name,
            attributes=attributes,
            input=input,
            output=output,
            metadata=metadata,
        )

    def start_generation(
        self,
        *,
        trace_context: Optional[Dict[str, str]] = None,
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
                remote_parent_span = Langfuse._create_remote_parent_span(
                    trace_id=trace_id, parent_span_id=parent_span_id
                )

                # TODO: check why previous context is not reset correctly
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
        trace_context: Optional[Dict[str, str]] = None,  # TODO: improve typing
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

        if trace_context:
            trace_id = trace_context.get("trace_id", None)
            parent_span_id = trace_context.get("parent_span_id", None)

            if trace_id:
                remote_parent_span = Langfuse._create_remote_parent_span(
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
                )

        return self._start_as_current_otel_span_with_processed_media(
            as_type="generation",
            name=name,
            attributes=attributes,
            input=input,
            output=output,
            metadata=metadata,
        )

    @contextmanager
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
            ) as langfuse_span:
                if remote_parent_span is not None:
                    langfuse_span._otel_span.set_attribute(
                        LangfuseSpanAttributes.AS_ROOT, True
                    )

                yield langfuse_span

    @contextmanager
    def _start_as_current_otel_span_with_processed_media(
        self,
        *,
        name: str,
        attributes: Dict[str, str],
        as_type: Optional[Literal["generation", "span"]] = None,
        input: Optional[Any] = None,
        output: Optional[Any] = None,
        metadata: Optional[Any] = None,
    ):
        with self.tracer.start_as_current_span(
            name=name, attributes=attributes
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

    def get_current_span(self) -> Optional[otel_trace_api.Span]:
        current_span = otel_trace_api.get_current_span()

        if current_span is otel_trace_api.INVALID_SPAN:
            langfuse_logger.warning(
                "Current span not found. Please verify you have started a span before trying to update it."
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
        if not self.tracing_enabled:
            langfuse_logger.debug("Tracing is disabled. Skipping span update.")
            return

        current_otel_span = self.get_current_span()

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
        if not self.tracing_enabled:
            langfuse_logger.debug("Tracing is disabled. Skipping span update.")
            return

        current_otel_span = self.get_current_span()

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
        if not self.tracing_enabled:
            langfuse_logger.debug("Tracing is disabled. Skipping trace update.")
            return

        current_otel_span = self.get_current_span()

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

    def _create_observation_id(self) -> str:
        span_id_int = RandomIdGenerator().generate_span_id()

        return self._format_otel_span_id(span_id_int)

    def create_trace_id(self) -> str:
        trace_id_int = RandomIdGenerator().generate_trace_id()

        return self._format_otel_trace_id(trace_id_int)

    def _get_otel_trace_id(self, otel_span: otel_trace_api.Span):
        span_context = otel_span.get_span_context()

        return self._format_otel_trace_id(span_context.trace_id)

    def _get_otel_span_id(self, otel_span: otel_trace_api.Span):
        span_context = otel_span.get_span_context()

        return self._format_otel_span_id(span_context.span_id)

    def _format_otel_span_id(self, span_id_int: int) -> str:
        return format(span_id_int, "016x")

    def _format_otel_trace_id(self, trace_id_int: int) -> str:
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
        if not self.tracing_enabled:
            return

        score_id = score_id or self._create_observation_id()

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
            langfuse_logger.exception(e)

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
        current_span = self.get_current_span()

        if current_span is not None:
            self.create_score(
                trace_id=self._get_otel_trace_id(current_span),
                observation_id=self._get_otel_span_id(current_span),
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
        current_span = self.get_current_span()

        if current_span is not None:
            self.create_score(
                trace_id=self._get_otel_trace_id(current_span),
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
        self.langfuse_tracer.flush()

    def shutdown(self):
        self.langfuse_tracer.shutdown()
