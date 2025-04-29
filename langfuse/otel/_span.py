from abc import ABC, abstractmethod
from datetime import datetime
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Union,
    cast,
    overload,
)

from opentelemetry import trace as otel_trace_api

from langfuse.model import PromptClient

if TYPE_CHECKING:
    from langfuse.otel import Langfuse

from langfuse.otel.attributes import (
    LangfuseSpanAttributes,
    create_generation_attributes,
    create_span_attributes,
    create_trace_attributes,
)

from ..types import MapValue, ScoreDataType, SpanLevel
from ._logger import langfuse_logger


class LangfuseSpanWrapper(ABC):
    def __init__(
        self,
        *,
        otel_span: otel_trace_api.Span,
        langfuse_client: "Langfuse",
        as_type: Literal["span", "generation"],
        input: Optional[Any] = None,
        output: Optional[Any] = None,
        metadata: Optional[Any] = None,
    ):
        self._otel_span = otel_span
        self._otel_span.set_attribute(LangfuseSpanAttributes.OBSERVATION_TYPE, as_type)
        self._langfuse_client = langfuse_client

        self.trace_id = self._langfuse_client._get_otel_trace_id(otel_span)
        self.observation_id = self._langfuse_client._get_otel_span_id(otel_span)

        # Handle media only if span is sampled
        if self._otel_span.is_recording:
            media_processed_input = self._process_media_and_apply_mask(
                data=input, field="input", span=self._otel_span
            )
            media_processed_output = self._process_media_and_apply_mask(
                data=output, field="output", span=self._otel_span
            )
            media_processed_metadata = self._process_media_and_apply_mask(
                data=metadata, field="metadata", span=self._otel_span
            )

            attributes = {
                LangfuseSpanAttributes.OBSERVATION_INPUT: media_processed_input,
                LangfuseSpanAttributes.OBSERVATION_OUTPUT: media_processed_output,
                LangfuseSpanAttributes.OBSERVATION_METADATA: media_processed_metadata,
            }

            self._otel_span.set_attributes(
                {k: v for k, v in attributes.items() if v is not None}
            )

    def end(self, *, end_time: Optional[int] = None):
        self._otel_span.end(end_time=end_time)

    @abstractmethod
    def update(self, **kwargs):
        pass

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
    ):
        if not self._otel_span.is_recording():
            return

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
        self._langfuse_client.create_score(
            name=name,
            value=cast(str, value),
            trace_id=self.trace_id,
            observation_id=self.observation_id,
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
        self._langfuse_client.create_score(
            name=name,
            value=cast(str, value),
            trace_id=self.trace_id,
            score_id=score_id,
            data_type=cast(Literal["CATEGORICAL"], data_type),
            comment=comment,
            config_id=config_id,
        )

    def _set_processed_span_attributes(
        self,
        *,
        span: otel_trace_api.Span,
        as_type: Optional[Literal["span", "generation"]] = None,
        input: Optional[Any] = None,
        output: Optional[Any] = None,
        metadata: Optional[Any] = None,
    ):
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

        span.set_attributes(media_processed_attributes)

    def _process_media_and_apply_mask(
        self,
        *,
        data: Optional[Any] = None,
        span: otel_trace_api.Span,
        field: Union[Literal["input"], Literal["output"], Literal["metadata"]],
    ):
        return self._mask_attribute(
            data=self._process_media_in_attribute(data=data, span=span, field=field)
        )

    def _mask_attribute(self, *, data):
        if not self._langfuse_client._mask:
            return data

        try:
            return self._langfuse_client._mask(data=data)
        except Exception as e:
            langfuse_logger.error(f"Mask function failed with error: {e}")

            return "<fully masked due to failed mask function>"

    def _process_media_in_attribute(
        self,
        *,
        data: Optional[Any] = None,
        span: otel_trace_api.Span,
        field: Union[Literal["input"], Literal["output"], Literal["metadata"]],
    ):
        media_processed_attribute = self._langfuse_client.langfuse_tracer._media_manager._find_and_process_media(
            data=data,
            field=field,
            trace_id=self.trace_id,
            observation_id=self.observation_id,
            project_id=self._langfuse_client.langfuse_tracer.project_id,
        )

        return media_processed_attribute


class LangfuseSpan(LangfuseSpanWrapper):
    def __init__(
        self,
        *,
        otel_span: otel_trace_api.Span,
        langfuse_client: "Langfuse",
        input: Optional[Any] = None,
        output: Optional[Any] = None,
        metadata: Optional[Any] = None,
    ):
        super().__init__(
            otel_span=otel_span,
            as_type="span",
            langfuse_client=langfuse_client,
            input=input,
            output=output,
            metadata=metadata,
        )

    def update(
        self,
        *,
        input: Optional[Any] = None,
        output: Optional[Any] = None,
        metadata: Optional[Any] = None,
        version: Optional[str] = None,
        level: Optional[SpanLevel] = None,
        status_message: Optional[str] = None,
        **kwargs,
    ):
        if not self._otel_span.is_recording():
            return

        processed_input = self._process_media_and_apply_mask(
            data=input, field="input", span=self._otel_span
        )
        processed_output = self._process_media_and_apply_mask(
            data=output, field="output", span=self._otel_span
        )
        processed_metadata = self._process_media_and_apply_mask(
            data=metadata, field="metadata", span=self._otel_span
        )

        attributes = create_span_attributes(
            input=processed_input,
            output=processed_output,
            metadata=processed_metadata,
            version=version,
            level=level,
            status_message=status_message,
        )

        self._otel_span.set_attributes(attributes=attributes)

    def start_span(
        self,
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

        with otel_trace_api.use_span(self._otel_span):
            new_otel_span = self._langfuse_client.tracer.start_span(
                name=name, attributes=attributes
            )

            if new_otel_span.is_recording:
                self._set_processed_span_attributes(
                    span=new_otel_span,
                    as_type="span",
                    input=input,
                    output=output,
                    metadata=metadata,
                )

        return LangfuseSpan(
            otel_span=new_otel_span, langfuse_client=self._langfuse_client
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
    ):
        attributes = create_span_attributes(
            input=input,
            output=output,
            metadata=metadata,
            version=version,
            level=level,
            status_message=status_message,
        )

        return self._langfuse_client._create_span_with_parent_context(
            name=name,
            attributes=attributes,
            as_type="span",
            remote_parent_span=None,
            parent=self._otel_span,
            input=input,
            output=output,
            metadata=metadata,
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

        with otel_trace_api.use_span(self._otel_span):
            new_otel_span = self._langfuse_client.tracer.start_span(
                name=name, attributes=attributes
            )

            if new_otel_span.is_recording:
                self._set_processed_span_attributes(
                    span=new_otel_span,
                    as_type="generation",
                    input=input,
                    output=output,
                    metadata=metadata,
                )

        return LangfuseGeneration(
            otel_span=new_otel_span, langfuse_client=self._langfuse_client
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

        return self._langfuse_client._create_span_with_parent_context(
            name=name,
            attributes=attributes,
            as_type="generation",
            remote_parent_span=None,
            parent=self._otel_span,
            input=input,
            output=output,
            metadata=metadata,
        )


class LangfuseGeneration(LangfuseSpanWrapper):
    def __init__(
        self,
        *,
        otel_span: otel_trace_api.Span,
        langfuse_client: "Langfuse",
        input: Optional[Any] = None,
        output: Optional[Any] = None,
        metadata: Optional[Any] = None,
    ):
        super().__init__(
            otel_span=otel_span,
            as_type="generation",
            langfuse_client=langfuse_client,
            input=input,
            output=output,
            metadata=metadata,
        )

    def update(
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
        **kwargs,
    ):
        if not self._otel_span.is_recording():
            return

        processed_input = self._process_media_and_apply_mask(
            data=input, field="input", span=self._otel_span
        )
        processed_output = self._process_media_and_apply_mask(
            data=output, field="output", span=self._otel_span
        )
        processed_metadata = self._process_media_and_apply_mask(
            data=metadata, field="metadata", span=self._otel_span
        )

        attributes = create_generation_attributes(
            input=processed_input,
            output=processed_output,
            metadata=processed_metadata,
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

        self._otel_span.set_attributes(attributes=attributes)
