import base64
from typing import Optional

from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from langfuse.otel._logger import logger
from langfuse.otel._utils import span_formatter
from langfuse.otel.constants import LANGFUSE_TRACER_NAME
from langfuse.version import __version__ as langfuse_version


class LangfuseSpanProcessor(BatchSpanProcessor):
    def __init__(
        self,
        *,
        public_key: str,
        secret_key: str,
        host: str,
        timeout: Optional[int] = None,
        flush_at: Optional[int] = None,
        flush_interval: Optional[float] = None,
    ):
        self.public_key = public_key

        langfuse_span_exporter = OTLPSpanExporter(
            endpoint=f"{host}/api/public/otel/v1/traces",
            headers={
                "Authorization": "Basic "
                + base64.b64encode(f"{public_key}:{secret_key}".encode("utf-8")).decode(
                    "ascii"
                ),
                "x_langfuse_sdk_name": "python",
                "x_langfuse_sdk_version": langfuse_version,
                "x_langfuse_public_key": public_key,
            },
            timeout=timeout,
        )

        super().__init__(
            span_exporter=langfuse_span_exporter,
            export_timeout_millis=timeout * 1_000 if timeout else None,
            max_export_batch_size=flush_at,
            schedule_delay_millis=flush_interval,
        )

    def on_end(self, span: ReadableSpan) -> None:
        # Only export spans that belong to the scoped project
        # This is important to not send spans to wrong project in multi-project setups
        if self._is_langfuse_span(span) and not self._is_langfuse_project_span(span):
            logger.debug(
                f"Skipping span from different project (current processor is for project '{self.public_key}'): {span_formatter(span)}"
            )
            return

        logger.debug(f"Processing span:\n{span_formatter(span)}")

        super().on_end(span)

    @staticmethod
    def _is_langfuse_span(span: ReadableSpan) -> bool:
        return (
            span.instrumentation_scope is not None
            and LANGFUSE_TRACER_NAME in span.instrumentation_scope.name
        )

    def _is_langfuse_project_span(self, span: ReadableSpan) -> bool:
        if not LangfuseSpanProcessor._is_langfuse_span(span):
            return False

        if span.instrumentation_scope is not None:
            public_key_in_span = span.instrumentation_scope.name.split(":")[-1]

            return public_key_in_span == self.public_key

        return False
