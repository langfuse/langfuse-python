"""Langfuse enrichment interceptor for Temporal.

This interceptor does **not** create its own spans. Span creation is owned
by ``temporalio.contrib.opentelemetry.OpenTelemetryInterceptor``, which the
plugin registers ahead of this one. We only decorate the current span with
Langfuse-specific attributes (session_id, user_id, tags, metadata, and
optional payload captures) so that Langfuse's OTel ingestion can map the
span onto a trace/observation with the right metadata.

Implemented defensively: if any enrichment fails, it is logged and swallowed
so that the application workflow or activity is never broken by a tracing
error.

``temporalio`` is imported lazily inside the class bodies so that the base
``langfuse.temporal`` module can be imported (and unit tested) without
Temporal installed.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from .attributes import (
    LANGFUSE_ENVIRONMENT,
    LANGFUSE_INPUT,
    LANGFUSE_METADATA_PREFIX,
    LANGFUSE_OUTPUT,
    LANGFUSE_RELEASE,
    LANGFUSE_SESSION_ID,
    LANGFUSE_TAGS,
    LANGFUSE_USER_ID,
    LANGFUSE_VERSION,
    TEMPORAL_ACTIVITY_ID,
    TEMPORAL_ACTIVITY_TYPE,
    TEMPORAL_ATTEMPT,
    TEMPORAL_IS_LOCAL_ACTIVITY,
    TEMPORAL_NAMESPACE,
    TEMPORAL_RUN_ID,
    TEMPORAL_TASK_QUEUE,
    TEMPORAL_WORKFLOW_ID,
    TEMPORAL_WORKFLOW_TYPE,
)
from .config import FactoryContext, LangfusePluginConfig
from .redaction import prepare_payload

logger = logging.getLogger("langfuse.temporal")


def _safe_set(span: Any, key: str, value: Any) -> None:
    """Attach an attribute to the current span without ever raising."""
    if span is None or value is None:
        return
    try:
        span.set_attribute(key, value)
    except Exception:
        logger.debug("Failed to set span attribute %s", key, exc_info=True)


def _json_list(values: Any) -> Optional[str]:
    if values is None:
        return None
    try:
        import json

        return json.dumps(list(values))
    except Exception:
        return None


def _invoke_factory(factory: Any, ctx: FactoryContext) -> Any:
    if factory is None:
        return None
    try:
        return factory(ctx)
    except Exception:
        # Factories run inside the workflow sandbox for workflow spans, so
        # we must not let them escape. We log at debug to avoid tripping
        # logging instrumentation in hot paths.
        logger.debug("Langfuse factory raised", exc_info=True)
        return None


class _Enrichment:
    """Shared helpers that operate against the currently active OTel span.

    Kept as a plain class (no Temporal base class) so that tests can
    exercise the enrichment logic without importing ``temporalio``.
    """

    def __init__(self, config: LangfusePluginConfig) -> None:
        self._config = config

    # ------------------------------------------------------------------
    # Attribute application helpers
    # ------------------------------------------------------------------
    def _current_span(self) -> Any:
        from opentelemetry import trace

        return trace.get_current_span()

    def _apply_static(self, span: Any) -> None:
        cfg = self._config
        if cfg.environment:
            _safe_set(span, LANGFUSE_ENVIRONMENT, cfg.environment)
        if cfg.release:
            _safe_set(span, LANGFUSE_RELEASE, cfg.release)
        if cfg.version:
            _safe_set(span, LANGFUSE_VERSION, cfg.version)

        if cfg.static_metadata:
            for k, v in cfg.static_metadata.items():
                _safe_set(span, LANGFUSE_METADATA_PREFIX + str(k), _coerce_attr(v))

    def _apply_factories(
        self, span: Any, kind: str, info: Any, input_value: Any
    ) -> None:
        cfg = self._config
        ctx = FactoryContext(kind=kind, info=info, input=input_value)

        session_id = _invoke_factory(cfg.session_id_factory, ctx)
        if session_id is None and info is not None:
            # Default: session = workflow_id. This matches the trace model
            # documented in the requirements: a Langfuse trace is one run,
            # a Langfuse session groups all runs of the same workflow_id.
            session_id = getattr(info, "workflow_id", None)
        _safe_set(span, LANGFUSE_SESSION_ID, session_id)

        user_id = _invoke_factory(cfg.user_id_factory, ctx)
        _safe_set(span, LANGFUSE_USER_ID, user_id)

        tags = list(cfg.resolved_static_tags())
        factory_tags = _invoke_factory(cfg.tags_factory, ctx)
        if factory_tags:
            tags.extend(factory_tags)
        if tags:
            _safe_set(span, LANGFUSE_TAGS, _json_list(tags))

        metadata = _invoke_factory(cfg.metadata_factory, ctx)
        if metadata:
            for k, v in metadata.items():
                _safe_set(span, LANGFUSE_METADATA_PREFIX + str(k), _coerce_attr(v))

    # ------------------------------------------------------------------
    # Public entry points, called by the wrapper interceptors below.
    # ------------------------------------------------------------------
    def on_client_start_workflow(self, workflow_type: Optional[str], args: Any) -> None:
        span = self._current_span()
        self._apply_static(span)
        self._apply_factories(
            span,
            kind="client_start_workflow",
            info=None,
            input_value=args,
        )
        if (
            self._config.capture.capture_workflow_inputs
            and self._config.capture.should_capture_workflow(workflow_type)
        ):
            payload = prepare_payload(
                args,
                redact=self._config.capture.redact,
                size_limit_bytes=self._config.capture.size_limit_bytes,
            )
            _safe_set(span, LANGFUSE_INPUT, payload)

    def on_workflow_execute(self, info: Any, args: Any) -> None:
        span = self._current_span()
        self._apply_static(span)
        _safe_set(span, TEMPORAL_WORKFLOW_ID, getattr(info, "workflow_id", None))
        _safe_set(span, TEMPORAL_RUN_ID, getattr(info, "run_id", None))
        _safe_set(span, TEMPORAL_WORKFLOW_TYPE, getattr(info, "workflow_type", None))
        _safe_set(span, TEMPORAL_NAMESPACE, getattr(info, "namespace", None))
        _safe_set(span, TEMPORAL_TASK_QUEUE, getattr(info, "task_queue", None))
        _safe_set(span, TEMPORAL_ATTEMPT, getattr(info, "attempt", None))
        self._apply_factories(span, kind="workflow", info=info, input_value=args)

        workflow_type = getattr(info, "workflow_type", None)
        if (
            self._config.capture.capture_workflow_inputs
            and self._config.capture.should_capture_workflow(workflow_type)
        ):
            payload = prepare_payload(
                args,
                redact=self._config.capture.redact,
                size_limit_bytes=self._config.capture.size_limit_bytes,
            )
            _safe_set(span, LANGFUSE_INPUT, payload)

    def on_workflow_complete(self, info: Any, result: Any) -> None:
        workflow_type = getattr(info, "workflow_type", None)
        if (
            self._config.capture.capture_workflow_outputs
            and self._config.capture.should_capture_workflow(workflow_type)
        ):
            span = self._current_span()
            payload = prepare_payload(
                result,
                redact=self._config.capture.redact,
                size_limit_bytes=self._config.capture.size_limit_bytes,
            )
            _safe_set(span, LANGFUSE_OUTPUT, payload)

    def on_activity_execute(self, info: Any, args: Any, is_local: bool) -> None:
        span = self._current_span()
        self._apply_static(span)
        _safe_set(span, TEMPORAL_WORKFLOW_ID, getattr(info, "workflow_id", None))
        _safe_set(span, TEMPORAL_RUN_ID, getattr(info, "workflow_run_id", None))
        _safe_set(span, TEMPORAL_ACTIVITY_ID, getattr(info, "activity_id", None))
        _safe_set(span, TEMPORAL_ACTIVITY_TYPE, getattr(info, "activity_type", None))
        _safe_set(span, TEMPORAL_NAMESPACE, getattr(info, "workflow_namespace", None))
        _safe_set(span, TEMPORAL_TASK_QUEUE, getattr(info, "task_queue", None))
        _safe_set(span, TEMPORAL_ATTEMPT, getattr(info, "attempt", None))
        _safe_set(span, TEMPORAL_IS_LOCAL_ACTIVITY, bool(is_local))
        self._apply_factories(span, kind="activity", info=info, input_value=args)

        activity_type = getattr(info, "activity_type", None)
        if (
            self._config.capture.capture_activity_inputs
            and self._config.capture.should_capture_activity(activity_type)
        ):
            payload = prepare_payload(
                args,
                redact=self._config.capture.redact,
                size_limit_bytes=self._config.capture.size_limit_bytes,
            )
            _safe_set(span, LANGFUSE_INPUT, payload)

    def on_activity_complete(self, info: Any, result: Any) -> None:
        activity_type = getattr(info, "activity_type", None)
        if (
            self._config.capture.capture_activity_outputs
            and self._config.capture.should_capture_activity(activity_type)
        ):
            span = self._current_span()
            payload = prepare_payload(
                result,
                redact=self._config.capture.redact,
                size_limit_bytes=self._config.capture.size_limit_bytes,
            )
            _safe_set(span, LANGFUSE_OUTPUT, payload)


def _coerce_attr(value: Any) -> Any:
    """Coerce arbitrary metadata values into OTel-attribute-compatible types.

    OTel attribute values must be primitives or homogeneous sequences. Anything
    else gets stringified.
    """
    if isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, (list, tuple)) and all(
        isinstance(v, (str, bool, int, float)) for v in value
    ):
        return list(value)
    try:
        import json

        return json.dumps(value, default=str, ensure_ascii=False)
    except Exception:
        return repr(value)


def build_langfuse_interceptor(config: LangfusePluginConfig) -> Any:
    """Return a Temporal ``Interceptor`` instance that enriches spans.

    Imported lazily so this module is importable without ``temporalio``.
    """
    import temporalio.activity
    import temporalio.client
    import temporalio.worker
    import temporalio.workflow

    enrichment = _Enrichment(config)

    class _LangfuseClientOutbound(temporalio.client.OutboundInterceptor):
        async def start_workflow(self, input: Any) -> Any:  # type: ignore[override]
            try:
                enrichment.on_client_start_workflow(
                    getattr(input, "workflow", None), getattr(input, "args", None)
                )
            except Exception:
                logger.debug("client start_workflow enrichment failed", exc_info=True)
            return await super().start_workflow(input)

    class _LangfuseActivityInbound(temporalio.worker.ActivityInboundInterceptor):
        async def execute_activity(self, input: Any) -> Any:  # type: ignore[override]
            try:
                info = temporalio.activity.info()
                is_local = getattr(info, "is_local", False)
                enrichment.on_activity_execute(
                    info, getattr(input, "args", None), bool(is_local)
                )
            except Exception:
                logger.debug("activity enrichment failed", exc_info=True)
            result = await super().execute_activity(input)
            try:
                info = temporalio.activity.info()
                enrichment.on_activity_complete(info, result)
            except Exception:
                logger.debug("activity complete enrichment failed", exc_info=True)
            return result

    class _LangfuseWorkflowInbound(temporalio.worker.WorkflowInboundInterceptor):
        async def execute_workflow(self, input: Any) -> Any:  # type: ignore[override]
            try:
                info = temporalio.workflow.info()
                enrichment.on_workflow_execute(info, getattr(input, "args", None))
            except Exception:
                # workflow.info() is deterministic & replay-safe; this
                # path should never blow up the workflow.
                pass
            result = await super().execute_workflow(input)
            try:
                info = temporalio.workflow.info()
                enrichment.on_workflow_complete(info, result)
            except Exception:
                pass
            return result

    class LangfuseTracingInterceptor(
        temporalio.client.Interceptor, temporalio.worker.Interceptor
    ):
        def intercept_client(
            self, next: "temporalio.client.OutboundInterceptor"
        ) -> "temporalio.client.OutboundInterceptor":
            return _LangfuseClientOutbound(next)

        def intercept_activity(
            self, next: "temporalio.worker.ActivityInboundInterceptor"
        ) -> "temporalio.worker.ActivityInboundInterceptor":
            return _LangfuseActivityInbound(next)

        def workflow_interceptor_class(self, input: Any) -> Any:
            return _LangfuseWorkflowInbound

    return LangfuseTracingInterceptor()


__all__ = ["build_langfuse_interceptor"]
