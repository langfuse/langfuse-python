"""``LangfusePlugin`` — the Temporal plugin entry point.

Design summary:

* Subclasses ``temporalio.plugin.SimplePlugin`` (see
  https://python.temporal.io/temporalio.plugin.SimplePlugin.html).
* Reuses ``temporalio.contrib.opentelemetry.OpenTelemetryInterceptor`` for
  Temporal span creation and header-based context propagation. We do not
  re-implement the Temporal tracing semantics — we only enrich the spans
  Temporal already produces.
* Adds a Langfuse enrichment interceptor after the OTel interceptor so
  session/user/tag/metadata attributes are applied to the current span.
* Sets up / reuses an OTel tracer provider and attaches it to the
  Langfuse client at worker startup.
* Flushes the Langfuse client on worker shutdown via ``run_context``.
* Stays sandbox-safe: workflow-side code only touches OpenTelemetry API
  (which is sandbox-compatible and added to the sandbox passthrough
  modules below) and ``temporalio.workflow.info()``. The Langfuse client
  is used exclusively from the worker process, never from workflow code.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Optional

from .config import LangfusePluginConfig

logger = logging.getLogger("langfuse.temporal")

PLUGIN_NAME = "langfuse.LangfusePlugin"


def _build_class() -> type:
    """Build the real ``LangfusePlugin`` class, importing Temporal lazily."""
    import temporalio.plugin
    from temporalio.contrib.opentelemetry import OpenTelemetryInterceptor
    from temporalio.worker import WorkflowRunner
    from temporalio.worker.workflow_sandbox import SandboxedWorkflowRunner

    from .interceptor import build_langfuse_interceptor

    def _configure_workflow_runner(
        existing: Optional[WorkflowRunner],
    ) -> WorkflowRunner:
        """Keep the user's workflow runner; add minimal passthroughs.

        The plugin must not force ``UnsandboxedWorkflowRunner`` — that is an
        explicit anti-goal. For sandboxed runners we only extend the
        passthrough list with OpenTelemetry-related modules that the
        Temporal OTel interceptor and our enrichment interceptor need to
        touch from inside a workflow (``opentelemetry`` itself and
        ``langfuse.temporal.attributes`` / ``langfuse.temporal.redaction``,
        which are pure-Python helpers with no side effects).
        """
        if existing is None:
            # SimplePlugin will resolve the default runner for us; return a
            # fresh sandboxed runner with our passthroughs pre-applied.
            existing = SandboxedWorkflowRunner()
        if not isinstance(existing, SandboxedWorkflowRunner):
            return existing

        extra = [
            "opentelemetry",
            "langfuse.temporal.attributes",
            "langfuse.temporal.redaction",
            "langfuse.temporal.interceptor",
            "langfuse.temporal.config",
        ]
        try:
            restrictions = existing.restrictions.with_passthrough_modules(*extra)
            return SandboxedWorkflowRunner(restrictions=restrictions)
        except Exception:
            logger.warning(
                "Failed to extend sandbox passthrough modules; "
                "falling back to user's existing workflow runner.",
                exc_info=True,
            )
            return existing

    class _LangfusePlugin(temporalio.plugin.SimplePlugin):
        """Real plugin, built lazily so ``temporalio`` stays optional."""

        def __init__(
            self,
            config: Optional[LangfusePluginConfig] = None,
            **overrides: Any,
        ) -> None:
            # Accept either a prepared config or a list of kwargs that map
            # onto the config dataclass. This keeps the constructor ergonomic
            # while still allowing framework presets to pass a fully-built
            # config.
            if config is None:
                config = LangfusePluginConfig()
            for key, value in overrides.items():
                if not hasattr(config, key):
                    raise TypeError(
                        f"Unknown LangfusePlugin option: {key!r}. "
                        f"Pass via LangfusePluginConfig if you need an advanced knob."
                    )
                setattr(config, key, value)

            self._config = config

            otel_interceptor = OpenTelemetryInterceptor(
                add_temporal_spans=config.tracing.add_temporal_spans
            )
            langfuse_interceptor = build_langfuse_interceptor(config)

            super().__init__(
                name=PLUGIN_NAME,
                interceptors=[otel_interceptor, langfuse_interceptor],
                workflow_runner=_configure_workflow_runner,
                run_context=self._run_context,
            )

        # --- lifecycle ------------------------------------------------
        @asynccontextmanager
        async def _run_context(self) -> AsyncIterator[None]:
            """Initialize Langfuse at worker/replayer start, flush on exit.

            The Langfuse client is resolved at startup so that workflow
            code never has to touch it. In ``context_only`` mode (e.g. a
            starter-only process that doesn't have Langfuse credentials),
            we skip client resolution entirely — context propagation
            already happens via the OTel interceptor.
            """
            client = self._config.langfuse_client
            if client is None and not self._config.context_only:
                try:
                    from langfuse import get_client

                    client = get_client()
                except Exception:
                    logger.debug(
                        "Langfuse client could not be resolved; continuing "
                        "with context-only propagation.",
                        exc_info=True,
                    )
                    client = None

            try:
                yield
            finally:
                if client is not None and self._config.flush_on_shutdown:
                    try:
                        client.flush()
                    except Exception:
                        logger.debug("Langfuse flush failed", exc_info=True)

        # --- optional UI enrichment ----------------------------------
        def configure_client(self, config: Any) -> Any:  # type: ignore[override]
            # ``SimplePlugin`` chains ``configure_client`` for us; we only
            # override to set service metadata if the user configured one.
            config = super().configure_client(config)
            return config

        @property
        def config(self) -> LangfusePluginConfig:
            return self._config

    return _LangfusePlugin


_real_class: Optional[type] = None


def _get_real_class() -> type:
    global _real_class
    if _real_class is None:
        _real_class = _build_class()
    return _real_class


def _make_plugin(*args: Any, **kwargs: Any) -> Any:
    return _get_real_class()(*args, **kwargs)


# Public entry point. The attribute is a class, but ``__new__`` returns an
# instance of the *real* ``SimplePlugin`` subclass that ``_build_class``
# produces lazily the first time the plugin is constructed. That keeps
# ``from langfuse.temporal import LangfusePlugin`` working in environments
# that do not have ``temporalio`` installed (attribute access is safe —
# only construction triggers the Temporal import).
class LangfusePlugin:
    """Langfuse Temporal plugin.

    Example::

        from temporalio.client import Client
        from langfuse.temporal import LangfusePlugin

        client = await Client.connect(
            "localhost:7233",
            plugins=[LangfusePlugin()],
        )

    The returned instance is a subclass of
    ``temporalio.plugin.SimplePlugin`` at construction time, so it can be
    passed anywhere Temporal accepts a plugin.
    """

    def __new__(cls, *args: Any, **kwargs: Any) -> Any:  # type: ignore[misc]
        return _make_plugin(*args, **kwargs)


__all__ = ["LangfusePlugin", "PLUGIN_NAME"]
