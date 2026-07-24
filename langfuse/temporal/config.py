"""Configuration surface for the Langfuse Temporal plugin.

Organized into logical groups so the main plugin constructor does not have
to explode into a giant positional API. Each knob has a conservative default
so that installing the plugin with no arguments gives you safe, metadata-only
Temporal tracing routed to Langfuse.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, List, Mapping, Optional, Sequence

from .redaction import RedactCallback

# Factory signatures. They are called at span-creation time inside the
# Langfuse enrichment interceptor and must be side-effect free: they run
# inside the Temporal workflow sandbox for workflow spans and must not touch
# the network, wall-clock time, or mutable globals.
IdFactory = Callable[["FactoryContext"], Optional[str]]
TagFactory = Callable[["FactoryContext"], Optional[Sequence[str]]]
MetadataFactory = Callable[["FactoryContext"], Optional[Mapping[str, Any]]]


@dataclass
class FactoryContext:
    """Deterministic context passed to user-provided factory callbacks.

    ``kind`` is one of ``"client_start_workflow"``, ``"workflow"``,
    ``"activity"``. ``info`` carries whatever Temporal info object is
    available on the current side (the client input, ``workflow.Info``, or
    ``activity.Info``). ``input`` is the raw Temporal args tuple; it is made
    available so factories can, e.g., pull a user_id off a request object,
    but factories must not mutate it.
    """

    kind: str
    info: Any = None
    input: Any = None


@dataclass
class CaptureConfig:
    """Controls for payload capture on Temporal spans.

    All flags default to ``False`` so that installing the plugin never
    accidentally exports sensitive workflow/activity payloads to Langfuse.
    When a flag is enabled, the corresponding payload is serialized, passed
    through :attr:`redact`, truncated to :attr:`size_limit_bytes`, and
    attached to the relevant span.
    """

    capture_workflow_inputs: bool = False
    capture_workflow_outputs: bool = False
    capture_activity_inputs: bool = False
    capture_activity_outputs: bool = False

    size_limit_bytes: Optional[int] = 32 * 1024
    redact: Optional[RedactCallback] = None

    # Allow/deny lists by Temporal name. A workflow or activity name appears
    # on the denylist wins: it is never captured. When an allowlist is
    # non-empty, only names on the allowlist are captured.
    workflow_allowlist: Optional[Sequence[str]] = None
    workflow_denylist: Optional[Sequence[str]] = None
    activity_allowlist: Optional[Sequence[str]] = None
    activity_denylist: Optional[Sequence[str]] = None

    def should_capture_workflow(self, workflow_type: Optional[str]) -> bool:
        return _should_capture(
            workflow_type, self.workflow_allowlist, self.workflow_denylist
        )

    def should_capture_activity(self, activity_type: Optional[str]) -> bool:
        return _should_capture(
            activity_type, self.activity_allowlist, self.activity_denylist
        )


def _should_capture(
    name: Optional[str],
    allowlist: Optional[Sequence[str]],
    denylist: Optional[Sequence[str]],
) -> bool:
    if name is None:
        # With no name to match against, fall back to the user's allowlist
        # intent: if they specified an allowlist we default-deny, otherwise
        # default-allow.
        return not allowlist
    if denylist and name in denylist:
        return False
    if allowlist:
        return name in allowlist
    return True


@dataclass
class TracingConfig:
    """Controls for which Temporal surfaces produce spans.

    The base plugin always instruments client ``start_workflow`` and worker
    ``execute_workflow`` / ``execute_activity``. The remaining surfaces
    (signals, queries, updates, local activities) are opt-in because they
    can be extremely high-volume in production and often duplicate info
    already present on the parent workflow run.
    """

    add_temporal_spans: bool = True
    trace_signals: bool = False
    trace_queries: bool = False
    trace_updates: bool = True
    trace_local_activities: bool = True


@dataclass
class UIEnrichmentConfig:
    """Controls for correlating Temporal UI fields with Langfuse.

    ``memo_trace_id`` asks the plugin to add the Langfuse ``trace_id`` to
    the Temporal workflow memo so operators can jump from Temporal UI to
    the Langfuse trace. ``search_attribute_key``, when set, does the same
    thing via a custom search attribute (which must already be registered
    in the target namespace).
    """

    memo_trace_id: bool = False
    search_attribute_key: Optional[str] = None


@dataclass
class LangfusePluginConfig:
    """Full configuration for :class:`langfuse.temporal.LangfusePlugin`.

    Using a dataclass keeps the plugin constructor readable and lets
    framework presets build a config once and reuse it.
    """

    # Tracing ownership.
    tracer_provider: Optional[Any] = None
    use_existing_otel: bool = True

    # Langfuse client. When omitted, the plugin uses :func:`langfuse.get_client`
    # at worker startup. Workflow code never touches this object — it is
    # used only for flushing at worker shutdown.
    langfuse_client: Optional[Any] = None
    flush_on_shutdown: bool = True

    # Tracing scope.
    tracing: TracingConfig = field(default_factory=TracingConfig)

    # Privacy.
    capture: CaptureConfig = field(default_factory=CaptureConfig)

    # Correlation.
    session_id_factory: Optional[IdFactory] = None
    user_id_factory: Optional[IdFactory] = None
    tags_factory: Optional[TagFactory] = None
    metadata_factory: Optional[MetadataFactory] = None

    # Static defaults applied to every span (cheap, always safe).
    static_tags: Sequence[str] = field(default_factory=list)
    static_metadata: Mapping[str, Any] = field(default_factory=dict)
    environment: Optional[str] = None
    release: Optional[str] = None
    version: Optional[str] = None

    # UI enrichment.
    ui: UIEnrichmentConfig = field(default_factory=UIEnrichmentConfig)

    # Deployment mode. When ``True`` the plugin installs tracing/context
    # propagation but does not require/initialize a Langfuse exporter,
    # which is the right shape for starter-only processes.
    context_only: bool = False

    def resolved_static_tags(self) -> List[str]:
        return list(self.static_tags)


__all__ = [
    "CaptureConfig",
    "FactoryContext",
    "IdFactory",
    "LangfusePluginConfig",
    "MetadataFactory",
    "TagFactory",
    "TracingConfig",
    "UIEnrichmentConfig",
]
