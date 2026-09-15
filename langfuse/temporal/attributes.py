"""OpenTelemetry span attribute keys used by the Langfuse Temporal plugin.

These are the canonical attribute keys that the plugin writes onto the
OpenTelemetry spans produced by ``temporalio.contrib.opentelemetry``. They are
consumed by Langfuse's OTel ingestion path, which maps selected attributes
onto Langfuse trace / observation fields (``session_id``, ``user_id``,
``tags``, ``metadata``, ``input``, ``output``, etc.).

Keeping them centralized prevents typos and makes it easy to audit exactly
which attributes the plugin can emit.
"""

from __future__ import annotations

# Langfuse core correlation attributes. These mirror the attribute keys used
# by the Langfuse OTel SDK and must stay compatible with Langfuse ingestion.
LANGFUSE_SESSION_ID = "langfuse.session.id"
LANGFUSE_USER_ID = "langfuse.user.id"
LANGFUSE_TAGS = "langfuse.tags"
LANGFUSE_METADATA_PREFIX = "langfuse.metadata."
LANGFUSE_ENVIRONMENT = "langfuse.environment"
LANGFUSE_RELEASE = "langfuse.release"
LANGFUSE_VERSION = "langfuse.version"

# Optional payload capture. These are only written when the plugin is
# explicitly configured to capture payloads.
LANGFUSE_INPUT = "langfuse.observation.input"
LANGFUSE_OUTPUT = "langfuse.observation.output"

# Temporal-specific metadata. These are ingested by Langfuse as generic span
# attributes (and surfaced as observation metadata) and are always safe to
# emit because they are identifiers, not payload bodies.
TEMPORAL_WORKFLOW_ID = "temporal.workflow.id"
TEMPORAL_RUN_ID = "temporal.workflow.run_id"
TEMPORAL_WORKFLOW_TYPE = "temporal.workflow.type"
TEMPORAL_NAMESPACE = "temporal.namespace"
TEMPORAL_TASK_QUEUE = "temporal.task_queue"
TEMPORAL_ACTIVITY_ID = "temporal.activity.id"
TEMPORAL_ACTIVITY_TYPE = "temporal.activity.type"
TEMPORAL_ATTEMPT = "temporal.attempt"
TEMPORAL_PARENT_WORKFLOW_ID = "temporal.parent.workflow_id"
TEMPORAL_PARENT_RUN_ID = "temporal.parent.run_id"
TEMPORAL_IS_LOCAL_ACTIVITY = "temporal.activity.is_local"
TEMPORAL_IS_REPLAYING = "temporal.workflow.is_replaying"


__all__ = [
    "LANGFUSE_SESSION_ID",
    "LANGFUSE_USER_ID",
    "LANGFUSE_TAGS",
    "LANGFUSE_METADATA_PREFIX",
    "LANGFUSE_ENVIRONMENT",
    "LANGFUSE_RELEASE",
    "LANGFUSE_VERSION",
    "LANGFUSE_INPUT",
    "LANGFUSE_OUTPUT",
    "TEMPORAL_WORKFLOW_ID",
    "TEMPORAL_RUN_ID",
    "TEMPORAL_WORKFLOW_TYPE",
    "TEMPORAL_NAMESPACE",
    "TEMPORAL_TASK_QUEUE",
    "TEMPORAL_ACTIVITY_ID",
    "TEMPORAL_ACTIVITY_TYPE",
    "TEMPORAL_ATTEMPT",
    "TEMPORAL_PARENT_WORKFLOW_ID",
    "TEMPORAL_PARENT_RUN_ID",
    "TEMPORAL_IS_LOCAL_ACTIVITY",
    "TEMPORAL_IS_REPLAYING",
]
