"""Langfuse integration for Temporal (``temporalio``).

This package provides a Temporal plugin that emits OpenTelemetry spans for
Temporal client/workflow/activity operations and routes them to Langfuse
with Langfuse-specific attributes (session id, user id, tags, metadata,
and optional input/output captures).

Public API::

    from langfuse.temporal import LangfusePlugin, LangfusePluginConfig

Usage (minimal)::

    from temporalio.client import Client
    from langfuse.temporal import LangfusePlugin

    client = await Client.connect(
        "localhost:7233",
        plugins=[LangfusePlugin()],
    )

The plugin is worker-capable, so when attached to the ``Client`` it is
automatically carried over to any ``Worker`` constructed from that client.
Do not re-attach it on the worker — see the guide for details.

Framework presets::

    from langfuse.temporal.presets import (
        langfuse_openai_agents_plugins,
        langfuse_pydantic_ai_plugins,
    )

Installation::

    pip install "langfuse[temporal]"

See the Langfuse docs for the full configuration surface, sandbox /
replay caveats, and framework-specific guides.
"""

from __future__ import annotations

from .attributes import (
    LANGFUSE_INPUT,
    LANGFUSE_METADATA_PREFIX,
    LANGFUSE_OUTPUT,
    LANGFUSE_SESSION_ID,
    LANGFUSE_TAGS,
    LANGFUSE_USER_ID,
)
from .config import (
    CaptureConfig,
    FactoryContext,
    LangfusePluginConfig,
    TracingConfig,
    UIEnrichmentConfig,
)
from .plugin import PLUGIN_NAME, LangfusePlugin

__all__ = [
    "LangfusePlugin",
    "LangfusePluginConfig",
    "CaptureConfig",
    "TracingConfig",
    "UIEnrichmentConfig",
    "FactoryContext",
    "PLUGIN_NAME",
    "LANGFUSE_SESSION_ID",
    "LANGFUSE_USER_ID",
    "LANGFUSE_TAGS",
    "LANGFUSE_INPUT",
    "LANGFUSE_OUTPUT",
    "LANGFUSE_METADATA_PREFIX",
]
