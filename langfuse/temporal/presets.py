"""Explicit framework presets for common Temporal + Langfuse topologies.

These are intentionally thin helpers. The base :class:`LangfusePlugin` stays
framework-agnostic; the presets only document the *recommended composition*
for a given framework and return a list of plugins + a small set of setup
steps so users do not have to re-read the docs every time.

The presets do not:

* replace the framework's own Temporal plugin (``OpenAIAgentsPlugin``,
  ``PydanticAIPlugin``, etc.);
* auto-detect which frameworks are installed;
* silently import framework-level tracers — any such import is guarded and
  opt-in.

Users compose the returned list into ``Client.connect(plugins=[...])``.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional

from .config import LangfusePluginConfig
from .plugin import LangfusePlugin

logger = logging.getLogger("langfuse.temporal")


def langfuse_openai_agents_plugins(
    *,
    config: Optional[LangfusePluginConfig] = None,
    model_params: Optional[Any] = None,
    openai_agents_plugin: Optional[Any] = None,
) -> List[Any]:
    """Return plugins for Temporal + OpenAI Agents SDK + Langfuse.

    Temporal's ``OpenAIAgentsPlugin`` keeps ownership of durable agent
    execution (model activities, MCP lifecycle, data conversion). The
    Langfuse plugin only adds Temporal-level tracing and correlation. Pass
    ``openai_agents_plugin`` if you already built one; otherwise one is
    constructed with ``ModelActivityParameters`` defaults.

    This helper does *not* enable OpenAI Agents tracing exporters — that
    setup belongs to the application and should follow the Langfuse
    OpenAI Agents integration guide.
    """
    plugins: List[Any] = []

    if openai_agents_plugin is None:
        try:
            from temporalio.contrib.openai_agents import (
                ModelActivityParameters,
                OpenAIAgentsPlugin,
            )

            openai_agents_plugin = OpenAIAgentsPlugin(
                model_params=model_params or ModelActivityParameters()
            )
        except ImportError:
            logger.warning(
                "temporalio.contrib.openai_agents is not installed; "
                "install 'temporalio[openai-agents]' to use this preset."
            )
            openai_agents_plugin = None

    if openai_agents_plugin is not None:
        plugins.append(openai_agents_plugin)

    plugins.append(LangfusePlugin(config=config))
    return plugins


def langfuse_pydantic_ai_plugins(
    *,
    config: Optional[LangfusePluginConfig] = None,
    pydantic_ai_plugin: Optional[Any] = None,
    instrument_all: bool = True,
) -> List[Any]:
    """Return plugins for Temporal + Pydantic AI + Langfuse.

    Pydantic AI's ``PydanticAIPlugin`` owns durable execution. Pydantic AI's
    OTel-native instrumentation is what produces the framework-level spans
    (model calls, tool calls). The Langfuse plugin stitches those spans
    into the Temporal span tree so that Langfuse sees a single connected
    trace per run.

    If ``instrument_all`` is ``True`` we best-effort call
    ``Agent.instrument_all()`` to enable Pydantic AI's framework-level
    instrumentation. This is a no-op if Pydantic AI is not installed.
    """
    plugins: List[Any] = []

    if pydantic_ai_plugin is None:
        try:
            from pydantic_ai.durable_exec.temporal import PydanticAIPlugin

            pydantic_ai_plugin = PydanticAIPlugin()
        except ImportError:
            logger.warning(
                "pydantic_ai is not installed; install "
                "'pydantic-ai[temporal]' to use this preset."
            )
            pydantic_ai_plugin = None

    if pydantic_ai_plugin is not None:
        plugins.append(pydantic_ai_plugin)

    if instrument_all:
        try:
            from pydantic_ai import Agent

            Agent.instrument_all()
        except Exception:
            logger.debug("Pydantic AI instrument_all() not available", exc_info=True)

    plugins.append(LangfusePlugin(config=config))
    return plugins


__all__ = [
    "langfuse_openai_agents_plugins",
    "langfuse_pydantic_ai_plugins",
]
