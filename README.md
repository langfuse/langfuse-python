![GitHub Banner](https://github.com/langfuse/langfuse-python/assets/2834609/3c36488e-6fe6-4a82-b0f5-5419250ddf86)

# Langfuse Python SDK (legacy v2)

[![MIT License](https://img.shields.io/badge/License-MIT-red.svg?style=flat-square)](https://opensource.org/licenses/MIT)
[![CI test status](https://img.shields.io/github/actions/workflow/status/langfuse/langfuse-python/ci.yml?style=flat-square&label=All%20tests)](https://github.com/langfuse/langfuse-python/actions/workflows/ci.yml?query=branch%3Amain)
[![PyPI Version](https://img.shields.io/pypi/v/langfuse.svg?style=flat-square&label=pypi+langfuse)](https://pypi.python.org/pypi/langfuse)
[![GitHub Repo stars](https://img.shields.io/github/stars/langfuse/langfuse?style=flat-square&logo=GitHub&label=langfuse%2Flangfuse)](https://github.com/langfuse/langfuse)
[![Discord](https://img.shields.io/discord/1111061815649124414?style=flat-square&logo=Discord&logoColor=white&label=Discord&color=%23434EE4)](https://discord.gg/7NXusRtqYU)
[![YC W23](https://img.shields.io/badge/Y%20Combinator-W23-orange?style=flat-square)](https://www.ycombinator.com/companies/langfuse)

> [!IMPORTANT]
> **`langfuse` v2 is not the latest Langfuse Python SDK.** This branch publishes the **deprecated** v2 client (`Langfuse()`, `trace()`, `span()`, `generation()`, `api.trace.list`). Unpinned `pip install langfuse` installs the current v4. Do **not** use v2 for new instrumentation.
>
> **Latest Python SDK (OpenTelemetry)**, developed on [`main`](https://github.com/langfuse/langfuse-python/tree/main): `pip install langfuse`. Use `get_client()` + `start_as_current_observation` / `@observe`. Docs: https://langfuse.com/docs/observability/sdk/overview. Upgrade: [v2 to v3](https://langfuse.com/docs/observability/sdk/upgrade-path/python-v2-to-v3), [v3 to v4](https://langfuse.com/docs/observability/sdk/upgrade-path/python-v3-to-v4).
>
> Query traces with [Observations API v2](https://langfuse.com/docs/api-and-data-platform/features/observations-api) (`GET /api/public/v2/observations`, [API reference](https://api.reference.langfuse.com/#tag/observationsv2)), not `GET /api/public/traces` / `api.trace.list`.
>
> **Data freshness:** the only real-time path is OpenTelemetry ingestion (`langfuse` >= 4.7.0, or any OTel exporter sending `x-langfuse-ingestion-version: 4`) combined with the [Observations API v2](https://langfuse.com/docs/api-and-data-platform/features/observations-api) and [Metrics API v2](https://langfuse.com/docs/metrics/features/metrics-api). All other GET APIs from all SDK versions can be up to 10 minutes behind.

## Installation

This branch is the **legacy v2** SDK, in maintenance mode with critical bug fixes only. For new work, install the current v4 (`pip install langfuse`). To pin this legacy major: `pip install "langfuse>=2,<3"`.

## Docs

- Current SDK (v4): https://langfuse.com/docs/observability/sdk/overview
- Upgrade from this major: https://langfuse.com/docs/observability/sdk/upgrade-path/python-v2-to-v3
- Machine-readable docs index (for AI agents): https://langfuse.com/llms.txt
- REST API reference (Observations API v2, Metrics API v2, Scores API v3): https://api.reference.langfuse.com
- Data freshness and real-time ingestion: https://langfuse.com/docs/compatibility#faq-delay

## Legacy v2 interfaces (do not use for new work)

- `@observe()` decorator ([docs](https://langfuse.com/docs/sdk/python/decorators))
- Low-level tracing SDK ([docs](https://langfuse.com/docs/sdk/python/low-level-sdk))
- Wrapper of Langfuse public API

Integrations (legacy)

- OpenAI SDK ([docs](https://langfuse.com/docs/integrations/openai))
- LlamaIndex ([docs](https://langfuse.com/docs/integrations/llama-index))
- LangChain ([docs](https://langfuse.com/docs/integrations/langchain))
