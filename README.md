<img width="2400" height="600" alt="hero-b" src="https://github.com/user-attachments/assets/4005eb1b-539d-4d35-9683-3a61ec9d9301" />

# Langfuse Python SDK

[![MIT License](https://img.shields.io/badge/License-MIT-red.svg?style=flat-square)](https://opensource.org/licenses/MIT)
[![CI test status](https://img.shields.io/github/actions/workflow/status/langfuse/langfuse-python/ci.yml?style=flat-square&label=All%20tests)](https://github.com/langfuse/langfuse-python/actions/workflows/ci.yml?query=branch%3Amain)
[![PyPI Version](https://img.shields.io/pypi/v/langfuse.svg?style=flat-square&label=pypi+langfuse)](https://pypi.python.org/pypi/langfuse)
[![GitHub Repo stars](https://img.shields.io/github/stars/langfuse/langfuse?style=flat-square&logo=GitHub&label=langfuse%2Flangfuse)](https://github.com/langfuse/langfuse)
[![Discord](https://img.shields.io/discord/1111061815649124414?style=flat-square&logo=Discord&logoColor=white&label=Discord&color=%23434EE4)](https://discord.gg/7NXusRtqYU)
[![YC W23](https://img.shields.io/badge/Y%20Combinator-W23-orange?style=flat-square)](https://www.ycombinator.com/companies/langfuse)

The [Langfuse](https://langfuse.com) Python SDK covers the full platform: **observability/tracing** (OpenTelemetry-based, with OpenAI and LangChain integrations), **datasets & experiments** (offline evaluation and regression testing of prompt/model changes, including [CI via GitHub Actions](https://github.com/langfuse/experiment-action)), **LLM-as-a-judge and custom evaluations/scores**, **prompt management**, and a **full REST API client**.

## Installation

> [!IMPORTANT]
> The SDK was rewritten in v4 and released in March 2026. Refer to the [v4 migration guide](https://langfuse.com/docs/observability/sdk/upgrade-path/python-v3-to-v4) for instructions on updating your code.

```
pip install langfuse
```

## Quickstart

```python
# env: LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_BASE_URL

from langfuse import get_client

langfuse = get_client()

# Create a span using a context manager
with langfuse.start_as_current_observation(as_type="span", name="process-request") as span:
    # Your processing logic here
    span.update(output="Processing complete")

    # Create a nested generation for an LLM call
    with langfuse.start_as_current_observation(as_type="generation", name="llm-response", model="gpt-5.6") as generation:
        # Your LLM call logic here
        generation.update(output="Generated response")

# All spans are automatically closed when exiting their context blocks


# Flush events in short-lived applications
langfuse.flush()
```

## Docs

- SDK guide: https://langfuse.com/docs/observability/sdk/overview
- Full documentation: https://langfuse.com/docs
- Machine-readable docs index (for AI agents): https://langfuse.com/llms.txt
- API reference of this package: https://python.reference.langfuse.com
