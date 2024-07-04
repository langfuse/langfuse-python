![GitHub Banner](https://github.com/langfuse/langfuse-python/assets/2834609/3c36488e-6fe6-4a82-b0f5-5419250ddf86)

# Langfuse Python SDK

[![MIT License](https://img.shields.io/badge/License-MIT-red.svg?style=flat-square)](https://opensource.org/licenses/MIT)
[![CI test status](https://img.shields.io/github/actions/workflow/status/langfuse/langfuse-python/ci.yml?style=flat-square&label=All%20tests)](https://github.com/langfuse/langfuse-python/actions/workflows/ci.yml?query=branch%3Amain)
[![PyPI Version](https://img.shields.io/pypi/v/langfuse.svg?style=flat-square&label=pypi+langfuse)](https://pypi.python.org/pypi/langfuse)
[![GitHub Repo stars](https://img.shields.io/github/stars/langfuse/langfuse?style=flat-square&logo=GitHub&label=langfuse%2Flangfuse)](https://github.com/langfuse/langfuse)
[![Discord](https://img.shields.io/discord/1111061815649124414?style=flat-square&logo=Discord&logoColor=white&label=Discord&color=%23434EE4)](https://discord.gg/7NXusRtqYU)
[![YC W23](https://img.shields.io/badge/Y%20Combinator-W23-orange?style=flat-square)](https://www.ycombinator.com/companies/langfuse)

## Installation

> [!IMPORTANT]
> The SDK was rewritten in v2 and released on December 17, 2023. Refer to the [v2 migration guide](https://langfuse.com/docs/sdk/python/low-level-sdk#upgrading-from-v1xx-to-v2xx) for instructions on updating your code.

```
pip install langfuse
```

## Docs

- Decorators: https://langfuse.com/docs/sdk/python/decorators
- Low-level SDK: https://langfuse.com/docs/sdk/python/low-level-sdk
- Langchain integration: https://langfuse.com/docs/integrations/langchain/tracing

## Interfaces

Interfaces:

- `@observe()` decorator ([docs](https://langfuse.com/docs/sdk/python/decorators))
- Low-level tracing SDK ([docs](https://langfuse.com/docs/sdk/python/low-level-sdk))
- Wrapper of Langfuse public API

Integrations

- OpenAI SDK ([docs](https://langfuse.com/docs/integrations/openai))
- LlamaIndex ([docs](https://langfuse.com/docs/integrations/llama-index))
- LangChain ([docs](https://langfuse.com/docs/integrations/langchain))
