# Langfuse Python SDK

[![MIT License](https://img.shields.io/badge/License-MIT-red.svg?style=flat-square)](https://opensource.org/licenses/MIT)
[![CI test status](https://img.shields.io/github/actions/workflow/status/langfuse/langfuse-python/ci.yml?style=flat-square&label=All%20tests)](https://github.com/langfuse/langfuse-python/actions/workflows/ci.yml?query=branch%3Amain)
[![PyPI Version](https://img.shields.io/pypi/v/langfuse.svg?style=flat-square&label=pypi+langfuse)](https://pypi.python.org/pypi/langfuse)
[![GitHub Repo stars](https://img.shields.io/github/stars/langfuse/langfuse?style=flat-square&logo=GitHub&label=langfuse%2Flangfuse)](https://github.com/langfuse/langfuse)
[![Discord](https://img.shields.io/discord/1111061815649124414?style=flat-square&logo=Discord&logoColor=white&label=Discord&color=%23434EE4)](https://discord.gg/7NXusRtqYU)
[![YC W23](https://img.shields.io/badge/Y%20Combinator-W23-orange?style=flat-square)](https://www.ycombinator.com/companies/langfuse)

```
pip install langfuse
```

Full documentation: https://docs.langfuse.com/sdk/python

Langchain documentation: https://docs.langfuse.com/langchain

## Development

### Install dependencies

```
poetry install
```

### Update openapi spec

1. Generate Fern Python SDK in [langfuse](https://github.com/langfuse/langfuse) and copy the files generated in `generated/python` into the `langfuse/api` folder in this repo.
2. Rebuild and deploy the package to PyPi.

### Deployment

1. poetry version patch
2. poetry build
3. poetry publish
