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

### Add Poetry plugins

```
poetry self add poetry-dotenv-plugin
poetry self add poetry-bumpversion
```

### Install dependencies

```
poetry install
```

### Add Pre-commit

```
poetry run pre-commit install
```

### Tests

#### Setup

- Add .env based on .env.example

#### Run

- Run all
  ```
  poetry run pytest -s -v --log-cli-level=INFO
  ```
- Run a specific test
  ```
  poetry run pytest -s -v --log-cli-level=INFO tests/test_sdk.py::test_flush
  ```
- E2E tests involving OpenAI and Serp API are usually skipped, remove skip decorators in [tests/test_langchain.py](tests/test_langchain.py) to run them.

### Update openapi spec

1. Generate Fern Python SDK in [langfuse](https://github.com/langfuse/langfuse) and copy the files generated in `generated/python` into the `langfuse/api` folder in this repo.
2. Rebuild and deploy the package to PyPi.

### Publish release

1. `poetry version patch`
   - `poetry version prepatch` for pre-release versions
2. `poetry install`
3. `poetry build`
4. `poetry publish`
   - Create PyPi API token: https://pypi.org/manage/account/token/
   - Setup: `poetry config pypi-token.pypi your-api-token`
