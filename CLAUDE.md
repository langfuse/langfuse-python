# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the Langfuse Python SDK, a client library for accessing the Langfuse observability platform. The SDK provides integration with OpenTelemetry (OTel) for tracing, automatic instrumentation for popular LLM frameworks (OpenAI, Langchain, etc.), and direct API access to Langfuse's features.

## Development Commands

### Setup
```bash
# Install Poetry plugins (one-time setup)
poetry self add poetry-dotenv-plugin
poetry self add poetry-bumpversion

# Install all dependencies including optional extras
poetry install --all-extras

# Setup pre-commit hooks
poetry run pre-commit install
```

### Testing
```bash
# Run all tests with verbose output
poetry run pytest -s -v --log-cli-level=INFO

# Run a specific test
poetry run pytest -s -v --log-cli-level=INFO tests/test_core_sdk.py::test_flush

# Run tests in parallel (faster)
poetry run pytest -s -v --log-cli-level=INFO -n auto
```

### Code Quality
```bash
# Format code with Ruff
poetry run ruff format .

# Run linting (development config)
poetry run ruff check .

# Run type checking
poetry run mypy .

# Run pre-commit hooks manually
poetry run pre-commit run --all-files
```

### Building and Releasing
```bash
# Build the package
poetry build

# Run release script (handles versioning, building, tagging, and publishing)
poetry run release

# Generate documentation
poetry run pdoc -o docs/ --docformat google --logo "https://langfuse.com/langfuse_logo.svg" langfuse
```

## Architecture

### Core Components

- **`langfuse/_client/`**: Main SDK implementation built on OpenTelemetry
  - `client.py`: Core Langfuse client with OTel integration
  - `span.py`: LangfuseSpan, LangfuseGeneration, LangfuseEvent classes
  - `observe.py`: Decorator for automatic instrumentation
  - `datasets.py`: Dataset management functionality

- **`langfuse/api/`**: Auto-generated Fern API client
  - Contains all API resources and types
  - Generated from OpenAPI spec - do not manually edit these files

- **`langfuse/_task_manager/`**: Background processing
  - Media upload handling and queue management
  - Score ingestion consumer

- **Integration modules**:
  - `langfuse/openai.py`: OpenAI instrumentation
  - `langfuse/langchain/`: Langchain integration via CallbackHandler

### Key Design Patterns

The SDK is built on OpenTelemetry for observability, using:
- Spans for tracing LLM operations
- Attributes for metadata (see `LangfuseOtelSpanAttributes`)
- Resource management for efficient batching and flushing

The client follows an async-first design with automatic batching of events and background flushing to the Langfuse API.

## Configuration

Environment variables (defined in `_client/environment_variables.py`):
- `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY`: API credentials
- `LANGFUSE_HOST`: API endpoint (defaults to https://cloud.langfuse.com)
- `LANGFUSE_DEBUG`: Enable debug logging
- `LANGFUSE_TRACING_ENABLED`: Enable/disable tracing
- `LANGFUSE_SAMPLE_RATE`: Sampling rate for traces

## Testing Notes

- Create `.env` file based on `.env.template` for integration tests
- E2E tests with external APIs (OpenAI, SERP) are typically skipped in CI
- Remove `@pytest.mark.skip` decorators in test files to run external API tests
- Tests use `respx` for HTTP mocking and `pytest-httpserver` for test servers

## Important Files

- `pyproject.toml`: Poetry configuration, dependencies, and tool settings
- `ruff.toml`: Local development linting config (stricter)
- `ci.ruff.toml`: CI linting config (more permissive)
- `langfuse/version.py`: Version string (updated by release script)

## API Generation

The `langfuse/api/` directory is auto-generated from the Langfuse OpenAPI specification using Fern. To update:

1. Generate new SDK in main Langfuse repo
2. Copy generated files from `generated/python` to `langfuse/api/`
3. Run `poetry run ruff format .` to format the generated code

## Testing Guidelines

### Approach to Test Changes
- Don't remove functionality from existing unit tests just to make tests pass. Only change the test, if underlying code changes warrant a test change.

## Python Code Rules

### Exception Handling
- Exception must not use an f-string literal, assign to variable first
