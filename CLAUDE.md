# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the **Langfuse Python SDK**, an observability and analytics platform for AI applications. It provides tracing, evaluation, and analytics for LLM applications through multiple interfaces: decorators, low-level SDK, and integrations with popular AI libraries.

## Development Commands

### Environment Setup
```bash
# Using UV (preferred)
uv venv --python 3.12
source .venv/bin/activate
uv sync

# Using Poetry (legacy)
poetry install --all-extras
poetry run pre-commit install
```

### Testing
```bash
# Run all tests
poetry run pytest -s -v --log-cli-level=INFO

# Run specific test
poetry run pytest -s -v --log-cli-level=INFO tests/test_core_sdk.py::test_flush

# Run with UV
uv run pytest -s -v --log-cli-level=INFO
```

### Memory for Running Unit Tests
- To run unit tests you must always use the env file, use: `UV_ENV_FILE=.env uv run pytest -s -v --log-cli-level=INFO tests/TESTFILE::TEST_NAME`

### Code Quality
```bash
# Format code
ruff format .

# Run linter (development config)
ruff check .

# Run linter (CI config)
ruff check --config ci.ruff.toml .
```

### Documentation
```bash
# Generate SDK reference docs
poetry run pdoc -o docs/ --docformat google --logo "https://langfuse.com/langfuse_logo.svg" langfuse

# Serve docs locally
poetry run pdoc --docformat google --logo "https://langfuse.com/langfuse_logo.svg" langfuse
```

## Architecture Overview

### Core Components

**Main SDK Client** (`langfuse/_client/client.py`)
- Built on OpenTelemetry foundations
- Provides span management for tracing (LangfuseSpan, LangfuseGeneration, LangfuseEvent)
- Thread-safe singleton pattern with multi-project support
- Handles both sync and async operations

**Resource Manager** (`langfuse/_client/resource_manager.py`)
- Central coordination hub implementing thread-safe singleton
- Manages OpenTelemetry setup, API clients, background workers
- Handles media upload and score ingestion queues
- Provides graceful shutdown and resource cleanup

**API Layer** (`langfuse/api/`)
- Auto-generated from OpenAPI spec using Fern
- Provides complete typed client for Langfuse API
- Organized by resources (prompts, datasets, observations, etc.)

### SDK Interfaces

**Three primary interaction patterns:**

1. **Decorator Pattern** (`@observe`)
   ```python
   @observe(as_type="generation")
   def my_llm_function():
       # Automatically traced
   ```

2. **Low-level Client API**
   ```python
   langfuse = Langfuse()
   span = langfuse.start_span(name="operation")
   ```

3. **Integration Libraries**
   ```python
   from langfuse.openai import openai  # Drop-in replacement
   ```

### Integration Architecture

**OpenAI Integration** (`langfuse/openai.py`)
- Drop-in replacement supporting both v0.x and v1.x
- Wraps all completion methods (chat, completions, streaming, async)
- Automatic metrics collection (tokens, cost, latency)

**LangChain Integration** (`langfuse/langchain/`)
- CallbackHandler pattern for chain tracing
- Automatic span creation for chains, tools, and agents
- UUID mapping between LangChain runs and Langfuse spans

### Background Processing

**Task Manager** (`langfuse/_task_manager/`)
- Media upload processing with configurable workers
- Score ingestion with batching and retry logic
- Queue-based architecture with backpressure handling

## Key Development Patterns

### Multi-Project Safety
- Thread-safe singleton per public key prevents data leakage
- Project-scoped span processor filters ensure data isolation
- Disabled clients returned when project context is ambiguous

### OpenTelemetry Foundation
- All tracing built on OTel primitives for standards compliance
- Custom span processor for Langfuse-specific export
- Proper context propagation across async boundaries

### Testing Approach
- Comprehensive unit and integration tests
- API mocking through test wrappers
- Concurrency testing with ThreadPoolExecutor
- Proper resource cleanup in test teardown

## File Structure Notes

- `langfuse/_client/` - Core client implementation and tracing logic
- `langfuse/api/` - Auto-generated API client (excluded from linting)
- `langfuse/_utils/` - Utility functions for serialization, error handling, etc.
- `langfuse/_task_manager/` - Background processing workers
- `tests/` - Comprehensive test suite with integration tests
- `static/` - Test assets and sample files

## Configuration

### Environment Variables
Key environment variables for development:
- `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` - API credentials
- `LANGFUSE_HOST` - Langfuse instance URL
- `LANGFUSE_DEBUG` - Enable debug logging
- `LANGFUSE_TRACING_ENABLED` - Enable/disable tracing

### Test Configuration
- Tests require `.env` file based on `.env.template`
- Some E2E tests are skipped by default (remove decorators to run)
- CI uses different ruff config (`ci.ruff.toml`)

## Release Process

```bash
# Automated release
poetry run release

# Manual release steps
poetry version patch
poetry build
git commit -am "chore: release v{version}"
git tag v{version}
git push --tags
poetry publish
```

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
