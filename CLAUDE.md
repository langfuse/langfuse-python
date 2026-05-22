# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

Langfuse Python SDK. OpenTelemetry-based tracing with decorator-style instrumentation, an API client, and integrations for OpenAI and LangChain. Supports Python 3.10+.

## Architecture

```
langfuse/
├── _client/              # Core SDK
│   ├── client.py         # Main Langfuse class, OTel integration
│   ├── span.py           # Span types (LangfuseSpan, LangfuseGeneration, LangfuseEvent, ...)
│   ├── observe.py        # @observe decorator for automatic tracing
│   ├── get_client.py     # get_client() singleton, multi-project support
│   ├── propagation.py    # Context propagation for trace/span IDs
│   ├── resource_manager.py  # Resource lifecycle, shutdown/flushing
│   ├── span_processor.py # OTel span processor
│   └── datasets.py       # Dataset client for experiments
├── _task_manager/        # Background workers (media uploads, score ingestion)
├── _utils/               # Serialization, HTTP, error handling, prompt cache
├── api/                  # Generated Fern API client — DO NOT HAND-EDIT
├── openai.py             # OpenAI instrumentation
├── langchain/            # LangChain callback integration
├── experiment.py         # Experiment runner
├── batch_evaluation.py   # Batch evaluation framework
└── __init__.py           # Public exports
```

Key patterns:
- Singleton client via `LangfuseResourceManager._instances` keyed by `public_key`
- `@observe(name, as_type)` decorator wraps sync/async functions with automatic span creation
- Background workers handle media uploads and score ingestion asynchronously
- `langfuse/api/` is auto-generated from Fern — never hand-edit

## Commands

```bash
uv sync --locked                                    # Install dependencies
uv run --frozen pytest -n auto tests/unit           # Unit tests (parallel, no server)
uv run --frozen pytest -n 4 tests/e2e -m "not serial_e2e"   # E2E parallel
uv run --frozen pytest tests/e2e -m "serial_e2e"    # E2E serial-only
uv run --frozen pytest -n 4 tests/live_provider -m "live_provider"  # Live provider tests
uv run --frozen ruff check .                        # Lint
uv run --frozen ruff check --fix .                  # Lint with auto-fix
uv run --frozen ruff format .                       # Format
uv run --frozen ruff format --check .               # Format check
uv run --frozen mypy langfuse --no-error-summary    # Type check
bash scripts/codex/quick-check.sh                   # Broad local validation
uv build --no-sources                               # Build distribution
```

Run a single test: `uv run --frozen pytest tests/unit/test_file.py::test_function_name`

## Test Structure

- `tests/unit/` — deterministic local tests, no server required. Marker: `@pytest.mark.unit`
- `tests/e2e/` — requires running Langfuse server. Markers: `@pytest.mark.e2e`, `@pytest.mark.serial_e2e`
- `tests/live_provider/` — calls real LLM providers. Marker: `@pytest.mark.live_provider`
- `tests/support/` — shared e2e helpers
- `tests/conftest.py` — fixtures including `InMemorySpanExporter`, `langfuse_memory_client`, `get_span`, `find_spans`

Use `langfuse_memory_client` fixture for unit tests — it provides a pre-configured client with in-memory span export for assertions.

## Verification Matrix

| Change scope | Minimum verification |
|---|---|
| Core SDK (`langfuse/_client/`) | Unit tests + lint + mypy |
| Integration (`openai.py`, `langchain/`) | Unit tests + relevant e2e tests |
| Generated API (`langfuse/api/`) | Regenerate from Fern, never hand-edit |
| Cross-cutting | `bash scripts/codex/quick-check.sh` |

## Key Rules

- E2E tests: use bounded polling from `tests/support/`, never raw `sleep()`.
- Tests must be independent and parallel-safe.
- Shutdown/flush lifecycle is critical — watch for race conditions and dropped events.
- `LANGFUSE_BASE_URL` is the canonical env var; `LANGFUSE_HOST` is deprecated.
- Conventional Commits required for PR titles.
- Pre-commit hooks: ruff check, ruff format, mypy (run `uv run pre-commit install`).

## Environment Variables for Tests

```
LANGFUSE_BASE_URL="http://localhost:3000"
LANGFUSE_PUBLIC_KEY="pk-lf-1234567890"
LANGFUSE_SECRET_KEY="sk-lf-1234567890"
OPENAI_API_KEY=<for live_provider tests>
ANTHROPIC_API_KEY=<for live_provider tests>
```
