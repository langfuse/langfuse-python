# Codex Guidelines for Langfuse Python

This is the canonical root agent guide for the repo. The root `AGENTS.md`
should remain only as a discovery symlink so tools that require that filename
continue to work while `.agents/` stays the source of truth.

Langfuse Python SDK guidance for fast, safe code changes.

## Maintenance Contract

- `AGENTS.md` is a living document.
- Update this file in the same PR when repo-level architecture, workflows,
  verification requirements, release processes, or agent setup conventions
  materially change.
- Update this file when user feedback adds a durable repo-level instruction that
  future agents should follow.
- Keep root guidance concise, specific, and easy to verify. If repo-wide
  guidance grows large or becomes task-specific, move that detail into shared
  skills or future nested `AGENTS.md` files closer to the relevant code.
- If no durable guidance changed, do not edit AGENTS files.

## Project Overview

This repository contains the Langfuse Python SDK, a client library for
accessing the Langfuse observability platform. The SDK integrates with
OpenTelemetry for tracing, provides automatic instrumentation for popular LLM
frameworks, and exposes a generated API client for the Langfuse platform.

## Project Structure

```text
langfuse-python/
├─ langfuse/_client/         # Core SDK implementation built on OpenTelemetry
├─ langfuse/api/             # Generated Fern API client (do not hand-edit)
├─ langfuse/_task_manager/   # Background upload and ingestion helpers
├─ langfuse/langchain/       # LangChain integration
├─ tests/                    # Test suite
├─ static/                   # Test fixtures and sample content
└─ .agents/                  # Canonical shared agent instructions and config
```

High-signal entry points:

- `langfuse/_client/client.py`: core Langfuse client with OTel integration
- `langfuse/_client/span.py`: observation/span abstractions
- `langfuse/_client/observe.py`: decorator-based instrumentation
- `langfuse/openai.py`: OpenAI instrumentation helpers
- `langfuse/langchain/CallbackHandler.py`: LangChain integration
- `langfuse/api/`: generated API surface copied from the main Langfuse repo

## Instruction Design

- Root `AGENTS.md` should cover durable repo-wide expectations only: setup,
  verification, architecture, security, generated files, and handoff rules.
- Prefer concrete, testable instructions over vague phrasing. Name the exact
  command, path, module, or condition whenever possible.
- Keep stable tone/role guidance separate from task-specific examples. For
  complex prompts or reusable workflows, place examples in skills or referenced
  docs instead of bloating the root guide.
- Add nearby nested guidance only when a subdirectory truly needs different
  rules. Put the override as close as possible to the specialized code.
- Use shared skills for recurring task-specific workflows that should not be
  loaded into context on every task.

## Build, Test, and Development Commands

- Agent environment bootstrap: `bash .agents/scripts/codex/setup.sh`
- Install dependencies: `bash .agents/scripts/install.sh --all-extras`
- Sync generated agent shims: `python3 .agents/scripts/sync-agent-shims.py`
- Verify generated agent shims: `python3 .agents/scripts/sync-agent-shims.py --check`
- Install pre-commit hooks: `poetry run pre-commit install`
- Run all tests: `poetry run pytest -s -v --log-cli-level=INFO`
- Run tests in parallel: `poetry run pytest -s -v --log-cli-level=INFO -n auto`
- Run one test: `poetry run pytest -s -v --log-cli-level=INFO tests/test_core_sdk.py::test_flush`
- Format code: `poetry run ruff format .`
- Lint code: `poetry run ruff check .`
- Type-check: `poetry run mypy langfuse --no-error-summary`
- Run pre-commit across the repo: `poetry run pre-commit run --all-files`
- Build package: `poetry build`
- Generate docs: `poetry run pdoc -o docs/ --docformat google --logo "https://langfuse.com/langfuse_logo.svg" langfuse`

Minimum verification matrix:

| Change scope | Minimum verification |
| --- | --- |
| `langfuse/_client/**` | `poetry run ruff check .` + `poetry run mypy langfuse --no-error-summary` + targeted pytest coverage |
| `langfuse/api/**` | verify source update path from main repo + `poetry run ruff format .` + targeted API tests |
| Integration modules (`langfuse/openai.py`, `langfuse/langchain/**`) | targeted tests for the touched integration + lint + latest official provider docs review if behavior or API usage changed |
| Test-only changes | targeted pytest coverage for the updated tests |
| Agent setup files (`.agents/**`) | `python3 .agents/scripts/sync-agent-shims.py` + `python3 .agents/scripts/sync-agent-shims.py --check` |

CI notes:

- Linting runs via `astral-sh/ruff-action`.
- Type checking runs on Python 3.13 with Poetry, `.venv` caching, and the agent
  shim sync/check step.
- The main test matrix runs on Python 3.10 through 3.14.
- Integration CI clones the main `langfuse/langfuse` repo, boots Dockerized
  services, seeds the server with `pnpm`, and then runs this SDK's pytest suite
  against that local server.
- If a change plausibly depends on server behavior, call out whether it was only
  covered by unit tests locally and whether full CI is the real end-to-end
  verification path.

## Architecture

### Core Components

- `langfuse/_client/`: main SDK implementation built on OpenTelemetry
  - `client.py`: core Langfuse client
  - `span.py`: span, generation, and event classes
  - `observe.py`: decorator for automatic instrumentation
  - `datasets.py`: dataset management functionality
- `langfuse/api/`: auto-generated Fern API client
- `langfuse/_task_manager/`: background processing for uploads and ingestion
- `langfuse/openai.py`: OpenAI instrumentation
- `langfuse/langchain/`: LangChain integration

### Key Design Patterns

- The SDK is built on OpenTelemetry for observability.
- Spans are the core tracing primitive.
- Attributes carry trace metadata. See `LangfuseOtelSpanAttributes`.
- The client batches work and flushes asynchronously to the Langfuse API.

## Generated Files

- `langfuse/api/**` is generated from the main Langfuse repo. Do not edit it by
  hand unless the task is explicitly about generated client updates.
- `docs/` output from `pdoc` is generated. Regenerate it instead of editing
  rendered output directly.
- Agent/tool shims at `.mcp.json`, `.claude/settings.json`, `.claude/skills/*`,
  `.cursor/mcp.json`, `.cursor/environment.json`, `.vscode/mcp.json`,
  `.codex/config.toml`, and `.codex/environments/environment.toml` are local
  generated artifacts. Update `.agents/config.json` or `.agents/skills/**`
  instead of editing them by hand.
- `AGENTS.md` and `CLAUDE.md` at the repo root are compatibility symlinks. Edit
  `.agents/AGENTS.md`, not the symlink target path directly.

## Configuration

Environment variables are defined in
`langfuse/_client/environment_variables.py`.

Common ones:

- `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY`: API credentials
- `LANGFUSE_HOST`: API endpoint, defaults to `https://cloud.langfuse.com`
- `LANGFUSE_DEBUG`: enable debug logging
- `LANGFUSE_TRACING_ENABLED`: enable or disable tracing
- `LANGFUSE_SAMPLE_RATE`: sampling rate for traces

Security/config notes:

- Keep credentials and machine-specific secrets in environment variables or
  local untracked files, never in committed agent config.
- The shared Claude settings intentionally deny reading `./.env` and
  `./.env.*`, and they do not auto-approve Bash commands by default. If a task
  genuinely requires inspecting local env overrides or shell access, get
  explicit user approval first instead of weakening the default policy.
- For authenticated MCP servers or provider-specific config additions, prefer
  secret injection via environment variables rather than committed inline
  tokens.

## Testing Guidelines

- Keep tests independent and parallel-safe.
- Do not weaken or delete meaningful assertions just to make tests pass.
- When fixing a bug, write or update the regression test first when feasible.
- E2E tests involving external APIs are often skipped in CI. Document when
  manual coverage is still needed.
- Use `respx` and `pytest-httpserver` for HTTP mocking when possible.
- Prefer the narrowest useful test invocation first, then widen coverage when a
  change touches shared tracing, batching, or provider integrations.

## API Generation

The `langfuse/api/` directory is generated from the Langfuse OpenAPI
specification via Fern.

Update flow:

1. Generate the Python SDK in the main `langfuse/langfuse` repo.
2. Copy the generated files from `generated/python` into `langfuse/api/`.
3. Run `poetry run ruff format .`.
4. Run targeted verification for any touched endpoints or types.

## Release Guidelines

- Releases are automated via GitHub Actions.
- The release workflow updates `pyproject.toml` and `langfuse/version.py`,
  builds the package, publishes to PyPI, and creates a GitHub release.
- Do not change release/versioning flow without updating this file and
  `CONTRIBUTING.md`.

## Agent-specific Notes

- `.agents/AGENTS.md` is the canonical root guide.
- Root `AGENTS.md` is a symlink to `.agents/AGENTS.md`.
- Root `CLAUDE.md` is a compatibility symlink to `AGENTS.md`.
- Shared agent/tool config lives in `.agents/config.json`.
- Shared agent setup documentation lives in `.agents/README.md`.
- Shared skills live under `.agents/skills/`.
- `.agents/scripts/` is the home for repo-owned agent bootstrap and sync
  tooling.
- `python3 .agents/scripts/sync-agent-shims.py` regenerates tool-specific config
  shims for Claude, Cursor, VS Code, Codex, and shared MCP discovery files.
- Tool-specific directories such as `.claude/`, `.cursor/`, `.codex/`, and
  `.vscode/` remain because those tools discover project settings from fixed
  paths.
- Cursor discovery should continue to work through the generated
  `.cursor/environment.json` and `.cursor/mcp.json` shims plus the root
  `AGENTS.md` symlink. Do not hand-edit those generated files.
- This file should stay concise. Anthropic recommends keeping persistent project
  memory under roughly 200 lines, and both Anthropic and OpenAI guidance favor
  specific, well-structured instructions over long prose.
- If future `.cursor/rules/*.mdc` files are added, keep them as thin wrappers
  around shared `AGENTS.md` guidance or shared skills instead of making them the
  only source of durable repo guidance.
- Shared skill index: [`skills/README.md`](skills/README.md)
- When changing OpenAI or Anthropic integrations, prompts, or documented usage:
  check the latest official provider docs first, keep prompts simple and direct,
  preserve clear separation between stable instructions and task-specific
  examples, and mention any provider-facing verification you did not run.

Official references to start from:

- OpenAI AGENTS guide: <https://developers.openai.com/codex/guides/agents-md>
- OpenAI prompting guide: <https://developers.openai.com/api/docs/guides/prompting>
- OpenAI reasoning best practices: <https://developers.openai.com/api/docs/guides/reasoning-best-practices>
- Anthropic Claude Code memory guide: <https://docs.anthropic.com/en/docs/claude-code/memory>
- Anthropic Claude Code MCP guide: <https://docs.anthropic.com/en/docs/claude-code/mcp>
- Anthropic prompting best practices: <https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices>

## Git Notes

- Do not use destructive git commands such as `reset --hard` unless explicitly
  requested.
- Do not revert unrelated working tree changes.
- Keep commits focused and atomic.

## Python Code Rules

- Exception messages must not inline f-string literals directly in the `raise`.
  Assign the string to a variable first if formatting is required.
