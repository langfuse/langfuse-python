# Langfuse Python SDK Review Checklist

Use this checklist for `/review`, PR review, or self-review before handoff.

## Priorities

- Findings first: correctness bugs, regressions, security/privacy risks, performance issues with real impact, and missing tests for risky behavior.
- Keep line references tight and actionable.
- If no findings, say so explicitly and mention any residual risk or unrun verification.

## SDK Correctness

- Public SDK behavior should remain backwards compatible unless the PR is explicitly breaking.
- Prefer `LANGFUSE_BASE_URL`; `LANGFUSE_HOST` is deprecated and should only appear in compatibility paths or tests.
- Check shutdown, flushing, background task, and resource-manager changes for races, dropped events/scores/media, daemon-thread leaks, and hanging interpreter shutdown.
- OpenTelemetry changes should preserve context propagation, span parenting, exporter-local testability, and idempotent instrumentation setup.
- OpenAI and LangChain instrumentation should avoid brittle assertions on provider internals; prefer stable exporter-local behavior in unit tests.

## API And Generated Code

- Do not hand-edit `langfuse/api/`; regenerate it from the upstream Fern/OpenAPI source.
- Public API or serialization changes should include tests for request shape, response shape, and backwards-compatible aliases when relevant.
- Update README examples, `.env.template`, or generated reference docs when changed behavior would make them stale.

## Tests And CI

- Unit tests must not require a running Langfuse server.
- E2E tests should use bounded polling helpers from `tests/support/`, not raw `sleep()`.
- New e2e files must be named `tests/e2e/test_*.py` so mechanical CI sharding includes them.
- Use `serial_e2e` only for tests that are unsafe with shared-server concurrency.
- Live-provider tests should assert stable provider-facing behavior, not exact observation counts unless counts are the behavior under test.

## Python Style

- Exception messages should not inline f-string literals in `raise` statements; build the message in a variable first.
- Keep edits ASCII-only unless the file already uses Unicode or Unicode is clearly required.
- Keep changes scoped; avoid opportunistic refactors.
- Never commit secrets or credentials.
