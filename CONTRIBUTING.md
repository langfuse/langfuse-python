# Contributing

## Development

### Install dependencies

```bash
uv sync --locked
```

### Add pre-commit

```bash
uv run pre-commit install
```

### Quality checks

```bash
uv run --frozen ruff check .
uv run --frozen ruff format .
uv run --frozen mypy langfuse --no-error-summary
```

For a broad local confidence check, run:

```bash
bash scripts/codex/quick-check.sh
```

### Tests

Unit tests do not require a running Langfuse server:

```bash
uv run --frozen pytest -n auto --dist worksteal tests/unit
```

E2E tests require a running Langfuse server and environment variables based on `.env.template`:

```bash
uv run --frozen pytest -n 4 --dist worksteal tests/e2e -m "not serial_e2e"
uv run --frozen pytest tests/e2e -m "serial_e2e"
```

Live-provider tests make real provider calls and require provider API keys:

```bash
uv run --frozen pytest -n 4 --dist worksteal tests/live_provider -m "live_provider"
```

Run a specific test with:

```bash
uv run --frozen pytest tests/unit/test_resource_manager.py::test_pause_signals_score_consumer_shutdown
```

## Codex Cloud Setup

This repository includes repo-owned Codex setup so agents can start from a reproducible environment.

Recommended Codex UI configuration:

1. Create a Codex cloud environment for this repository.
2. Set the setup script to:

   ```bash
   bash scripts/codex/setup.sh
   ```

3. Set the maintenance script to:

   ```bash
   bash scripts/codex/maintenance.sh
   ```

4. Keep agent internet access disabled by default, or allow only the domains required for the task.
5. Add secrets and environment variables in the Codex UI instead of committing them.

## Pull Requests

PR titles and commit messages must follow Conventional Commits:

```text
type(scope): description
type: description
```

Common types include `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `build`, `ci`, `chore`, `revert`, and `security`.

Before opening a PR:

- Self-review the diff and use `code_review.md` for the repo-specific checklist.
- Keep changes focused and avoid unrelated refactors.
- Add or update tests for behavior changes.
- List the verification commands you ran in the PR description.

### Update OpenAPI spec

The generated API client in `langfuse/api/` must not be hand-edited. Regenerate it from the upstream Fern/OpenAPI source.

### Publish release

Releases are automated via GitHub Actions using PyPI Trusted Publishing (OIDC).

To create a release:

1. Go to [Actions > Release Python SDK](https://github.com/langfuse/langfuse-python/actions/workflows/release.yml)
2. Click "Run workflow"
3. Select the version bump type:
   - `patch` - Bug fixes (1.0.0 → 1.0.1)
   - `minor` - New features (1.0.0 → 1.1.0)
   - `major` - Breaking changes (1.0.0 → 2.0.0)
   - `prepatch`, `preminor`, or `premajor` - Pre-release versions (for example 1.0.0 → 1.0.1a1)
4. For pre-releases, select the type: `alpha`, `beta`, or `rc`
5. Click "Run workflow"

The workflow will automatically:
- Bump the version in `pyproject.toml`
- Build the package
- Publish to PyPI
- Create a git tag and GitHub release with auto-generated release notes

### SDK Reference

Note: The generated SDK reference is currently work in progress.

The SDK reference is generated via pdoc. The docs dependency group is installed on demand when you run the documentation commands.

To update the reference, run the following command:

```sh
uv run --group docs pdoc -o docs/ --docformat google --logo "https://langfuse.com/langfuse_logo.svg" langfuse
```

To run the reference locally, you can use the following command:

```sh
uv run --group docs pdoc --docformat google --logo "https://langfuse.com/langfuse_logo.svg" langfuse
```

## Credits

Thanks to the PostHog team for the awesome work on [posthog-python](https://github.com/PostHog/posthog-python). This project is based on it as it was the best starting point to build an async Python SDK.
