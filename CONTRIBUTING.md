# Contributing

## Development

### Install dependencies

```
uv sync
```

### Add Pre-commit

```
uv run pre-commit install
```

### Type Checking

To run type checking on the langfuse package, run:
```sh
uv run mypy langfuse --no-error-summary
```

### Tests

#### Setup

- Add .env based on .env.template

#### Run

- Run all

  ```
  uv run --env-file .env pytest -s -v --log-cli-level=INFO
  ```

- Run a specific test

  ```
  uv run --env-file .env pytest -s -v --log-cli-level=INFO tests/test_core_sdk.py::test_flush
  ```

- E2E tests involving OpenAI and Serp API are usually skipped, remove skip decorators in [tests/test_langchain.py](tests/test_langchain.py) to run them.

### Update openapi spec

A PR with the changes is automatically created upon changing the Spec in the langfuse repo.

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
