# Contributing

## Development

### Add Poetry plugins

```
poetry self add poetry-dotenv-plugin
```

### Install dependencies

```
poetry install --all-extras
```

### Add Pre-commit

```
poetry run pre-commit install
```

### Type Checking

To run type checking on the langfuse package, run:
```sh
poetry run mypy langfuse --no-error-summary
```

### Tests

#### Setup

- Add .env based on .env.template

#### Run

- Run all

  ```
  poetry run pytest -s -v --log-cli-level=INFO
  ```

- Run a specific test

  ```
  poetry run pytest -s -v --log-cli-level=INFO tests/test_core_sdk.py::test_flush
  ```

- E2E tests involving OpenAI and Serp API are usually skipped, remove skip decorators in [tests/test_langchain.py](tests/test_langchain.py) to run them.

### Update openapi spec

1. Generate Fern Python SDK in [langfuse](https://github.com/langfuse/langfuse) and copy the files generated in `generated/python` into the `langfuse/api` folder in this repo.
2. Execute the linter by running `poetry run ruff format .`
3. Rebuild and deploy the package to PyPi.

### Publish release

Releases are automated via GitHub Actions using PyPI Trusted Publishing (OIDC).

To create a release:

1. Go to [Actions > Release Python SDK](https://github.com/langfuse/langfuse-python/actions/workflows/release.yml)
2. Click "Run workflow"
3. Select the version bump type:
   - `patch` - Bug fixes (1.0.0 → 1.0.1)
   - `minor` - New features (1.0.0 → 1.1.0)
   - `major` - Breaking changes (1.0.0 → 2.0.0)
   - `prerelease` - Pre-release versions (1.0.0 → 1.0.0a1)
4. For pre-releases, select the type: `alpha`, `beta`, or `rc`
5. Click "Run workflow"

The workflow will automatically:
- Bump the version in `pyproject.toml` and `langfuse/version.py`
- Build the package
- Publish to PyPI
- Create a git tag and GitHub release with auto-generated release notes

### SDK Reference

Note: The generated SDK reference is currently work in progress.

The SDK reference is generated via pdoc. You need to have all extra dependencies installed to generate the reference.

```sh
poetry install --all-extras
```

To update the reference, run the following command:

```sh
poetry run pdoc -o docs/ --docformat google --logo "https://langfuse.com/langfuse_logo.svg" langfuse
```

To run the reference locally, you can use the following command:

```sh
poetry run pdoc --docformat google --logo "https://langfuse.com/langfuse_logo.svg" langfuse
```

## Credits

Thanks to the PostHog team for the awesome work on [posthog-python](https://github.com/PostHog/posthog-python). This project is based on it as it was the best starting point to build an async Python SDK.
