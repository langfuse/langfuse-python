# Contributing

## Development

### Add Poetry plugins

```
poetry self add poetry-dotenv-plugin
poetry self add poetry-bumpversion
```

### Install dependencies

```
poetry install --all-extras
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
  poetry run pytest -s -v --log-cli-level=INFO tests/test_core_sdk.py::test_flush
  ```
- E2E tests involving OpenAI and Serp API are usually skipped, remove skip decorators in [tests/test_langchain.py](tests/test_langchain.py) to run them.

### Update openapi spec

1. Generate Fern Python SDK in [langfuse](https://github.com/langfuse/langfuse) and copy the files generated in `generated/python` into the `langfuse/api` folder in this repo.
2. Execute the linter by running `poetry run ruff format .`
3. Rebuild and deploy the package to PyPi.

### Publish release

1. `poetry version patch`
   - `poetry version prepatch` for pre-release versions
2. `poetry install`
3. `poetry build`
4. `git commit -am "chore: release v{version}"`
5. `git push`
6. `git tag v{version}`
7. `git push --tags`
8. `poetry publish`
   - Create PyPi API token: https://pypi.org/manage/account/token/
   - Setup: `poetry config pypi-token.pypi your-api-token`
9. Create a release on GitHub with the changelog

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
