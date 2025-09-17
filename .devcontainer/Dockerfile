# Dev container Dockerfile
FROM mcr.microsoft.com/devcontainers/python:3.12-bullseye

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -

# Install Poetry Plugins
RUN poetry self add poetry-dotenv-plugin || true && poetry self add poetry-bumpversion || true
