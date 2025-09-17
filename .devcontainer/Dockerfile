# Dev container Dockerfile
FROM mcr.microsoft.com/devcontainers/universal:4-noble

# Install essential utilities including coreutils
RUN apt-get update && apt-get install -y \
    coreutils \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -
