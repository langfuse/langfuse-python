#!/bin/bash

set -e

echo "🚀 Setting up Langfuse Python SDK development environment..."

# Install project dependencies including all extras
echo "📚 Installing project dependencies..."
poetry self add poetry-dotenv-plugin || true
poetry self add poetry-bumpversion || true
poetry install --all-extras

# Setup pre-commit hooks
echo "🪝 Setting up pre-commit hooks..."
poetry run pre-commit install

# Create a basic .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "📝 Creating .env file from template..."
    cp .env.template .env
fi

echo "✅ Development environment setup complete!"
echo ""
echo "🎯 Quick start commands:"
echo "  poetry run pytest -s -v --log-cli-level=INFO    # Run tests"
echo "  poetry run ruff format .                        # Format code"
echo "  poetry run ruff check .                         # Lint code"
echo "  poetry run mypy .                               # Type check"
echo "  poetry run pre-commit run --all-files           # Run pre-commit"
echo ""
