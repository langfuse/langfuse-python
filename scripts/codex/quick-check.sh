#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

uv run --frozen ruff check .
uv run --frozen mypy langfuse --no-error-summary
uv run --frozen pytest -n auto --dist worksteal tests/unit
