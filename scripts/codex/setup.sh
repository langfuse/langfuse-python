#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

if ! command -v uv >/dev/null 2>&1; then
  python3 -m pip install --user "uv==0.11.2"
  export PATH="$HOME/.local/bin:$PATH"
fi

uv sync --locked
uv run --frozen python --version
