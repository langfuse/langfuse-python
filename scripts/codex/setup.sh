#!/usr/bin/env bash

set -euo pipefail

poetry config virtualenvs.create true --local
poetry config virtualenvs.in-project true --local
poetry install --all-extras

bash scripts/postinstall.sh
