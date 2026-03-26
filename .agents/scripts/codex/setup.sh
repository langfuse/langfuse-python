#!/usr/bin/env bash

set -euo pipefail

poetry config virtualenvs.create true --local
poetry config virtualenvs.in-project true --local

bash .agents/scripts/install.sh --all-extras
