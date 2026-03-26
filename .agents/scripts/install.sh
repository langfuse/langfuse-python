#!/usr/bin/env bash

set -euo pipefail

poetry install "$@"

bash .agents/scripts/postinstall.sh
