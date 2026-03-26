#!/usr/bin/env bash

set -euo pipefail

if [[ ! -f "scripts/agents/sync-agent-shims.py" ]]; then
  echo "Skipping agent shim sync: scripts/agents/sync-agent-shims.py is not present in this install context."
  exit 0
fi

python3 scripts/agents/sync-agent-shims.py
python3 scripts/agents/sync-agent-shims.py --check
