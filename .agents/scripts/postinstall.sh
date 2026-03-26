#!/usr/bin/env bash

set -euo pipefail

if [[ ! -f ".agents/scripts/sync-agent-shims.py" ]]; then
  echo "Skipping agent shim sync: .agents/scripts/sync-agent-shims.py is not present in this install context."
  exit 0
fi

python3 .agents/scripts/sync-agent-shims.py
python3 .agents/scripts/sync-agent-shims.py --check
