#!/usr/bin/env bash
# Start the local operator dashboard.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
CONFIG_FILE="${DT_CONFIG:-$PROJECT_ROOT/config/system.env}"

set -a
source "$CONFIG_FILE"
set +a

export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

HOST_IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
echo "[dashboard] http://${HOST_IP:-127.0.0.1}:$DASHBOARD_PORT"
PORT="$DASHBOARD_PORT" exec python -m digital_twin.dashboard
