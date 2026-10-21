#!/usr/bin/env bash
# Start the operator dashboard; mapper and twin are started from it.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"
set -a
source "${DT_CONFIG:-config/system.env}"     # ROVER_IP, DASHBOARD_PORT, MAPPER_PYTHON
set +a
export PYTHONPATH="$PROJECT_ROOT/src"        # not ROS's: the twin job runs Isaac's Python, which it breaks

HOST_IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
echo "[dashboard] http://${HOST_IP:-127.0.0.1}:$DASHBOARD_PORT"
exec "${MAPPER_PYTHON:-python3}" -m rover_twin.ops.dashboard
