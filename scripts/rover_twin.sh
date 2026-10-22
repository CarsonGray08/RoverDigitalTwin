#!/usr/bin/env bash
# Start the mapper and the twin as one session sharing one run directory.
# Extra arguments go to the mapper, e.g. --replay runs/X/phone.scanlog --speed 1.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"               # paths in the config are relative to the repo
set -a
source "${DT_CONFIG:-config/system.env}"     # host paths: ISAAC_SIM_ROOT
set +a
: "${ISAAC_SIM_ROOT:?set ISAAC_SIM_ROOT in config/system.env}"
CONFIG="${RT_CONFIG:-config/rover_twin.toml}"
RUN_DIR="runs/$(date +%Y%m%d-%H%M%S)"
mkdir -p "$RUN_DIR"

# Isaac resolves the Newton kit's extension paths relative to its apps folder.
NEWTON_KIT="$ISAAC_SIM_ROOT/apps/isaacsim.exp.base.python.newton.kit"
cp "$SCRIPT_DIR/isaacsim.exp.base.python.newton.kit" "$NEWTON_KIT"
export PYTHONPATH="$PROJECT_ROOT/src"        # not ROS's: it breaks Isaac's Python

echo "[rover-twin] run: $RUN_DIR"
"${MAPPER_PYTHON:-python3}" -m rover_twin.mapper.main --config "$CONFIG" --run-dir "$RUN_DIR" "$@" &
MAPPER_PID=$!
"$ISAAC_SIM_ROOT/python.sh" -m rover_twin.twin.main --config "$CONFIG" --run-dir "$RUN_DIR" \
  --experience "$NEWTON_KIT" &
TWIN_PID=$!

signal_both() {
  kill "-$1" "$MAPPER_PID" 2>/dev/null || true
  # python.sh runs Isaac's python as a child and does not pass signals on.
  pkill "-$1" -P "$TWIN_PID" 2>/dev/null || kill "-$1" "$TWIN_PID" 2>/dev/null || true
}

cleanup() {
  trap - EXIT INT TERM
  signal_both INT
  # 20 s to save the run's outputs; then forced, so neither keeps its ports.
  for _ in $(seq 40); do
    kill -0 "$MAPPER_PID" 2>/dev/null || kill -0 "$TWIN_PID" 2>/dev/null || break
    sleep 0.5
  done
  signal_both TERM
  wait "$MAPPER_PID" "$TWIN_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM
wait -n "$MAPPER_PID" "$TWIN_PID"
