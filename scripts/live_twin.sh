#!/usr/bin/env bash
# Start the supported live mapper and Isaac validator as one managed session.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
CONFIG_FILE="${DT_CONFIG:-$PROJECT_ROOT/config/system.env}"

set -a
source "$CONFIG_FILE"
set +a

export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

SESSION_STAMP="$(date +%Y%m%d-%H%M%S)-$$-${RANDOM}"
MAP_DIR="${MAP_ROOT}/session-${SESSION_STAMP}"
SESSION_ID="${MAP_DIR##*/}"
MOUNT_PATH="$PROJECT_ROOT/config/phone_mount.json"

mkdir -p "$MAP_ROOT"
ln -sfn "$MAP_DIR" "$MAP_ROOT/latest"

echo "[live-twin] session: $SESSION_ID"
echo "[live-twin] map: $MAP_DIR"
echo "[live-twin] phone: $PHONE_PORT  control: $CONTROL_PORT  validator: $VALIDATOR_PORT"

export DT_MAP_DIR="$MAP_DIR"
export DT_SESSION_ID="$SESSION_ID"
export DT_MOUNT_PATH="$MOUNT_PATH"

python -m digital_twin.mapper &
MAPPER_PID=$!

"$SCRIPT_DIR/isaac_validator.sh" &
VALIDATOR_PID=$!

cleanup() {
  trap - EXIT INT TERM
  kill "$MAPPER_PID" "$VALIDATOR_PID" 2>/dev/null || true
  wait "$MAPPER_PID" "$VALIDATOR_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# Ending either process ends the session and cleans up its partner.
wait -n "$MAPPER_PID" "$VALIDATOR_PID"
