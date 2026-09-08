#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
CONFIG_FILE="${DT_CONFIG:-$PROJECT_ROOT/config/system.env}"
set -a
source "$CONFIG_FILE"
set +a

ISAAC_SIM_ROOT="${ISAAC_SIM_ROOT:-/home/ca152656/isaac-sim-6.0.0}"
SCENE_PATH="${SCENE_PATH:-$PROJECT_ROOT/assets/rover.usd}"
if [[ "$SCENE_PATH" != /* ]]; then
  SCENE_PATH="$PROJECT_ROOT/$SCENE_PATH"
fi
# Launch with the Newton-physics python app kit so the session runs Newton
# from startup (PhysX never loads), instead of switching engines after the
# stock PhysX kit has booted. The kit lives in this repo; it must be copied
# into the Isaac apps folder so its ${app}-relative extension paths resolve.
NEWTON_KIT_NAME="isaacsim.exp.base.python.newton.kit"
NEWTON_KIT="$ISAAC_SIM_ROOT/apps/$NEWTON_KIT_NAME"
cp "$SCRIPT_DIR/$NEWTON_KIT_NAME" "$NEWTON_KIT"

export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export DT_SCENE_PATH="$SCENE_PATH"
export DT_NEWTON_KIT="$NEWTON_KIT"

exec "$ISAAC_SIM_ROOT/python.sh" -m digital_twin.isaac_service
