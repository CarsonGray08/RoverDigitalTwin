#!/usr/bin/env bash
# Start the operator dashboard and the Cloudflare tunnel that publishes it.
# Ctrl-C stops both.
#
# Set in .env (kept out of git): APP_USERNAME and APP_PASSWORD, the login;
# TUNNEL_CONFIG, a cloudflared config whose ingress sends the public hostname
# to the dashboard's port; and CLOUDFLARED if cloudflared is not on PATH.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"
set -a
source "${DT_CONFIG:-config/system.env}"     # DASHBOARD_PORT
[ -f .env ] && source .env
set +a
: "${APP_USERNAME:?set APP_USERNAME and APP_PASSWORD in .env: through the tunnel the dashboard is public}"
: "${APP_PASSWORD:?set APP_USERNAME and APP_PASSWORD in .env: through the tunnel the dashboard is public}"
: "${TUNNEL_CONFIG:?set TUNNEL_CONFIG in .env to the cloudflared config that routes to port $DASHBOARD_PORT}"
CLOUDFLARED="${CLOUDFLARED:-cloudflared}"
command -v "$CLOUDFLARED" > /dev/null || { echo "[remote] no cloudflared: set CLOUDFLARED in .env" >&2; exit 1; }

TUNNEL_LOG="${TUNNEL_LOG:-/tmp/rover_twin_tunnel.log}"
# The tunnel named in the config; a short grace period so Ctrl-C does not wait 30 s.
"$CLOUDFLARED" tunnel --no-autoupdate --grace-period 2s --config "$TUNNEL_CONFIG" run > "$TUNNEL_LOG" 2>&1 &
TUNNEL_PID=$!
trap 'kill "$TUNNEL_PID" 2> /dev/null || true; wait "$TUNNEL_PID" 2> /dev/null || true' EXIT

# The hostnames the config routes to the dashboard's port.
HOSTS="$(awk -v port=":$DASHBOARD_PORT" '$1 == "-" && $2 == "hostname:" { host = $3 }
  $1 == "service:" && index($2, port) && host { print "https://" host; host = "" }' "$TUNNEL_CONFIG")"
echo "[remote] ${HOSTS:-no hostname in $TUNNEL_CONFIG routes to port $DASHBOARD_PORT} (tunnel log: $TUNNEL_LOG)"

"$SCRIPT_DIR/dashboard.sh"
