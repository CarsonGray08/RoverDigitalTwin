#!/usr/bin/env python3
"""Operator dashboard for the live digital twin.

Launch and stop the twin, run a baseline scan or validated drive, adjust mapper
settings, and view the phone camera and rover pose. Uses Python's HTTP server
and binds to ``0.0.0.0`` by default for access from the lab network.

Run ``./scripts/dashboard.sh`` and open the printed URL.
"""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urlparse
from urllib.request import Request, urlopen

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from digital_twin.live_pose import read_pose  # noqa: E402
from digital_twin.executor import DEFAULT_COMMAND_MODE  # noqa: E402
from digital_twin.live_twin_session import read_session  # noqa: E402
from digital_twin.mesh_tiles import load_manifest  # noqa: E402

DIST_DIR = Path(os.environ.get("DASHBOARD_DIST_DIR", REPO / "dashboard")).resolve()
STATE_DIR = Path(os.environ.get("DASHBOARD_STATE_DIR", "/tmp/dt_dashboard"))
LOG_DIR = STATE_DIR / "logs"
PARAMS_PATH = STATE_DIR / "params.json"

HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8090"))
APP_USERNAME = os.environ.get("APP_USERNAME")
APP_PASSWORD = os.environ.get("APP_PASSWORD")
MAX_BODY_BYTES = 1_048_576
PROXY_TIMEOUT_S = 5.0

# Validated drives are gated by the twin, but bound the goal anyway so a typo
# can't send the rover across the room in one request.
MAX_GOAL_DISTANCE_M = 3.0

# Only settings that an operator may need to change in the field are exposed.
# Tracking, fill, session, and Newton settings are fixed by the supported
# mapper/validator profiles rather than duplicated as dashboard switches.
PARAM_SPEC: dict[str, dict[str, Any]] = {
    "map_root": {
        "env": "MAP_ROOT", "type": "str", "default": "/tmp/nvblox_map",
        "group": "Connection", "label": "Map root directory",
        "help": "Each launch writes an isolated session-* subdir under this root.",
    },
    "phone_port": {
        "env": "PHONE_PORT", "type": "int", "default": 9099,
        "group": "Connection", "label": "Phone stream port",
        "help": "TCP port the iPhone app connects to.",
    },
    "control_port": {
        "env": "CONTROL_PORT", "type": "int", "default": 9100,
        "group": "Connection", "label": "Mapper control port",
        "help": "Loopback control plane (scan start/stop, camera preview).",
    },
    "validator_port": {
        "env": "VALIDATOR_PORT", "type": "int", "default": 8765,
        "group": "Connection", "label": "Validator port",
        "help": "Isaac validate_plan / observation endpoint.",
    },
    "command_mode": {
        "env": "COMMAND_MODE", "type": "choice", "default": DEFAULT_COMMAND_MODE,
        "choices": ["continual", "step"],
        "group": "Rover", "label": "Rover command mode",
        "help": "Commands go through the rover HTTP controller. Continual is "
                "the default and stops from live odometry; step is bounded.",
    },
    "voxel_size_m": {
        "env": "VOXEL_SIZE_M", "type": "float", "default": 0.02,
        "group": "Mapping", "label": "Voxel size (m)",
        "help": "0.02 is the tuned bench default. 0.01 = ~8x memory/compute.",
    },
    "min_depth_confidence": {
        "env": "MIN_DEPTH_CONFIDENCE", "type": "choice", "choices": [0, 1, 2], "default": 2,
        "group": "Mapping", "label": "Min depth confidence",
        "help": "Drop ARKit depth below this (2 = high-confidence only).",
    },
    "tile_size_m": {
        "env": "TILE_SIZE_M", "type": "float", "default": 4.0,
        "group": "Mapping", "label": "Tile size (m)",
    },
    "map_update_interval_s": {
        "env": "MAP_UPDATE_INTERVAL_S", "type": "float", "default": 0.5,
        "group": "Mapping", "label": "Mesh update interval (s)",
    },
    "stabilize_floor": {
        "env": "STABILIZE_FLOOR", "type": "bool", "default": True,
        "group": "Scan behavior", "label": "Stabilize floor",
    },
}


def _coerce_param(name: str, value: Any) -> Any:
    spec = PARAM_SPEC[name]
    kind = spec["type"]
    if kind == "bool":
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "on")
        return bool(value)
    if kind == "int":
        return int(value)
    if kind == "float":
        return float(value)
    if kind == "choice":
        choices = spec["choices"]
        coerced = type(choices[0])(value) if not isinstance(value, type(choices[0])) else value
        if coerced not in choices:
            raise ValueError(f"{name} must be one of {choices}")
        return coerced
    return str(value)


def load_params() -> dict[str, Any]:
    merged = {name: spec["default"] for name, spec in PARAM_SPEC.items()}
    if PARAMS_PATH.exists():
        try:
            saved = json.loads(PARAMS_PATH.read_text())
        except (OSError, ValueError):
            saved = {}
        for key, value in saved.items():
            if key in PARAM_SPEC:
                try:
                    merged[key] = _coerce_param(key, value)
                except (TypeError, ValueError):
                    pass
    return merged


def save_params(overrides: dict[str, Any]) -> dict[str, Any]:
    current = load_params()
    for key, value in overrides.items():
        if key not in PARAM_SPEC:
            raise ValueError(f"unknown param {key!r}")
        current[key] = _coerce_param(key, value)
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = PARAMS_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(current, indent=2))
    os.replace(tmp, PARAMS_PATH)
    return current


def env_overrides(params: dict[str, Any]) -> dict[str, str]:
    overrides = {}
    for name, spec in PARAM_SPEC.items():
        value = params.get(name, spec["default"])
        if spec["type"] == "bool":
            overrides[spec["env"]] = "1" if value else "0"
        else:
            overrides[spec["env"]] = str(value)
    return overrides


@dataclass
class ManagedProcess:
    """One long-running child (the twin launcher, a 360 sweep) plus its log tail.

    ``start_new_session=True`` + signaling the whole process group mirrors
    live_twin.sh's own trap: that script forks the mapper and the Isaac
    validator as its own children, so signaling only the shell leaves them
    running. Stop escalates SIGINT -> SIGTERM -> SIGKILL: SIGINT first because
    the motion jobs perform cleanup on it -- digital_twin.scan publishes
    STOP to an armed rover on KeyboardInterrupt, and the mapper finalizes the
    map -- exactly like the operator's Ctrl-C in a terminal.
    """

    name: str
    popen: subprocess.Popen | None = None
    started_at: float | None = None
    returncode: int | None = None
    log_lines: deque = field(default_factory=lambda: deque(maxlen=400))
    lock: threading.Lock = field(default_factory=threading.Lock)

    def _running_locked(self) -> bool:
        return self.popen is not None and self.popen.poll() is None

    def is_running(self) -> bool:
        with self.lock:
            return self._running_locked()

    def start(self, cmd: list[str], env: dict[str, str]) -> None:
        with self.lock:
            if self._running_locked():
                raise RuntimeError(f"{self.name} is already running")
            LOG_DIR.mkdir(parents=True, exist_ok=True)
            self.log_lines.clear()
            self.returncode = None
            # Python fully block-buffers stdout when it isn't a tty (i.e. always,
            # once captured through this pipe), so short-lived scripts like
            # digital_twin.scan -- few prints, most without flush=True -- would
            # show nothing in the log tail until they exited. Force every child
            # (including whatever python live_twin.sh's bash forks) unbuffered.
            env = dict(env)
            env["PYTHONUNBUFFERED"] = "1"
            self.popen = subprocess.Popen(
                cmd,
                cwd=str(REPO),
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            self.started_at = time.time()
            popen = self.popen
        threading.Thread(target=self._pump, args=(popen,), daemon=True).start()

    def _pump(self, popen: subprocess.Popen) -> None:
        log_path = LOG_DIR / f"{self.name}.log"
        with open(log_path, "ab") as log_file:
            assert popen.stdout is not None
            for raw_line in popen.stdout:
                log_file.write(raw_line)
                with self.lock:
                    self.log_lines.append(raw_line.decode("utf-8", "replace").rstrip("\n"))
        popen.wait()
        with self.lock:
            self.returncode = popen.returncode

    def stop(self, sigint_grace_s: float = 8.0, sigterm_grace_s: float = 8.0) -> None:
        with self.lock:
            popen = self.popen
        if popen is None or popen.poll() is not None:
            return
        escalation = (
            (signal.SIGINT, sigint_grace_s),
            (signal.SIGTERM, sigterm_grace_s),
            (signal.SIGKILL, 5.0),
        )
        for sig, grace_s in escalation:
            try:
                os.killpg(os.getpgid(popen.pid), sig)
            except ProcessLookupError:
                return
            try:
                popen.wait(timeout=grace_s)
                return
            except subprocess.TimeoutExpired:
                continue

    def status(self) -> dict[str, Any]:
        with self.lock:
            running = self._running_locked()
            pid = self.popen.pid if (self.popen is not None and running) else None
            return {
                "name": self.name,
                "running": running,
                "pid": pid,
                "returncode": self.returncode,
                "started_at": self.started_at,
                "uptime_s": (time.time() - self.started_at) if running and self.started_at else None,
                "log_tail": list(self.log_lines)[-120:],
            }


JOBS: dict[str, ManagedProcess] = {
    "twin": ManagedProcess("twin"),
    "scan360": ManagedProcess("scan360"),
    "drive_goal": ManagedProcess("drive_goal"),
}


def build_status() -> dict[str, Any]:
    params = load_params()
    map_root = Path(params["map_root"])
    latest = map_root / "latest"
    session_dir: Path | None = None
    if latest.exists():
        try:
            session_dir = latest.resolve()
        except OSError:
            session_dir = None

    pose = read_pose(session_dir) if session_dir is not None else None
    session = read_session(session_dir) if session_dir is not None else None
    manifest = load_manifest(session_dir) if session_dir is not None else {}

    return {
        "jobs": {name: job.status() for name, job in JOBS.items()},
        "params": params,
        "map_dir": str(session_dir) if session_dir is not None else None,
        "session_id": session.get("session_id") if session else None,
        "tracking_normal": bool(session.get("tracking_normal")) if session else False,
        "pose": pose,
        "manifest_rev": manifest.get("rev") if manifest else None,
        "tile_count": len(manifest.get("tiles", [])) if manifest else None,
        "server_time_s": time.time(),
    }


def fetch_observation(validator_port: int) -> tuple[int, dict[str, Any]]:
    request = Request(f"http://127.0.0.1:{validator_port}/observation", method="GET")
    try:
        with urlopen(request, timeout=PROXY_TIMEOUT_S) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        return exc.code, {"error": exc.reason}
    except (URLError, TimeoutError, OSError) as exc:
        return 502, {"error": f"validator unreachable: {exc}"}
    except json.JSONDecodeError as exc:
        return 502, {"error": f"invalid validator response: {exc}"}


def fetch_camera_frame(control_port: int) -> tuple[int, bytes, str, dict[str, str]]:
    """Proxy the mapper's phone camera preview (``GET /camera/latest`` in
    nvblox_live_map.py). Unlike ``/observation``, this is the real phone
    image -- a color keyframe when one is fresh, a depth-preview PNG between
    keyframes -- and comes from the mapper's control port (not the
    validator's). Forwards the mapper's X-Frame-* headers so the UI can show
    frame age and color/depth source."""
    request = Request(f"http://127.0.0.1:{control_port}/camera/latest", method="GET")
    try:
        with urlopen(request, timeout=PROXY_TIMEOUT_S) as response:
            headers = {
                key: response.headers[key]
                for key in ("X-Frame-Timestamp", "X-Frame-Source")
                if response.headers.get(key)
            }
            content_type = response.headers.get("Content-Type", "image/jpeg")
            return response.status, response.read(), content_type, headers
    except HTTPError as exc:
        body = exc.read()
        return exc.code, body, exc.headers.get("Content-Type", "application/json"), {}
    except (URLError, TimeoutError, OSError) as exc:
        body = json.dumps({"error": f"mapper unreachable: {exc}"}).encode("utf-8")
        return 502, body, "application/json", {}


def post_mapper_control(control_port: int, path: str) -> None:
    """POST a control action to the mapper (scan start/stop, integration pause).

    Raises ``RuntimeError`` with an operator-readable message; the mapper's 409
    on ``/scan/start`` specifically means no phone is connected to it."""
    request = Request(
        f"http://127.0.0.1:{control_port}/{path.lstrip('/')}", data=b"", method="POST")
    try:
        with urlopen(request, timeout=PROXY_TIMEOUT_S) as response:
            if response.status != 200:
                raise RuntimeError(f"mapper /{path} returned HTTP {response.status}")
    except HTTPError as exc:
        if exc.code == 409:
            raise RuntimeError(
                "the phone is not connected to the mapper -- open the LiveScanner "
                "app and tap Connect, then retry") from exc
        raise RuntimeError(f"mapper /{path} failed: HTTP {exc.code}") from exc
    except (URLError, TimeoutError, OSError) as exc:
        raise RuntimeError(
            f"mapper unreachable ({exc}) -- is the live twin running?") from exc


class DashboardHandler(BaseHTTPRequestHandler):
    server_version = "DigitalTwinDashboard/1.0"

    def log_message(self, fmt: str, *args: object) -> None:
        sys.stderr.write("%s - - [%s] %s\n" % (self.address_string(), self.log_date_time_string(), fmt % args))

    # -- auth / body helpers -------------------------------------------------
    def authorized(self) -> bool:
        path = urlparse(self.path).path
        if path == "/healthz" or not (APP_USERNAME and APP_PASSWORD):
            return True
        expected = "Basic " + base64.b64encode(f"{APP_USERNAME}:{APP_PASSWORD}".encode()).decode()
        if self.headers.get("Authorization") == expected:
            return True
        self.send_response(401)
        self.send_header("WWW-Authenticate", 'Basic realm="Digital Twin Dashboard"')
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.end_headers()
        self.wfile.write(b"Authentication required\n")
        return False

    def read_json_body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0") or "0")
        if length <= 0:
            return {}
        if length > MAX_BODY_BYTES:
            raise ValueError("request body is too large")
        raw = self.rfile.read(length)
        body = json.loads(raw.decode("utf-8")) if raw.strip() else {}
        if not isinstance(body, dict):
            raise ValueError("JSON body must be an object")
        return body

    def send_json(self, payload: Any, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    # -- routing -------------------------------------------------------------
    def do_GET(self) -> None:
        if not self.authorized():
            return
        path = urlparse(self.path).path
        if path == "/healthz":
            self.send_json({"ok": True})
            return
        if path == "/api/status":
            self.send_json(build_status())
            return
        if path == "/api/observation":
            params = load_params()
            status, payload = fetch_observation(int(params["validator_port"]))
            self.send_json(payload, status=status)
            return
        if path == "/api/camera":
            params = load_params()
            status, body, content_type, frame_headers = fetch_camera_frame(
                int(params["control_port"]))
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            for key, value in frame_headers.items():
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(body)
            return
        if path == "/api/params":
            self.send_json(load_params())
            return
        if path == "/api/param_spec":
            self.send_json(PARAM_SPEC)
            return
        self.serve_static()

    def do_POST(self) -> None:
        if not self.authorized():
            return
        path = urlparse(self.path).path
        handlers = {
            "/api/jobs/twin/start": self.handle_twin_start,
            "/api/jobs/twin/stop": self.handle_twin_stop,
            "/api/jobs/scan360/start": self.handle_scan360_start,
            "/api/jobs/scan360/stop": self.handle_scan360_stop,
            "/api/jobs/drive_goal/start": self.handle_drive_goal_start,
            "/api/jobs/drive_goal/stop": self.handle_drive_goal_stop,
            "/api/tracking/start": self.handle_tracking_start,
            "/api/tracking/stop": self.handle_tracking_stop,
            "/api/params": self.handle_params_post,
        }
        handler = handlers.get(path)
        if handler is None:
            self.send_json({"error": "not_found"}, status=404)
            return
        try:
            handler()
        except ValueError as exc:
            self.send_json({"error": str(exc)}, status=400)
        except RuntimeError as exc:
            self.send_json({"error": str(exc)}, status=409)

    # -- POST handlers -------------------------------------------------------
    def handle_twin_start(self) -> None:
        body = self.read_json_body()
        overrides = body.get("params", {})
        if not isinstance(overrides, dict):
            raise ValueError("params must be an object")
        params = save_params(overrides) if overrides else load_params()
        env = os.environ.copy()
        env.update(env_overrides(params))
        JOBS["twin"].start(["bash", str(REPO / "scripts" / "live_twin.sh")], env)
        self.send_json({"status": "started"}, status=202)

    def handle_twin_stop(self) -> None:
        # Stop escalation can take many seconds (Isaac shutdown); don't hold
        # the HTTP request open for it -- the status poll shows the result.
        threading.Thread(target=JOBS["twin"].stop, daemon=True).start()
        self.send_json({"status": "stopping"}, status=202)

    def handle_scan360_start(self) -> None:
        body = self.read_json_body()
        arm = bool(body.get("arm", False))
        rover_ip = str(body.get("rover_ip") or "").strip()
        if arm and not rover_ip:
            raise ValueError("arm requires rover_ip")
        params = load_params()
        cmd = [
            sys.executable,
            "-m", "digital_twin.scan",
            "--yes",
            "--control-url", f"http://127.0.0.1:{params['control_port']}",
            "--turn-command-mode", params["command_mode"],
        ]
        degrees = body.get("degrees")
        if degrees is not None:
            cmd += ["--degrees", str(float(degrees))]
        if body.get("track_after", True):
            # Keep the twin following the rover once the map is done (scan
            # resumed, integration paused) instead of freezing its pose.
            cmd += ["--track-after"]
        if arm:
            cmd += ["--telemetry-stop"]
        if arm:
            cmd += ["--rover-ip", rover_ip, "--arm"]
        JOBS["scan360"].start(cmd, os.environ.copy())
        self.send_json({"status": "started"}, status=202)

    def handle_scan360_stop(self) -> None:
        threading.Thread(target=JOBS["scan360"].stop, daemon=True).start()
        self.send_json({"status": "stopping"}, status=202)

    def handle_drive_goal_start(self) -> None:
        """Twin-validated drive: every command is rehearsed
        against the live twin and only approved commands reach the rover. It
        resumes the scan itself (localization-only), so it also restores
        tracking. ``--yes`` because a dashboard job cannot answer the
        per-command confirm prompt; the button press is the confirmation."""
        body = self.read_json_body()
        arm = bool(body.get("arm", False))
        rover_ip = str(body.get("rover_ip") or "").strip()
        if arm and not rover_ip:
            raise ValueError("arm requires rover_ip")
        distance_m = float(body.get("distance_m", 1.0))
        if not 0.0 < distance_m <= MAX_GOAL_DISTANCE_M:
            raise ValueError(f"distance_m must be in (0, {MAX_GOAL_DISTANCE_M}]")
        params = load_params()
        cmd = [
            sys.executable,
            "-m", "digital_twin.drive",
            "--yes",
            "--distance-m", str(distance_m),
            "--map-dir", str(params["map_root"]),
            "--control-url", f"http://127.0.0.1:{params['control_port']}",
            "--validator-url", f"http://127.0.0.1:{params['validator_port']}/validate_plan",
            "--command-mode", params["command_mode"],
        ]
        if arm:
            cmd += ["--telemetry-stop"]
        if arm:
            cmd += ["--rover-ip", rover_ip, "--arm"]
        JOBS["drive_goal"].start(cmd, os.environ.copy())
        self.send_json({"status": "started", "armed": arm}, status=202)

    def handle_drive_goal_stop(self) -> None:
        threading.Thread(target=JOBS["drive_goal"].stop, daemon=True).start()
        self.send_json({"status": "stopping"}, status=202)

    def handle_tracking_start(self) -> None:
        """Resume live localization without rewriting the map: pause integration
        FIRST so no frame can fuse in the gap, then restart the phone scan."""
        params = load_params()
        control_port = int(params["control_port"])
        post_mapper_control(control_port, "map/integration/pause")
        post_mapper_control(control_port, "scan/start")
        self.send_json({"status": "tracking"})

    def handle_tracking_stop(self) -> None:
        params = load_params()
        post_mapper_control(int(params["control_port"]), "scan/stop")
        self.send_json({"status": "stopped"})

    def handle_params_post(self) -> None:
        body = self.read_json_body()
        self.send_json(save_params(body))

    # -- static files --------------------------------------------------------
    def serve_static(self) -> None:
        target = self.static_path()
        if target is None:
            self.send_json({"error": "not_found"}, status=404)
            return
        content_type = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
        data = target.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        # Everything no-cache: a cached app.js/style.css from before an edit,
        # mixed with a fresh index.html, silently breaks the page (e.g. CSS
        # that hides an image until JS the browser never refetched marks it
        # loaded). These files are tiny; caching buys nothing on a LAN.
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(data)

    def static_path(self) -> Path | None:
        raw_path = unquote(urlparse(self.path).path)
        if raw_path == "/":
            raw_path = "/index.html"
        candidate = (DIST_DIR / raw_path.lstrip("/")).resolve()
        if DIST_DIR not in candidate.parents and candidate != DIST_DIR:
            return None
        if not candidate.is_file():
            return None
        return candidate


def main() -> int:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer((HOST, PORT), DashboardHandler)
    print(f"[dashboard] serving {DIST_DIR} at http://{HOST}:{PORT}")
    print(f"[dashboard] state dir: {STATE_DIR}")
    if not (APP_USERNAME and APP_PASSWORD):
        print("[dashboard] WARNING: no APP_USERNAME/APP_PASSWORD set -- unauthenticated. "
              "Fine on a trusted local network; set both before tunneling this publicly.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        for job in JOBS.values():
            job.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
