"""Operator dashboard: start the twin, watch the map and the rover, and run a
360 scan or a drive, behind an arm switch.

    scripts/dashboard.sh                # prints http://<workstation>:8090

Motion goes through the drive tool (``rover_twin.ops.drive``): every command
is rehearsed on the live map first and stopped on the live pose, and while
disarmed the plan is only rehearsed. The page loads disarmed. It listens on
every interface for the lab network; set APP_USERNAME and APP_PASSWORD
before exposing it any further.
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
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import unquote, urlparse
from urllib.request import urlopen

from rover_twin import config
from rover_twin.ops.rover import Rover, RoverError

REPO = Path(__file__).resolve().parents[3]
STATIC = REPO / "dashboard"
STATE = Path(os.environ.get("DASHBOARD_STATE_DIR", "/tmp/rover_twin_dashboard"))
SPEED = 30                                   # the twin's command speed; the real rover's is --level
SCAN = [{"command": "LEFT", "speed": SPEED, "distance": 90}] * 4    # rehearsed a quarter at a time
COMMANDS = ("FWD", "REV", "LEFT", "RIGHT")


class Job:
    """One child process and the tail of its output. Stopped like Ctrl-C in a
    terminal (SIGINT to its whole process group), which the drive tool answers
    by stopping the rover and rover_twin.sh by stopping mapper and twin."""

    def __init__(self, name: str) -> None:
        self.name, self.popen, self.lock = name, None, threading.Lock()
        self.lines: deque[str] = deque(maxlen=300)

    def running(self) -> bool:
        return self.popen is not None and self.popen.poll() is None

    def start(self, cmd: list[str]) -> None:
        with self.lock:
            if self.running():
                raise RuntimeError(f"{self.name} is already running")
            self.lines.clear()
            self.popen = subprocess.Popen(cmd, cwd=REPO, env={**os.environ, "PYTHONUNBUFFERED": "1"},
                                          stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                          stderr=subprocess.STDOUT, start_new_session=True)
        threading.Thread(target=self._pump, args=(self.popen,), daemon=True).start()

    def stop(self) -> None:
        popen = self.popen
        for sig, grace_s in ((signal.SIGINT, 10.0), (signal.SIGTERM, 5.0), (signal.SIGKILL, 5.0)):
            if popen is None or popen.poll() is not None:
                return
            try:
                os.killpg(popen.pid, sig)
                popen.wait(timeout=grace_s)
            except ProcessLookupError:
                return
            except subprocess.TimeoutExpired:
                continue

    def status(self) -> dict:
        return {"running": self.running(), "returncode": None if self.popen is None else self.popen.poll(),
                "log": list(self.lines)[-120:]}

    def _pump(self, popen: subprocess.Popen) -> None:
        for raw in popen.stdout:
            self.lines.append(raw.decode("utf-8", "replace").rstrip("\n"))


class Dashboard:
    def __init__(self, cfg: config.Config, twin_url: str) -> None:
        self.cfg, self.twin_url = cfg, twin_url.rstrip("/")
        self.twin, self.motion = Job("twin"), Job("motion")
        STATE.mkdir(parents=True, exist_ok=True)

    def status(self) -> dict:
        code, body = self.twin_get("/observation", 2.0)
        return {"twin_job": self.twin.status(), "motion_job": self.motion.status(),
                "observation": json.loads(body) if code == 200 else None,
                "rover_ip": os.environ.get("ROVER_IP", "")}

    def twin_get(self, path: str, timeout_s: float) -> tuple[int, bytes]:
        try:
            with urlopen(self.twin_url + path, timeout=timeout_s) as response:
                return response.status, response.read()
        except HTTPError as exc:
            return exc.code, exc.read()
        except OSError as exc:
            return 502, json.dumps({"error": f"the twin is not answering ({exc})"}).encode()

    def start_twin(self) -> None:
        self.twin.start(["bash", str(REPO / "scripts" / "rover_twin.sh")])

    def start_motion(self, body: dict) -> None:
        """A 360 scan or one command, rehearsed on the twin; flown if armed."""
        if body.get("kind") == "scan":
            commands = SCAN
        else:
            command, amount = body.get("command"), float(body.get("amount", 0))
            if command not in COMMANDS or amount <= 0:
                raise ValueError(f"a drive is one of {COMMANDS} with a positive amount")
            units = self.cfg.rover.distance_units_per_m if command in ("FWD", "REV") else 1.0
            commands = [{"command": command, "speed": SPEED, "distance": round(amount * units)}]
        plan = STATE / "plan.json"
        plan.write_text(json.dumps({"commands": commands}))
        cmd = [sys.executable, "-m", "rover_twin.ops.drive", str(plan), "--twin", self.twin_url,
               "--log", str(REPO / "runs" / "drive_log.jsonl")]
        if body.get("arm"):
            rover_ip = str(body.get("rover_ip") or "").strip()
            if not rover_ip:
                raise ValueError("an armed drive needs the rover's IP")
            cmd += ["--arm", "--rover-ip", rover_ip, "--level", str(int(body.get("level", 1)))]
            if os.environ.get("ROVER_NAME"):
                cmd += ["--rover-name", os.environ["ROVER_NAME"]]
        (REPO / "runs").mkdir(exist_ok=True)
        self.motion.start(cmd)

    def stop_motion(self, rover_ip: str) -> None:
        """Stop the drive tool (it stops the rover itself) and send the rover
        a STOP directly as well, in case the tool is not answering."""
        if rover_ip:
            try:
                Rover(rover_ip, name=os.environ.get("ROVER_NAME", "")).stop()
            except RoverError:
                pass
        self.motion.stop()


def serve(dashboard: Dashboard, host: str, port: int) -> ThreadingHTTPServer:
    login = os.environ.get("APP_USERNAME"), os.environ.get("APP_PASSWORD")
    expected = "Basic " + base64.b64encode(f"{login[0]}:{login[1]}".encode()).decode() if all(login) else None

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if not self._authorized():
                return
            path = urlparse(self.path).path
            if path == "/api/status":
                self._json(200, dashboard.status())
            elif path == "/api/health":
                self._send(*dashboard.twin_get("/health", 2.0), "application/json")
            elif path == "/api/map":                  # waits behind rehearsals, like any /map
                self._send(*dashboard.twin_get("/map", 30.0), "application/json")
            else:
                self._static(path)

        def do_POST(self) -> None:
            if not self._authorized():
                return
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))) or b"{}")
            actions = {
                "/api/twin/start": dashboard.start_twin,
                "/api/twin/stop": lambda: threading.Thread(target=dashboard.twin.stop, daemon=True).start(),
                "/api/motion/start": lambda: dashboard.start_motion(body),
                "/api/motion/stop": lambda: dashboard.stop_motion(str(body.get("rover_ip") or "").strip()),
            }
            action = actions.get(urlparse(self.path).path)
            if action is None:
                self._json(404, {"error": "not_found"})
                return
            try:
                action()
                self._json(200, {"ok": True})
            except ValueError as exc:
                self._json(400, {"error": str(exc)})
            except RuntimeError as exc:
                self._json(409, {"error": str(exc)})

        def _authorized(self) -> bool:
            if expected is None or self.headers.get("Authorization") == expected:
                return True
            self.send_response(401)
            self.send_header("WWW-Authenticate", 'Basic realm="rover twin"')
            self.end_headers()
            return False

        def _static(self, path: str) -> None:
            target = (STATIC / unquote(path).lstrip("/")).resolve() if path != "/" else STATIC / "index.html"
            if STATIC not in target.parents or not target.is_file():
                self._json(404, {"error": "not_found"})
                return
            self._send(200, target.read_bytes(), mimetypes.guess_type(target.name)[0] or "application/octet-stream")

        def _json(self, status: int, payload: dict) -> None:
            self._send(status, json.dumps(payload).encode(), "application/json")

        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args) -> None:
            return

    server = ThreadingHTTPServer((host, port), Handler)
    threading.Thread(target=server.serve_forever, name="dashboard-http", daemon=True).start()
    return server


def main() -> None:
    cfg = config.load(REPO / "config" / "rover_twin.toml")
    dashboard = Dashboard(cfg, f"http://127.0.0.1:{cfg.twin.port}")
    server = serve(dashboard, "0.0.0.0", int(os.environ.get("DASHBOARD_PORT", "8090")))
    if not os.environ.get("APP_USERNAME"):
        print("dashboard: no APP_USERNAME/APP_PASSWORD, so anyone on this network can open it", flush=True)
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        dashboard.motion.stop()
        dashboard.twin.stop()


if __name__ == "__main__":
    main()
