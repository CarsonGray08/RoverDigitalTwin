"""Operator dashboard: start the twin, watch the map, the rover, the phone's
camera and the twin's view, and run a 360 scan or a drive, behind an arm switch. The scan's turns are flown on the
live pose without the twin's rehearsal; a drive is rehearsed first.

    scripts/dashboard.sh                # prints http://<workstation>:8090

Motion goes through the drive tool (``rover_twin.ops.drive``): every command
is rehearsed on the live map first and stopped on the live pose, and while
disarmed the plan is only rehearsed. The page loads disarmed. It listens on
every interface for the lab network; set APP_USERNAME and APP_PASSWORD
before exposing it any further. Through a Cloudflare tunnel, plain HTTP is
redirected to HTTPS before the login is asked for.
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
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import unquote, urlparse
from urllib.request import Request, urlopen

from rover_twin import config
from rover_twin.ops.rover import Rover, RoverError

REPO = Path(__file__).resolve().parents[3]
STATIC = REPO / "dashboard"
STATE = Path(os.environ.get("DASHBOARD_STATE_DIR", f"/tmp/rover_twin_dashboard-{os.getuid()}"))  # per account
SPEED = 30                                   # the twin's command speed; the real rover's is --level
SCAN = [{"command": "LEFT", "speed": SPEED, "distance": 45}] * 8    # a smooth-wheel 90 needs more than MAX_STEPS
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
        """Until the whole group is gone, not just its leader: a child left
        behind keeps its ports, and the next start cannot use them."""
        popen = self.popen
        if popen is None:
            return
        for sig, grace_s in ((signal.SIGINT, 25.0), (signal.SIGTERM, 5.0), (signal.SIGKILL, 5.0)):
            try:
                os.killpg(popen.pid, sig)
            except ProcessLookupError:
                return
            deadline = time.monotonic() + grace_s
            while time.monotonic() < deadline:
                popen.poll()                      # reap the leader, or the group lives on as its zombie
                try:
                    os.killpg(popen.pid, 0)
                except ProcessLookupError:
                    return
                time.sleep(0.2)

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

    def image(self, url: str) -> tuple[int, bytes, str, dict]:
        """A JPEG from the mapper or the twin with its X-Age-S, or why there is none (JSON)."""
        try:
            with urlopen(url, timeout=2.0) as response:
                age = response.headers.get("X-Age-S")
                return (response.status, response.read(), response.headers.get("Content-Type", "image/jpeg"),
                        {"X-Age-S": age} if age else {})
        except HTTPError as exc:
            return exc.code, exc.read(), "application/json", {}
        except OSError as exc:
            return 502, json.dumps({"error": f"not answering ({exc})"}).encode(), "application/json", {}

    def start_twin(self) -> None:
        self.twin.start(["bash", str(REPO / "scripts" / "rover_twin.sh")])

    def start_motion(self, body: dict) -> None:
        """A 360 scan or one command, rehearsed on the twin; flown if armed."""
        rehearse = True
        if body.get("kind") == "scan":
            self.start_phone_scan()
            commands, rehearse = SCAN, False
        else:
            command, amount = body.get("command"), float(body.get("amount", 0))
            if command not in COMMANDS or amount <= 0:
                raise ValueError(f"a drive is one of {COMMANDS} with a positive amount")
            units = self.cfg.rover.distance_units_per_m if command in ("FWD", "REV") else 1.0
            commands = [{"command": command, "speed": SPEED, "distance": round(amount * units)}]
        plan = STATE / "plan.json"
        plan.write_text(json.dumps({"commands": commands, "rehearse": rehearse}))
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

    def start_phone_scan(self) -> None:
        """Tell the phone, through the mapper, to start scanning (harmless if it already is)."""
        self._mapper_control("/scan/start", "start the phone's scan")

    def clear_map(self) -> None:
        """Forget the map; it grows again from what the phone sees next."""
        self._idle("clearing the map")
        self._mapper_control("/map/clear", "clear the map")

    def reset_tracking(self) -> None:
        """Restart the phone's ARKit tracking; the map is cleared and anchors
        again, at the start pose, once the rover holds still."""
        self._idle("resetting the phone's tracking")
        self._mapper_control("/tracking/reset", "reset the phone's tracking")

    def _idle(self, what: str) -> None:
        if self.motion.running():
            raise RuntimeError(f"stop the scan or drive before {what}")

    def _mapper_control(self, path: str, what: str) -> None:
        url = f"http://127.0.0.1:{self.cfg.phone.control_port}{path}"
        try:
            with urlopen(Request(url, b"", method="POST"), timeout=2.0):
                pass
        except HTTPError as exc:
            raise RuntimeError(f"could not {what}: {exc.read().decode(errors='replace')}") from exc
        except OSError as exc:
            raise RuntimeError(f"the mapper is not answering ({exc}): start the mapper and twin first") from exc

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
            if self._insecure() or not self._authorized():
                return
            path = urlparse(self.path).path
            if path == "/api/status":
                self._json(200, dashboard.status())
            elif path == "/api/health":
                self._send(*dashboard.twin_get("/health", 2.0), "application/json")
            elif path == "/api/map":                  # waits behind rehearsals, like any /map
                self._send(*dashboard.twin_get("/map", 30.0), "application/json")
            elif path == "/api/camera":
                self._send(*dashboard.image(f"http://127.0.0.1:{dashboard.cfg.phone.control_port}/camera.jpg"))
            elif path == "/api/twin/view":
                self._send(*dashboard.image(dashboard.twin_url + "/view.jpg"))
            else:
                self._static(path)

        def do_POST(self) -> None:
            if self._insecure() or not self._authorized():
                return
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))) or b"{}")
            actions = {
                "/api/twin/start": dashboard.start_twin,
                "/api/twin/stop": lambda: threading.Thread(target=dashboard.twin.stop, daemon=True).start(),
                "/api/motion/start": lambda: dashboard.start_motion(body),
                "/api/motion/stop": lambda: dashboard.stop_motion(str(body.get("rover_ip") or "").strip()),
                "/api/map/clear": dashboard.clear_map,
                "/api/tracking/reset": dashboard.reset_tracking,
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

        def _insecure(self) -> bool:
            """Cloudflare's edge says how the browser reached it; plain HTTP goes
            to HTTPS first, so the login never crosses the internet in clear."""
            if json.loads(self.headers.get("Cf-Visitor") or "{}").get("scheme") != "http":
                return False
            self.send_response(308)
            self.send_header("Location", f"https://{self.headers.get('Host', '')}{self.path}")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return True

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

        def _send(self, status: int, body: bytes, content_type: str, headers: dict | None = None) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-cache")
            for key, value in (headers or {}).items():
                self.send_header(key, value)
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
