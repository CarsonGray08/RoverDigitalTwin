import json
import time
import urllib.error
import urllib.request

from support import TEST_CONFIG, saved_map

from rover_twin.ops import dashboard as dash
from rover_twin.twin import offline


def call(url, path, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(url + path, data, {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def test_the_dashboard_shows_the_twin_and_rehearses_without_arming(tmp_path, monkeypatch):
    monkeypatch.setattr(dash, "STATE", tmp_path)
    monkeypatch.setattr(dash, "REPO", tmp_path)            # the drive log goes here, not in runs/
    path = saved_map(tmp_path / "map.npz", (-1.0, 3.0, -1.5, 1.5, 0.0))
    _, twin = offline.start(path, TEST_CONFIG, "127.0.0.1", 0)
    board = dash.Dashboard(TEST_CONFIG, f"http://127.0.0.1:{twin.server_address[1]}")
    server = dash.serve(board, "127.0.0.1", 0)
    url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        status, s = call(url, "/api/status")
        assert status == 200 and s["observation"]["terrain_tiles"] > 0 and not s["motion_job"]["running"]
        assert call(url, "/api/map")[1]["shape"][0] > 0
        assert call(url, "/api/motion/start", {"kind": "drive", "command": "FWD", "amount": 0.5, "arm": True})[0] == 400

        # Disarmed, a drive is the drive tool rehearsing: offline there is no
        # live pose, and it says so instead of moving anything.
        assert call(url, "/api/motion/start", {"kind": "drive", "command": "FWD", "amount": 0.5})[0] == 200
        deadline = time.monotonic() + 30
        while board.motion.running() and time.monotonic() < deadline:
            time.sleep(0.1)
        log = "\n".join(call(url, "/api/status")[1]["motion_job"]["log"])
        assert "no fresh live pose" in log and "--arm" not in json.dumps(board.motion.popen.args)
        plan = json.loads((tmp_path / "plan.json").read_text())
        assert plan["commands"] == [{"command": "FWD", "speed": 30, "distance": 11500}]
    finally:
        server.shutdown()
        twin.shutdown()
