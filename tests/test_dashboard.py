import base64
import json
import time
from dataclasses import replace
import urllib.error
import urllib.request

from support import TEST_CONFIG, saved_map

from rover_twin.mapper import main as mapper_main
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


def test_the_dashboard_shows_the_phone_camera_and_says_when_the_twin_has_no_view(tmp_path, monkeypatch):
    monkeypatch.setattr(dash, "STATE", tmp_path)

    class Phone:
        def send(self, kind):
            return False

    control = mapper_main.serve_control(Phone(), 0, lambda: (b"\xff\xd8camera", time.monotonic()))
    _, twin = offline.start(saved_map(tmp_path / "map.npz", (-1.0, 1.0, -1.0, 1.0, 0.0)), TEST_CONFIG, "127.0.0.1", 0)
    cfg = replace(TEST_CONFIG, phone=replace(TEST_CONFIG.phone, control_port=control.server_address[1]))
    board = dash.Dashboard(cfg, f"http://127.0.0.1:{twin.server_address[1]}")
    server = dash.serve(board, "127.0.0.1", 0)
    url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        with urllib.request.urlopen(url + "/api/camera", timeout=5) as response:
            assert response.read() == b"\xff\xd8camera" and float(response.headers["X-Age-S"]) < 2.0
        assert call(url, "/api/twin/view")[0] == 404                   # the offline twin draws nothing
        control.shutdown()
        assert call(url, "/api/camera")[0] == 502                      # no mapper
    finally:
        server.shutdown()
        twin.shutdown()


def test_through_the_tunnel_plain_http_goes_to_https_before_the_login(tmp_path, monkeypatch):
    monkeypatch.setattr(dash, "STATE", tmp_path)
    monkeypatch.setenv("APP_USERNAME", "operator")
    monkeypatch.setenv("APP_PASSWORD", "secret")
    server = dash.serve(dash.Dashboard(TEST_CONFIG, "http://127.0.0.1:9"), "127.0.0.1", 0)
    url = f"http://127.0.0.1:{server.server_address[1]}"

    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *args):
            return None

    opener = urllib.request.build_opener(NoRedirect)

    def get(headers):
        try:
            with opener.open(urllib.request.Request(url + "/api/status", headers=headers), timeout=5) as response:
                return response.status, response.headers
        except urllib.error.HTTPError as exc:
            return exc.code, exc.headers

    host = {"Host": "twin.example.org"}
    try:
        status, headers = get({**host, "Cf-Visitor": '{"scheme":"http"}'})
        assert status == 308 and headers["Location"] == "https://twin.example.org/api/status"
        assert get({**host, "Cf-Visitor": '{"scheme":"https"}'})[0] == 401          # then the login
        assert get({})[0] == 401                                                     # on the lab network too
        login = "Basic " + base64.b64encode(b"operator:secret").decode()
        assert get({**host, "Cf-Visitor": '{"scheme":"https"}', "Authorization": login})[0] == 200
    finally:
        server.shutdown()
