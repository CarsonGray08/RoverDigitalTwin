from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

from digital_twin import _isaac_runtime, _mapper_runtime, dashboard, isaac_service, mapper, scan
from digital_twin.models import CommandName
from digital_twin.executor import DEFAULT_COMMAND_MODE, continual_twist_components
from digital_twin.rover_command_model import RoverCommandCalibration


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_phone_mount_uses_the_calibrated_rover_frame():
    mount = json.loads((PROJECT_ROOT / "config" / "phone_mount.json").read_text())
    assert mount["rover_forward_axis"] == "-y"
    assert mount["camera_in_rover"]["translation_m"] == [0.0, -0.1, 0.22]


def test_dashboard_exposes_runtime_settings(monkeypatch, tmp_path):
    monkeypatch.setattr(dashboard, "PARAMS_PATH", tmp_path / "params.json")
    assert set(dashboard.PARAM_SPEC) == {
        "map_root",
        "phone_port",
        "control_port",
        "validator_port",
        "command_mode",
        "voxel_size_m",
        "min_depth_confidence",
        "tile_size_m",
        "map_update_interval_s",
        "stabilize_floor",
    }
    saved = dashboard.save_params({"voxel_size_m": "0.03"})
    assert saved["voxel_size_m"] == pytest.approx(0.03)
    assert dashboard.env_overrides(saved)["VOXEL_SIZE_M"] == "0.03"
    assert dashboard.env_overrides(saved)["STABILIZE_FLOOR"] == "1"
    assert saved["command_mode"] == DEFAULT_COMMAND_MODE == "continual"
    assert dashboard.env_overrides(saved)["COMMAND_MODE"] == "continual"


def test_scan_uses_four_bounded_turn_segments():
    commands = scan.turn_segments(RoverCommandCalibration(), 360.0, 90.0)
    assert len(commands) == 4
    assert all(command.command is CommandName.LEFT for command in commands)
    assert all(command.distance == 24_000 for command in commands)


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        (CommandName.FWD, (1.0, 0.0)),
        (CommandName.REV, (-1.0, 0.0)),
        (CommandName.LEFT, (0.0, 1.0)),
        (CommandName.RIGHT, (0.0, -1.0)),
        (CommandName.STOP, (0.0, 0.0)),
    ],
)
def test_continual_command_twist_mapping(command, expected):
    from digital_twin.models import CommandStep

    assert continual_twist_components(CommandStep(command, 1, 100)) == expected


def test_mapper_wrapper_configures_mapping_and_localization(monkeypatch, tmp_path):
    monkeypatch.setenv("DT_MAP_DIR", str(tmp_path))
    monkeypatch.setenv("DT_SESSION_ID", "test-session")
    args = mapper.runtime_arguments()
    assert "--use-cuvslam" in args
    assert args[args.index("--cuvslam-variant") + 1] == "depth-gray"
    assert "--resumable-scan" in args
    assert "--stabilize-floor" in args
    assert args[args.index("--fill-mode") + 1] == "delaunay"
    assert "--replay" not in args

    observed = []
    monkeypatch.setattr(
        _mapper_runtime,
        "main",
        lambda: observed.extend(sys.argv[1:]) or 17,
    )
    assert mapper.main([]) == 17
    assert observed == args


def test_validator_wrapper_configures_live_terrain(monkeypatch, tmp_path):
    for name, value in {
        "DT_SCENE_PATH": tmp_path / "rover.usd",
        "DT_MAP_DIR": tmp_path / "map",
        "DT_MOUNT_PATH": tmp_path / "mount.json",
        "DT_NEWTON_KIT": tmp_path / "newton.kit",
    }.items():
        monkeypatch.setenv(name, str(value))
    args = isaac_service.runtime_arguments()
    assert "--track-live-pose" in args
    assert "--load-terrain" in args
    assert "--hide-static-scene-at-start" in args
    assert "--disable-static-scene-collision" in args
    assert "--live-scan" not in args

    observed = []
    monkeypatch.setattr(
        _isaac_runtime,
        "main",
        lambda: observed.extend(sys.argv[1:]) or 23,
    )
    assert isaac_service.main([]) == 23
    assert observed == args


def test_launchers_use_the_supported_profile_wrappers():
    live_launcher = (PROJECT_ROOT / "scripts" / "live_twin.sh").read_text()
    isaac_launcher = (PROJECT_ROOT / "scripts" / "isaac_validator.sh").read_text()
    assert "python -m digital_twin.mapper" in live_launcher
    assert "-m digital_twin.isaac_service" in isaac_launcher
    assert "-m digital_twin._isaac_runtime" not in isaac_launcher


def test_hidden_static_payloads_are_unloaded_without_touching_rover_payloads():
    class Prim:
        def GetPath(self):
            return "/World/RASSOR"

    class Stage:
        def __init__(self):
            self.unloaded = []

        def GetLoadSet(self):
            return [
                "/World/RASSOR/body_payload",
                "/World/Environment/old_scan",
            ]

        def Unload(self, path):
            self.unloaded.append(path)

    validator = _isaac_runtime.IsaacSceneValidator.__new__(
        _isaac_runtime.IsaacSceneValidator
    )
    validator.hide_static_scene_at_start = True
    validator.rover_prim = Prim()
    validator.stage = Stage()

    validator._unload_hidden_static_payloads()

    assert validator.stage.unloaded == ["/World/Environment/old_scan"]


def test_runtime_does_not_import_the_parent_research_package():
    offenders = []
    for path in (PROJECT_ROOT / "src").rglob("*.py"):
        if "digital_twin_planner" in path.read_text():
            offenders.append(path.relative_to(PROJECT_ROOT))
    assert offenders == []


def test_large_research_artifacts_are_outside_the_handoff_project():
    forbidden = {"paper", "runs", "snapshots"}
    assert forbidden.isdisjoint(path.name for path in PROJECT_ROOT.iterdir())


def test_dashboard_has_no_unvalidated_motion_controls():
    client_source = (PROJECT_ROOT / "dashboard" / "app.js").read_text()
    html_source = (PROJECT_ROOT / "dashboard" / "index.html").read_text()
    assert '"/api/drive"' not in client_source
    assert "jog-pad" not in html_source
    assert "scan-telemetry" not in html_source
    assert "goal-telemetry" not in html_source


@pytest.mark.parametrize("arm", [False, True])
@pytest.mark.parametrize("command_mode", ["continual", "step"])
@pytest.mark.parametrize(
    ("handler_name", "job_name", "module", "mode_flag"),
    [
        ("handle_scan360_start", "scan360", "digital_twin.scan", "--turn-command-mode"),
        ("handle_drive_goal_start", "drive_goal", "digital_twin.drive", "--command-mode"),
    ],
)
def test_dashboard_motion_commands(monkeypatch, arm, command_mode, handler_name,
                                   job_name, module, mode_flag):
    params = {name: spec["default"] for name, spec in dashboard.PARAM_SPEC.items()}
    params["command_mode"] = command_mode
    monkeypatch.setattr(dashboard, "load_params", lambda: params)
    job = Mock()
    monkeypatch.setattr(dashboard, "JOBS", {job_name: job})
    handler = dashboard.DashboardHandler.__new__(dashboard.DashboardHandler)
    handler.read_json_body = Mock(return_value={"arm": arm, "rover_ip": "192.168.0.157"})
    handler.send_json = Mock()

    getattr(handler, handler_name)()

    job.start.assert_called_once()
    cmd, _env = job.start.call_args.args
    assert cmd[:3] == [sys.executable, "-m", module]
    assert cmd[cmd.index(mode_flag) + 1] == command_mode
    assert ("--arm" in cmd) is arm
    assert ("--telemetry-stop" in cmd) is arm
    assert ("--rover-ip" in cmd) is arm
    if arm:
        assert cmd[cmd.index("--rover-ip") + 1] == "192.168.0.157"


@pytest.mark.parametrize("handler_name", ["handle_scan360_start", "handle_drive_goal_start"])
def test_dashboard_rejects_armed_motion_without_rover_ip(monkeypatch, handler_name):
    jobs = {name: Mock() for name in ("scan360", "drive_goal")}
    monkeypatch.setattr(dashboard, "JOBS", jobs)
    handler = dashboard.DashboardHandler.__new__(dashboard.DashboardHandler)
    handler.read_json_body = Mock(return_value={"arm": True})

    with pytest.raises(ValueError, match="arm requires rover_ip"):
        getattr(handler, handler_name)()

    for job in jobs.values():
        job.start.assert_not_called()


def test_dashboard_rejects_unvalidated_drive_route():
    handler = dashboard.DashboardHandler.__new__(dashboard.DashboardHandler)
    handler.path = "/api/drive"
    handler.authorized = Mock(return_value=True)
    handler.send_json = Mock()

    handler.do_POST()

    handler.send_json.assert_called_once_with({"error": "not_found"}, status=404)


def test_research_and_diagnostic_entry_points_are_not_packaged():
    package = PROJECT_ROOT / "src" / "digital_twin"
    removed = {
        "advisor_gate_trials.py",
        "cli.py",
        "cuvslam_sweep.py",
        "fidelity_trials.py",
        "orchestrator.py",
        "validator_server.py",
    }
    assert all(not (package / name).exists() for name in removed)
