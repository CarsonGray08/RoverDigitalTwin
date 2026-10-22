"""Replay regressions: synthetic scans through the full mapper, scored against
their ground truth with the same metrics the benchmark recordings will use."""

import json
import math

import numpy as np
import pytest
from scipy.spatial import cKDTree
from support import TEST_CONFIG, CollectingLink, FakeTsdf, map_scan, truth_of

from rover_twin import frames
from rover_twin.link.messages import LivePose, Tiles
from rover_twin.mapper import main as mapper_main
from rover_twin.mapper.pipeline import Mapper
from rover_twin.metrics import report
from rover_twin.phone import protocol, recording
from rover_twin.phone.protocol import Arrival, Event, Msg, PhonePose


def scored(scanlog, run_dir):
    return report.score(report.load_run(run_dir), report.Truth.load(truth_of(scanlog)))


def test_exact_depth_maps_the_room_exactly(scan, tmp_path):
    log = scan("spin", noise_per_m=0.0)
    map_scan(log, tmp_path)
    s = scored(log, tmp_path)
    assert max(s["floor_rms_m"], s["floor_step_m"], s["seam_p95_m"]) < 1e-4
    assert s["yaw_error_deg"] < 1e-6
    assert s["box_error_m"] <= 0.025          # one 2 cm cell: the grid's resolution


def test_a_phone_tilted_down_on_its_mount_still_maps_the_room_exactly(scan, tmp_path, capsys):
    # The config says level; the phone points 12 deg down. Measured at the
    # anchor, the tilt neither shifts the rover nor lowers the floor.
    log = scan("spin", noise_per_m=0.0, pitch_deg=12.0)
    link = CollectingLink()
    mapper = map_scan(log, tmp_path, link)
    assert mapper.watch.values["mount_pitch_deg"] == pytest.approx(12.0, abs=0.01)
    assert "tilted 12.0 deg down" in capsys.readouterr().out
    floor = mapper.grid.points()
    floor = floor[np.abs(floor[:, 2]) < 0.05]
    assert np.median(floor[:, 2]) == pytest.approx(0.0, abs=1e-3)        # not 1.3 cm under the floor
    rover = np.array([json.loads(line)["rover"] for line in (tmp_path / "trajectory.jsonl").open()])
    assert np.hypot(rover[:, 0], rover[:, 1]).max() < 1e-3               # spins on the spot, not on a circle
    heights = [m.z_m for m in link.sent if isinstance(m, LivePose)]
    assert np.abs(heights).max() < 1e-3                                  # the live pose stands on the floor


def test_noisy_spin_meets_the_fidelity_targets(scan, tmp_path):
    log = scan("spin")
    map_scan(log, tmp_path)
    s = scored(log, tmp_path)
    for metric in ("floor_rms_m", "floor_step_m", "seam_p95_m", "yaw_error_deg"):
        assert s[metric] <= report.TARGETS[metric], metric
    # Depth noise acts along the ray, so vertical faces bleed ~2 sigma toward
    # the camera in a grid without visibility clearing: held here, the 2 cm
    # target not met.
    assert s["box_error_m"] < 0.15


def test_the_loop_closure_removes_odometry_drift(scan, tmp_path):
    log = scan("loop", yaw_drift_deg_s=0.1, z_drift_m_s=0.001)   # 3.2 deg and 3.2 cm by the end of the loop
    mapper = map_scan(log, tmp_path)
    s = scored(log, tmp_path)
    assert mapper.graph.closures
    assert mapper.graph.height() == pytest.approx(-0.032, abs=0.004)    # the height drift, measured back out
    for metric in ("yaw_error_deg", "drift_pct", "seam_p95_m", "floor_rms_m"):
        assert s[metric] <= report.TARGETS[metric], metric


def test_reset_position_moves_the_map_with_the_new_anchor(scan, tmp_path):
    link = CollectingLink()
    mapper = map_scan(scan("spin", noise_per_m=0.0), tmp_path, link)
    before = mapper.grid.points()
    sent_before = len(link.sent)
    # Reset Position with ARKit now reporting the rover 0.5 m along twin +x
    # from where the map was built: the rover there becomes the start pose.
    T_W_A = mapper.alignment.T_W_A
    shift = frames.rot_z(0.0, (0.5, 0.0, 0.0))
    T_A_C = frames.rigid_inverse(T_W_A) @ shift @ T_W_A @ mapper.track[-1].T_A_C
    t = mapper.track[-1].t_phone
    messages = [Event(Msg.REALIGN, t)] + [PhonePose(t + 0.2 * i, T_A_C, protocol.TRACKING_NORMAL) for i in range(6)]
    for m in messages:
        mapper.process(Arrival(0.0, protocol.encode(m)[protocol.LENGTH.size:]))

    resets = [m for m in link.sent[sent_before:] if isinstance(m, Tiles)]
    assert len(resets) == 1 and resets[0].reset and len(resets[0].patches) == len(mapper.grid.tiles)
    # The live pose follows its terrain at once: the last frame, 0.5 m behind the new start.
    pose = link.sent[link.sent.index(resets[0]) + 1]
    assert isinstance(pose, LivePose) and pose.rev == resets[0].rev
    assert (pose.pose.x, pose.pose.y) == pytest.approx((-0.5, 0.0), abs=1e-3)
    after = mapper.grid.points()
    distance, _ = cKDTree(before - [0.5, 0.0, 0.0]).query(after)
    assert np.percentile(distance, 99) < 0.005     # every re-fused cell is where the shifted map had it
    # Keyframes alone cover less at range than every frame did.
    assert len(after) > 0.8 * len(before)


def test_limited_tracking_stops_the_map_and_the_live_pose_until_it_recovers(scan, tmp_path):
    log = scan("spin", noise_per_m=0.0)
    link = CollectingLink()
    mapper = map_scan(log, tmp_path, link)
    frame = next(a for a in reversed(list(recording.replay(log, speed=0)))
                 if isinstance(protocol.decode(a.payload), protocol.RgbdFrame))
    T_A_C, t = mapper.track[-1].T_A_C, mapper.track[-1].t_phone

    def pose(tracking):
        return Arrival(0.0, protocol.encode(PhonePose(t, T_A_C, tracking))[protocol.LENGTH.size:])

    sent = len(link.sent)
    mapper.process(pose(1))                      # ARKit: limited
    mapper.process(frame)
    assert len(link.sent) == sent                # nothing fused, no live pose: it ages and a drive stops
    mapper.process(pose(protocol.TRACKING_NORMAL))
    mapper.process(frame)
    assert isinstance(link.sent[-1], LivePose)


def test_mapper_process_replays_a_recording(scan, tmp_path, monkeypatch):
    log = scan("drive")
    cfg = tmp_path / "rover_twin.toml"
    cfg.write_text(mapper_main.DEFAULT_CONFIG.read_text() + "\n[link]\nport = 0\n")
    monkeypatch.setattr(mapper_main, "NvbloxTsdf", lambda *a, **k: FakeTsdf())
    run = tmp_path / "run"
    mapper_main.main(["--config", str(cfg), "--replay", str(log), "--speed", "0", "--run-dir", str(run)])
    metrics = json.loads((run / "metrics.json").read_text())
    trajectory = (run / "trajectory.jsonl").read_text().splitlines()
    assert {"decoded", "fused", "tiles_sent", "pose_sent"} <= set(metrics["latency_ms"])
    assert len(trajectory) == metrics["latency_ms"]["pose_sent"]["n"] > 30
    assert (run / "config.toml").read_text() == cfg.read_text()
    end = json.loads(trajectory[-1])["rover"]
    assert end == pytest.approx([1.6, 0.0, 0.0], abs=1e-6)


def test_poses_from_before_the_scan_do_not_anchor_the_map(scan, tmp_path):
    # LiveScanner streams poses from launch, while the phone is still in a
    # hand: only once RGB-D streams (it is mounted, scanning) may they anchor.
    log = scan("spin", noise_per_m=0.0)
    arrivals = list(recording.replay(log, speed=0))
    pose = next(m for m in map(protocol.decode, (a.payload for a in arrivals)) if isinstance(m, PhonePose))
    turned = frames.rot_z(math.radians(150.0), (0.3, 0.0, 0.0))        # about gravity, in ARKit's world
    handheld = frames.rigid_inverse(frames.M_AXIS) @ turned @ frames.M_AXIS @ pose.T_A_C
    link = CollectingLink()
    mapper = Mapper(TEST_CONFIG, FakeTsdf(), link, tmp_path)
    for i in range(10):
        mapper.process(Arrival(0.0, protocol.encode(
            PhonePose(pose.t_phone - 5.0 + 0.2 * i, handheld, protocol.TRACKING_NORMAL))[protocol.LENGTH.size:]))
    assert mapper.alignment.T_W_A is None
    mapper.run(arrivals)
    first = next(m for m in link.sent if isinstance(m, LivePose))
    assert (first.pose.x, first.pose.y, first.pose.yaw_deg) == pytest.approx((0.0, 0.0, 0.0), abs=1e-3)


def test_depth_of_the_rover_itself_is_not_mapped(tmp_path):
    # A phone pointed down sees a front wheel at the edge of the frame. Within
    # the footprint, it is dropped; the floor ahead is kept.
    mapper = Mapper(TEST_CONFIG, FakeTsdf(), CollectingLink(), tmp_path)
    intr = frames.Intrinsics(180.0, 180.0, 128.0, 96.0, 256, 192)
    depth = np.full((192, 256), 2.0, np.float32)
    c, s_ = math.cos(math.radians(17.0)), math.sin(math.radians(17.0))
    tilt = np.eye(4)
    tilt[1:3, 1:3] = [[c, s_], [-s_, c]]                     # 17 deg down, as on the lab rover
    rc = TEST_CONFIG.rover
    mapper.mount = mount = frames.Mount(mapper.mount.T_R_C @ tilt, mapper.mount.forward)
    left = np.cross([0.0, 0.0, 1.0], mount.forward)
    wheel = (rc.footprint_half_length_m + 0.02) * mount.forward + 0.12 * left + [0.0, 0.0, 0.2]
    x, y, z = frames.transform(mount.T_C_R, wheel[None])[0]
    u, v = round(intr.fx * x / z + intr.cx), round(intr.fy * y / z + intr.cy)
    depth[v, u] = z
    frame = protocol.RgbdFrame(0.0, np.eye(4), depth, np.full(depth.shape, 2, np.uint8), b"", intr)
    usable = mapper._usable_depth(frame)
    assert usable[v, u] == 0.0
    usable[v - 2:v + 3, u - 2:u + 3] = 1.0                          # the floor right behind it goes as an edge
    assert (usable > 0).all()


def test_flying_pixels_at_a_depth_edge_are_dropped_and_grazing_floor_is_kept(tmp_path):
    # An object 0.5 m away against floor 1.2 m behind it (as a wheel against
    # the floor): the sensor reports the pixels along the edge in between.
    # Floor far away at a grazing angle steps a few percent a pixel and stays.
    mapper = Mapper(TEST_CONFIG, FakeTsdf(), CollectingLink(), tmp_path)
    intr = frames.Intrinsics(180.0, 180.0, 128.0, 96.0, 256, 192)
    depth = np.repeat(np.linspace(5.0, 2.0, 192, dtype=np.float32)[:, None], 256, axis=1)   # grazing floor
    depth[100:140, 60:100] = 1.2
    depth[100:140, 100:140] = 0.5                                   # an object in front of it
    depth[100:140, 100] = 0.8                                       # the edge, reported in between
    frame = protocol.RgbdFrame(0.0, np.eye(4), depth, np.full(depth.shape, 2, np.uint8), b"", intr)
    usable = mapper._usable_depth(frame)
    assert (usable[100:140, 99:101] == 0).all()                     # the edge, and the floor just behind it
    assert (usable[100:140, 101:140] > 0).all()                     # the object, to its outline
    assert (usable[105:135, 63:97] > 0).all()
    assert (usable[:90] > 0).all()                                  # every row of the grazing floor
    # A dark tyre reads low-confidence: dropped itself, it still marks the edge.
    confidence = np.full(depth.shape, 2, np.uint8)
    confidence[100:140, 101:140] = 0
    dark = mapper._usable_depth(protocol.RgbdFrame(0.0, np.eye(4), depth, confidence, b"", intr))
    assert (dark[100:140, 99:140] == 0).all() and (dark[105:135, 63:97] > 0).all()
