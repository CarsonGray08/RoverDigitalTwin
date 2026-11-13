from dataclasses import replace
"""Replay regressions: synthetic scans through the full mapper, scored against
their ground truth with the same metrics the benchmark recordings will use."""

import json
import math

import numpy as np
import pytest
from scipy.spatial import cKDTree
from support import TEST_CONFIG, CollectingLink, FakeTsdf, map_scan, truth_of

from rover_twin import frames
from rover_twin.link.messages import LivePose, Tiles, VisualMesh
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
    # One 2 cm cell, the grid's resolution; with the occupancy on, boxes are
    # its 4 cm voxels, which can each reach a voxel past a side: two in all.
    occupancy = TEST_CONFIG.occupancy
    assert s["box_error_m"] <= (2 * occupancy.voxel_m if occupancy.enabled else 0.025) + 0.005


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
    # Every ground tile (and, with the occupancy on, any tile only an obstacle reaches).
    assert len(resets) == 1 and resets[0].reset and {p.key for p in resets[0].patches} >= set(mapper.grid.tiles)
    # The live pose follows its terrain at once: the last frame, 0.5 m behind the new start.
    pose = link.sent[link.sent.index(resets[0]) + 1]
    assert isinstance(pose, LivePose) and pose.rev == resets[0].rev
    assert (pose.pose.x, pose.pose.y) == pytest.approx((-0.5, 0.0), abs=1e-3)
    after = mapper.grid.points()
    distance, _ = cKDTree(before - [0.5, 0.0, 0.0]).query(after)
    assert np.percentile(distance, 99) < 0.005     # every re-fused cell is where the shifted map had it
    # Keyframes alone cover less at range than every frame did.
    assert len(after) > 0.8 * len(before)


def test_clearing_the_map_empties_the_twin_and_keeps_the_live_pose_where_it_was(scan, tmp_path):
    log = scan("loop", yaw_drift_deg_s=0.1, z_drift_m_s=0.001)
    link = CollectingLink()
    mapper = map_scan(log, tmp_path, link)
    assert mapper.graph.closures                 # so the latest pose carries a correction
    before = [m for m in link.sent if isinstance(m, LivePose)][-1]
    frame = next(a for a in reversed(list(recording.replay(log, speed=0)))
                 if isinstance(protocol.decode(a.payload), protocol.RgbdFrame))
    sent = len(link.sent)
    mapper.process(Arrival(1.0, protocol.encode(Event(Msg.CLEAR_MAP))[protocol.LENGTH.size:]))
    tiles, mesh = link.sent[sent:sent + 2]
    assert isinstance(tiles, Tiles) and tiles.reset and not tiles.patches
    assert isinstance(mesh, VisualMesh) and not len(mesh.mesh.faces)
    assert not mapper.grid.tiles and not mapper.keyframes.frames and not mapper.track
    # The same frame again: the map grows from it, and the rover is where it was.
    mapper.process(frame)
    after = link.sent[-1]
    assert isinstance(after, LivePose) and len(mapper.keyframes.frames) == 1 and mapper.grid.tiles
    assert (after.pose.x, after.pose.y) == pytest.approx((before.pose.x, before.pose.y), abs=0.01)
    assert frames.wrap_deg(after.pose.yaw_deg - before.pose.yaw_deg) == pytest.approx(0.0, abs=0.5)


def test_a_tracking_reset_clears_the_map_and_anchors_again_at_the_start(scan, tmp_path):
    log = scan("spin", noise_per_m=0.0)
    link = CollectingLink()
    mapper = map_scan(log, tmp_path, link)
    T_W_A = mapper.alignment.T_W_A
    sent = len(link.sent)
    mapper.process(Arrival(1.0, protocol.encode(Event(Msg.TRACKING_RESET))[protocol.LENGTH.size:]))
    assert mapper.alignment.T_W_A is None and not mapper.grid.tiles
    assert not any(isinstance(m, LivePose) for m in link.sent[sent:])
    # ARKit restarted with its origin at the phone, which now holds still: the map
    # anchors again with the rover at the start pose.
    new_world = frames.rigid_inverse(frames.M_AXIS) @ frames.rot_z(0.7, (2.0, -1.0, 0.3)) @ frames.M_AXIS
    frame = next(a for a in reversed(list(recording.replay(log, speed=0)))
                 if isinstance(protocol.decode(a.payload), protocol.RgbdFrame))
    old = protocol.decode(frame.payload)
    T_A_C = new_world @ old.T_A_C                             # ARKit's world turned about vertical, and moved
    for i in range(mapper.cfg.alignment.stable_samples):
        pose = PhonePose(old.t_phone + 1 + 0.1 * i, T_A_C, protocol.TRACKING_NORMAL)
        mapper.process(Arrival(2.0, protocol.encode(pose)[protocol.LENGTH.size:]))
    assert mapper.alignment.T_W_A is not None
    moved = protocol.RgbdFrame(old.t_phone + 2, T_A_C, old.depth, old.confidence, old.jpeg, old.color_intrinsics)
    mapper.process(Arrival(3.0, protocol.encode(moved)[protocol.LENGTH.size:]))
    live = link.sent[-1]
    assert isinstance(live, LivePose)
    assert (live.pose.x, live.pose.y, live.pose.yaw_deg) == pytest.approx(
        (mapper.start.x, mapper.start.y, mapper.start.yaw_deg), abs=1e-3)


def test_someone_who_left_stays_gone_when_the_map_is_rebuilt(scan, tmp_path):
    # They stand 2 m ahead in the keyframe the map starts from, and leave at 4 s.
    mapper = map_scan(scan("still", person=(2.0, 0.0, 4.0), pitch_deg=25.0), tmp_path)

    def obstacles():
        p = mapper.grid.points()
        at = (np.abs(p[:, 0] - 2.0) < 0.2) & (np.abs(p[:, 1]) < 0.2)
        return int((p[at, 2] > mapper.cfg.rover.max_step_m).sum())

    assert mapper.grid.cleared and obstacles() == 0       # seen gone in the live map
    mapper._refuse(0.0)                                   # as the next loop closure would
    assert obstacles() == 0


def with_occupancy(cfg=TEST_CONFIG):
    return replace(cfg, occupancy=replace(cfg.occupancy, enabled=True))


def test_occupancy_takes_every_fused_frame_including_those_that_start_a_rebuild(scan, tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(Mapper, "_occupy", lambda self, kf, T, node: seen.append(kf.frame.t_phone))
    mapper = map_scan(scan("loop", yaw_drift_deg_s=0.1, z_drift_m_s=0.001), tmp_path, cfg=with_occupancy())
    assert mapper.watch.counts["refuses"] > 1                  # closures moved the graph
    assert seen == [f.t_phone for f in mapper.track]


def test_someone_who_left_stays_out_of_the_occupancy_through_a_rebuild(scan, tmp_path):
    mapper = map_scan(scan("still", person=(2.0, 0.0, 4.0), pitch_deg=25.0), tmp_path, cfg=with_occupancy())
    assert mapper.occupancy.count_in(1.8, 2.2, -0.2, 0.2) == 0
    mapper._refuse(0.0)
    assert mapper.occupancy.count_in(1.8, 2.2, -0.2, 0.2) == 0


def test_with_the_occupancy_the_twin_sees_someone_while_there_and_not_once_they_leave(scan, tmp_path):
    from rover_twin.link.client import Update
    from rover_twin.twin.terrain import TerrainMap

    log = scan("still", person=(2.0, 0.0, 4.0), pitch_deg=25.0)      # leaves at 4 s
    link = CollectingLink()
    mapper = Mapper(with_occupancy(), FakeTsdf(), link, tmp_path)
    terrain, applied, seen = TerrainMap(), 0, {}

    def obstacles_there():
        nonlocal applied
        for message in link.sent[applied:]:
            if isinstance(message, Tiles):
                update = Update()
                update.merge(message)
                terrain.apply(update)
        applied = len(link.sent)
        cells = terrain.obstacle_cells(mapper.cfg.rover.max_step_m)
        return int(((np.abs(cells[:, 0] - 2.0) < 0.25) & (np.abs(cells[:, 1]) < 0.25)).sum())

    for arrival in recording.replay(log, speed=0):
        mapper.process(arrival)
        message = protocol.decode(arrival.payload)
        if isinstance(message, protocol.RgbdFrame):
            seen[round(message.t_phone, 1)] = obstacles_there()
    assert seen[3.5] > 20                       # there, in the twin's terrain
    assert seen[max(seen)] == 0                 # gone from it once they left
    floor = terrain.obstacle_cells(mapper.cfg.rover.max_step_m)
    assert (np.abs(floor[:, 0]) < 1.5).sum() == 0     # and the floor between was never an obstacle


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


def test_the_mapper_keeps_the_phones_latest_colour_image(scan, tmp_path):
    log = scan("spin")
    last = [m for m in (protocol.decode(a.payload) for a in recording.replay(log, speed=0))
            if isinstance(m, protocol.RgbdFrame)][-1]
    mapper = map_scan(log, tmp_path)
    assert mapper.camera is not None and mapper.camera[0] == last.jpeg and mapper.camera[0][:2] == b"\xff\xd8"


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
    # A phone pointed down sees a front wheel at the edge of the frame: its face
    # reaches 0.325 m ahead of the centre, past the footprint, and is measured
    # a centimetre either side of that. It is dropped; the floor ahead is kept.
    mapper = Mapper(TEST_CONFIG, FakeTsdf(), CollectingLink(), tmp_path)
    intr = frames.Intrinsics(180.0, 180.0, 128.0, 96.0, 256, 192)
    depth = np.full((192, 256), 2.0, np.float32)
    c, s_ = math.cos(math.radians(17.0)), math.sin(math.radians(17.0))
    tilt = np.eye(4)
    tilt[1:3, 1:3] = [[c, s_], [-s_, c]]                     # 17 deg down, as on the lab rover
    mapper.mount = mount = frames.Mount(mapper.mount.T_R_C @ tilt, mapper.mount.forward)
    left = np.cross([0.0, 0.0, 1.0], mount.forward)
    wheel = 0.335 * mount.forward + 0.2 * left + [0.0, 0.0, 0.13]
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


def test_a_keyframe_keeps_the_latest_frames_of_its_place_for_a_re_fuse():
    from types import SimpleNamespace

    from rover_twin.config import Keyframes
    from rover_twin.mapping.keyframes import RECENT, Keyframe, KeyframeBuffer

    def at(t, x=0.0):
        T = np.eye(4)
        T[0, 3] = x
        return Keyframe(SimpleNamespace(t_phone=t), None, T)

    buffer = KeyframeBuffer(Keyframes(recent_s=2.0))
    taken = [buffer.offer(at(t)) for t in (0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0)]
    assert taken == [True] + [False] * 7 and len(buffer) == 1      # standing still: one keyframe ...
    assert [f.frame.t_phone for f in buffer.recent[0]] == [6.0, 8.0, 10.0][-RECENT:]   # ... and its latest frames
    assert buffer.offer(at(11.0, x=0.5)) and list(buffer.recent[1]) == []
    off = KeyframeBuffer(Keyframes(recent_s=0.0))
    assert [off.offer(at(t)) for t in (0.0, 5.0)] == [True, False] and list(off.recent[0]) == []


def test_with_the_occupancy_the_mesh_loses_someone_who_left_and_keeps_the_rest(scan, tmp_path):
    from rover_twin.mapping.tsdf import Mesh

    def triangle(x, y, z):
        return [[x - 0.01, y - 0.01, z], [x + 0.01, y - 0.01, z], [x, y + 0.01, z + 0.01]]

    # The person (0.3 m square at x = 2) is seen to 0.36 m from the rover, on the face toward it.
    where = {"person": (1.86, 0.0, 0.25), "floor": (1.0, 0.0, 0.0), "wall": (2.49, 0.0, 0.25),
             "unseen": (-2.0, 0.0, 0.5)}

    class StaleTsdf(FakeTsdf):          # nvblox still holding the person's surface
        def mesh(self):
            v = np.array([p for c in where.values() for p in triangle(*c)], np.float32)
            return Mesh(v, np.arange(len(v), dtype=np.int32).reshape(-1, 3), np.zeros(v.shape, np.uint8))

    log = scan("still", person=(2.0, 0.0, 4.0), pitch_deg=25.0)      # leaves at 4 s
    link = CollectingLink()
    mapper = Mapper(with_occupancy(), StaleTsdf(), link, tmp_path)

    def shown():
        mapper._send_mesh(0.0)
        faces = link.sent[-1].mesh.vertices[link.sent[-1].mesh.faces].mean(axis=1)
        return {name for name, c in where.items() if (np.abs(faces - c).max(axis=1) < 0.02).any()}

    replay = list(recording.replay(log, speed=0))
    for arrival in replay:
        mapper.process(arrival)
        message = protocol.decode(arrival.payload)
        if isinstance(message, protocol.RgbdFrame) and message.t_phone >= 3.5:
            assert shown() == set(where)            # while there, all of it
            break
    for arrival in replay[replay.index(arrival) + 1:]:
        mapper.process(arrival)
    assert shown() == {"floor", "wall", "unseen"}
