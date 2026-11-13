"""The mapper: the phone stream in; terrain tiles, live pose and visual mesh out.

Each RGB-D frame runs these stages, timed from its arrival::

    decode -> motion gate -> keyframe buffer + pose graph -> tsdf + elevation -> tiles, then pose

Each frame carries ARKit's pose. The pose graph corrects it: in the plane
when a keyframe closes a loop, and in height wherever two keyframes see the
same floor (ARKit's height drifts a few cm over a 360; measuring it on the
floor, rather than pinning the rover to an assumed floor, keeps the terrain
the rover drives over). Separate phone poses anchor the map into the twin, once
the phone is scanning (it streams poses from launch, while still in a hand),
and say whether ARKit is tracking: while it is not, frames are neither fused nor
published as the live pose, which then ages until the drive loop stops the
rover. When the graph moves or the anchor changes (the first one, or Reset
Position), the keyframe buffer is re-fused at the corrected poses and the
twin gets the rebuilt terrain: the map moves instead of being lost. Mapping
never pauses.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Protocol

import numpy as np
from scipy import ndimage

from rover_twin import frames
from rover_twin.config import Config
from rover_twin.frames import Mount, Pose2D
from rover_twin.link.messages import LivePose, Message, Patch, Tiles, VisualMesh
from rover_twin.mapping.elevation import OBSERVED, ElevationGrid, Key, View
from rover_twin.mapping.keyframes import Keyframe, KeyframeBuffer
from rover_twin.mapping.occupancy import OccupancyGrid
from rover_twin.mapping.tsdf import Mesh, Tsdf
from rover_twin.metrics.latency import Stopwatch
from rover_twin.phone.protocol import (TRACKING_NORMAL, Arrival, Event, Msg, PhonePose, ProtocolError, RgbdFrame,
                                       decode)
from rover_twin.pose.alignment import Alignment
from rover_twin.pose.graph import PoseGraph

SAMPLE_STRIDE = 8     # pixel stride of the per-frame surface samples kept for the seam metric
GRAPH_STRIDE = 4      # pixel stride of a keyframe's surface for registration
REFUSE_M, REFUSE_DEG = 0.01, 0.2    # re-fuse when a closure moves or lifts any keyframe this much
# Around the footprint, still too close for the camera to see ground: it is the
# rover. The wheels reach 0.325 m ahead of and behind its centre, past the
# footprint, and the phone sees the front ones' faces there, with noise.
SELF_PAD_M = 0.08
EDGE_RATIO = 0.1      # farther than its nearest neighbour by this much of its depth: behind an edge
EDGE_SIZE = 5         # pixels: the neighbourhood, wide enough to see across a band of flying pixels
G_T_A = frames.M_AXIS   # the graph's frame: ARKit's world turned Z up, so it ignores the anchor
A_T_G = frames.rigid_inverse(G_T_A)
VETO_RING = 1         # cells round a cleared one whose older evidence a rebuild also drops: a closure's shift


class Link(Protocol):
    def send(self, message: Message) -> None: ...

    def snapshot_requested(self) -> bool: ...


@dataclass(frozen=True, eq=False)
class Tracked:
    """A fused frame, kept so outputs can be re-expressed at the final poses."""

    t_phone: float
    T_A_C: np.ndarray
    node: int                # the keyframe whose correction it takes
    samples: np.ndarray      # camera-frame points, float32


class Mapper:
    def __init__(self, cfg: Config, tsdf: Tsdf, link: Link, run_dir: Path, clock=time.monotonic) -> None:
        self.cfg, self.tsdf, self.link, self.run_dir = cfg, tsdf, link, run_dir
        self.clock = clock
        self.mount = Mount.from_config(cfg)
        self.start = Pose2D(cfg.start.x_m, cfg.start.y_m, cfg.start.yaw_deg)
        self.alignment = Alignment(cfg.alignment, self.mount, self.start, cfg.start.z_m)
        self.keyframes = KeyframeBuffer(cfg.keyframes)
        self.grid = ElevationGrid(cfg.elevation)
        self.occupancy = OccupancyGrid(cfg.occupancy) if cfg.occupancy.enabled else None
        self.graph = PoseGraph()
        self.track: list[Tracked] = []
        # Cells the live map saw gone: each in its keyframe's ARKit frame (x, y,
        # ground), with the keyframe and the phone time, so a rebuild puts it
        # where that keyframe's corrected pose now does.
        self._gone_at: list[tuple[np.ndarray, int, float]] = []
        self.watch = Stopwatch(clock)
        self.rev = 0
        self._last_motion: tuple[float, Pose2D] | None = None
        self._tracking = False      # the latest phone pose had normal tracking
        self._scanning = False      # RGB-D has streamed: the phone is mounted, so poses may anchor
        self._last_pose: LivePose | None = None
        self._last_mesh: VisualMesh | None = None
        self.camera: tuple[bytes, float] | None = None   # the latest colour image and when it came, for the dashboard
        self._mesh_due = clock() + cfg.tsdf.mesh_interval_s

    def run(self, arrivals: Iterable[Arrival]) -> None:
        try:
            for arrival in arrivals:
                self.process(arrival)
        finally:
            self.snapshot()

    def process(self, arrival: Arrival) -> None:
        if self.link.snapshot_requested():
            self._send_snapshot()
        try:
            message = decode(arrival.payload)
        except ProtocolError:
            self.watch.count("malformed")
            return
        if isinstance(message, RgbdFrame):
            self._scanning = True
            self.camera = (message.jpeg, time.monotonic())
            self.watch.since("decoded", arrival.t)
            self._on_frame(message, arrival.t)
        elif isinstance(message, PhonePose):
            self._tracking = message.tracking == TRACKING_NORMAL
            if self._scanning and self.alignment.offer(message):
                self._anchored(message.T_A_C)
                self._refuse(arrival.t)
        elif isinstance(message, Event) and message.kind == Msg.REALIGN:
            self.watch.count("realign")
            self.alignment.reset()
        elif isinstance(message, Event) and message.kind == Msg.CLEAR_MAP:
            self.watch.count("map_clears")
            self._clear(arrival.t, keep_anchor=True)
        elif isinstance(message, Event) and message.kind == Msg.TRACKING_RESET:
            self.watch.count("tracking_resets")
            self._clear(arrival.t, keep_anchor=False)
        elif isinstance(message, Event) and message.kind == Msg.FINALIZE:
            self.snapshot()
        # IMU samples are recorded, not used: ARKit's pose already fuses the IMU.

    def _on_frame(self, frame: RgbdFrame, t: float) -> None:
        depth = self._usable_depth(frame)
        T_A_C, T_W_A = frame.T_A_C, self.alignment.T_W_A
        if T_W_A is None:
            self.watch.count("unaligned")
            return
        if not self._tracking:          # ARKit's pose may be frozen or drifting
            self.watch.count("tracking_limited")
            return
        if not self._plausible(frame.t_phone, frames.rover_pose(T_W_A @ T_A_C, self.mount)):
            self.watch.count("implausible_motion")
            return
        keyframe = Keyframe(frame, depth, T_A_C)
        moved = False
        if self.keyframes.offer(keyframe):
            self.watch.count("keyframes")
            moved = self._add_node(keyframe)
        node = len(self.graph.estimate) - 1
        samples = frames.backproject(depth, frame.depth_intrinsics, SAMPLE_STRIDE).astype(np.float32)
        self.track.append(Tracked(frame.t_phone, T_A_C, node, samples))
        if moved:
            self._refuse(t)
            self._occupy(keyframe, self._world(T_A_C, node), node)    # this frame too, at its corrected pose
            if changed := self._occupancy_tiles():
                self._send_tiles(changed, t, reset=False)
            return
        T_W_C = self._world(T_A_C, node)
        cleared = self.grid.cleared
        dirty = self._fuse(keyframe, T_W_C)
        self._occupy(keyframe, T_W_C, node)
        dirty |= self._occupancy_tiles()
        if self.grid.cleared > cleared and self.keyframes.changed:
            self.keyframes.changed[node] = True
            self._remember_gone(node, frame.t_phone)
        rover = frames.rover_pose(T_W_C, self.mount)
        self.watch.since("fused", t)
        if dirty:
            self._send_tiles(dirty, t, reset=False)
            self.watch.since("tiles_sent", t)
        self._last_pose = LivePose(self.rev, t, frame.t_phone, rover, self._ground_z(T_W_C))
        self.link.send(self._last_pose)
        self.watch.since("pose_sent", t)
        if self.clock() >= self._mesh_due:
            self._send_mesh(t)

    def _add_node(self, kf: Keyframe) -> bool:
        """Add a keyframe to the pose graph; True if a closure moved the graph
        enough to re-fuse. Its surface goes in the graph's frame, with heights
        in the twin frame, where the floor is near 0, so walls can be told from floor."""
        t0 = self.clock()
        T_G_C = G_T_A @ kf.T_A_C
        rover = frames.rover_pose(T_G_C, self.mount)
        camera = frames.backproject(kf.depth, kf.frame.depth_intrinsics, GRAPH_STRIDE)
        points = frames.transform(T_G_C, camera)
        points[:, 2] = frames.transform(self.alignment.T_W_A @ kf.T_A_C, camera)[:, 2]
        before, heights = np.array(self.graph.estimate), np.array(self.graph.heights)
        closed = self.graph.add(np.array([rover.x, rover.y, math.radians(rover.yaw_deg)]), points)
        self.watch.took("graph", self.clock() - t0)
        lifted = np.abs(np.array(self.graph.heights[:len(heights)]) - heights)
        if not closed:
            return bool((lifted > REFUSE_M).any())
        self.watch.count("closures")
        shift = np.abs(np.array(self.graph.estimate[:len(before)]) - before)
        return bool((shift[:, :2] > REFUSE_M).any() or (np.degrees(shift[:, 2]) > REFUSE_DEG).any()
                    or (lifted > REFUSE_M).any())

    def _world(self, T_A_C: np.ndarray, node: int) -> np.ndarray:
        """``T_W_C`` corrected by keyframe ``node``'s graph estimate, in the plane and in height."""
        return self._W_T_A(node) @ T_A_C

    def _W_T_A(self, node: int) -> np.ndarray:
        """ARKit's world in the twin, as keyframe ``node``'s correction places it."""
        d = self.graph.correction(node)
        return self.alignment.T_W_A @ A_T_G @ frames.rot_z(d[2], (d[0], d[1], self.graph.height(node))) @ G_T_A

    def _remember_gone(self, node: int, t_phone: float) -> None:
        cells, ground = self.grid.last_cleared
        world = np.column_stack([(cells + 0.5) * self.cfg.elevation.cell_m, ground])
        self._gone_at.append((frames.transform(frames.rigid_inverse(self._W_T_A(node)), world), node, t_phone))

    def _veto(self) -> None:
        """Hand the grid every cell seen gone, where the graph now puts it, and
        the ring round it, so re-fusing older frames does not bring it back."""
        if not self._gone_at:
            self.grid.set_veto(np.zeros((0, 2), np.int64), np.zeros(0), np.zeros(0))
            return
        ring = np.stack(np.mgrid[-VETO_RING:VETO_RING + 1, -VETO_RING:VETO_RING + 1], -1).reshape(-1, 2)
        cells, times, ground = [], [], []
        for points, node, t in self._gone_at:
            world = frames.transform(self._W_T_A(node), points)
            at = np.floor(world[:, :2] / self.cfg.elevation.cell_m).astype(np.int64)
            cells.append((at[:, None, :] + ring).reshape(-1, 2))
            times.append(np.full(len(at) * len(ring), t))
            ground.append(np.repeat(world[:, 2], len(ring)))
        self.grid.set_veto(np.concatenate(cells), np.concatenate(times), np.concatenate(ground))

    def _fuse(self, kf: Keyframe, T_W_C: np.ndarray) -> set[Key]:
        intr = kf.frame.depth_intrinsics
        t0 = self.clock()
        self.tsdf.integrate(kf.depth, intr, T_W_C, kf.frame.jpeg, kf.frame.color_intrinsics)
        t1 = self.clock()
        points = frames.transform(T_W_C, frames.backproject(kf.depth, intr))
        if self.occupancy is not None:          # the grid is the ground; obstacles are the occupancy's
            points = points[points[:, 2] < self.cfg.occupancy.z_min_m]
        view = View(kf.frame.depth, kf.frame.confidence, intr, T_W_C)     # all it measured, to see past
        dirty = self.grid.integrate(points, T_W_C[:3, 3], view, kf.frame.t_phone)
        dirty |= self._infer_under(T_W_C)
        dirty |= self.grid.infer_footprint((self.start.x, self.start.y), self.cfg.start.z_m)
        self.watch.took("tsdf", t1 - t0)
        self.watch.took("elevation", self.clock() - t1)
        return dirty

    def _infer_under(self, T_W_C: np.ndarray) -> set[Key]:
        """The rover stands on ground no sensor sees. Where the ring around it
        is not seen yet (a turn in place drifts the centre off the start
        footprint), its own wheels give the height, but only on the start's
        floor: a phone carried in a hand is not standing on anything."""
        rover = frames.rover_pose(T_W_C, self.mount)
        ground_z = self._ground_z(T_W_C)
        on_floor = abs(ground_z - self.cfg.start.z_m) < self.cfg.elevation.step_m
        return self.grid.infer_footprint((rover.x, rover.y), ground_z if on_floor else None)

    def _anchored(self, T_A_C: np.ndarray) -> None:
        """Every rover pose from here on uses the mount as measured at the anchor."""
        self.mount = self.alignment.mount
        R = (G_T_A @ T_A_C)[:3, :3]
        pitch, roll = math.degrees(math.asin(-R[2, 2])), math.degrees(math.asin(R[2, 0]))
        self.watch.values["mount_pitch_deg"], self.watch.values["mount_roll_deg"] = round(pitch, 2), round(roll, 2)
        print(f"mapper: map anchored; phone tilted {pitch:.1f} deg down, {roll:.1f} deg roll", flush=True)

    def _ground_z(self, T_W_C: np.ndarray) -> float:
        """The ground under the rover: the rover frame's origin is on it."""
        return float((T_W_C @ self.mount.T_C_R)[2, 3])

    def _refuse(self, t: float) -> None:
        """Rebuild every output at the current anchor from the keyframe buffer:
        terrain, then the visual mesh and the live pose, so the twin never
        holds a pose or mesh from the old anchor next to the new terrain."""
        t0 = self.clock()
        self.tsdf.clear()
        self.grid.clear()
        self._veto()
        self._move_occupancy()
        for k, kf in enumerate(self.keyframes):
            recent = self.keyframes.recent[k] if self.keyframes.changed[k] else ()
            for frame in (kf, *recent):        # then what its place looks like now, if it changed
                self._fuse(frame, self._world(frame.T_A_C, k))
        for k, kf in enumerate(self.keyframes):    # where it stood, now that the rings around are seen
            self._infer_under(self._world(kf.T_A_C, k))
        self._last_motion = None
        self._send_tiles(self._all_tiles(), t, reset=True)
        self._send_mesh(t)
        if self.track:
            last = self.track[-1]
            T_W_C = self._world(last.T_A_C, last.node)
            self._last_pose = LivePose(self.rev, t, last.t_phone, frames.rover_pose(T_W_C, self.mount),
                                       self._ground_z(T_W_C))
            self.link.send(self._last_pose)
        self.watch.took("refuse", self.clock() - t0)
        self.watch.count("refuses")

    def _clear(self, t: float, keep_anchor: bool) -> None:
        """Forget the map: keyframes, pose graph, terrain and mesh. The twin
        gets an empty terrain and mesh, and the map grows again from the next
        frame. Keeping the anchor, the latest graph correction moves into it,
        so the live pose does not jump. Without it (ARKit restarted, so every
        old pose is in another world), the map anchors again once the phone
        holds still, and until then there is no live pose."""
        if keep_anchor and self.alignment.T_W_A is not None and self.graph.estimate:
            d = self.graph.correction()
            correction = frames.rot_z(d[2], (d[0], d[1], self.graph.height()))
            self.alignment.T_W_A = self.alignment.T_W_A @ A_T_G @ correction @ G_T_A
        elif not keep_anchor:
            self.alignment.reset()
            self._last_pose = None
        self.keyframes = KeyframeBuffer(self.cfg.keyframes)
        self.graph = PoseGraph()
        self.track.clear()
        self._gone_at.clear()
        self._veto()
        if self.occupancy is not None:
            self.occupancy.clear()
        self.tsdf.clear()
        self.grid.clear()
        self._last_motion = None
        self._send_tiles(set(), t, reset=True)
        self._last_mesh = None
        self.link.send(VisualMesh(t, Mesh.empty()))
        print(f"mapper: map cleared{'' if keep_anchor else ' for a tracking reset: hold the rover still'}",
              flush=True)

    def _plausible(self, t_phone: float, rover: Pose2D) -> bool:
        """Reject a pose implying motion the rover cannot make: a tracking jump
        that would otherwise be fused into the map."""
        if self._last_motion is not None:
            t0, last = self._last_motion
            dt = max(t_phone - t0, 0.05)
            odo = self.cfg.odometry
            if (math.hypot(rover.x - last.x, rover.y - last.y) / dt > odo.max_speed_mps
                    or abs(frames.wrap_deg(rover.yaw_deg - last.yaw_deg)) / dt > odo.max_yaw_rate_deg_s):
                return False
        self._last_motion = (t_phone, rover)
        return True

    def _move_occupancy(self) -> None:
        """Put the occupancy where the graph (or the anchor) now puts each keyframe."""
        if self.occupancy is not None and self.graph.estimate:
            t0 = self.clock()
            self.occupancy.move([self._W_T_A(k) for k in range(len(self.graph.estimate))])
            self.watch.took("occupancy_move", self.clock() - t0)

    def _occupy(self, keyframe: Keyframe, T_W_C: np.ndarray, node: int) -> None:
        """Fuse a live frame into the occupancy (none during a rebuild: it moves instead)."""
        if self.occupancy is None:
            return
        t0 = self.clock()
        frame = keyframe.frame
        hits = frames.transform(T_W_C, frames.backproject(keyframe.depth, frame.depth_intrinsics))
        self.occupancy.integrate(hits, self._free_depth(frame), frame.depth_intrinsics, T_W_C, node,
                                 self._W_T_A(node), frame.t_phone)
        self.watch.took("occupancy", self.clock() - t0)

    def _free_depth(self, frame: RgbdFrame) -> np.ndarray:
        """Depth a pixel may see past an occupancy voxel with: measured, of the
        configured confidence, in range, and not behind a depth edge (a flying
        pixel lies between the surfaces, beyond the nearer one)."""
        depth = frame.depth
        measured = np.isfinite(depth) & (depth > 0)
        nearest = ndimage.minimum_filter(np.where(measured, depth, np.inf), size=EDGE_SIZE)
        ok = measured & (depth <= self.cfg.depth.max_range_m) & ~(depth - nearest > EDGE_RATIO * depth)
        if frame.confidence is not None:
            ok &= frame.confidence >= self.cfg.occupancy.free_confidence
        return np.where(ok, depth, 0.0).astype(np.float32)

    def _usable_depth(self, frame: RgbdFrame) -> np.ndarray:
        depth = frame.depth
        measured = np.isfinite(depth) & (depth > 0)
        # Flying pixels: at a depth edge (a wheel against the floor behind it)
        # the sensor reports depths in between, high-confidence, which land as
        # bumps beside the object. They are farther than the object's own edge,
        # which is kept, so the edge is found in every measured depth: a dark
        # tyre's own pixels are low-confidence. Distant floor seen at a grazing
        # angle steps less than EDGE_RATIO from one pixel to the next and is kept.
        nearest = ndimage.minimum_filter(np.where(measured, depth, np.inf), size=EDGE_SIZE)
        bad = ~measured | (depth > self.cfg.depth.max_range_m) | (depth - nearest > EDGE_RATIO * depth)
        if frame.confidence is not None:
            bad |= frame.confidence < self.cfg.depth.min_confidence
        depth = np.where(bad, 0.0, depth)
        return np.where(self._on_rover(depth, frame.depth_intrinsics), 0.0, depth).astype(np.float32)

    def _on_rover(self, depth: np.ndarray, intr) -> np.ndarray:
        """Pixels that see the rover itself, such as a wheel at the edge of a
        phone pointed down: in plan inside its footprint (and SELF_PAD_M),
        ground the camera cannot see from where it is mounted. Fixed in the
        rover frame, so no pose is needed."""
        v, u = np.mgrid[0:depth.shape[0], 0:depth.shape[1]]
        camera = np.stack([(u - intr.cx) / intr.fx * depth, (v - intr.cy) / intr.fy * depth, depth], -1)
        body = frames.transform(self.mount.T_R_C, camera.reshape(-1, 3))
        forward = self.mount.forward
        left = np.cross([0.0, 0.0, 1.0], forward)
        rc = self.cfg.rover
        inside = ((np.abs(body @ forward) < rc.footprint_half_length_m + SELF_PAD_M)
                  & (np.abs(body @ left) < rc.footprint_half_width_m + SELF_PAD_M))
        return inside.reshape(depth.shape) & (depth > 0)

    def _patch(self, key: Key) -> tuple[np.ndarray, np.ndarray]:
        """A tile's collider patch: the ground, and over it, with the occupancy
        on, the top of whatever is occupied above each cell."""
        height, state = self.grid.patch(key)
        if self.occupancy is None:
            return height, state
        k, cell = self.grid.n + 1, self.cfg.elevation.cell_m
        ij = np.asarray(key) * self.grid.n + np.stack(np.mgrid[0:k, 0:k], -1).reshape(-1, 2)
        top = self.occupancy.tops((ij + 0.5) * cell).reshape(k, k)
        over = np.isfinite(top)
        height = np.where(over, np.fmax(height, top), height).astype(np.float32)
        return height, np.where(over, OBSERVED, state).astype(np.uint8)

    def _terrain(self) -> ElevationGrid:
        """The terrain as the twin has it: with the occupancy on, the ground
        and what is occupied over it (the grid itself holds only the ground)."""
        if self.occupancy is None:
            return self.grid
        terrain, n = ElevationGrid(self.cfg.elevation), self.grid.n
        keys = set(self.grid.tiles) | self._occupancy_tiles(self.occupancy.columns()[0])
        for key in keys:
            height, state = self._patch(key)
            weight = self.grid.tiles[key][1] if key in self.grid.tiles else np.ones((n, n))
            terrain.tiles[key] = (np.nan_to_num(height[:n, :n]), weight, state[:n, :n])
        return terrain

    def _occupancy_tiles(self, columns: np.ndarray | None = None) -> set[Key]:
        """Tiles whose patches cover the occupancy columns that changed (or ``columns``)."""
        if self.occupancy is None:
            return set()
        if columns is None:
            columns = self.occupancy.changed()
        if not len(columns):
            return set()
        s, cell, n = self.cfg.occupancy.voxel_m, self.cfg.elevation.cell_m, self.grid.n
        lo = np.floor(columns * s / cell + 1e-6).astype(np.int64) - 1    # -1: the apron of the patch before
        hi = np.ceil((columns + 1) * s / cell - 1e-6).astype(np.int64) - 1
        keys = {(int(a), int(b)) for x0, y0, x1, y1 in np.column_stack([lo, hi]) // n
                for a in range(x0, x1 + 1) for b in range(y0, y1 + 1)}
        return keys

    def _all_tiles(self) -> set[Key]:
        """Every tile with ground or an occupied column in it."""
        if self.occupancy is None:
            return set(self.grid.tiles)
        self.occupancy.settle()                     # all of it is about to be sent
        return set(self.grid.tiles) | self._occupancy_tiles(self.occupancy.columns()[0])

    def _send_tiles(self, keys: set[Key], t: float, reset: bool) -> None:
        self.rev += 1
        patches = tuple(Patch(k, *self._patch(k)) for k in sorted(keys))
        self.link.send(Tiles(self.rev, t, reset, self.cfg.elevation.cell_m, self.grid.n, patches))

    def _send_mesh(self, t: float) -> None:
        t0 = self.clock()
        mesh = self.tsdf.mesh()
        if self.occupancy is not None and len(mesh.faces):
            # nvblox is slow to carve away someone who left: drop the faces
            # where the occupancy has since seen empty, as the terrain does.
            mesh = mesh.keep(~self.occupancy.seen_empty(mesh.vertices[mesh.faces].mean(axis=1)))
        self.watch.took("mesh_extract", self.clock() - t0)
        self._mesh_due = self.clock() + self.cfg.tsdf.mesh_interval_s
        self._last_mesh = VisualMesh(t, mesh) if len(mesh.faces) else None
        if self._last_mesh is not None:
            self.link.send(self._last_mesh)
            self.watch.since("mesh_sent", t)

    def _send_snapshot(self) -> None:
        """Everything a newly connected twin needs, in stream order, stamped
        with the arrival of the newest frame behind it."""
        t = self._last_pose.t_arrival if self._last_pose is not None else self.clock()
        self._send_tiles(self._all_tiles(), t, reset=True)
        for message in (self._last_mesh, self._last_pose):
            if message is not None:
                self.link.send(message)

    def snapshot(self) -> None:
        """Write the run's outputs, expressed at the final anchor and poses."""
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.watch.write(self.run_dir / "metrics.json")
        if self.alignment.T_W_A is None or not self.track:
            return
        self._terrain().save(self.run_dir / "elevation.npz")
        with open(self.run_dir / "trajectory.jsonl", "w") as fh:
            for f in self.track:
                T_W_C = self._world(f.T_A_C, f.node)
                rover = frames.rover_pose(T_W_C, self.mount)
                fh.write(json.dumps({"t_phone": f.t_phone, "camera": T_W_C.ravel().round(6).tolist(),
                                     "rover": [rover.x, rover.y, rover.yaw_deg]}) + "\n")
        points = [frames.transform(self._world(f.T_A_C, f.node), f.samples) for f in self.track]
        np.savez_compressed(self.run_dir / "samples.npz", points=np.concatenate(points).astype(np.float32),
                            frames=np.repeat(np.arange(len(points)), [len(p) for p in points]))
        mesh = self.tsdf.mesh()
        if len(mesh.faces):
            np.savez_compressed(self.run_dir / "mesh.npz", vertices=mesh.vertices, faces=mesh.faces,
                                colors=mesh.colors)
