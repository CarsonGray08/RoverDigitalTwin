"""Synthetic scans: ray-cast depth of a known room, streamed as the phone would.

Until the benchmark recordings exist, this is how replay tests exercise the
mapper and its metrics end to end. The rover follows a path in a walled room
with boxes on the floor; the mounted phone's depth is rendered exactly, and
the pose it reports can be given a yaw drift to show the metrics catch it.
Writes the recording and its ground truth, ``<name>.truth.toml``, in the same
format a tape-measured benchmark recording uses.

usage: python tools/synthetic_scan.py {spin,drive,loop} OUT.scanlog [--yaw-drift DEG_PER_S]
"""

from __future__ import annotations

import argparse
import io
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from rover_twin import config, frames
from rover_twin.frames import M_AXIS, Intrinsics, Mount, Pose2D
from rover_twin.phone import protocol
from rover_twin.phone.protocol import Arrival, Event, Msg, PhonePose, RgbdFrame
from rover_twin.phone.recording import Recorder

ROOM = (np.array([-2.5, -3.0, 0.0]), np.array([2.5, 3.0, 2.4]))   # inner walls; floor at z = 0


@dataclass(frozen=True)
class Box:
    name: str
    center: tuple[float, float]
    size: tuple[float, float, float]     # length, width, height
    yaw_deg: float


BOXES = (
    Box("crate", (1.4, -1.2), (0.40, 0.30, 0.25), 20.0),
    Box("bin", (-1.2, 1.4), (0.30, 0.30, 0.40), 0.0),
    Box("step", (-1.0, -1.5), (0.60, 0.40, 0.12), -35.0),
)

# The ARKit session started somewhere else: its world is yawed, offset and
# 0.9 m below the phone's starting height relative to the twin.
T_W_A_TRUE = frames.rot_z(0.6, (0.4, -0.3, 0.9)) @ M_AXIS
HOLD_S = 1.2     # stationary start, long enough for the alignment gate


def knots(kind: str) -> list[tuple[float, float, float, float]]:
    """(t, x, y, unwrapped yaw_deg) knots of the rover path, linear in between."""
    path = [(0.0, 0.0, 0.0, 0.0), (HOLD_S, 0.0, 0.0, 0.0)]

    def go(seconds, x, y, yaw):
        path.append((path[-1][0] + seconds, x, y, yaw))

    if kind == "spin":
        go(12.0, 0.0, 0.0, 360.0)          # 30 deg/s turn in place
    elif kind == "drive":
        go(6.4, 1.6, 0.0, 0.0)             # 0.25 m/s
    elif kind == "loop":
        for i, (x, y) in enumerate([(1.2, 0.0), (1.2, 1.2), (0.0, 1.2), (0.0, 0.0)]):
            go(4.8, x, y, 90.0 * i)
            go(3.0, x, y, 90.0 * (i + 1))
    else:
        raise ValueError(kind)
    go(0.6, *path[-1][1:])
    return path


def rover_at(path, t: float) -> Pose2D:
    k = np.asarray(path)
    return Pose2D(*(float(np.interp(t, k[:, 0], k[:, i])) for i in (1, 2, 3)))


def render(T_W_C: np.ndarray, intr: Intrinsics, rng, noise_per_m: float, max_range_m: float = 5.0) -> np.ndarray:
    """Z-depth of the room and boxes seen from ``T_W_C`` (CV axes)."""
    v, u = np.mgrid[0:intr.height, 0:intr.width]
    rays = np.stack([(u - intr.cx) / intr.fx, (v - intr.cy) / intr.fy, np.ones(u.shape)], -1).reshape(-1, 3)
    d = rays @ T_W_C[:3, :3].T     # z component of each camera ray is 1, so t is z-depth
    o = T_W_C[:3, 3]
    t = _slab(o, d, *ROOM)[1]
    for box in BOXES:
        R = frames.rot_z(-math.radians(box.yaw_deg))[:3, :3]
        half = np.array([box.size[0] / 2, box.size[1] / 2, box.size[2]])
        near, far = _slab(R @ (o - [*box.center, 0.0]), d @ R.T, -half * [1, 1, 0], half)
        t = np.where((near <= far) & (near > 0) & (near < t), near, t)
    t = t + rng.normal(0.0, noise_per_m * t)
    return np.where(t < max_range_m, t, 0.0).reshape(intr.height, intr.width).astype(np.float32)


def _slab(o, d, lo, hi):
    with np.errstate(divide="ignore", invalid="ignore"):
        a, b = (lo - o) / d, (hi - o) / d
    return np.fmax.reduce(np.fmin(a, b), axis=1), np.fmin.reduce(np.fmax(a, b), axis=1)


def write(kind: str, path: str | Path, *, yaw_drift_deg_s: float = 0.0, z_drift_m_s: float = 0.0, width: int = 256,
          rgbd_hz: float = 10.0, noise_per_m: float = 0.01, seed: int = 0, pitch_deg: float = 0.0) -> Path:
    """Write ``path`` and its truth file; returns the truth file's path.
    ``pitch_deg`` tilts the phone down on its mount from the configured level."""
    path = Path(path)
    lab = config.load(Path(__file__).parents[1] / "config" / "rover_twin.toml")
    mount = Mount.from_config(lab)
    if pitch_deg:      # about the camera's own x axis, at the same mounting point: +z toward +y (down)
        c, s_ = math.cos(math.radians(pitch_deg)), math.sin(math.radians(pitch_deg))
        tilt = np.eye(4)
        tilt[1:3, 1:3] = [[c, s_], [-s_, c]]
        mount = Mount(mount.T_R_C @ tilt, mount.forward)
    rng = np.random.default_rng(seed)
    s = width / 256
    depth_intr = Intrinsics(210.0 * s, 210.0 * s, 128.0 * s, 96.0 * s, width, width * 3 // 4)
    color_intr = depth_intr.scaled(2 * width, 2 * depth_intr.height)
    jpeg = io.BytesIO()
    Image.new("L", (color_intr.width, color_intr.height), 128).save(jpeg, "JPEG")
    path_knots = knots(kind)
    end = path_knots[-1][0]
    # Odometry with a drifting heading integrates every step of the true
    # motion in the wrong direction, so position error grows along the path.
    fine = np.arange(0.0, end + 0.002, 0.001)
    true_xy = np.array([[(p := rover_at(path_knots, t)).x, p.y] for t in fine])
    heading_error = np.radians(yaw_drift_deg_s * np.maximum(0.0, fine - HOLD_S))
    c, s_ = np.cos(heading_error[1:]), np.sin(heading_error[1:])
    step = np.diff(true_xy, axis=0)
    drifted_xy = np.vstack([true_xy[:1], true_xy[0] + np.cumsum(
        np.column_stack([c * step[:, 0] - s_ * step[:, 1], s_ * step[:, 0] + c * step[:, 1]]), axis=0)])

    def camera(x: float, y: float, yaw_deg: float) -> np.ndarray:
        body = math.radians(yaw_deg) - math.atan2(mount.forward[1], mount.forward[0])
        return frames.rot_z(body, (x, y, 0.0)) @ mount.T_R_C

    def reported(t: float) -> tuple[np.ndarray, np.ndarray]:
        """(true T_W_C, the phone's T_A_C as its drifting odometry sees it)."""
        rover = rover_at(path_knots, t)
        x, y = (np.interp(t, fine, drifted_xy[:, i]) for i in (0, 1))
        drifted = camera(x, y, rover.yaw_deg + math.degrees(np.interp(t, fine, heading_error)))
        drifted[2, 3] += z_drift_m_s * max(0.0, t - HOLD_S)          # odometry's height drifts too
        return camera(rover.x, rover.y, rover.yaw_deg), frames.rigid_inverse(T_W_A_TRUE) @ drifted

    messages = [(t, False) for t in np.arange(0.0, end, 0.2)]                # phone pose, 5 Hz
    messages += [(t, True) for t in np.arange(0.05, end, 1.0 / rgbd_hz)]     # RGB-D frames
    recorder = Recorder(path)
    for t, is_rgbd in sorted(messages):
        T_W_C, T_A_C = reported(t)
        if not is_rgbd:
            message = PhonePose(t, T_A_C, protocol.TRACKING_NORMAL)
        else:
            depth = render(T_W_C, depth_intr, rng, noise_per_m)
            message = RgbdFrame(t, T_A_C, depth, np.full(depth.shape, 2, np.uint8), jpeg.getvalue(), color_intr)
        recorder.write(Arrival(t, protocol.encode(message)[protocol.LENGTH.size:]))
    recorder.write(Arrival(end, protocol.encode(Event(Msg.FINALIZE, end))[protocol.LENGTH.size:]))
    recorder.close()

    final = rover_at(path_knots, end)
    truth = path.with_name(path.stem + ".truth.toml")
    lines = [f'kind = "{kind}"', "", "[end]", f"x_m = {final.x}", f"y_m = {final.y}",
             f"yaw_deg = {frames.wrap_deg(final.yaw_deg)}"]
    for box in BOXES if kind != "drive" else ():     # a straight drive never faces them all
        lines += ["", "[[boxes]]", f'name = "{box.name}"', f"center_m = {list(box.center)}",
                  f"size_m = {list(box.size)}"]
    truth.write_text("\n".join(lines) + "\n")
    return truth


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("kind", choices=("spin", "drive", "loop"))
    parser.add_argument("out", type=Path)
    parser.add_argument("--yaw-drift", type=float, default=0.0, help="reported yaw drift, deg/s")
    parser.add_argument("--z-drift", type=float, default=0.0, help="reported height drift, m/s")
    parser.add_argument("--width", type=int, default=256, help="depth image width (4:3)")
    args = parser.parse_args()
    print(write(args.kind, args.out, yaw_drift_deg_s=args.yaw_drift, z_drift_m_s=args.z_drift, width=args.width))
