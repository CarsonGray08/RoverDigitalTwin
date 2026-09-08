"""Atomic single-pose transport across the mapper -> Isaac process boundary.

The phone's TCP socket has one owner: the mapper (``digital_twin.mapper``,
host Python + GPU), which derives the twin-frame rover pose from each camera
pose (``phone_mount.rover_pose_from_camera`` with the established ``T_align``)
and publishes it here. The Isaac validator reads it to track the physical rover.
During plan validation, the simulated rover may temporarily follow the proposed
commands; this file remains the physical pose source.

This mirrors ``mesh_tiles``' filesystem transport and atomic-write discipline:
the pose lands in the same ``--map-dir`` as ``manifest.json``/``tiles/``, and a
reader never sees a half-written file (write-to-temp + ``os.replace``). A
monotonic ``seq`` lets the consumer detect fresh updates; ``timestamp_s`` lets
it detect a stalled producer. Pure stdlib + ``Pose`` so it is unit-testable
off-GPU and off-Isaac.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

from digital_twin.models import Pose

POSE_NAME = "rover_pose.json"


def pose_path(out_dir) -> Path:
    return Path(out_dir) / POSE_NAME


def write_pose(
    out_dir,
    pose: Pose,
    *,
    seq: int,
    timestamp_s: float,
    tracking_normal: bool = True,
    z: float = 0.0,
) -> None:
    """Atomically publish the twin-frame rover ``pose`` to ``out_dir``.

    ``seq`` is a monotonic counter (consumer detects a new pose by a change);
    ``timestamp_s`` is wall-clock seconds (consumer detects a stalled producer);
    ``tracking_normal`` flags whether the upstream camera tracking was healthy
    when this pose was produced.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "x": float(pose.x),
        "y": float(pose.y),
        "yaw_deg": float(pose.yaw_deg),
        "z": float(z),
        "seq": int(seq),
        "timestamp_s": float(timestamp_s),
        "tracking_normal": bool(tracking_normal),
    }
    path = pose_path(out_dir)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as fh:
        json.dump(payload, fh)
    os.replace(tmp, path)  # atomic on POSIX: reader sees old or new, never partial


def read_pose(out_dir) -> dict | None:
    """Return the latest published pose payload, or ``None`` if absent/unreadable
    (missing file, or a transient mid-write race -- the caller retries next poll)."""
    try:
        return json.loads(pose_path(out_dir).read_text())
    except (FileNotFoundError, OSError, ValueError):
        return None


def pose_from_payload(payload: dict) -> Pose:
    """Build a :class:`Pose` from a payload returned by :func:`read_pose`."""
    return Pose(
        x=float(payload["x"]),
        y=float(payload["y"]),
        yaw_deg=float(payload["yaw_deg"]),
    )


@dataclass
class LivePoseFileProvider:
    """PoseProvider backed by ``rover_pose.json`` in a live map directory."""

    map_dir: Path
    max_age_s: float = 2.0
    require_tracking_normal: bool = True

    def current_pose(self) -> Pose | None:
        payload = read_pose(self.map_dir)
        if payload is None:
            return None
        if self.require_tracking_normal and not payload.get("tracking_normal", False):
            return None
        timestamp_s = payload.get("timestamp_s")
        if timestamp_s is not None and time.time() - float(timestamp_s) > self.max_age_s:
            return None
        return pose_from_payload(payload)
