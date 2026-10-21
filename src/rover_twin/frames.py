"""Coordinate frames and camera geometry shared by the mapper and the twin.

Frames (right-handed, metres):

- ``A``  ARKit world, Y up: the phone's visual-inertial odometry reports
  camera poses in this frame.
- ``W``  twin world, Z up: the Isaac stage.
- ``C``  camera in CV axes (+x right, +y down, +z forward), as nvblox
  expects. The phone's own camera axes (+y up, -z forward) are converted
  away at the wire boundary and in the mount calibration; nothing downstream
  sees them.
- ``R``  rover body.

``T_X_Y`` maps points from ``Y`` into ``X`` (column vectors, ``p_X = T_X_Y @ p_Y``).
The whole map reaches the twin through one rigid alignment ``T_W_A``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from rover_twin import config

# ARKit world (Y up) -> twin world (Z up): (x, y, z) -> (x, -z, y).
M_AXIS = np.array(
    [[1.0, 0.0, 0.0, 0.0],
     [0.0, 0.0, -1.0, 0.0],
     [0.0, 1.0, 0.0, 0.0],
     [0.0, 0.0, 0.0, 1.0]]
)

# CV camera axes expressed in the phone's camera axes (flip Y and Z). Its own
# inverse.
PHONE_T_CV = np.diag([1.0, -1.0, -1.0, 1.0])

_FORWARD_AXES = {
    "x": (1.0, 0.0, 0.0), "-x": (-1.0, 0.0, 0.0),
    "y": (0.0, 1.0, 0.0), "-y": (0.0, -1.0, 0.0),
}


@dataclass(frozen=True)
class Pose2D:
    """Planar rover pose in the twin frame."""

    x: float
    y: float
    yaw_deg: float


@dataclass(frozen=True)
class Intrinsics:
    """Pinhole intrinsics for an image of ``width`` x ``height`` pixels."""

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    def scaled(self, width: int, height: int) -> "Intrinsics":
        sx, sy = width / self.width, height / self.height
        return Intrinsics(self.fx * sx, self.fy * sy, self.cx * sx, self.cy * sy, width, height)


def from_wire(flat16) -> np.ndarray:
    """The phone sends ``simd_float4x4`` in memory order, which is column-major."""
    return np.asarray(flat16, dtype=np.float64).reshape(4, 4).T


def to_wire(T: np.ndarray) -> tuple[float, ...]:
    return tuple(float(v) for v in np.asarray(T).T.reshape(16))


def phone_camera_pose(flat16) -> np.ndarray:
    """A pose from the wire (phone camera axes, frame A) as ``T_A_C``."""
    return from_wire(flat16) @ PHONE_T_CV


def rigid_inverse(T: np.ndarray) -> np.ndarray:
    out = np.eye(4)
    out[:3, :3] = T[:3, :3].T
    out[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return out


def rot_z(yaw_rad: float, translation=(0.0, 0.0, 0.0)) -> np.ndarray:
    c, s = math.cos(yaw_rad), math.sin(yaw_rad)
    T = np.eye(4)
    T[:2, :2] = [[c, -s], [s, c]]
    T[:3, 3] = translation
    return T


def rotation_wxyz(q) -> np.ndarray:
    w, x, y, z = np.asarray(q, dtype=np.float64) / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def wrap_deg(angle: float) -> float:
    return (angle + 180.0) % 360.0 - 180.0


@dataclass(frozen=True)
class Mount:
    """The phone's fixed pose on the rover: ``T_R_C`` and the driving direction."""

    T_R_C: np.ndarray
    forward: np.ndarray

    @classmethod
    def from_config(cls, cfg: config.Config) -> "Mount":
        """The phone camera pose is measured in the phone's camera axes; it is
        converted to CV axes here."""
        T_R_phone = np.eye(4)
        T_R_phone[:3, :3] = rotation_wxyz(cfg.mount.quaternion_wxyz)
        T_R_phone[:3, 3] = cfg.mount.translation_m
        return cls(T_R_phone @ PHONE_T_CV, forward_vector(cfg.rover.forward_axis))

    @property
    def T_C_R(self) -> np.ndarray:
        return rigid_inverse(self.T_R_C)


def forward_vector(axis: str) -> np.ndarray:
    return np.array(_FORWARD_AXES[axis])


def rover_pose(T_W_C: np.ndarray, mount: Mount) -> Pose2D:
    """Planar rover pose from a camera pose in the twin frame."""
    T_W_R = T_W_C @ mount.T_C_R
    forward = T_W_R[:3, :3] @ mount.forward
    yaw = math.degrees(math.atan2(forward[1], forward[0]))
    return Pose2D(float(T_W_R[0, 3]), float(T_W_R[1, 3]), wrap_deg(yaw))


def planar_alignment(T_A_C: np.ndarray, mount: Mount, start: Pose2D, start_z: float = 0.0) -> np.ndarray:
    """``T_W_A`` that puts the rover seen at ``T_A_C`` exactly at ``start``.

    Both worlds are gravity-aligned once the axes are swapped, so the only
    freedom is heading and position: the alignment is a yaw plus translation
    and never tilts the map from one noisy frame.
    """
    S = M_AXIS @ T_A_C @ mount.T_C_R
    forward = S[:3, :3] @ mount.forward
    dyaw = math.radians(start.yaw_deg) - math.atan2(forward[1], forward[0])
    rotated = rot_z(dyaw)[:3, :3] @ S[:3, 3]
    return rot_z(dyaw, (start.x - rotated[0], start.y - rotated[1], start_z - rotated[2])) @ M_AXIS


def backproject(depth: np.ndarray, intr: Intrinsics, stride: int = 1) -> np.ndarray:
    """Camera-frame points (N, 3) for the valid (> 0, finite) depth pixels."""
    d = depth[::stride, ::stride]
    v, u = np.nonzero(np.isfinite(d) & (d > 0.0))
    z = d[v, u].astype(np.float64)
    return np.column_stack([(u * stride - intr.cx) / intr.fx * z, (v * stride - intr.cy) / intr.fy * z, z])


def transform(T: np.ndarray, points: np.ndarray) -> np.ndarray:
    return points @ T[:3, :3].T + T[:3, 3]
