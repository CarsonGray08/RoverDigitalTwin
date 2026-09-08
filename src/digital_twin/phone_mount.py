"""Phone-mount calibration: convert a streamed phone camera pose into the
rover-body pose in the twin frame.

Coordinate frames:

- ``A``  ARKit world (right-handed, Y-up, metres).
- ``C``  phone camera frame.
- ``R``  rover body frame.
- ``W``  twin/Isaac world (Z-up).

The runtime conversion is ``T_W_R = T_W_A * T_A_C * T_C_R`` with
``T_W_A = M_axis * T_align``. ``M_axis`` is the fixed ARKit-Y-up -> twin-Z-up
remap (the same ``(x, y, z) -> (x, -z, y)`` swap the live-scan mesh path already
applies); ``T_align`` is the rigid world alignment computed from the initial
phone pose and the rover's starting pose. Conversion helpers use identity
alignment when no ``t_align`` is supplied.

``T_C_R`` is the fixed mount calibration. The user measures the *camera pose in
the rover body frame* (``T_R_C``): its translation is the lever arm (camera
offset from the rover centre) and its rotation is the camera orientation on the
rover. We invert it to get ``T_C_R``.

All matrices here use the **column-vector** convention (``p' = M @ p``, 4x4 as a
tuple of 4 rows). The streamed transform from ``live_scan`` is stored row-major /
row-vector (``p' = p @ H``); :func:`mat4_from_host_transform` transposes it into
this convention at the boundary.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from digital_twin.geometry import normalize_yaw_deg, yaw_error_deg
from digital_twin.models import Pose

# A 4x4 matrix as a tuple of 4 rows, column-vector convention.
Mat4 = tuple[tuple[float, float, float, float], ...]
Vec3 = tuple[float, float, float]

_ORTHONORMAL_TOL = 1e-4
_DET_TOL = 1e-3

# Fixed ARKit-Y-up -> twin-Z-up axis remap, as a column-vector matrix:
#   M_axis @ (x, y, z, 1) = (x, -z, y, 1)
# This matches live_scan.arkit_world_to_usd_point. It is a proper rotation
# (det +1): ARKit and the twin are both right-handed, only the up-axis differs.
M_AXIS: Mat4 = (
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, -1.0, 0.0),
    (0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 1.0),
)

# Camera-axis basis change used by nvblox_feeder: CV/nvblox camera coords
# (+x right, +y down, +z forward) expressed in ARKit/phone camera coords
# (+x right, +y up, -z forward). It is its own inverse.
CV_TO_ARKIT_CAM: Mat4 = (
    (1.0, 0.0, 0.0, 0.0),
    (0.0, -1.0, 0.0, 0.0),
    (0.0, 0.0, -1.0, 0.0),
    (0.0, 0.0, 0.0, 1.0),
)

_FORWARD_AXES: dict[str, Vec3] = {
    "x": (1.0, 0.0, 0.0),
    "-x": (-1.0, 0.0, 0.0),
    "y": (0.0, 1.0, 0.0),
    "-y": (0.0, -1.0, 0.0),
    "z": (0.0, 0.0, 1.0),
    "-z": (0.0, 0.0, -1.0),
}


def identity_mat4() -> Mat4:
    return (
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )


def mat4_multiply(a: Mat4, b: Mat4) -> Mat4:
    return tuple(
        tuple(sum(a[i][k] * b[k][j] for k in range(4)) for j in range(4))
        for i in range(4)
    )


def host_from_mat4(matrix: Mat4) -> tuple[float, ...]:
    """Inverse of :func:`mat4_from_host_transform`: flatten a column-vector
    ``Mat4`` into the host's row-major / row-vector 16-float layout (translation
    at flat indices 12-14). Use this to feed a ``Mat4`` into the live-scan
    registration or to store it as a streamed transform."""

    return tuple(matrix[k % 4][k // 4] for k in range(16))


def mat4_from_host_transform(flat16: Sequence[float]) -> Mat4:
    """Convert a streamed transform (row-major, row-vector, translation at flat
    indices 12-14) into a column-vector ``Mat4`` (``p' = M @ p``).

    The host stores ``H`` such that ``p_world_row = p_local_row @ H``; the
    column-vector equivalent is ``p_world_col = H^T @ p_local_col``, so the
    column-vector matrix is the transpose of the row-major flat layout.
    """

    if len(flat16) != 16:
        raise ValueError("host transform must contain 16 floats")
    return tuple(tuple(float(flat16[j * 4 + i]) for j in range(4)) for i in range(4))


def transform_point(matrix: Mat4, point: Vec3) -> Vec3:
    x, y, z = point
    return (
        matrix[0][0] * x + matrix[0][1] * y + matrix[0][2] * z + matrix[0][3],
        matrix[1][0] * x + matrix[1][1] * y + matrix[1][2] * z + matrix[1][3],
        matrix[2][0] * x + matrix[2][1] * y + matrix[2][2] * z + matrix[2][3],
    )


def transform_direction(matrix: Mat4, direction: Vec3) -> Vec3:
    x, y, z = direction
    return (
        matrix[0][0] * x + matrix[0][1] * y + matrix[0][2] * z,
        matrix[1][0] * x + matrix[1][1] * y + matrix[1][2] * z,
        matrix[2][0] * x + matrix[2][1] * y + matrix[2][2] * z,
    )


def rigid_inverse(matrix: Mat4) -> Mat4:
    """Inverse of a rigid (rotation + translation) transform: ``R^T`` with
    translation ``-R^T t``. Assumes ``matrix`` is rigid (validate first)."""

    r = [[matrix[i][j] for j in range(3)] for i in range(3)]
    t = (matrix[0][3], matrix[1][3], matrix[2][3])
    # R^T
    rt = [[r[j][i] for j in range(3)] for i in range(3)]
    # -R^T t
    nt = tuple(-(rt[i][0] * t[0] + rt[i][1] * t[1] + rt[i][2] * t[2]) for i in range(3))
    return (
        (rt[0][0], rt[0][1], rt[0][2], nt[0]),
        (rt[1][0], rt[1][1], rt[1][2], nt[1]),
        (rt[2][0], rt[2][1], rt[2][2], nt[2]),
        (0.0, 0.0, 0.0, 1.0),
    )


def _rotation_3x3(matrix: Mat4) -> list[list[float]]:
    return [[matrix[i][j] for j in range(3)] for i in range(3)]


def assert_rigid(matrix: Mat4, *, name: str = "transform") -> None:
    """Assert the rotation block is a proper (right-handed) orthonormal rotation.

    This is the convention/handedness check the plan calls for: a det of -1
    means a mirrored / left-handed mount spec — the classic silent bug that
    produces a plausible-looking but rotation-mirrored pose. We refuse it rather
    than assume the caller got the convention right.
    """

    r = _rotation_3x3(matrix)
    # Orthonormality: R^T R == I.
    for i in range(3):
        for j in range(3):
            dot = sum(r[k][i] * r[k][j] for k in range(3))
            expected = 1.0 if i == j else 0.0
            if abs(dot - expected) > _ORTHONORMAL_TOL:
                raise ValueError(
                    f"{name} rotation is not orthonormal (column {i}.{j} dot={dot:.5f}); "
                    "check the mount calibration"
                )
    det = (
        r[0][0] * (r[1][1] * r[2][2] - r[1][2] * r[2][1])
        - r[0][1] * (r[1][0] * r[2][2] - r[1][2] * r[2][0])
        + r[0][2] * (r[1][0] * r[2][1] - r[1][1] * r[2][0])
    )
    if abs(det - 1.0) > _DET_TOL:
        raise ValueError(
            f"{name} rotation has determinant {det:.5f}, expected +1. A determinant "
            "of -1 means a left-handed / mirrored frame — fix the mount convention "
            "(likely a flipped axis or swapped quaternion order)."
        )


def quaternion_wxyz_to_mat4(quat: Sequence[float], translation: Vec3) -> Mat4:
    """Build a rigid ``Mat4`` from a unit quaternion (w, x, y, z) and translation."""

    if len(quat) != 4:
        raise ValueError("quaternion must be (w, x, y, z)")
    w, x, y, z = (float(v) for v in quat)
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm < 1e-9:
        raise ValueError("quaternion has near-zero norm")
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    tx, ty, tz = translation
    return (
        (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w), float(tx)),
        (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w), float(ty)),
        (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y), float(tz)),
        (0.0, 0.0, 0.0, 1.0),
    )


@dataclass(frozen=True)
class PhoneMountCalibration:
    """Fixed rigid mount of the camera on the rover.

    ``T_R_C`` is the camera pose expressed in the rover body frame (translation =
    lever arm). ``rover_forward_axis`` names the rover-frame axis that points in
    the driving direction; it is used only to extract yaw.
    """

    T_R_C: Mat4
    rover_forward_axis: str = "x"

    @property
    def lever_arm_m(self) -> Vec3:
        return (self.T_R_C[0][3], self.T_R_C[1][3], self.T_R_C[2][3])

    @property
    def T_C_R(self) -> Mat4:
        return rigid_inverse(self.T_R_C)

    @property
    def forward_vec(self) -> Vec3:
        return _FORWARD_AXES[self.rover_forward_axis]

    @classmethod
    def identity(cls) -> "PhoneMountCalibration":
        return cls(T_R_C=identity_mat4(), rover_forward_axis="x")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PhoneMountCalibration":
        forward = str(data.get("rover_forward_axis", "x")).lower()
        if forward not in _FORWARD_AXES:
            raise ValueError(
                f"rover_forward_axis must be one of {sorted(_FORWARD_AXES)}, got {forward!r}"
            )
        if "matrix_T_R_C" in data:
            flat = data["matrix_T_R_C"]
            if len(flat) != 16:
                raise ValueError("matrix_T_R_C must contain 16 floats (row-major, column-vector)")
            t_r_c = tuple(tuple(float(flat[i * 4 + j]) for j in range(4)) for i in range(4))
        elif "camera_in_rover" in data:
            block = data["camera_in_rover"]
            translation = block.get("translation_m", [0.0, 0.0, 0.0])
            if len(translation) != 3:
                raise ValueError("camera_in_rover.translation_m must be [x, y, z] metres")
            translation = (float(translation[0]), float(translation[1]), float(translation[2]))
            quat = block.get("quaternion_wxyz", [1.0, 0.0, 0.0, 0.0])
            t_r_c = quaternion_wxyz_to_mat4(quat, translation)
        else:
            raise ValueError(
                "mount JSON must contain either 'camera_in_rover' "
                "(translation_m + quaternion_wxyz) or 'matrix_T_R_C'"
            )
        assert_rigid(t_r_c, name="T_R_C")
        return cls(T_R_C=t_r_c, rover_forward_axis=forward)

    @classmethod
    def load(cls, path: str | Path) -> "PhoneMountCalibration":
        data = json.loads(Path(path).read_text())
        return cls.from_dict(data)


def rotation_z(yaw_rad: float, translation: Vec3 = (0.0, 0.0, 0.0)) -> Mat4:
    """Rotation about the twin up-axis (Z) with optional translation, as a
    column-vector ``Mat4``."""

    c, s = math.cos(yaw_rad), math.sin(yaw_rad)
    tx, ty, tz = translation
    return (
        (c, -s, 0.0, float(tx)),
        (s, c, 0.0, float(ty)),
        (0.0, 0.0, 1.0, float(tz)),
        (0.0, 0.0, 0.0, 1.0),
    )


def compute_t_align(
    camera_transform_host_start: Sequence[float],
    calibration: PhoneMountCalibration,
    rover_start_pose: Pose,
    start_z: float = 0.0,
) -> Mat4:
    """Compute the rigid world alignment ``T_align``.

    At the first normal ARKit frame the rover is assumed to sit at
    ``rover_start_pose`` in the twin. ``T_align`` is constrained to a **planar**
    transform (yaw about twin-Z + translation): ARKit and the twin are both
    gravity-aligned after ``M_axis``, so the only real misalignment is heading +
    position. Keeping it planar avoids tilting the whole twin (and its floor)
    from one noisy frame. The resulting ``T_align`` maps the start camera pose
    exactly onto ``rover_start_pose``.
    """

    t_a_c = mat4_from_host_transform(camera_transform_host_start)
    # S: rover pose in twin axes, before world alignment.
    s = mat4_multiply(M_AXIS, mat4_multiply(t_a_c, calibration.T_C_R))
    origin_s = transform_point(s, (0.0, 0.0, 0.0))
    forward_s = transform_direction(s, calibration.forward_vec)
    yaw_s = math.atan2(forward_s[1], forward_s[0])
    delta_yaw = math.radians(rover_start_pose.yaw_deg) - yaw_s
    c, ss = math.cos(delta_yaw), math.sin(delta_yaw)
    # Where origin_s lands after the rotation (before translation).
    rotated = (
        c * origin_s[0] - ss * origin_s[1],
        ss * origin_s[0] + c * origin_s[1],
        origin_s[2],
    )
    translation = (
        rover_start_pose.x - rotated[0],
        rover_start_pose.y - rotated[1],
        start_z - rotated[2],
    )
    return rotation_z(delta_yaw, translation)


def registration_delta(old_registration: Sequence[float], new_t_align: Mat4) -> tuple[float, ...]:
    """Row-major delta that re-maps already-transformed twin points from an old
    registration to a new ``T_align``.

    Existing live-scan vertices were stored as ``R_old @ usd_point``; we want
    ``T_align @ usd_point``. The delta ``M = T_align @ inv(R_old)`` satisfies
    ``new = M @ old``. Returned in the host row-major layout so it can be applied
    with ``transform_point_row_major`` (which performs the column-vector action
    of its argument). Used to re-anchor the map when alignment changes so pose
    and mesh stay registered to one shared transform.
    """

    r_old = mat4_from_host_transform(old_registration)
    delta = mat4_multiply(new_t_align, rigid_inverse(r_old))
    return host_from_mat4(delta)


def rover_pose_from_camera(
    camera_transform_host: Sequence[float],
    calibration: PhoneMountCalibration,
    t_align: Mat4 | None = None,
) -> Pose:
    """Compute the twin-frame rover pose (x, y, yaw_deg) from a streamed camera
    transform ``T_A_C`` and the mount calibration.

    ``t_align`` is the rigid world alignment ``T_align``; defaults to
    identity, which gives a rover pose correct up to that alignment.
    """

    align = t_align if t_align is not None else identity_mat4()
    t_a_c = mat4_from_host_transform(camera_transform_host)
    # T_W_R = T_align * M_axis * T_A_C * T_C_R
    t_w_r = mat4_multiply(
        align,
        mat4_multiply(M_AXIS, mat4_multiply(t_a_c, calibration.T_C_R)),
    )
    origin = transform_point(t_w_r, (0.0, 0.0, 0.0))
    forward_world = transform_direction(t_w_r, calibration.forward_vec)
    yaw = math.degrees(math.atan2(forward_world[1], forward_world[0]))
    return Pose(x=origin[0], y=origin[1], yaw_deg=normalize_yaw_deg(yaw))


def rover_pose_from_twin_camera_matrix(
    camera_to_twin: Sequence[Sequence[float]],
    calibration: PhoneMountCalibration,
) -> Pose:
    """Compute the twin-frame rover pose from an nvblox/CV camera pose.

    Online cuVSLAM returns ``T_W_Ccv`` directly after the mapper anchors its
    odometry frame into the twin. The mount calibration is measured in the
    phone/ARKit camera frame, so first convert the CV camera axes back to ARKit
    camera axes, then apply ``T_C_R``. Unlike :func:`rover_pose_from_camera`, no
    ARKit-world remap or ``T_align`` is applied here.
    """

    if len(camera_to_twin) != 4 or any(len(row) != 4 for row in camera_to_twin):
        raise ValueError("camera_to_twin must be a 4x4 matrix")
    t_w_c: Mat4 = tuple(
        tuple(float(camera_to_twin[i][j]) for j in range(4))  # type: ignore[misc]
        for i in range(4)
    )  # type: ignore[assignment]
    t_w_c_arkit = mat4_multiply(t_w_c, CV_TO_ARKIT_CAM)
    t_w_r = mat4_multiply(t_w_c_arkit, calibration.T_C_R)
    origin = transform_point(t_w_r, (0.0, 0.0, 0.0))
    forward_world = transform_direction(t_w_r, calibration.forward_vec)
    yaw = math.degrees(math.atan2(forward_world[1], forward_world[0]))
    return Pose(x=origin[0], y=origin[1], yaw_deg=normalize_yaw_deg(yaw))


@dataclass
class AlignmentStabilityGate:
    """Require a short stationary normal-tracking window before anchoring.

    ARKit can report ``TRACKING_NORMAL`` while the operator is still moving the
    phone or while the first few poses are settling. Anchoring the twin to that
    first pose makes the whole scan appear shifted or yawed relative to the
    rover. This gate uses the unaligned rover pose estimate only as a stability
    signal; the actual anchor still uses the latest camera transform.
    """

    samples_required: int = 5
    max_translation_m: float = 0.03
    max_yaw_deg: float = 3.0
    _samples: list[Pose] = field(default_factory=list)

    def reset(self) -> None:
        self._samples.clear()

    def add(
        self,
        camera_transform_host: Sequence[float],
        calibration: PhoneMountCalibration,
    ) -> tuple[bool, str]:
        sample = rover_pose_from_camera(camera_transform_host, calibration)
        if self.samples_required <= 1:
            self._samples = [sample]
            return True, "stable disabled"
        self._samples.append(sample)
        if len(self._samples) > self.samples_required:
            self._samples.pop(0)
        if len(self._samples) < self.samples_required:
            return False, f"{len(self._samples)}/{self.samples_required} stable samples"
        latest = self._samples[-1]
        max_translation = max(
            math.hypot(p.x - latest.x, p.y - latest.y) for p in self._samples
        )
        max_yaw = max(yaw_error_deg(p.yaw_deg, latest.yaw_deg) for p in self._samples)
        if max_translation <= self.max_translation_m and max_yaw <= self.max_yaw_deg:
            return True, (
                f"{len(self._samples)} stable samples "
                f"(span {max_translation:.3f} m, {max_yaw:.1f} deg)"
            )
        return False, (
            f"waiting for stable pose "
            f"(span {max_translation:.3f} m/{self.max_translation_m:.3f} m, "
            f"{max_yaw:.1f}/{self.max_yaw_deg:.1f} deg)"
        )
