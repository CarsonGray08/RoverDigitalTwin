"""Deterministic IMU-yaw loop closure for recorded scanlog replays."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np

from digital_twin.live_scan import (
    MSG_DEPTH_FRAME,
    MSG_IMU,
    MSG_KEYFRAME_RGBD,
    MSG_PHONE_POSE,
    TRACKING_NORMAL,
    decode_payload,
)
from digital_twin.models import Pose
from digital_twin.nvblox_feeder import (
    arkit_camera_pose_to_nvblox,
    rover_camera_pose_to_nvblox,
)
from digital_twin.phone_mount import (
    PhoneMountCalibration,
    compute_t_align,
    rotation_z,
    rover_pose_from_camera,
)
from digital_twin.scan_recording import iter_records


_AXIS_INDEX = {"x": 0, "y": 1, "z": 2}


@dataclass(frozen=True)
class ReplayImuYawClosure:
    """Pose lookup for replay frames corrected by gyro yaw and loop closure.

    The iPhone log already contains synchronized RGB-D, ARKit poses, and IMU.
    For the problematic 360 benchmark, ARKit under-reports yaw; the gyro carries
    the rotation. This class keeps ARKit's local translation shape but removes
    accumulated start/end translation drift and replaces yaw with integrated gyro
    yaw, optionally normalized to a known loop angle.
    """

    poses_by_timestamp: dict[float, np.ndarray]
    raw_final_yaw_deg: float
    applied_final_yaw_deg: float
    translation_drift_m: tuple[float, float]

    @classmethod
    def from_scanlog(
        cls,
        path: str | Path,
        calibration: PhoneMountCalibration,
        twin_start_pose: Pose,
        twin_start_z: float,
        *,
        gyro_axis: str = "x",
        target_yaw_degrees: float = 360.0,
        attitude_mode: str = "planar",
        translation_scale: float = 1.0,
    ) -> "ReplayImuYawClosure":
        axis = _AXIS_INDEX.get(gyro_axis.lower())
        if axis is None:
            raise ValueError(f"unknown gyro axis {gyro_axis!r}; use x, y, or z")
        if attitude_mode not in ("planar", "arkit"):
            raise ValueError("attitude_mode must be 'planar' or 'arkit'")

        t_align = None
        imu_ts: list[float] = []
        imu_rates: list[float] = []
        frames: list[tuple[float, float, float, float, np.ndarray]] = []
        for _offset_s, payload in iter_records(path):
            frame = decode_payload(payload)
            if (
                frame.message_type == MSG_PHONE_POSE
                and frame.tracking_state == TRACKING_NORMAL
                and t_align is None
            ):
                t_align = compute_t_align(
                    frame.anchor_transform, calibration, twin_start_pose, twin_start_z)
            elif frame.message_type == MSG_IMU:
                imu_ts.append(float(frame.timestamp_s))
                imu_rates.append(float(frame.imu_gyro[axis]))
            elif (
                frame.message_type in (MSG_KEYFRAME_RGBD, MSG_DEPTH_FRAME)
                and frame.depth_bytes
                and t_align is not None
            ):
                rover = rover_pose_from_camera(
                    frame.anchor_transform, calibration, t_align)
                base_pose = arkit_camera_pose_to_nvblox(frame.anchor_transform, t_align)
                frames.append((
                    float(frame.timestamp_s),
                    float(rover.x),
                    float(rover.y),
                    float(rover.yaw_deg),
                    base_pose,
                ))

        if t_align is None:
            raise ValueError(f"{path}: no normal phone pose to establish replay alignment")
        if len(imu_ts) < 2:
            raise ValueError(f"{path}: not enough IMU samples for yaw closure")
        if len(frames) < 2:
            raise ValueError(f"{path}: not enough RGB-D/depth frames for yaw closure")

        imu_ts_arr = np.asarray(imu_ts, dtype=np.float64)
        imu_rates_arr = np.asarray(imu_rates, dtype=np.float64)
        order = np.argsort(imu_ts_arr)
        imu_ts_arr = imu_ts_arr[order]
        imu_rates_arr = imu_rates_arr[order]
        cumulative = np.zeros_like(imu_ts_arr)
        dt = np.diff(imu_ts_arr)
        cumulative[1:] = np.cumsum(
            (imu_rates_arr[:-1] + imu_rates_arr[1:]) * 0.5 * dt)

        frame_ts = np.asarray([item[0] for item in frames], dtype=np.float64)
        raw_yaw = np.interp(frame_ts, imu_ts_arr, cumulative)
        raw_yaw = raw_yaw - raw_yaw[0]
        raw_final = float(raw_yaw[-1])
        if abs(raw_final) < math.radians(5.0):
            raise ValueError(f"{path}: integrated IMU yaw is too small for closure")

        applied_yaw = raw_yaw.copy()
        if abs(target_yaw_degrees) > 0.0:
            target = math.radians(float(target_yaw_degrees))
            applied_yaw *= target / raw_final

        xy = np.asarray([[item[1], item[2]] for item in frames], dtype=np.float64)
        start_xy = xy[0].copy()
        drift = xy[-1] - start_xy
        denom = applied_yaw[-1] if abs(applied_yaw[-1]) > 1e-9 else 1.0
        alpha = np.clip(applied_yaw / denom, 0.0, 1.0)
        closed_xy = xy - alpha[:, None] * drift[None, :]
        scale = float(translation_scale)
        corrected_xy = start_xy[None, :] + scale * (closed_xy - start_xy[None, :])

        poses: dict[float, np.ndarray] = {}
        for frame_info, yaw, pos in zip(frames, applied_yaw, corrected_xy):
            ts = frame_info[0]
            rover_tf = rotation_z(
                yaw,
                (float(pos[0]), float(pos[1]), float(twin_start_z)),
            )
            pose = rover_camera_pose_to_nvblox(rover_tf, calibration)
            if attitude_mode == "arkit":
                _ts, rover_x, rover_y, rover_yaw_deg, base_pose = frame_info
                arkit_rover_tf = rotation_z(
                    math.radians(rover_yaw_deg),
                    (rover_x, rover_y, float(twin_start_z)),
                )
                arkit_nominal = rover_camera_pose_to_nvblox(
                    arkit_rover_tf, calibration)
                pose = _replace_nominal_yaw_preserving_arkit_attitude(
                    base_pose, arkit_nominal, pose)
            poses[round(float(ts), 6)] = pose

        return cls(
            poses_by_timestamp=poses,
            raw_final_yaw_deg=math.degrees(raw_final),
            applied_final_yaw_deg=math.degrees(float(applied_yaw[-1])),
            translation_drift_m=(float(drift[0]), float(drift[1])),
        )

    def pose_for_frame(self, frame) -> np.ndarray | None:
        return self.poses_by_timestamp.get(round(float(frame.timestamp_s), 6))


def _replace_nominal_yaw_preserving_arkit_attitude(
    base_pose: np.ndarray,
    old_nominal_pose: np.ndarray,
    new_nominal_pose: np.ndarray,
) -> np.ndarray:
    """Use closed-loop nominal translation/yaw while keeping ARKit roll/pitch.

    The planar correction is intentionally constrained, but a real scan can still
    have phone pitch/roll changes from the mount or operator. This decomposes the
    ARKit camera rotation into nominal yaw plus residual attitude, removes any
    residual yaw, and reapplies the remaining attitude to the corrected nominal
    yaw. Translation stays loop-corrected.
    """
    base = np.asarray(base_pose, dtype=np.float64).reshape(4, 4)
    old_nominal = np.asarray(old_nominal_pose, dtype=np.float64).reshape(4, 4)
    new_nominal = np.asarray(new_nominal_pose, dtype=np.float64).reshape(4, 4)
    residual = base[:3, :3] @ old_nominal[:3, :3].T
    residual_yaw = math.atan2(residual[1, 0], residual[0, 0])
    residual_no_yaw = _rz(-residual_yaw) @ residual

    out = new_nominal.copy()
    out[:3, :3] = residual_no_yaw @ new_nominal[:3, :3]
    return out.astype(np.float32)


def _rz(yaw_rad: float) -> np.ndarray:
    c = math.cos(yaw_rad)
    s = math.sin(yaw_rad)
    return np.asarray(
        [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
