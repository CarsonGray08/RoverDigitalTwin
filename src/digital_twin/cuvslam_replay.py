"""cuVSLAM pose lookup for deterministic RGB-D scanlog replay."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np

from digital_twin.live_scan import (
    MSG_KEYFRAME_RGBD,
    MSG_PHONE_POSE,
    TRACKING_NORMAL,
    decode_payload,
)
from digital_twin.models import Pose
from digital_twin.nvblox_feeder import (
    apply_confidence_mask,
    arkit_camera_pose_to_nvblox,
    decode_depth_map,
    decode_jpeg_rgb,
)
from digital_twin.phone_mount import PhoneMountCalibration, compute_t_align
from digital_twin.scan_recording import iter_records


@dataclass(frozen=True)
class ReplayCuvslamTrajectory:
    """cuVSLAM RGB-D camera poses keyed by scanlog frame timestamp.

    cuVSLAM's first world frame is the first camera frame. For nvblox replay we
    anchor that frame to the normal twin alignment computed from the scanlog's
    first good phone pose, then left-multiply every cuVSLAM pose into the twin
    frame. The result is a camera-to-twin-world pose in the same CV camera-axis
    convention used by :mod:`digital_twin.nvblox_feeder`.
    """

    poses_by_timestamp: dict[float, np.ndarray]
    valid_poses: int
    total_frames: int
    final_translation_norm_m: float
    final_yaw_y_deg: float
    final_yaw_y_unwrapped_deg: float

    @classmethod
    def from_scanlog(
        cls,
        path: str | Path,
        calibration: PhoneMountCalibration,
        twin_start_pose: Pose,
        twin_start_z: float,
        *,
        variant: str = "depth-gray",
        min_confidence: int = 1,
        depth_scale_factor: float = 1000.0,
        use_gpu: bool = True,
        async_sba: bool = False,
        use_motion_model: bool = True,
        use_denoising: bool = False,
        max_frame_delta_s: float = 5.0,
        close_translation: bool = False,
        translation_scale: float = 1.0,
    ) -> "ReplayCuvslamTrajectory":
        import cuvslam

        frames: list[object] = []
        t_align = None
        for _offset_s, payload in iter_records(path):
            frame = decode_payload(payload)
            if (
                frame.message_type == MSG_PHONE_POSE
                and frame.tracking_state == TRACKING_NORMAL
                and t_align is None
            ):
                t_align = compute_t_align(
                    frame.anchor_transform, calibration, twin_start_pose, twin_start_z)
            elif frame.message_type == MSG_KEYFRAME_RGBD and frame.depth_bytes:
                frames.append(frame)

        if t_align is None:
            raise ValueError(f"{path}: no normal phone pose to establish replay alignment")
        if not frames:
            raise ValueError(f"{path}: no RGB-D frames for cuVSLAM replay")

        first = frames[0]
        camera = _camera_for_variant(cuvslam, first, variant)
        config = cuvslam.Tracker.OdometryConfig()
        config.odometry_mode = cuvslam.Tracker.OdometryMode.RGBD
        config.use_gpu = bool(use_gpu)
        config.async_sba = bool(async_sba)
        config.use_motion_model = bool(use_motion_model)
        config.use_denoising = bool(use_denoising)
        config.max_frame_delta_s = float(max_frame_delta_s)
        config.enable_observations_export = False
        config.enable_landmarks_export = False
        rgbd = cuvslam.Tracker.OdometryRGBDSettings()
        rgbd.depth_scale_factor = float(depth_scale_factor)
        rgbd.depth_camera_id = 0
        rgbd.enable_depth_stereo_tracking = False
        config.rgbd_settings = rgbd
        tracker = cuvslam.Tracker(cuvslam.Rig([camera]), config)

        anchor_pose = arkit_camera_pose_to_nvblox(first.anchor_transform, t_align)
        raw_poses: list[tuple[float, np.ndarray, float]] = []
        t0_s = float(first.timestamp_s)
        valid = 0
        last_translation = np.zeros(3, dtype=np.float64)
        last_yaw_y_rad: float | None = None
        yaw_y_unwrapped_rad = 0.0
        last_yaw_y_deg = math.nan

        for frame in frames:
            image, depth_mm = _prepare_cuvslam_inputs(frame, variant, min_confidence)
            timestamp_ns = int(round((float(frame.timestamp_s) - t0_s) * 1e9))
            pose_estimate, _slam_pose = tracker.track(timestamp_ns, [image], depths=[depth_mm])
            world_from_rig = pose_estimate.world_from_rig
            if world_from_rig is None:
                continue
            pose = world_from_rig.pose
            t_c0_ci = _pose_to_matrix(pose.rotation, pose.translation)
            valid += 1
            last_translation = np.asarray(pose.translation, dtype=np.float64)
            q = np.asarray(pose.rotation, dtype=np.float64)
            last_yaw_y_deg = _yaw_about_y_deg(q)
            yaw_y_rad = math.radians(last_yaw_y_deg)
            if last_yaw_y_rad is None:
                yaw_y_unwrapped_rad = yaw_y_rad
            else:
                delta = yaw_y_rad - last_yaw_y_rad
                while delta > math.pi:
                    delta -= 2.0 * math.pi
                while delta < -math.pi:
                    delta += 2.0 * math.pi
                yaw_y_unwrapped_rad += delta
            last_yaw_y_rad = yaw_y_rad
            raw_poses.append((round(float(frame.timestamp_s), 6), t_c0_ci, yaw_y_unwrapped_rad))

        if valid == 0:
            raise ValueError(f"{path}: cuVSLAM did not return any valid RGB-D poses")

        anchor = np.asarray(anchor_pose, dtype=np.float64).reshape(4, 4)
        poses: dict[float, np.ndarray] = {}
        final_yaw = raw_poses[-1][2] if raw_poses else 0.0
        drift = raw_poses[-1][1][:3, 3].copy() if raw_poses else np.zeros(3)
        scale = float(translation_scale)
        for ts_key, t_c0_ci, yaw_unwrapped in raw_poses:
            adjusted = t_c0_ci.copy()
            alpha = 0.0
            if close_translation and abs(final_yaw) > 1e-9:
                alpha = float(np.clip(yaw_unwrapped / final_yaw, 0.0, 1.0))
            adjusted[:3, 3] = scale * (adjusted[:3, 3] - alpha * drift)
            poses[ts_key] = (anchor @ adjusted).astype(np.float32)

        return cls(
            poses_by_timestamp=poses,
            valid_poses=valid,
            total_frames=len(frames),
            final_translation_norm_m=float(np.linalg.norm(last_translation)),
            final_yaw_y_deg=float(last_yaw_y_deg),
            final_yaw_y_unwrapped_deg=float(math.degrees(yaw_y_unwrapped_rad)),
        )

    def pose_for_frame(self, frame) -> np.ndarray | None:
        return self.poses_by_timestamp.get(round(float(frame.timestamp_s), 6))


def _camera_for_variant(cuvslam, frame, variant: str):
    fx, fy, cx, cy = [float(v) for v in frame.intrinsics]
    if variant.startswith("depth-"):
        sx = float(frame.depth_width) / float(frame.image_width)
        sy = float(frame.depth_height) / float(frame.image_height)
        return cuvslam.Camera(
            size=[int(frame.depth_width), int(frame.depth_height)],
            focal=[fx * sx, fy * sy],
            principal=[cx * sx, cy * sy],
        )
    return cuvslam.Camera(
        size=[int(frame.image_width), int(frame.image_height)],
        focal=[fx, fy],
        principal=[cx, cy],
    )


def _prepare_cuvslam_inputs(frame, variant: str, min_confidence: int) -> tuple[np.ndarray, np.ndarray]:
    from PIL import Image

    rgb = decode_jpeg_rgb(frame.image_bytes)
    depth_m = decode_depth_map(frame.depth_bytes, frame.depth_width, frame.depth_height)
    depth_m = apply_confidence_mask(depth_m, frame.confidence_bytes, min_confidence)
    depth_mm = np.clip(depth_m * 1000.0, 0.0, 65535.0).astype(np.uint16)

    if variant == "depth-gray":
        image = np.array(
            Image.fromarray(rgb)
            .resize((int(frame.depth_width), int(frame.depth_height)), Image.Resampling.BILINEAR)
            .convert("L"),
            dtype=np.uint8,
        )
        return image, depth_mm
    if variant == "depth-rgb":
        image = np.array(
            Image.fromarray(rgb).resize(
                (int(frame.depth_width), int(frame.depth_height)), Image.Resampling.BILINEAR
            ),
            dtype=np.uint8,
        )
        return image, depth_mm
    if variant == "full-gray":
        image = np.array(Image.fromarray(rgb).convert("L"), dtype=np.uint8)
        return image, _resize_depth(depth_mm, int(frame.image_width), int(frame.image_height))
    if variant == "full-rgb":
        return rgb, _resize_depth(depth_mm, int(frame.image_width), int(frame.image_height))
    raise ValueError(
        f"unknown cuVSLAM replay variant {variant!r}; use depth-gray, depth-rgb, full-gray, or full-rgb"
    )


def _resize_depth(depth_mm: np.ndarray, width: int, height: int) -> np.ndarray:
    from PIL import Image

    return np.array(
        Image.fromarray(depth_mm).resize((width, height), Image.Resampling.NEAREST),
        dtype=np.uint16,
    )


def _pose_to_matrix(rotation, translation) -> np.ndarray:
    q = np.asarray(rotation, dtype=np.float64)
    x, y, z, w = [float(v) for v in q]
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )
    mat[:3, 3] = np.asarray(translation, dtype=np.float64)
    return mat


def _yaw_about_y_deg(q: np.ndarray) -> float:
    x, y, z, w = [float(v) for v in q]
    r02 = 2.0 * (x * z + w * y)
    r22 = 1.0 - 2.0 * (x * x + y * y)
    return math.degrees(math.atan2(r02, r22))
