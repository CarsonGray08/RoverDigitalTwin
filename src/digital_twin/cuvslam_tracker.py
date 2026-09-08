"""Incremental cuVSLAM RGB-D tracking for live/replay scan frames."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from digital_twin.nvblox_feeder import (
    apply_confidence_mask,
    arkit_camera_pose_to_nvblox,
    decode_depth_map,
    decode_jpeg_rgb,
)


@dataclass
class CuvslamTrackStats:
    frames: int = 0
    valid_poses: int = 0
    dropped_poses: int = 0
    final_translation_norm_m: float = 0.0
    final_yaw_y_deg: float = 0.0
    final_yaw_y_unwrapped_deg: float = 0.0


class LiveCuvslamRgbdTracker:
    """Stateful cuVSLAM RGB-D odometry for incoming iPhone RGB-D keyframes.

    The first valid cuVSLAM rig pose is identity in cuVSLAM's odometry frame.
    We anchor that first rig pose to the frame's ARKit-derived camera pose in
    the twin frame, then apply subsequent cuVSLAM relative motion from there.
    """

    def __init__(
        self,
        *,
        variant: str = "depth-gray",
        min_confidence: int = 1,
        depth_scale_factor: float = 1000.0,
        use_gpu: bool = True,
        async_sba: bool = False,
        use_motion_model: bool = True,
        use_denoising: bool = False,
        max_frame_delta_s: float = 5.0,
    ) -> None:
        self.variant = variant
        self.min_confidence = int(min_confidence)
        self.depth_scale_factor = float(depth_scale_factor)
        self.use_gpu = bool(use_gpu)
        self.async_sba = bool(async_sba)
        self.use_motion_model = bool(use_motion_model)
        self.use_denoising = bool(use_denoising)
        self.max_frame_delta_s = float(max_frame_delta_s)
        self._cuvslam = __import__("cuvslam")
        self._tracker = None
        self._anchor_pose: np.ndarray | None = None
        self._t0_s: float | None = None
        self._last_yaw_y_rad: float | None = None
        self._yaw_y_unwrapped_rad = 0.0
        self.stats = CuvslamTrackStats()

    @property
    def initialized(self) -> bool:
        return self._tracker is not None and self._anchor_pose is not None

    def track_frame(self, frame, t_align) -> np.ndarray | None:
        """Track ``frame`` and return camera-to-twin pose, or None on failure."""
        if not getattr(frame, "depth_bytes", b"") or not getattr(frame, "image_bytes", b""):
            return None
        if self._tracker is None:
            self._tracker = self._make_tracker(frame)
            self._anchor_pose = arkit_camera_pose_to_nvblox(frame.anchor_transform, t_align)
            self._t0_s = float(frame.timestamp_s)
        image, depth_mm = _prepare_cuvslam_inputs(frame, self.variant, self.min_confidence)
        timestamp_ns = int(round((float(frame.timestamp_s) - float(self._t0_s)) * 1e9))
        self.stats.frames += 1
        pose_estimate, _slam_pose = self._tracker.track(
            timestamp_ns, [image], depths=[depth_mm])
        world_from_rig = pose_estimate.world_from_rig
        if world_from_rig is None:
            self.stats.dropped_poses += 1
            return None
        pose = world_from_rig.pose
        rel = _pose_to_matrix(pose.rotation, pose.translation)
        out = np.asarray(self._anchor_pose, dtype=np.float64).reshape(4, 4) @ rel
        self.stats.valid_poses += 1
        translation = np.asarray(pose.translation, dtype=np.float64)
        self.stats.final_translation_norm_m = float(np.linalg.norm(translation))
        yaw_y = _yaw_about_y_deg(np.asarray(pose.rotation, dtype=np.float64))
        self.stats.final_yaw_y_deg = float(yaw_y)
        self._update_unwrapped_yaw(math.radians(yaw_y))
        self.stats.final_yaw_y_unwrapped_deg = float(math.degrees(self._yaw_y_unwrapped_rad))
        return out.astype(np.float32)

    def _make_tracker(self, frame):
        cuvslam = self._cuvslam
        camera = _camera_for_variant(cuvslam, frame, self.variant)
        config = cuvslam.Tracker.OdometryConfig()
        config.odometry_mode = cuvslam.Tracker.OdometryMode.RGBD
        config.use_gpu = self.use_gpu
        config.async_sba = self.async_sba
        config.use_motion_model = self.use_motion_model
        config.use_denoising = self.use_denoising
        config.max_frame_delta_s = self.max_frame_delta_s
        config.enable_observations_export = False
        config.enable_landmarks_export = False
        rgbd = cuvslam.Tracker.OdometryRGBDSettings()
        rgbd.depth_scale_factor = self.depth_scale_factor
        rgbd.depth_camera_id = 0
        rgbd.enable_depth_stereo_tracking = False
        config.rgbd_settings = rgbd
        return cuvslam.Tracker(cuvslam.Rig([camera]), config)

    def _update_unwrapped_yaw(self, yaw_y_rad: float) -> None:
        if self._last_yaw_y_rad is None:
            self._yaw_y_unwrapped_rad = yaw_y_rad
        else:
            delta = yaw_y_rad - self._last_yaw_y_rad
            while delta > math.pi:
                delta -= 2.0 * math.pi
            while delta < -math.pi:
                delta += 2.0 * math.pi
            self._yaw_y_unwrapped_rad += delta
        self._last_yaw_y_rad = yaw_y_rad


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
        f"unknown cuVSLAM variant {variant!r}; use depth-gray, depth-rgb, full-gray, or full-rgb"
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
