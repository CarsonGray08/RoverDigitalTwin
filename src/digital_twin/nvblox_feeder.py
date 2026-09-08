"""Feeder: live-scan iPhone frames -> NvbloxMapperBridge.

Bridges the wire protocol/ARKit conventions to the transport-agnostic
:class:`~digital_twin.nvblox_bridge.NvbloxMapperBridge`:

1. **Pose**: ARKit streams a camera-to-ARKit-world transform (row-major /
   row-vector, Y-up). nvblox wants a camera-to-**twin-world** pose in the
   computer-vision camera convention (+x right, +y down, +z forward). The
   conversion is ``T_W_Ccv = T_align @ M_axis @ T_A_C @ B`` where ``M_axis``
   is the ARKit-Y-up -> twin-Z-up remap shared with the rest of the system
   (see :mod:`digital_twin.phone_mount`), ``T_align`` is the rigid
   world alignment, and ``B`` re-expresses the CV camera axes in ARKit camera
   axes (flip Y and Z). This makes the nvblox map land in the same twin frame
   as the rover pose and the old live-scan mesh.
2. **Intrinsics**: keyframe intrinsics are at JPEG resolution; depth is at its
   own (lower) resolution, so they are scaled to the depth image. MSG_DEPTH_FRAME
   already carries depth-resolution intrinsics, so it needs no scaling.
3. **Confidence**: ARKit per-pixel confidence (0/1/2) masks out unreliable depth.
"""

from __future__ import annotations

import numpy as np

from digital_twin.nvblox_bridge import CameraIntrinsics, NvbloxMapperBridge
from digital_twin.models import Pose
from digital_twin.phone_mount import (
    M_AXIS,
    PhoneMountCalibration,
    mat4_multiply,
    rotation_z,
    rover_pose_from_camera,
)

# Camera-axis basis change: a point in nvblox/CV camera coords (+x right, +y
# down, +z forward) expressed in ARKit camera coords (+x right, +y up, -z
# forward). Right-multiplied onto the ARKit camera-to-world pose so the result's
# columns are the CV camera axes nvblox expects.
_CV_TO_ARKIT_CAM = np.array(
    [[1.0, 0.0, 0.0, 0.0],
     [0.0, -1.0, 0.0, 0.0],
     [0.0, 0.0, -1.0, 0.0],
     [0.0, 0.0, 0.0, 1.0]],
    dtype=np.float64,
)
# ARKit world (Y-up) -> twin world (Z-up), as a numpy column-vector matrix.
_M_AXIS_NP = np.array(M_AXIS, dtype=np.float64)


def _t_align_np(t_align) -> np.ndarray:
    if t_align is None:
        return np.eye(4, dtype=np.float64)
    return np.asarray(t_align, dtype=np.float64).reshape(4, 4)


def arkit_camera_pose_to_nvblox(camera_transform, t_align=None) -> np.ndarray:
    """Convert a streamed ARKit camera transform into a camera-to-twin-world 4x4
    pose in nvblox's CV camera convention. ``camera_transform`` is the 16-float
    row-major/row-vector layout from the live-scan header; ``t_align`` is the
    rigid ARKit->twin alignment (column-vector ``Mat4`` or 4x4), or None for
    identity."""
    flat = np.asarray(camera_transform, dtype=np.float64).reshape(16)
    # Row-major/row-vector (p' = p @ H) -> column-vector (p' = M @ p) is a
    # transpose, matching phone_mount.mat4_from_host_transform.
    t_a_c = flat.reshape(4, 4).T
    # T_align is applied OUTERMOST, after M_axis -- matching
    # phone_mount.rover_pose_from_camera (T_W = T_align @ M_axis @ T_A_C @ ...).
    # With identity T_align the order is moot, but it matters once aligned.
    t_w_a = _t_align_np(t_align) @ _M_AXIS_NP
    return (t_w_a @ t_a_c @ _CV_TO_ARKIT_CAM).astype(np.float32)


def turn_in_place_camera_pose_to_nvblox(
    camera_transform,
    calibration: PhoneMountCalibration,
    t_align,
    locked_rover_pose: Pose,
    locked_rover_z: float = 0.0,
) -> np.ndarray:
    """Camera pose for a startup 360 where the rover pivots in place.

    ARKit translation is the weak part of a slow turn-in-place scan: small VIO
    translation drift makes the map spiral even though the rover body should
    stay at one XY position. For that special scan mode, trust only ARKit yaw,
    lock the rover body origin to the known startup pose, and reconstruct the
    mounted camera pose from the calibrated lever arm.
    """
    observed = rover_pose_from_camera(camera_transform, calibration, t_align)
    locked_rover = rotation_z(
        np.radians(observed.yaw_deg),
        (locked_rover_pose.x, locked_rover_pose.y, locked_rover_z),
    )
    return rover_camera_pose_to_nvblox(locked_rover, calibration)


def rover_camera_pose_to_nvblox(
    rover_transform,
    calibration: PhoneMountCalibration,
) -> np.ndarray:
    """Camera-to-twin pose for a known rover transform.

    ``rover_transform`` is a twin-frame rover body pose in the same 4x4
    convention as :func:`rotation_z`. The returned pose uses nvblox/CV camera
    axes, matching :func:`arkit_camera_pose_to_nvblox`.
    """
    t_w_c_arkit = np.asarray(
        mat4_multiply(rover_transform, calibration.T_R_C), dtype=np.float64
    ).reshape(4, 4)
    return (t_w_c_arkit @ _CV_TO_ARKIT_CAM).astype(np.float32)


def scale_intrinsics_to_depth(
    intrinsics, rgb_width: int, rgb_height: int, depth_width: int, depth_height: int
) -> CameraIntrinsics:
    """Scale (fx, fy, cx, cy) at RGB resolution to the depth image resolution."""
    fx, fy, cx, cy = intrinsics
    sx = float(depth_width) / float(rgb_width)
    sy = float(depth_height) / float(rgb_height)
    return CameraIntrinsics(
        fx=fx * sx, fy=fy * sy, cx=cx * sx, cy=cy * sy,
        width=int(depth_width), height=int(depth_height),
    )


def decode_depth_map(depth_bytes: bytes, width: int, height: int) -> np.ndarray:
    """Contiguous little-endian float32 depth bytes -> (H, W) meters array."""
    depth = np.frombuffer(depth_bytes, dtype="<f4")
    if depth.size != int(width) * int(height):
        raise ValueError(
            f"depth bytes ({depth.size}) != {width}x{height} ({width * height})"
        )
    return depth.reshape(int(height), int(width)).astype(np.float32, copy=True)


def apply_confidence_mask(
    depth: np.ndarray, confidence_bytes: bytes, min_confidence: int
) -> np.ndarray:
    """Zero out depth pixels below ``min_confidence`` (ARKit 0/1/2). nvblox treats
    0 as 'no return', so masking is the same as dropping those rays. No-op when
    confidence is absent or min_confidence <= 0."""
    if min_confidence <= 0 or not confidence_bytes:
        return depth
    conf = np.frombuffer(confidence_bytes, dtype=np.uint8)
    if conf.size != depth.size:
        return depth
    masked = depth.copy()
    masked[conf.reshape(depth.shape) < min_confidence] = 0.0
    return masked


def decode_jpeg_rgb(jpeg_bytes: bytes) -> np.ndarray:
    """Decode JPEG bytes to an (H, W, 3) uint8 RGB array. Pillow is imported
    lazily so the feeder has no hard image dependency for the depth-only path."""
    import io

    from PIL import Image

    with Image.open(io.BytesIO(jpeg_bytes)) as img:
        # np.array (not asarray) so the result is a writable copy -- torch.from_numpy
        # warns on the read-only buffer Pillow hands back otherwise.
        return np.array(img.convert("RGB"), dtype=np.uint8)


class NvbloxFeeder:
    """Drives a :class:`NvbloxMapperBridge` from decoded live-scan frames.

    Holds the current ``T_align`` (updated by the host when the twin alignment is
    (re)established) so frames land in the twin frame.
    """

    def __init__(
        self,
        bridge: NvbloxMapperBridge,
        min_confidence: int = 1,
        t_align=None,
    ) -> None:
        self.bridge = bridge
        self.min_confidence = int(min_confidence)
        self._t_align = t_align

    def set_alignment(self, t_align) -> None:
        self._t_align = t_align

    @property
    def t_align(self):
        """Current ARKit-world -> twin alignment (or None)."""
        return self._t_align

    def feed_depth_frame(self, frame) -> bool:
        """Integrate a decoded MSG_DEPTH_FRAME. Intrinsics are already depth-res."""
        if not frame.depth_bytes or frame.depth_width <= 0 or frame.depth_height <= 0:
            return False
        intr = CameraIntrinsics(
            fx=frame.intrinsics[0], fy=frame.intrinsics[1],
            cx=frame.intrinsics[2], cy=frame.intrinsics[3],
            width=int(frame.depth_width), height=int(frame.depth_height),
        )
        return self._integrate(frame, intr, rgb=None)

    def feed_keyframe_rgbd(self, frame, rgb: np.ndarray | None = None) -> bool:
        """Integrate the depth of a decoded MSG_KEYFRAME_RGBD (intrinsics are at
        JPEG resolution, so they're scaled to the depth image). Optional decoded
        ``rgb`` (at JPEG resolution) is fused as color."""
        if not frame.depth_bytes or frame.depth_width <= 0 or frame.depth_height <= 0:
            return False
        intr = scale_intrinsics_to_depth(
            frame.intrinsics, frame.image_width, frame.image_height,
            frame.depth_width, frame.depth_height,
        )
        return self._integrate(frame, intr, rgb=rgb, rgb_frame=frame)

    def feed_keyframe_color(self, frame, rgb: np.ndarray | None = None) -> bool:
        """Fuse a texture keyframe's color (only) onto the existing TSDF surface.
        ``rgb`` is the decoded (H, W, 3) image; if omitted and the frame carries a
        JPEG, it is decoded here. Returns False if there's no usable image."""
        if rgb is None:
            jpeg = getattr(frame, "image_bytes", b"")
            if not jpeg:
                return False
            rgb = decode_jpeg_rgb(jpeg)
        h, w = rgb.shape[:2]
        color_intr = CameraIntrinsics(
            fx=frame.intrinsics[0], fy=frame.intrinsics[1],
            cx=frame.intrinsics[2], cy=frame.intrinsics[3],
            width=int(w), height=int(h),
        )
        pose = arkit_camera_pose_to_nvblox(frame.anchor_transform, self._t_align)
        self.bridge.integrate_color(rgb, color_intr, pose)
        return True

    def integrate_at_pose(
        self, depth_m, depth_intr, T_world_cam,
        rgb=None, color_intr=None,
    ) -> bool:
        """Integrate a decoded depth image at an EXPLICIT camera-to-twin-world
        pose (e.g. from cuVSLAM), bypassing the ARKit ``anchor_transform`` path.
        ``T_world_cam`` is a 4x4 in nvblox's CV convention. Optional ``rgb`` (with
        ``color_intr``) is fused as color."""
        pose = np.asarray(T_world_cam, dtype=np.float32).reshape(4, 4)
        if rgb is not None:
            self.bridge.integrate_rgbd(
                depth_m, depth_intr, pose, rgb=rgb, color_intrinsics=color_intr)
        else:
            self.bridge.integrate_rgbd(depth_m, depth_intr, pose)
        return True

    def _integrate(self, frame, depth_intr, rgb=None, rgb_frame=None) -> bool:
        depth = decode_depth_map(frame.depth_bytes, frame.depth_width, frame.depth_height)
        depth = apply_confidence_mask(depth, frame.confidence_bytes, self.min_confidence)
        pose = arkit_camera_pose_to_nvblox(frame.anchor_transform, self._t_align)
        if rgb is not None and rgb_frame is not None:
            color_intr = CameraIntrinsics(
                fx=rgb_frame.intrinsics[0], fy=rgb_frame.intrinsics[1],
                cx=rgb_frame.intrinsics[2], cy=rgb_frame.intrinsics[3],
                width=int(rgb_frame.image_width), height=int(rgb_frame.image_height),
            )
            self.bridge.integrate_rgbd(
                depth, depth_intr, pose, rgb=rgb, color_intrinsics=color_intr
            )
        else:
            self.bridge.integrate_rgbd(depth, depth_intr, pose)
        return True
