"""ICP drift correction ("loop closure on top of ARKit") for the live nvblox map.

ARKit's per-frame VIO is good, but its pose drifts over a session, so when you
*revisit* a surface the new depth integrates at a slightly shifted pose and the
surface doubles. This snaps the live scan back onto the already-built map:
periodically ICP the current depth frame against the accumulated nvblox mesh and
fold the rigid correction into an accumulated drift transform applied to every
pose. Revisited geometry then lands on itself instead of ghosting.

This does NOT deform already-integrated geometry (nvblox can't); it keeps NEW
integration aligned, so doubling never accumulates. Corrections are small and
frequent, so seams stay negligible. Depth-only -- it rides the existing ARKit
depth stream, no RGB/IMU needed.
"""

from __future__ import annotations

import numpy as np


class IcpCorrector:
    """Accumulates a twin-space drift correction and refines it by ICP of the
    live depth against the nvblox map.

    Parameters
    ----------
    period_s:
        Minimum seconds between ICP runs (it's the heavier op; the correction is
        held between runs).
    voxel_m:
        Downsample size for both clouds (speed + robustness).
    max_dist_m:
        ICP correspondence distance -- cap to the expected drift, so far-apart
        (non-overlapping/new) geometry doesn't create false matches.
    min_fitness:
        Only apply a correction when ICP overlaps enough of the frame (so scanning
        brand-new area, with nothing to align to, leaves the drift unchanged).
    local_target_padding_m:
        Crop the accumulated map to the current ARKit-predicted frame bounds plus
        this padding before ICP. This prevents matching a similar-looking room
        elsewhere in the map.
    max_correction_translation_m:
        Reject one-shot corrections larger than this. This is an incremental
        drift corrector, not a global relocalizer.
    max_correction_degrees:
        Reject one-shot rotations larger than this.
    min_points:
        Skip frames/targets with too few points to register reliably.
    depth_stride:
        Pixel stride for the live depth frame used by ICP. The full-resolution
        depth is still integrated into nvblox; only the ICP sample is decimated.
    """

    def __init__(self, period_s: float = 2.0, voxel_m: float = 0.08,
                 max_dist_m: float = 0.15, min_fitness: float = 0.95,
                 local_target_padding_m: float = 0.45,
                 max_correction_translation_m: float = 0.15,
                 max_correction_degrees: float = 5.0,
                 min_points: int = 800, depth_stride: int = 4,
                 gravity_locked: bool = False) -> None:
        # gravity_locked: project each accepted correction onto yaw + translation
        # (zero roll/pitch) before folding it into the drift. The twin frame is
        # Z-up and ARKit is gravity-aligned, so the real drift is heading + XY(+Z),
        # not tilt. On a floor-dominated turn-in-place scan, point-to-point ICP is
        # under-constrained in pitch/roll and can rotate about a horizontal axis
        # while still matching the ground plane; those small tilts compound into a
        # tilted floor. Locking removes that degenerate DOF. Off by default so
        # well-constrained walking scans keep full 6-DOF correction.
        self.gravity_locked = bool(gravity_locked)
        self.period_s = float(period_s)
        self.voxel_m = float(voxel_m)
        self.max_dist_m = float(max_dist_m)
        self.min_fitness = float(min_fitness)
        self.local_target_padding_m = float(local_target_padding_m)
        self.max_correction_translation_m = float(max_correction_translation_m)
        self.max_correction_degrees = float(max_correction_degrees)
        self.min_points = int(min_points)
        self.depth_stride = max(1, int(depth_stride))
        self.drift = np.eye(4, dtype=np.float64)   # accumulated twin-space correction
        self._target = None                        # o3d cloud of the existing map
        self._last_run = -1e9
        self.last_fitness = 0.0
        self.last_attempted = False
        self.last_applied = False
        self.correction_count = 0
        self.last_translation_m = 0.0
        self.last_rotation_degrees = 0.0
        self.last_reject_reason = ""

    # -- target (the accumulated map) --------------------------------------
    def set_target(self, vertices: np.ndarray) -> None:
        """Refresh the ICP target from the current nvblox mesh vertices (twin frame)."""
        verts = np.asarray(vertices, dtype=np.float64)
        if len(verts) < self.min_points:
            self._target = None
            return
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(verts)
        self._target = pcd.voxel_down_sample(self.voxel_m)

    # -- per-frame ---------------------------------------------------------
    @staticmethod
    def _backproject(depth_m: np.ndarray, intr, pose: np.ndarray) -> np.ndarray:
        """Depth (H,W m) + pinhole intrinsics + camera-to-twin pose -> (N,3) twin
        points (CV camera frame: x right, y down, z forward)."""
        h, w = depth_m.shape
        z = depth_m
        valid = np.isfinite(z) & (z > 0)
        vv, uu = np.nonzero(valid)
        zz = z[vv, uu].astype(np.float64)
        x = (uu - intr.cx) / intr.fx * zz
        y = (vv - intr.cy) / intr.fy * zz
        pts_cam = np.stack([x, y, zz], axis=1)
        return pts_cam @ pose[:3, :3].T + pose[:3, 3]

    def process(self, depth_m: np.ndarray, intr, base_pose_twin: np.ndarray,
                now: float) -> np.ndarray:
        """Return the corrected camera-to-twin pose for this frame (accumulated
        drift applied), refining the drift by ICP at most every ``period_s``."""
        base = np.asarray(base_pose_twin, dtype=np.float64).reshape(4, 4)
        corrected = self.drift @ base
        self.last_attempted = False
        self.last_applied = False
        self.last_reject_reason = ""
        if self._target is None or now - self._last_run < self.period_s:
            return corrected.astype(np.float32)
        self._last_run = now
        self.last_attempted = True

        depth_for_icp = depth_m[::self.depth_stride, ::self.depth_stride]
        intr_for_icp = type(intr)(
            fx=intr.fx / self.depth_stride,
            fy=intr.fy / self.depth_stride,
            cx=intr.cx / self.depth_stride,
            cy=intr.cy / self.depth_stride,
            width=depth_for_icp.shape[1],
            height=depth_for_icp.shape[0],
        )
        pts = self._backproject(depth_for_icp, intr_for_icp, corrected)
        if len(pts) < self.min_points:
            self.last_reject_reason = "too few source points"
            return corrected.astype(np.float32)

        import open3d as o3d
        src = o3d.geometry.PointCloud()
        src.points = o3d.utility.Vector3dVector(pts)
        src = src.voxel_down_sample(self.voxel_m)
        target = self._local_target(o3d, pts)
        if target is None or len(target.points) < self.min_points:
            self.last_reject_reason = "too few local target points"
            return corrected.astype(np.float32)
        reg = o3d.pipelines.registration.registration_icp(
            src, target, self.max_dist_m, np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        )
        self.last_fitness = float(reg.fitness)
        delta = np.asarray(reg.transformation, dtype=np.float64)
        self.last_translation_m = float(np.linalg.norm(delta[:3, 3]))
        self.last_rotation_degrees = _rotation_degrees(delta[:3, :3])
        if reg.fitness < self.min_fitness:
            self.last_reject_reason = "low fitness"
            return corrected.astype(np.float32)
        if self.last_translation_m > self.max_correction_translation_m:
            self.last_reject_reason = "translation too large"
            return corrected.astype(np.float32)
        if self.last_rotation_degrees > self.max_correction_degrees:
            self.last_reject_reason = "rotation too large"
            return corrected.astype(np.float32)

        # last_rotation_degrees stays the FULL measured rotation (a useful signal
        # of how much tilt ICP wanted); locking only changes what we apply.
        if self.gravity_locked:
            delta = _yaw_translation_only(delta)

        self.drift = delta @ self.drift
        corrected = self.drift @ base
        self.last_applied = True
        self.correction_count += 1
        return corrected.astype(np.float32)

    def _local_target(self, o3d, source_points: np.ndarray):
        """Crop the map around the predicted live-frame bounds before ICP."""
        pad = self.local_target_padding_m
        lo = np.min(source_points, axis=0) - pad
        hi = np.max(source_points, axis=0) + pad
        aabb = o3d.geometry.AxisAlignedBoundingBox(lo, hi)
        return self._target.crop(aabb)


def _rotation_degrees(R: np.ndarray) -> float:
    trace = float(np.trace(R))
    cos_theta = np.clip((trace - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


def _yaw_translation_only(delta: np.ndarray) -> np.ndarray:
    """Project a rigid 4x4 onto yaw (about twin +Z) + full translation, dropping
    roll/pitch. For the small corrections that pass the gates, the yaw of the
    rotation is well-approximated by atan2(R[1,0], R[0,0])."""
    R = delta[:3, :3]
    yaw = np.arctan2(R[1, 0], R[0, 0])
    c, s = np.cos(yaw), np.sin(yaw)
    locked = np.eye(4, dtype=np.float64)
    locked[0, 0], locked[0, 1] = c, -s
    locked[1, 0], locked[1, 1] = s, c
    locked[:3, 3] = delta[:3, 3]
    return locked
