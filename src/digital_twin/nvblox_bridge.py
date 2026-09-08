"""Transport- and frame-agnostic wrapper around the NVIDIA nvblox GPU mapper.

Integrates posed RGB-D frames into a GPU TSDF and extracts a mesh. It
deliberately knows nothing about ARKit, the live-scan TCP protocol, or USD --
those live in the feeder/sink layers -- so it stays unit-testable and reusable.

Driven by the installed ``nvblox_torch`` bindings (real NVIDIA nvblox, same core
as Isaac ROS nvblox). Fusion updates the volume incrementally; mesh extraction
through the Python bindings returns the full mesh.

Pose convention: ``integrate_rgbd`` takes a **camera-to-world** 4x4 transform in
nvblox's camera convention (+x right, +y down, +z forward, i.e. standard computer
vision / OpenCV). Converting from ARKit's camera axes (+y up, -z forward) and
applying ``T_align`` to land in the twin frame is the feeder's job, not this
module's.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch


# Intrinsics for a pinhole camera, matching what the live-scan keyframe header
# already carries (fx, fy, cx, cy) plus the image size.
@dataclass(frozen=True)
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    def key(self) -> tuple:
        return (self.fx, self.fy, self.cx, self.cy, self.width, self.height)


@dataclass(frozen=True)
class MeshArrays:
    """Plain-numpy mesh, ready for USD/PLY/whatever. Colors are 0-255 uint8."""

    vertices: np.ndarray  # (N, 3) float32, world/twin frame
    faces: np.ndarray     # (M, 3) int32
    colors: np.ndarray    # (N, 3) uint8

    @property
    def is_empty(self) -> bool:
        return len(self.vertices) == 0 or len(self.faces) == 0


def nvblox_available() -> bool:
    """True if nvblox_torch + a CUDA device are usable. Lets callers / tests
    skip cleanly on a CPU-only box without importing at module load."""
    try:
        import torch  # noqa: F401

        import nvblox_torch  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:  # noqa: BLE001
        return False


class NvbloxMapperBridge:
    """Owns one nvblox ``Mapper`` and feeds posed RGB-D frames into it.

    Parameters
    ----------
    voxel_size_m:
        TSDF voxel size. 0.05 m is the nvblox default for room/building scale;
        drop toward 0.02-0.03 for finer geometry at higher memory cost.
    max_integration_distance_m:
        Depth beyond this is ignored. iPhone LiDAR is unreliable past ~5 m.
    device:
        CUDA device for the input tensors. nvblox itself is always on GPU.
    preallocated_blocks:
        nvblox preallocates this many voxel blocks *per layer* in its memory pool
        (nvblox default 2048). The Mapper always builds the full layer cake
        (TSDF/Color/Feature/Freespace/Occupancy/ESDF/Mesh) -- there is no Python
        knob to drop a layer -- but the layers we never feed (Feature is the
        expensive one) never grow past their preallocation. Dropping this to a
        small value reclaims that idle preallocation (~340 MB measured at 2 cm
        going 2048->16). The pool still expands (x2) on demand for the layers we
        do feed (TSDF/Mesh), so the only cost of a small value is a little extra
        reallocation on the first frames. None = leave nvblox's default.
    """

    def __init__(
        self,
        voxel_size_m: float = 0.05,
        max_integration_distance_m: float = 5.0,
        device: str = "cuda",
        preallocated_blocks: int | None = 16,
    ) -> None:
        import torch
        from nvblox_torch.mapper import Mapper
        from nvblox_torch.mapper_params import (
            BlockMemoryPoolParams,
            MapperParams,
            ProjectiveIntegratorParams,
        )

        if not torch.cuda.is_available():
            raise RuntimeError("NvbloxMapperBridge requires a CUDA device")

        self._torch = torch
        self._device = device
        self.voxel_size_m = float(voxel_size_m)

        integrator_params = ProjectiveIntegratorParams()
        integrator_params.projective_integrator_max_integration_distance_m = float(
            max_integration_distance_m
        )
        params = MapperParams()
        params.set_projective_integrator_params(integrator_params)
        if preallocated_blocks is not None:
            pool_params = BlockMemoryPoolParams()
            pool_params.num_preallocated_blocks = int(preallocated_blocks)
            params.set_block_memory_pool_params(pool_params)

        self._mapper = Mapper(voxel_sizes_m=self.voxel_size_m, mapper_parameters=params)
        # Sensors are immutable per (intrinsics, resolution); build once and reuse
        # so we don't reallocate a C++ sensor object every frame.
        self._sensor_cache: dict[tuple, object] = {}
        self._frame_count = 0
        self._mesh_dirty = False

    # -- ingestion ---------------------------------------------------------
    def _sensor_for(self, intr: CameraIntrinsics):
        sensor = self._sensor_cache.get(intr.key())
        if sensor is None:
            from nvblox_torch.sensor import Sensor

            sensor = Sensor.from_camera(
                fu=float(intr.fx),
                fv=float(intr.fy),
                cu=float(intr.cx),
                cv=float(intr.cy),
                width=int(intr.width),
                height=int(intr.height),
            )
            self._sensor_cache[intr.key()] = sensor
        return sensor

    def _as_pose(self, T_world_cam: np.ndarray):
        pose = np.asarray(T_world_cam, dtype=np.float32)
        if pose.shape != (4, 4):
            raise ValueError(f"T_world_cam must be 4x4, got {pose.shape}")
        # nvblox wants the pose on CPU (see the bindings' examples).
        return self._torch.from_numpy(np.ascontiguousarray(pose))

    def _as_depth(self, depth_m: np.ndarray, intr: CameraIntrinsics):
        torch = self._torch
        if isinstance(depth_m, np.ndarray):
            arr = np.ascontiguousarray(depth_m, dtype=np.float32)
            if arr.shape != (intr.height, intr.width):
                raise ValueError(
                    f"depth shape {arr.shape} != (h,w)=({intr.height},{intr.width})"
                )
            return torch.from_numpy(arr).to(self._device)
        return depth_m.to(self._device).float().contiguous()

    def _as_rgb(self, rgb: np.ndarray, intr: CameraIntrinsics):
        torch = self._torch
        if isinstance(rgb, np.ndarray):
            arr = np.ascontiguousarray(rgb, dtype=np.uint8)
            if arr.shape != (intr.height, intr.width, 3):
                raise ValueError(
                    f"rgb shape {arr.shape} != (h,w,3)=({intr.height},{intr.width},3)"
                )
            return torch.from_numpy(arr).to(self._device)
        return rgb.to(self._device).contiguous()

    def integrate_rgbd(
        self,
        depth_m: np.ndarray,
        intrinsics: CameraIntrinsics,
        T_world_cam: np.ndarray,
        rgb: np.ndarray | None = None,
        color_intrinsics: CameraIntrinsics | None = None,
    ) -> None:
        """Fuse one posed depth (and optional color) frame into the TSDF.

        depth_m: (H, W) float32 metric depth (0 / non-finite = no return).
        T_world_cam: 4x4 camera-to-world pose (CV camera convention).
        rgb: optional (Hc, Wc, 3) uint8 color.
        color_intrinsics: intrinsics for ``rgb`` when it is a different
            resolution than the depth (e.g. full-res JPEG vs LiDAR depth).
            Defaults to ``intrinsics`` (color matches depth resolution).
        """
        pose = self._as_pose(T_world_cam)
        depth_sensor = self._sensor_for(intrinsics)
        depth = self._as_depth(depth_m, intrinsics)
        self._mapper.add_depth_frame(depth, pose, depth_sensor)
        if rgb is not None:
            color_intr = color_intrinsics or intrinsics
            color_sensor = self._sensor_for(color_intr)
            self._mapper.add_color_frame(self._as_rgb(rgb, color_intr), pose, color_sensor)
        self._frame_count += 1
        self._mesh_dirty = True

    def integrate_color(
        self,
        rgb: np.ndarray,
        intrinsics: CameraIntrinsics,
        T_world_cam: np.ndarray,
    ) -> None:
        """Fuse a posed color frame onto the existing TSDF surface, *without* a
        paired depth frame. Used for the texture keyframes: geometry comes from
        the dense depth stream, so keyframes only need to contribute color."""
        pose = self._as_pose(T_world_cam)
        sensor = self._sensor_for(intrinsics)
        self._mapper.add_color_frame(self._as_rgb(rgb, intrinsics), pose, sensor)
        self._mesh_dirty = True

    # -- outputs -----------------------------------------------------------
    @property
    def frame_count(self) -> int:
        return self._frame_count

    def update_mesh(self) -> None:
        """Recompute the (incremental) color mesh from the current TSDF."""
        self._mapper.update_color_mesh()
        self._mesh_dirty = False

    def mesh_arrays(self, update: bool = True) -> MeshArrays:
        """Return the current mesh as plain numpy arrays in the world frame."""
        if update and self._mesh_dirty:
            self.update_mesh()
        mesh = self._mapper.get_color_mesh()
        vertices = _to_numpy(mesh.vertices()).astype(np.float32, copy=False)
        faces = _to_numpy(mesh.triangles()).astype(np.int32, copy=False)
        colors = _to_numpy(mesh.vertex_colors())
        if colors.size and colors.dtype != np.uint8:
            # nvblox color meshes may come back as float [0,1]; normalize to 0-255.
            colors = np.clip(colors[:, :3] * 255.0, 0, 255).astype(np.uint8)
        elif colors.size:
            colors = colors[:, :3].astype(np.uint8, copy=False)
        else:
            colors = np.zeros((len(vertices), 3), dtype=np.uint8)
        return MeshArrays(vertices=vertices, faces=faces, colors=colors)

    def save_mesh(self, path: str) -> None:
        """Write the current color mesh to disk (PLY/OBJ per nvblox's writer)."""
        if self._mesh_dirty:
            self.update_mesh()
        self._mapper.get_color_mesh().save(str(path))

    # nvblox_torch addresses the default single map by id 0 (same as get_color_mesh).
    _DEFAULT_MAPPER_ID = 0

    def save_map(self, path: str) -> None:
        """Serialize the full nvblox map (layers) for persistence/relocalization."""
        self._mapper.save_map(str(path), self._DEFAULT_MAPPER_ID)

    def load_map(self, path: str) -> None:
        """Load a serialized nvblox map into the default mapper."""
        self._mapper.load_from_file(str(path), self._DEFAULT_MAPPER_ID)
        self._mesh_dirty = True

    def clear(self) -> None:
        self._mapper.clear()
        self._frame_count = 0
        self._mesh_dirty = False


def _to_numpy(value) -> np.ndarray:
    """Best-effort tensor/array -> contiguous numpy on CPU."""
    if isinstance(value, np.ndarray):
        return np.ascontiguousarray(value)
    # torch.Tensor (the common case) and anything else with .cpu()/.numpy().
    try:
        return np.ascontiguousarray(value.detach().cpu().numpy())
    except AttributeError:
        return np.ascontiguousarray(np.asarray(value))
