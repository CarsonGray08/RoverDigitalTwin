"""nvblox TSDF: the visual map. Never in the physics path.

The mapper talks to :class:`Tsdf`, so it runs, and is tested, without a GPU.
:class:`NvbloxTsdf` needs ``nvblox_torch`` and CUDA (lab workstation only).
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from rover_twin.frames import Intrinsics


@dataclass(frozen=True, eq=False)
class Mesh:
    vertices: np.ndarray   # (N, 3) float32, twin frame
    faces: np.ndarray      # (M, 3) int32
    colors: np.ndarray     # (N, 3) uint8

    @staticmethod
    def empty() -> "Mesh":
        return Mesh(np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int32), np.zeros((0, 3), np.uint8))


class Tsdf(Protocol):
    def integrate(self, depth: np.ndarray, intrinsics: Intrinsics, T_W_C: np.ndarray, jpeg: bytes,
                  color_intrinsics: Intrinsics) -> None: ...

    def mesh(self) -> Mesh: ...

    def clear(self) -> None: ...


class NvbloxTsdf:
    """One nvblox mapper fed posed depth and colour (``T_W_C`` in CV axes).

    ``preallocated_blocks`` shrinks nvblox's per-layer block pool: the mapper
    always builds every layer, and the ones never fed keep their idle
    preallocation (~340 MB at 2 cm with the default 2048). Fed layers still
    grow on demand.
    """

    def __init__(self, voxel_m: float, max_range_m: float, preallocated_blocks: int = 16) -> None:
        import torch
        from nvblox_torch.mapper import Mapper
        from nvblox_torch.mapper_params import BlockMemoryPoolParams, MapperParams, ProjectiveIntegratorParams

        if not torch.cuda.is_available():
            raise RuntimeError("nvblox needs a CUDA device")
        integrator = ProjectiveIntegratorParams()
        integrator.projective_integrator_max_integration_distance_m = float(max_range_m)
        pool = BlockMemoryPoolParams()
        pool.num_preallocated_blocks = int(preallocated_blocks)
        params = MapperParams()
        params.set_projective_integrator_params(integrator)
        params.set_block_memory_pool_params(pool)
        self._torch = torch
        self._mapper = Mapper(voxel_sizes_m=float(voxel_m), mapper_parameters=params)
        self._sensors: dict[Intrinsics, object] = {}
        self._dirty = False

    def integrate(self, depth, intrinsics, T_W_C, jpeg, color_intrinsics) -> None:
        from PIL import Image

        torch = self._torch
        pose = torch.from_numpy(np.ascontiguousarray(T_W_C, np.float32))     # nvblox wants the pose on CPU
        self._mapper.add_depth_frame(torch.from_numpy(np.ascontiguousarray(depth, np.float32)).cuda(), pose,
                                     self._sensor(intrinsics))
        if jpeg:
            with Image.open(io.BytesIO(jpeg)) as img:
                rgb = np.array(img.convert("RGB"), np.uint8)
            self._mapper.add_color_frame(torch.from_numpy(rgb).cuda(), pose, self._sensor(color_intrinsics))
        self._dirty = True

    def mesh(self) -> Mesh:
        if self._dirty:
            self._mapper.update_color_mesh()
            self._dirty = False
        mesh = self._mapper.get_color_mesh()
        vertices = mesh.vertices().cpu().numpy()
        if len(vertices) == 0:
            return Mesh.empty()
        colors = mesh.vertex_colors().cpu().numpy()
        if colors.dtype != np.uint8:      # nvblox may return float colours in [0, 1]
            colors = np.clip(colors * 255.0, 0, 255)
        return Mesh(vertices.astype(np.float32), mesh.triangles().cpu().numpy().astype(np.int32),
                    colors[:, :3].astype(np.uint8))

    def clear(self) -> None:
        self._mapper.clear()
        self._dirty = True

    def _sensor(self, k: Intrinsics):
        if k not in self._sensors:
            from nvblox_torch.sensor import Sensor

            self._sensors[k] = Sensor.from_camera(fu=k.fx, fv=k.fy, cu=k.cx, cv=k.cy, width=k.width, height=k.height)
        return self._sensors[k]
