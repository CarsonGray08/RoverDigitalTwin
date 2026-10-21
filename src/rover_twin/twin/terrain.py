"""The live terrain in the twin: what the link delivers, kept twice.

:class:`TerrainMap` is the twin's copy of the elevation grid, for questions
validation asks (is the ground under this footprint known, does the body hit
anything). :class:`StageTerrain` puts the same patches on the USD stage as
static colliders, plus the nvblox mesh as visuals only.
"""

from __future__ import annotations

import math

import numpy as np

from rover_twin.frames import Pose2D
from rover_twin.link.client import Update
from rover_twin.link.messages import Patch
from rover_twin.mapping.elevation import OBSERVED, UNKNOWN
from rover_twin.mapping.tsdf import Mesh


class TerrainMap:
    def __init__(self) -> None:
        self.patches: dict[tuple[int, int], Patch] = {}
        self.cell_m, self.n = 0.0, 0

    def apply(self, update: Update) -> None:
        if update.reset:
            self.patches.clear()
        self.patches.update(update.patches)
        if update.patches:
            self.cell_m, self.n = update.cell_m, update.n

    def footprint(self, pose: Pose2D, half_length: float, half_width: float):
        """(heights, states) of the cells under a rover footprint at ``pose``
        (length along the driving direction)."""
        if not self.patches:
            return np.zeros(0), np.zeros(0, np.uint8)
        step = self.cell_m
        u, v = np.meshgrid(np.arange(-half_length, half_length + 1e-9, step),
                           np.arange(-half_width, half_width + 1e-9, step))
        yaw = math.radians(pose.yaw_deg)
        c, s = math.cos(yaw), math.sin(yaw)
        x = pose.x + c * u.ravel() - s * v.ravel()
        y = pose.y + s * u.ravel() + c * v.ravel()
        return self.cells(np.floor(np.column_stack([x, y]) / step).astype(np.int64))

    def cells(self, ij: np.ndarray):
        """(heights, states) of global cell indices; unknown where no tile."""
        heights, states = np.full(len(ij), np.nan), np.zeros(len(ij), np.uint8)
        keys, local = np.divmod(ij, self.n)
        for key in {tuple(k) for k in keys.tolist()}:
            patch = self.patches.get(key)
            if patch is None:
                continue
            rows = np.all(keys == key, axis=1)
            heights[rows] = patch.height[local[rows, 0], local[rows, 1]]
            states[rows] = patch.state[local[rows, 0], local[rows, 1]]
        return heights, states

    def dense(self, max_step: float) -> dict:
        """The whole map as one grid, for software that plans on it: cell
        [i][j] covers x0 + i*cell .. x0 + (i+1)*cell (and the same in y);
        height is None where unknown; obstacle marks what validation refuses."""
        if not self.patches:
            return {"cell_m": self.cell_m, "origin_m": [0.0, 0.0], "shape": [0, 0],
                    "height_m": [], "state": [], "obstacle": []}
        keys = np.array(list(self.patches))
        lo, hi = keys.min(axis=0) * self.n, (keys.max(axis=0) + 1) * self.n
        height = np.full(hi - lo, np.nan)
        state = np.zeros(hi - lo, np.uint8)
        for (tx, ty), p in self.patches.items():
            i, j = tx * self.n - lo[0], ty * self.n - lo[1]
            height[i:i + self.n, j:j + self.n] = p.height[:-1, :-1]
            state[i:i + self.n, j:j + self.n] = p.state[:-1, :-1]
        obstacle = _obstacle(height, state, max_step)
        return {"cell_m": self.cell_m, "origin_m": (lo * self.cell_m).tolist(), "shape": list(height.shape),
                "height_m": np.where(state == UNKNOWN, None, np.round(height, 4)).tolist(),
                "state": state.tolist(), "obstacle": obstacle.astype(np.uint8).tolist()}

    def obstacle_cells(self, max_step: float) -> np.ndarray:
        """Centres (x, y) of the obstacle cells (see :func:`_obstacle`)."""
        out = []
        for (tx, ty), p in self.patches.items():
            i, j = np.nonzero(_obstacle(p.height[:-1, :-1], p.state[:-1, :-1], max_step))
            out.append(np.column_stack([(tx * self.n + i + 0.5), (ty * self.n + j + 0.5)]) * self.cell_m)
        return np.concatenate(out) if out else np.zeros((0, 2))


def _obstacle(height: np.ndarray, state: np.ndarray, max_step: float) -> np.ndarray:
    """Observed cells ``max_step`` or more above the floor (z = 0), which the
    body hits, or below it, a drop. Heights are absolute: on uneven ground
    this should be measured from the local ground instead."""
    return (state == OBSERVED) & (np.abs(height) >= max_step)


class StageTerrain:
    """Elevation patches as static triangle-mesh colliders, one Mesh per tile.

    A tile's topology is fixed when its prim is made, so an update only
    rewrites the points (a few ms per tile on the sim thread).
    Unknown cells sink ``pit_m`` below the floor: physics never stands the
    rover on them, and validation refuses to plan over them anyway. The
    colliders are invisible; the operator sees the nvblox mesh.
    """

    def __init__(self, stage, root: str = "/World/Terrain", friction: float = 1.0, pit_m: float = 0.5) -> None:
        from pxr import UsdGeom, UsdPhysics, UsdShade

        self.stage, self.root, self.pit_m = stage, root, pit_m
        UsdGeom.Xform.Define(stage, root)
        material = UsdShade.Material.Define(stage, f"{root}/Regolith")
        physics = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
        physics.CreateStaticFrictionAttr().Set(float(friction))
        physics.CreateDynamicFrictionAttr().Set(float(friction))
        physics.CreateRestitutionAttr().Set(0.0)
        self._material = material
        self._tiles: dict[tuple[int, int], object] = {}

    def apply(self, update: Update) -> None:
        if update.reset:
            for key in self._tiles:
                self.stage.RemovePrim(self._path(key))
            self._tiles.clear()
        for key, patch in update.patches.items():
            self._points(key, update.n).Set(self._vertices(key, patch, update.n, update.cell_m))

    def _points(self, key, n):
        """The points attribute of the tile's mesh, creating the prim once."""
        if key in self._tiles:
            return self._tiles[key]
        from pxr import UsdGeom, UsdPhysics, UsdShade, Vt

        mesh = UsdGeom.Mesh.Define(self.stage, self._path(key))
        mesh.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
        i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        a = (i * (n + 1) + j).ravel()
        quads = np.column_stack([a, a + n + 1, a + n + 2, a, a + n + 2, a + 1])
        mesh.CreateFaceVertexCountsAttr().Set(Vt.IntArray.FromNumpy(np.full(2 * n * n, 3, np.int32)))
        mesh.CreateFaceVertexIndicesAttr().Set(Vt.IntArray.FromNumpy(quads.reshape(-1).astype(np.int32)))
        mesh.MakeInvisible()
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr().Set("none")
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(self._material, materialPurpose="physics")
        self._tiles[key] = mesh.CreatePointsAttr()
        return self._tiles[key]

    def _vertices(self, key, patch: Patch, n: int, cell_m: float):
        from pxr import Vt

        x, y = np.meshgrid((key[0] * n + np.arange(n + 1) + 0.5) * cell_m,
                           (key[1] * n + np.arange(n + 1) + 0.5) * cell_m, indexing="ij")
        known = patch.state != UNKNOWN
        floor = np.min(patch.height[known]) if known.any() else 0.0
        z = np.where(known, patch.height, floor - self.pit_m)
        return Vt.Vec3fArray.FromNumpy(np.stack([x, y, z], -1).reshape(-1, 3).astype(np.float32))

    def show(self, mesh: Mesh) -> None:
        from pxr import Sdf, UsdGeom, Vt

        prim = UsdGeom.Mesh.Define(self.stage, f"{self.root}/Visual")
        prim.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
        # astype copies to native order: Isaac's USD rejects the wire's explicit little-endian buffers
        prim.CreatePointsAttr().Set(Vt.Vec3fArray.FromNumpy(mesh.vertices.astype(np.float32)))
        prim.CreateFaceVertexCountsAttr().Set(Vt.IntArray.FromNumpy(np.full(len(mesh.faces), 3, np.int32)))
        prim.CreateFaceVertexIndicesAttr().Set(Vt.IntArray.FromNumpy(mesh.faces.reshape(-1).astype(np.int32)))
        colors = UsdGeom.PrimvarsAPI(prim).CreatePrimvar("displayColor", Sdf.ValueTypeNames.Color3fArray,
                                                         UsdGeom.Tokens.vertex)
        colors.Set(Vt.Vec3fArray.FromNumpy(mesh.colors.astype(np.float32) / 255.0))

    def _path(self, key) -> str:
        tx, ty = key
        return f"{self.root}/tile_{'m' if tx < 0 else ''}{abs(tx)}_{'m' if ty < 0 else ''}{abs(ty)}"
