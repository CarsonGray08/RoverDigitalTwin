"""Author the live nvblox tile transport into a USD stage (shared sink logic).

Reads the incremental per-tile transport written by ``digital_twin.mapper``
(``manifest.json`` + ``tiles/<name>.npz``) and authors each tile as a colored
Mesh prim, optionally as an exact triangle-mesh **collider** bound to a regolith
physics material so the rover physically drives on the reconstructed terrain.

The Isaac validator authors this terrain into the rover's stage so command
validation runs on the reconstructed surface.

``pxr`` is imported lazily inside the functions (only importable after an Isaac
``SimulationApp`` boots), so importing this module is safe off-Isaac; only the
authoring calls require Kit.
"""

from __future__ import annotations

import numpy as np

from digital_twin.mesh_tiles import load_manifest, tile_npz_path

TERRAIN_MATERIAL_NAME = "RegolithPhysicsMaterial"


def ensure_terrain_physics_material(
    stage, path: str, static_friction: float, dynamic_friction: float, restitution: float
) -> str:
    """Define (idempotently) the regolith physics material the rover's wheels grip
    against, and return its prim path. Re-defining each authoring pass is a no-op
    on an existing prim, so this is safe to call on every incremental update."""
    from pxr import UsdPhysics, UsdShade

    material = UsdShade.Material.Define(stage, path)
    phys = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    phys.CreateStaticFrictionAttr().Set(float(static_friction))
    phys.CreateDynamicFrictionAttr().Set(float(dynamic_friction))
    phys.CreateRestitutionAttr().Set(float(restitution))
    return path


def author_mesh_prim(
    stage, path: str, vertices, faces, colors,
    collision: bool = False, physics_material_path: str | None = None,
) -> None:
    """Author one colored polygon Mesh prim at ``path`` (in place: Define+Set, no
    teardown). ``faces`` are LOCAL indices into ``vertices``.

    With ``collision``, the prim is also made an EXACT triangle-mesh collider so
    the rover physically drives on the reconstructed surface (slope, rocks, and
    clearance are the real geometry, not a proxy). When ``physics_material_path``
    is given, the collider is bound to that physics material for friction."""
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade, Vt

    mesh = UsdGeom.Mesh.Define(stage, path)
    # Dense nvblox output is triangle soup. USD meshes default to Catmull-Clark
    # subdivision, which is expensive and wrong for scanned geometry.
    mesh.CreateSubdivisionSchemeAttr().Set("none")
    vertices = np.ascontiguousarray(vertices, dtype=np.float32)
    faces = np.ascontiguousarray(faces, dtype=np.int32)
    color_f = np.ascontiguousarray(colors[:, :3].astype(np.float32) / 255.0)
    primvar = UsdGeom.PrimvarsAPI(mesh.GetPrim()).CreatePrimvar(
        "displayColor", Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.vertex
    )
    try:
        mesh.CreatePointsAttr().Set(Vt.Vec3fArray.FromNumpy(vertices))
        mesh.CreateFaceVertexCountsAttr().Set(
            Vt.IntArray.FromNumpy(np.full(len(faces), 3, dtype=np.int32))
        )
        mesh.CreateFaceVertexIndicesAttr().Set(
            Vt.IntArray.FromNumpy(np.ascontiguousarray(faces.reshape(-1), dtype=np.int32))
        )
        primvar.Set(Vt.Vec3fArray.FromNumpy(color_f))
    except (AttributeError, TypeError):  # pragma: no cover - USD-version guard
        mesh.CreatePointsAttr().Set([Gf.Vec3f(float(p[0]), float(p[1]), float(p[2])) for p in vertices])
        mesh.CreateFaceVertexCountsAttr().Set([3] * len(faces))
        mesh.CreateFaceVertexIndicesAttr().Set([int(i) for i in faces.reshape(-1)])
        primvar.Set([Gf.Vec3f(float(r), float(g), float(b)) for r, g, b in color_f])
    lo, hi = vertices.min(axis=0), vertices.max(axis=0)
    mesh.CreateExtentAttr().Set([
        Gf.Vec3f(float(lo[0]), float(lo[1]), float(lo[2])),
        Gf.Vec3f(float(hi[0]), float(hi[1]), float(hi[2])),
    ])

    if not collision:
        return
    prim = mesh.GetPrim()
    UsdPhysics.CollisionAPI.Apply(prim)
    # "none" = exact triangle-mesh collider (no convex approximation). Valid
    # because terrain is a STATIC collider; the rover drives on the actual
    # reconstructed surface, so slope/rock/clearance are real geometry.
    UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr().Set("none")
    if physics_material_path is not None:
        material = UsdShade.Material(stage.GetPrimAtPath(physics_material_path))
        if material:
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(
                material, materialPurpose="physics")


def author_nvblox_tiles(stage, map_dir, root_path: str,
                        state: dict | None = None,
                        collision: bool = False,
                        terrain_static_friction: float = 1.0,
                        terrain_dynamic_friction: float = 1.0,
                        terrain_restitution: float = 0.0) -> tuple[int, int, int]:
    """Author the live nvblox mesh under ``root_path`` from the per-tile transport
    written by ``nvblox_live_map.py`` (``manifest.json`` + ``tiles/<name>.npz``),
    re-authoring only the tiles whose fingerprint changed since the last call.

    The producer already tiled the mesh and fingerprinted each tile, so the work
    here is purely O(changed tiles): load each changed tile's small npz and
    (re)author its Mesh prim; drop prims for tiles that vanished. That keeps the
    twin tracking the scan even as the mapped volume grows, because the RTX
    renderer only rebuilds the handful of prims at the frontier.

    ``state`` (caller-owned, persists across calls) maps tile name -> fingerprint
    (the manifest's value, a list). Returns (n_tiles, n_authored, n_removed).

    With ``collision``, every tile is authored as an exact triangle-mesh collider
    bound to a shared regolith physics material (friction/restitution knobs), so
    the rover physically drives on the live terrain."""
    from pxr import UsdGeom

    if state is None:
        state = {}
    manifest = load_manifest(map_dir)
    tiles = manifest.get("tiles", {})

    # First author this session: clear any prior prim (e.g. a stale single mesh).
    if not state:
        stage.RemovePrim(root_path)
    UsdGeom.Xform.Define(stage, root_path)

    material_path = None
    if collision:
        material_path = ensure_terrain_physics_material(
            stage, f"{root_path}/{TERRAIN_MATERIAL_NAME}",
            terrain_static_friction, terrain_dynamic_friction, terrain_restitution)

    authored = 0
    for name, fp in tiles.items():
        if state.get(name) == fp:
            continue
        try:
            data = np.load(tile_npz_path(map_dir, name))
        except (FileNotFoundError, OSError, ValueError):
            # Producer raced (rewriting this tile, or removed it after we read the
            # manifest). Leave state untouched so we retry on the next poll.
            continue
        author_mesh_prim(stage, f"{root_path}/{name}",
                         data["vertices"], data["faces"], data["colors"],
                         collision=collision, physics_material_path=material_path)
        state[name] = fp
        authored += 1

    # Drop prims for tiles no longer in the manifest.
    gone = [n for n in state if n not in tiles]
    for name in gone:
        stage.RemovePrim(f"{root_path}/{name}")
        del state[name]
    return len(tiles), authored, len(gone)
