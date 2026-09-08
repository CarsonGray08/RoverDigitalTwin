"""Fill the rover's occlusion disc in the final terrain mesh.

During a turn-in-place 360 the rover body occludes the floor directly beneath
it, leaving a roughly circular hole in the reconstructed mesh centred on the
rover start pose. This module closes that hole with a flat/locally-planar disc
of triangles that is *flush* with the surrounding floor, so the rover sits on
real geometry and Isaac's resting-Z lookup finds support there.

The patch is inferred from the surrounding floor and authored as ordinary
terrain geometry for both belly-clearance and plan validation. It is a
deliberate, post-scan closure step -- never run mid-scan while the rover is
still occluding its own footprint.

Pure numpy so it is unit-testable off-GPU/off-Isaac, mirroring
``floor_stabilize``. The mapper applies it on the final mesh only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Default appearance for the patch when the surrounding floor has no usable
# colour (mid grey); real scans always carry per-vertex colour and override it.
_DEFAULT_COLOR = (170, 170, 170)


@dataclass(frozen=True)
class FloorDiscPatch:
    """Reusable under-rover floor-fill overlay."""

    center: tuple[float, float]
    cell_m: float
    hole_cells: frozenset[tuple[int, int]]
    vertices: np.ndarray
    faces: np.ndarray
    colors: np.ndarray | None


def fill_floor_disc(
    vertices,
    faces,
    colors,
    center,
    max_radius_m: float,
    *,
    cell_m: float = 0.05,
    ring_margin_m: float = 0.35,
    min_ring_points: int = 12,
    seed_radius_m: float = 0.2,
    floor_band_m: float = 0.05,
    max_slope: float = 1.0,
    preserve_obstacles: bool = True,
    seam_overlap_cells: int = 0,
    seam_drop_m: float = 0.0,
):
    """Return ``(vertices, faces, colors)`` with the hole under ``center`` filled.

    ``center`` is the twin-frame ``(x, y)`` of the rover body (the occlusion
    centre). The hole is found *adaptively*: flood-fill the empty fill-cells
    connected to the centre, stopping at floor-like cells that belong to the
    surrounding floor, bounded only by ``max_radius_m`` as a safety cap against
    leaking through a gap in the floor ring. This fills the actual occlusion
    footprint whatever its size/shape, rather than a fixed disc. Disconnected
    scan speckles inside the footprint are treated as part of the occlusion
    instead of carving holes through the patch.
    The patch height is a robust plane fit to the floor band of the geometry
    surrounding the hole (median + MAD reject drops walls/rocks), so it sits
    flush with a level or gently tilted floor.

    ``seam_overlap_cells`` expands the patch a few grid cells into the detected
    surrounding floor. That matters because floor evidence is sampled: if the
    patch stops exactly at the first floor sample, the real mesh can still have a
    jagged moat at the edge. With ``preserve_obstacles=True`` above-floor cells
    are left unpatched for callers that must not cover real objects. With it
    disabled, original geometry in the filled cells is removed and replaced by
    the clean support patch; the live under-rover 360 workflow uses that mode
    because any geometry in the blind spot is usually reconstruction debris.

    A no-op (inputs returned unchanged) when there is no input geometry or there
    is too little surrounding floor to match against -- better an honest hole
    than a fabricated floor at a guessed height.
    """
    patch = build_floor_disc_patch(
        vertices,
        faces,
        colors,
        center,
        max_radius_m,
        cell_m=cell_m,
        ring_margin_m=ring_margin_m,
        min_ring_points=min_ring_points,
        seed_radius_m=seed_radius_m,
        floor_band_m=floor_band_m,
        max_slope=max_slope,
        preserve_obstacles=preserve_obstacles,
        seam_overlap_cells=seam_overlap_cells,
        seam_drop_m=seam_drop_m,
    )
    if patch is None:
        return vertices, faces, colors
    return apply_floor_disc_patch(
        vertices,
        faces,
        colors,
        patch,
        remove_original_hole=not preserve_obstacles,
    )


def build_floor_disc_patch(
    vertices,
    faces,
    colors,
    center,
    max_radius_m: float,
    *,
    cell_m: float = 0.05,
    ring_margin_m: float = 0.35,
    min_ring_points: int = 12,
    seed_radius_m: float = 0.2,
    floor_band_m: float = 0.05,
    max_slope: float = 1.0,
    preserve_obstacles: bool = True,
    seam_overlap_cells: int = 0,
    seam_drop_m: float = 0.0,
) -> FloorDiscPatch | None:
    """Build a reusable floor-fill patch from the current mesh.

    The expensive part is finding the hole and fitting the floor plane. Once this
    returns a patch, later mesh exports can call ``apply_floor_disc_patch`` to
    splice the saved overlay into fresh raw mesh extractions without detecting
    the hole again.
    """
    V = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
    F_in = np.asarray(faces).reshape(-1, 3)
    C_in = np.asarray(colors)
    if V.shape[0] == 0 or max_radius_m <= 0.0 or cell_m <= 0.0:
        return None
    cx, cy = float(center[0]), float(center[1])
    rel = V[:, :2] - (cx, cy)
    dist = np.hypot(rel[:, 0], rel[:, 1])

    # Floor band of the geometry around the hole, for a robust fill height/colour.
    band = dist <= max_radius_m + ring_margin_m
    if int(band.sum()) < min_ring_points:
        return None
    band_v = V[band]
    z = band_v[:, 2]
    zmed = float(np.median(z))
    zmad = float(np.median(np.abs(z - zmed)))
    floor_sel = z <= zmed + max(floor_band_m, 3.0 * zmad)
    if int(floor_sel.sum()) < min_ring_points:
        floor_sel = np.ones(len(band_v), dtype=bool)  # band is all floor-ish
    floor_v = band_v[floor_sel]
    plane = _fit_plane(floor_v, float(np.median(floor_v[:, 2])), max_slope)
    patch_color = _ring_color(C_in, len(V), band, floor_sel)

    floor_cells, obstacle_cells = _classify_surface_cells(
        V, F_in, cx, cy, cell_m, max_radius_m, plane, floor_band_m)
    barrier_cells = _outer_floor_barrier_cells(floor_cells, cell_m, max_radius_m)
    if not barrier_cells:
        # Sparse point-only inputs may not have an obvious connected outer floor
        # component. In that case fall back to the conservative old behavior for
        # floor-height cells only; above-floor fragments still do not block the
        # fill from reaching the rest of the footprint.
        barrier_cells = floor_cells
    hole = _flood_fill_hole(barrier_cells, cell_m, max_radius_m, seed_radius_m)
    if preserve_obstacles:
        hole.difference_update(obstacle_cells)
    if not hole:
        return None
    patch_cells = _expand_patch_into_seam(
        hole,
        barrier_cells,
        obstacle_cells if preserve_obstacles else set(),
        max(0, int(seam_overlap_cells)),
    )

    # Sit the patch a hair below the fitted floor so the retained seam floor wins
    # the overlap (no z-fight, no exposed patch edge) while the genuine hole still
    # reads as filled. seam_drop_m=0 keeps the patch exactly on the plane.
    a_p, b_p, c_p = plane
    patch_plane = (a_p, b_p, c_p - max(0.0, float(seam_drop_m)))
    new_v, new_f = _cells_to_mesh(patch_cells, cx, cy, cell_m, patch_plane)
    if len(new_f) == 0:
        return None

    has_color = C_in.ndim == 2 and C_in.shape[0] == len(V) and C_in.shape[1] >= 3
    patch_colors = None
    if has_color:
        patch_colors = np.tile(patch_color.astype(C_in.dtype), (len(new_v), 1))
    return FloorDiscPatch(
        center=(cx, cy),
        cell_m=float(cell_m),
        hole_cells=frozenset((int(i), int(j)) for i, j in hole),
        vertices=new_v.astype(np.float32),
        faces=new_f.astype(np.int64),
        colors=patch_colors,
    )


def apply_floor_disc_patch(
    vertices,
    faces,
    colors,
    patch: FloorDiscPatch,
    *,
    remove_original_hole: bool = True,
):
    """Append a saved fill patch to a freshly extracted mesh."""
    base_v = np.asarray(vertices, dtype=np.float32).reshape(-1, 3)
    base_f = np.asarray(faces).reshape(-1, 3)
    base_c = np.asarray(colors)
    if remove_original_hole:
        # Drop only the interior occlusion debris (the real hole), NEVER the
        # surrounding floor we overlap. Keeping the seam floor lets it render on
        # top of the slightly-lowered patch, burying the patch's straight grid
        # boundary so it can't show a T-junction crack against the irregular
        # nvblox floor mesh.
        base_v, base_f, base_c = _drop_original_cells(
            base_v,
            base_f,
            base_c,
            patch.center[0],
            patch.center[1],
            patch.cell_m,
            set(patch.hole_cells),
        )
    out_v = np.vstack([base_v, patch.vertices.astype(np.float32, copy=False)])
    out_f = np.vstack([
        base_f,
        (patch.faces + len(base_v)).astype(base_f.dtype, copy=False),
    ])
    if (
        patch.colors is not None
        and base_c.ndim == 2
        and base_c.shape[0] == len(base_v)
        and base_c.shape[1] >= 3
    ):
        out_c = np.vstack([base_c, patch.colors.astype(base_c.dtype, copy=False)])
    else:
        out_c = base_c
    return out_v, out_f, out_c


def weld_mesh(vertices, faces, colors, tol_m: float = 0.0005):
    """Merge coincident vertices so the mesh becomes a connected surface.

    nvblox emits per-block, *unwelded* vertices: adjacent triangles do not share
    indices, so the raw mesh is thousands of disconnected fragments with no real
    surface topology (every shared edge looks like a boundary). Welding within
    ``tol_m`` recovers the connected surface this module's continuity work needs,
    and is a strict improvement for downstream rendering/collision too. Colours
    are averaged per merged group; degenerate faces are dropped.
    """
    V = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
    F = np.asarray(faces).reshape(-1, 3)
    C = np.asarray(colors)
    if len(V) == 0:
        return vertices, faces, colors
    keys = np.round(V / float(tol_m)).astype(np.int64)
    _, inv = np.unique(keys, axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    n = int(inv.max()) + 1
    counts = np.bincount(inv, minlength=n).astype(np.float64)
    VW = np.column_stack([np.bincount(inv, V[:, i], minlength=n) / counts for i in range(3)])
    FW = inv[F]
    nondegen = (FW[:, 0] != FW[:, 1]) & (FW[:, 1] != FW[:, 2]) & (FW[:, 0] != FW[:, 2])
    FW = FW[nondegen]
    has_color = C.ndim == 2 and C.shape[0] == len(V) and C.shape[1] >= 3
    if has_color:
        CW = np.column_stack([
            np.bincount(inv, C[:, i].astype(np.float64), minlength=n) / counts
            for i in range(C.shape[1])
        ]).astype(C.dtype)
    else:
        CW = C
    return VW.astype(np.float32), FW.astype(np.int64), CW


def fill_floor_disc_delaunay(
    vertices,
    faces,
    colors,
    center,
    max_radius_m: float,
    *,
    cell_m: float = 0.05,
    ring_margin_m: float = 0.35,
    weld_tol_m: float = 0.0005,
    max_edge_m: float | None = None,
    min_ring_points: int = 12,
    floor_band_m: float = 0.05,
    max_slope: float = 1.0,
):
    """Weld the mesh, then Delaunay-remesh the rover footprint into one sheet.

    Loop-based hole filling is impossible on a reconstructed floor whose hole is a
    *sponge* (mostly-empty centre, scattered floor islands, no single rim). This
    instead retriangulates the footprint disc over {welded floor vertices + fill
    points in the empty cells}, so the fill connects to the real floor vertices
    and the local sponge becomes one continuous, shared-vertex surface. Floor
    vertices keep their measured height; empty cells get the fitted floor plane,
    so the result is flush. Original faces inside the disc are replaced; faces
    outside are kept and share the welded boundary vertices.

    Returns the welded mesh (a strict cleanup) even if there is nothing to fill.
    """
    from scipy.spatial import Delaunay
    try:
        from scipy.spatial import QhullError  # scipy >= 1.8
    except ImportError:  # pragma: no cover - older scipy
        from scipy.spatial.qhull import QhullError

    VW, FW, CW = weld_mesh(vertices, faces, colors, weld_tol_m)
    VW = np.asarray(VW, dtype=np.float64).reshape(-1, 3)
    FW = np.asarray(FW).reshape(-1, 3)
    CW = np.asarray(CW)
    if len(VW) == 0:
        return vertices, faces, colors

    welded = (VW.astype(np.float32), FW.astype(np.int64), CW)
    cx, cy = float(center[0]), float(center[1])
    rel = VW[:, :2] - (cx, cy)
    dist = np.hypot(rel[:, 0], rel[:, 1])
    r_floor = max_radius_m + ring_margin_m

    band = dist <= r_floor
    if int(band.sum()) < min_ring_points:
        return welded
    bz = VW[band, 2]
    zmed = float(np.median(bz))
    zmad = float(np.median(np.abs(bz - zmed)))
    floor_sel = bz <= zmed + max(floor_band_m, 3.0 * zmad)
    if int(floor_sel.sum()) < min_ring_points:
        floor_sel = np.ones(int(band.sum()), dtype=bool)
    plane = _fit_plane(VW[band][floor_sel], float(np.median(VW[band][floor_sel][:, 2])), max_slope)
    a, b, c = plane
    patch_color = _delaunay_patch_color(CW, len(VW), band, floor_sel)

    # Floor vertices that take part in the remesh: floor-band, within the ring.
    resid = VW[:, 2] - (a * VW[:, 0] + b * VW[:, 1] + c)
    floor_tol = max(float(floor_band_m), 0.75 * float(cell_m))
    floor_mask = (dist <= r_floor) & (np.abs(resid) <= floor_tol)
    floor_idx = np.where(floor_mask)[0]
    if len(floor_idx) < 3:
        return welded

    # Fill points: plane-height samples in the empty cells inside the footprint.
    occ = {(int(i), int(j)) for i, j in np.floor(rel[floor_mask] / cell_m).astype(np.int64)}
    fill_pts = _disc_fill_points(cx, cy, max_radius_m, cell_m, plane, occ)
    if len(fill_pts) == 0:
        return welded  # no hole; the welded cleanup still stands

    floor_xy = VW[floor_idx, :2]
    pts_xy = np.vstack([floor_xy, fill_pts[:, :2]])
    try:
        tri = Delaunay(pts_xy)
    except QhullError:
        return welded

    if max_edge_m is None:
        max_edge_m = 3.0 * cell_m
    simp = tri.simplices
    p = pts_xy
    e = np.maximum.reduce([
        np.linalg.norm(p[simp[:, 0]] - p[simp[:, 1]], axis=1),
        np.linalg.norm(p[simp[:, 1]] - p[simp[:, 2]], axis=1),
        np.linalg.norm(p[simp[:, 2]] - p[simp[:, 0]], axis=1),
    ])
    # Keep by ALL three vertices (not centroid) inside the disc, so the disc's
    # outer boundary lands exactly on the shared floor-vertex ring -- this hands
    # off to the retained floor with the fewest residual T-junctions.
    prad = np.hypot(p[:, 0] - cx, p[:, 1] - cy)
    keep = (e <= max_edge_m) & (prad[simp] <= max_radius_m + cell_m).all(axis=1)
    simp = simp[keep]
    if len(simp) == 0:
        return welded

    # Orient CCW in XY so the patch normal points +Z (up).
    txy = p[simp]
    area2 = ((txy[:, 1, 0] - txy[:, 0, 0]) * (txy[:, 2, 1] - txy[:, 0, 1])
             - (txy[:, 2, 0] - txy[:, 0, 0]) * (txy[:, 1, 1] - txy[:, 0, 1]))
    flip = area2 < 0
    simp[flip] = simp[flip][:, [0, 2, 1]]

    base = len(VW)
    local_to_global = np.concatenate([floor_idx, base + np.arange(len(fill_pts))])
    disc_faces = local_to_global[simp]

    # Replace original faces fully inside the footprint; keep faces with any
    # vertex outside (they share the welded boundary vertices with the disc, so
    # the handoff stays connected).
    vrad = np.hypot(VW[:, 0] - cx, VW[:, 1] - cy)
    keep_orig = ~(vrad[FW] <= max_radius_m).all(axis=1)

    out_v = np.vstack([VW, fill_pts]).astype(np.float32)
    out_f = np.vstack([FW[keep_orig], disc_faces]).astype(np.int64)
    has_color = CW.ndim == 2 and CW.shape[0] == len(VW) and CW.shape[1] >= 3
    if has_color:
        patch = np.tile(patch_color.astype(CW.dtype), (len(fill_pts), 1))
        out_c = np.vstack([CW, patch])
    else:
        out_c = CW
    return out_v, out_f, out_c


def _disc_fill_points(cx, cy, radius_m, cell_m, plane, occupied):
    a, b, c = plane
    n = int(np.ceil(radius_m / cell_m))
    pts = []
    r2 = radius_m * radius_m
    for i in range(-n, n + 1):
        for j in range(-n, n + 1):
            xc = (i + 0.5) * cell_m
            yc = (j + 0.5) * cell_m
            if xc * xc + yc * yc > r2 or (i, j) in occupied:
                continue
            x = cx + xc
            y = cy + yc
            pts.append((x, y, a * x + b * y + c))
    return np.asarray(pts, dtype=np.float64).reshape(-1, 3)


def _delaunay_patch_color(colors, n_vertices, band, floor_sel):
    if colors.ndim != 2 or colors.shape[0] != n_vertices or colors.shape[1] < 3:
        return np.asarray(_DEFAULT_COLOR, dtype=np.float64)
    cc = colors[band][floor_sel][:, :3]
    if len(cc) == 0:
        return np.asarray(_DEFAULT_COLOR, dtype=np.float64)
    return np.median(cc.astype(np.float64), axis=0)


def _flood_fill_hole(
    blocked: set[tuple[int, int]],
    cell_m: float,
    max_radius_m: float,
    seed_radius_m: float,
) -> set[tuple[int, int]]:
    """Cells connected to the centre, walled in by outer floor geometry.

    Seeds from any unblocked cell within ``seed_radius_m`` of the centre (so a
    stray vertex exactly at the origin doesn't block the fill), then 8-connected
    expands over empty cells out to ``max_radius_m``."""
    seed_cells = int(np.ceil(seed_radius_m / cell_m))
    seeds = [
        (a, b)
        for a in range(-seed_cells, seed_cells + 1)
        for b in range(-seed_cells, seed_cells + 1)
        if (a, b) not in blocked
        and ((a + 0.5) * cell_m) ** 2 + ((b + 0.5) * cell_m) ** 2 <= seed_radius_m ** 2
    ]
    hole: set[tuple[int, int]] = set()
    stack = list(seeds)
    r2 = max_radius_m * max_radius_m
    neighbors = [
        (1, 0), (-1, 0), (0, 1), (0, -1),
        (1, 1), (1, -1), (-1, 1), (-1, -1),
    ]
    while stack:
        a, b = stack.pop()
        if (a, b) in hole or (a, b) in blocked:
            continue
        if ((a + 0.5) * cell_m) ** 2 + ((b + 0.5) * cell_m) ** 2 > r2:
            continue
        hole.add((a, b))
        stack.extend((a + da, b + db) for da, db in neighbors)
    return hole


def _classify_surface_cells(
    vertices: np.ndarray,
    faces: np.ndarray,
    cx: float,
    cy: float,
    cell_m: float,
    max_radius_m: float,
    plane: tuple[float, float, float],
    floor_band_m: float,
) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    """Rasterize local floor and obstacle evidence into fill-grid cells.

    Vertices alone are too sparse for some nvblox meshes, so valid face centroids
    are included as additional surface samples. Classification is relative to the
    fitted fill plane: floor-like samples form the boundary; above-floor samples
    are optional exclusions but never flood-fill walls.
    """
    samples = [np.asarray(vertices, dtype=np.float64).reshape(-1, 3)]
    if faces.size:
        tri = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
        valid = (tri >= 0).all(axis=1) & (tri < len(vertices)).all(axis=1)
        if valid.any():
            samples.append(vertices[tri[valid]].mean(axis=1))
    pts = np.vstack(samples) if samples else np.zeros((0, 3), dtype=np.float64)
    if len(pts) == 0:
        return set(), set()

    rel_xy = pts[:, :2] - (cx, cy)
    dist = np.hypot(rel_xy[:, 0], rel_xy[:, 1])
    local = dist <= max_radius_m + cell_m
    if not local.any():
        return set(), set()

    pts = pts[local]
    rel_xy = rel_xy[local]
    a, b, c = plane
    residual = pts[:, 2] - (a * pts[:, 0] + b * pts[:, 1] + c)
    floor_tol = max(float(floor_band_m), 0.75 * float(cell_m))
    obstacle_tol = max(2.0 * floor_tol, 0.12)
    cell_ij = np.floor(rel_xy / cell_m).astype(np.int64)

    floor_cells = {
        (int(i), int(j))
        for (i, j), r in zip(cell_ij, residual)
        if abs(float(r)) <= floor_tol
    }
    obstacle_cells = {
        (int(i), int(j))
        for (i, j), r in zip(cell_ij, residual)
        if float(r) > obstacle_tol
    }
    return floor_cells, obstacle_cells


def _outer_floor_barrier_cells(
    floor_cells: set[tuple[int, int]],
    cell_m: float,
    max_radius_m: float,
) -> set[tuple[int, int]]:
    """Floor cells connected to the outer scan floor, not interior speckles."""
    if not floor_cells:
        return set()
    outer_r2 = max(0.0, max_radius_m - 2.0 * cell_m) ** 2
    remaining = set(floor_cells)
    barrier: set[tuple[int, int]] = set()
    neighbors = [
        (1, 0), (-1, 0), (0, 1), (0, -1),
        (1, 1), (1, -1), (-1, 1), (-1, -1),
    ]
    while remaining:
        start = remaining.pop()
        component = {start}
        stack = [start]
        touches_outer = _cell_radius2(start, cell_m) >= outer_r2
        while stack:
            cell = stack.pop()
            for da, db in neighbors:
                nxt = (cell[0] + da, cell[1] + db)
                if nxt not in remaining:
                    continue
                remaining.remove(nxt)
                component.add(nxt)
                stack.append(nxt)
                if _cell_radius2(nxt, cell_m) >= outer_r2:
                    touches_outer = True
        if touches_outer:
            barrier.update(component)
    return barrier


def _cell_radius2(cell: tuple[int, int], cell_m: float) -> float:
    a, b = cell
    return ((a + 0.5) * cell_m) ** 2 + ((b + 0.5) * cell_m) ** 2


def _expand_patch_into_seam(
    hole: set[tuple[int, int]],
    barrier_cells: set[tuple[int, int]],
    excluded_cells: set[tuple[int, int]],
    layers: int,
) -> set[tuple[int, int]]:
    """Add a narrow floor overlap ring so the patch closes sampled seams."""
    patch = set(hole)
    if layers <= 0 or not barrier_cells:
        return patch
    frontier = set(hole)
    neighbors = [
        (1, 0), (-1, 0), (0, 1), (0, -1),
        (1, 1), (1, -1), (-1, 1), (-1, -1),
    ]
    for _ in range(layers):
        next_frontier: set[tuple[int, int]] = set()
        for cell in frontier:
            for da, db in neighbors:
                nxt = (cell[0] + da, cell[1] + db)
                if nxt in patch or nxt in excluded_cells or nxt not in barrier_cells:
                    continue
                next_frontier.add(nxt)
        if not next_frontier:
            break
        patch.update(next_frontier)
        frontier = next_frontier
    return patch


def _drop_original_cells(
    vertices: np.ndarray,
    faces: np.ndarray,
    colors: np.ndarray,
    cx: float,
    cy: float,
    cell_m: float,
    cells: set[tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Remove original geometry in cells that will be replaced by the patch."""
    if len(vertices) == 0 or not cells:
        return vertices, faces, colors
    rel = vertices[:, :2] - (cx, cy)
    cell_ij = np.floor(rel / cell_m).astype(np.int64)
    drop = np.fromiter(
        ((int(i), int(j)) in cells for i, j in cell_ij),
        dtype=bool,
        count=len(vertices),
    )
    if not drop.any():
        return vertices, faces, colors

    keep = ~drop
    remap = np.full(len(vertices), -1, dtype=np.int64)
    remap[keep] = np.arange(int(keep.sum()), dtype=np.int64)
    out_v = vertices[keep]
    if faces.size:
        face_keep = keep[faces].all(axis=1)
        out_f = remap[faces[face_keep]].astype(faces.dtype, copy=False)
    else:
        out_f = faces
    if colors.ndim == 2 and colors.shape[0] == len(vertices):
        out_c = colors[keep]
    else:
        out_c = colors
    return out_v, out_f, out_c


def _fit_plane(points: np.ndarray, fallback_z: float, max_slope: float) -> tuple[float, float, float]:
    """Least-squares ``z = a*x + b*y + c`` over floor points; flat fallback.

    Falls back to a level plane at ``fallback_z`` when the fit is degenerate or
    the slope is implausibly steep (the ring picked up something non-floor)."""
    if len(points) < 3:
        return 0.0, 0.0, fallback_z
    a_mat = np.column_stack([points[:, 0], points[:, 1], np.ones(len(points))])
    try:
        coeffs, *_ = np.linalg.lstsq(a_mat, points[:, 2], rcond=None)
    except np.linalg.LinAlgError:
        return 0.0, 0.0, fallback_z
    a, b, c = (float(v) for v in coeffs)
    if not all(np.isfinite([a, b, c])) or abs(a) > max_slope or abs(b) > max_slope:
        return 0.0, 0.0, fallback_z
    return a, b, c


def _ring_color(colors: np.ndarray, n_vertices: int, ring: np.ndarray, floor_sel: np.ndarray) -> np.ndarray:
    if colors.ndim != 2 or colors.shape[0] != n_vertices or colors.shape[1] < 3:
        return np.asarray(_DEFAULT_COLOR, dtype=np.float64)
    ring_colors = colors[ring][floor_sel][:, :3]
    if len(ring_colors) == 0:
        return np.asarray(_DEFAULT_COLOR, dtype=np.float64)
    return np.median(ring_colors.astype(np.float64), axis=0)


def _cells_to_mesh(
    cells: set[tuple[int, int]],
    cx: float,
    cy: float,
    cell_m: float,
    plane: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Triangulate a set of fill-cells into a watertight patch on ``plane``."""
    a, b, c = plane
    node_index: dict[tuple[int, int], int] = {}
    verts: list[tuple[float, float, float]] = []

    def node(ia: int, ib: int) -> int:
        key = (ia, ib)
        idx = node_index.get(key)
        if idx is None:
            x = cx + ia * cell_m
            y = cy + ib * cell_m
            idx = len(verts)
            verts.append((x, y, a * x + b * y + c))
            node_index[key] = idx
        return idx

    faces: list[tuple[int, int, int]] = []
    for ia, ib in cells:
        n00 = node(ia, ib)
        n10 = node(ia + 1, ib)
        n11 = node(ia + 1, ib + 1)
        n01 = node(ia, ib + 1)
        # CCW from above so the patch normal points +Z (up).
        faces.append((n00, n10, n11))
        faces.append((n00, n11, n01))

    if not faces:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int64)
    return np.asarray(verts, dtype=np.float32), np.asarray(faces, dtype=np.int64)
