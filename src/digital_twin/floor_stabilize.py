"""Final support-floor stabilization for reconstructed terrain meshes."""

from __future__ import annotations

import numpy as np

from digital_twin.scan_loop_eval import fit_floor_plane, floor_height_cells


def stabilize_floor_vertices(
    vertices: np.ndarray,
    *,
    floor_band_m: float = 0.08,
    up_axis: tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> np.ndarray:
    """Snap the dominant low floor band onto one level support plane.

    This is a final mesh regularization step for replay/live terrain output. It
    does not move walls or obstacles outside the detected floor band; it removes
    duplicate floor sheets and small height bands that remain after pose closure
    and depth fusion noise. The support plane is constrained to world-up rather
    than the fitted floor normal so the output floor stays level in the twin.
    """
    verts = np.asarray(vertices, dtype=np.float64)
    if verts.ndim != 2 or verts.shape[1] != 3 or len(verts) < 3:
        return np.asarray(vertices).copy()
    up = np.asarray(up_axis, dtype=np.float64)
    normal, offset, floor_mask = fit_floor_plane(
        verts, up, floor_band_m=float(floor_band_m))
    signed_to_fit = verts @ normal + offset
    support_height = float(np.percentile(verts[:, 2], 10.0))
    if np.count_nonzero(floor_mask) >= 3:
        support_height = float(np.median(verts[floor_mask, 2]))
    snap_mask = floor_mask | (np.abs(signed_to_fit) <= float(floor_band_m))
    snap_mask |= np.abs(verts[:, 2] - support_height) <= float(floor_band_m) * 1.25
    out = verts.copy()
    out[snap_mask, 2] = support_height
    return out.astype(np.asarray(vertices).dtype, copy=False)


def stabilize_floor_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    colors: np.ndarray | None = None,
    *,
    floor_band_m: float = 0.08,
    grid_cell_m: float = 0.05,
    min_floor_component_area_m2: float = 0.05,
    up_axis: tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Stabilize the support floor and drop tiny disconnected floor islands."""
    stabilized = stabilize_floor_vertices(
        vertices, floor_band_m=floor_band_m, up_axis=up_axis)
    verts64 = np.asarray(stabilized, dtype=np.float64)
    faces_arr = np.asarray(faces)
    if faces_arr.ndim != 2 or faces_arr.shape[1] != 3 or len(faces_arr) == 0:
        return stabilized, faces_arr.copy(), None if colors is None else np.asarray(colors).copy()

    floor_vertex_mask, tiny_cells = _tiny_floor_island_mask(
        verts64,
        floor_band_m=floor_band_m,
        grid_cell_m=grid_cell_m,
        min_area_m2=min_floor_component_area_m2,
        up_axis=up_axis,
    )
    if not tiny_cells:
        return stabilized, faces_arr.copy(), None if colors is None else np.asarray(colors).copy()

    cell_ix = np.floor(verts64[:, :2] / float(grid_cell_m)).astype(np.int64)
    vertex_tiny = np.zeros(len(verts64), dtype=bool)
    for idx, is_floor in enumerate(floor_vertex_mask):
        if is_floor and (int(cell_ix[idx, 0]), int(cell_ix[idx, 1])) in tiny_cells:
            vertex_tiny[idx] = True
    # Drop every face that touches a tiny disconnected floor cell. Requiring all
    # three vertices to be tiny leaves sliver/fringe triangles alive, and those
    # retained vertices still count as disconnected floor cells in evaluation.
    drop_face = np.any(vertex_tiny[faces_arr], axis=1)
    kept_faces = faces_arr[~drop_face]
    return _compact_mesh(stabilized, kept_faces, colors)


def _tiny_floor_island_mask(
    vertices: np.ndarray,
    *,
    floor_band_m: float,
    grid_cell_m: float,
    min_area_m2: float,
    up_axis: tuple[float, float, float],
) -> tuple[np.ndarray, set[tuple[int, int]]]:
    normal, offset, floor_mask = fit_floor_plane(
        vertices, np.asarray(up_axis, dtype=np.float64), floor_band_m=floor_band_m)
    cells = floor_height_cells(vertices[floor_mask], normal, offset, grid_cell_m)
    if not cells:
        return floor_mask, set()
    remaining = set(cells)
    components: list[list[tuple[int, int]]] = []
    while remaining:
        start = remaining.pop()
        stack = [start]
        component = [start]
        while stack:
            ix, iy = stack.pop()
            for neighbor in ((ix + 1, iy), (ix - 1, iy), (ix, iy + 1), (ix, iy - 1)):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    stack.append(neighbor)
                    component.append(neighbor)
        components.append(component)
    if not components:
        return floor_mask, set()
    components.sort(key=len, reverse=True)
    tiny: set[tuple[int, int]] = set()
    min_cells = max(1, int(np.ceil(float(min_area_m2) / (float(grid_cell_m) ** 2))))
    for component in components[1:]:
        if len(component) < min_cells:
            tiny.update(component)
    return floor_mask, tiny


def _compact_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    colors: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    if len(faces) == 0:
        return (
            np.zeros((0, 3), dtype=np.asarray(vertices).dtype),
            np.zeros((0, 3), dtype=np.asarray(faces).dtype),
            None if colors is None else np.zeros((0,) + np.asarray(colors).shape[1:], dtype=np.asarray(colors).dtype),
        )
    used = np.unique(faces.reshape(-1))
    remap = np.full((len(vertices),), -1, dtype=np.int64)
    remap[used] = np.arange(len(used), dtype=np.int64)
    compact_faces = remap[faces].astype(np.asarray(faces).dtype, copy=False)
    compact_vertices = np.asarray(vertices)[used]
    compact_colors = None
    if colors is not None:
        colors_arr = np.asarray(colors)
        if len(colors_arr) == len(vertices):
            compact_colors = colors_arr[used]
        else:
            compact_colors = colors_arr.copy()
    return compact_vertices, compact_faces, compact_colors
