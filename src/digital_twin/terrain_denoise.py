"""Feature-preserving denoise for the live nvblox terrain mesh.

Why this exists: the reconstructed surface carries high-frequency TSDF noise at
the voxel scale (~cm). On a triangle-mesh collider that noise becomes phantom
micro-hazards -- the wheels catch on facets that aren't really there, which
corrupts exactly the lunar-traversability quantities the research is about
(slope, clearance, tip-over). A blanket lowpass would also erase REAL
wheel-scale hazards (rocks, steps), which is just as wrong.

The lever is a **bounded** Laplacian smoother: iteratively move each vertex
toward its neighbours' centroid (kills high-frequency noise), but clamp every
vertex's total displacement from its ORIGINAL position to ``max_displacement_m``.
Sub-voxel noise is a small, incoherent perturbation, so the clamp never binds and
it smooths away. A real rock/step is a large, *coherent* displacement of many
vertices; flattening it would require moving them much further than the clamp
allows, so the clamp preserves it. ``max_displacement_m`` is therefore the
noise-amplitude dial -- set it to the reconstruction noise scale (a small
fraction of the voxel size), NOT to the hazard scale.

Pure numpy, operates on ``(vertices, faces)`` and returns new vertices (faces,
colors, and vertex indexing are unchanged), so it runs producer-side in the
mapper before tiling and is unit-testable off-GPU.
"""

from __future__ import annotations

import numpy as np


def denoise_terrain_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    iterations: int = 10,
    step: float = 0.5,
    max_displacement_m: float = 0.01,
) -> np.ndarray:
    """Return denoised vertices (float32, same shape/order as ``vertices``).

    Parameters
    ----------
    vertices : (N, 3) array of vertex positions (metres, twin frame).
    faces : (M, 3) int array of triangle vertex indices.
    iterations : Laplacian smoothing passes.
    step : per-pass blend toward the neighbour centroid in [0, 1] (0 = no move).
    max_displacement_m : hard cap on how far any vertex may move from its
        original position. This is the noise-amplitude dial: noise smooths away
        below it, coherent real features (rocks/steps) are preserved because
        flattening them would exceed it.
    """
    verts = np.asarray(vertices, dtype=np.float64)
    tris = np.asarray(faces)
    if verts.shape[0] == 0 or tris.size == 0 or iterations <= 0 or step <= 0.0:
        return np.asarray(vertices, dtype=np.float32)

    n = verts.shape[0]
    original = verts.copy()
    moving = verts.copy()

    # Undirected adjacency as parallel (src, dst) index arrays: each triangle
    # edge contributes both directions, so a vertex's neighbour set is dst[src==i].
    a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
    src = np.concatenate([a, b, b, c, c, a])
    dst = np.concatenate([b, a, c, b, a, c])
    degree = np.bincount(src, minlength=n).astype(np.float64)
    has_neighbours = degree > 0
    degree_safe = np.where(has_neighbours, degree, 1.0)[:, None]

    for _ in range(int(iterations)):
        neighbour_sum = np.zeros_like(moving)
        np.add.at(neighbour_sum, src, moving[dst])
        centroid = neighbour_sum / degree_safe
        stepped = moving + step * (centroid - moving)
        # Vertices with no neighbours (isolated) stay put.
        stepped = np.where(has_neighbours[:, None], stepped, moving)
        # Clamp total displacement from the ORIGINAL position to the noise scale.
        delta = stepped - original
        dist = np.linalg.norm(delta, axis=1)
        scale = np.where(
            dist > max_displacement_m,
            max_displacement_m / np.maximum(dist, 1e-12),
            1.0,
        )
        moving = original + delta * scale[:, None]

    return moving.astype(np.float32)
