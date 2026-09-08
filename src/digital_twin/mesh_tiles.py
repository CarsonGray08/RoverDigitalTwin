"""Spatial 3D tiling + incremental on-disk transport for the live nvblox mesh.

nvblox hands us the WHOLE color mesh every update (the Python binding has no
block-level accessor) and *reorders* it as the TSDF refines, so a naive
"re-author the whole mesh" sink is O(whole map) every refresh and the cost grows
without bound as the room grows -- which is what makes the Isaac twin fall
further and further behind the scan.

This module makes the producer -> consumer transport incremental:

  * :func:`split_into_tiles` buckets the mesh into a 3D grid of ~``tile_size`` m
    tiles with a single ``lexsort`` (O(M log M)), not the old O(tiles*faces)
    per-tile boolean scan.
  * :func:`fingerprint` is an order-invariant, position-quantized hash, so a
    tile that's geometrically stable (already scanned, just reordered/jittered
    by nvblox) hashes the same across updates and is skipped.
  * :func:`write_tiles` (producer side) diffs against the previous fingerprints
    and writes only the CHANGED tiles as small per-tile ``.npz`` files plus a
    tiny ``manifest.json``; the consumer (:func:`load_manifest` +
    :func:`tile_npz_path`) reads the manifest and re-authors only those tiles.
    File writes and USD updates are limited to changed tiles; extraction,
    partitioning, and fingerprinting still process the full mesh.

Pure numpy + stdlib (no torch/USD), so it is unit-testable off-GPU. The USD
authoring lives in ``nvblox_usd.py``; GPU mesh extraction is coordinated by
``_mapper_runtime.py``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterator

import numpy as np

TILES_SUBDIR = "tiles"
MANIFEST_NAME = "manifest.json"
DEFAULT_TILE_SIZE_M = 2.0
# Quantum for the order-invariant fingerprint. nvblox jitters vertices sub-voxel
# as the TSDF refines; rounding to 5 mm keeps a settled tile hashing the same.
DEFAULT_FP_QUANT_M = 0.005

# A fingerprint is (n_vertices, n_faces, xor_hash, sum_hash) of python ints, so
# it round-trips through JSON (as a list) for the manifest.
Fingerprint = tuple


def tile_name(ijk) -> str:
    """USD-safe identifier for a tile index: no '-' ('m' for minus), no leading
    digit. e.g. ``(3, -1, 0) -> 'tile_3_m1_0'``."""
    return "tile_" + "_".join(str(int(v)).replace("-", "m") for v in ijk)


def fingerprint(vertices: np.ndarray, n_faces: int,
                quant: float = DEFAULT_FP_QUANT_M) -> Fingerprint:
    """Order-invariant, position-quantized content fingerprint of a tile.

    nvblox regenerates and REORDERS the whole mesh every update (and jitters
    vertices sub-voxel), so a byte hash flags every tile as changed. Quantizing
    positions to ``quant`` m and reducing order-independently (xor + sum) makes a
    geometrically stable tile -- one already scanned and moved past -- hash the
    same across updates, so we can skip it."""
    q = np.round(np.ascontiguousarray(vertices, np.float64) / quant).astype(np.int64)
    rk = ((q[:, 0] * 73856093) ^ (q[:, 1] * 19349663) ^ (q[:, 2] * 83492791)).astype(np.uint64)
    with np.errstate(over="ignore"):  # uint64 reductions wrap mod 2^64 (intended)
        return (int(len(q)), int(n_faces),
                int(np.bitwise_xor.reduce(rk)), int(np.add.reduce(rk)))


def split_into_tiles(
    vertices: np.ndarray, faces: np.ndarray, colors: np.ndarray,
    tile_size: float = DEFAULT_TILE_SIZE_M,
) -> Iterator[tuple[str, np.ndarray, np.ndarray, np.ndarray]]:
    """Yield ``(name, local_vertices, local_faces, local_colors)`` per tile.

    Each face goes to exactly one tile by its centroid; a vertex shared across
    tiles is duplicated into each (so a tile is a standalone sub-mesh and its
    fingerprint depends only on its own geometry). A single ``lexsort`` groups
    faces by tile -- O(M log M) -- instead of the old O(tiles*faces) per-tile
    boolean scan. ``tile_size <= 0`` collapses everything into one tile."""
    vertices = np.asarray(vertices, dtype=np.float32).reshape(-1, 3)
    faces = np.asarray(faces, dtype=np.int32).reshape(-1, 3)
    colors = np.asarray(colors, dtype=np.uint8).reshape(-1, 3)
    if len(vertices) == 0 or len(faces) == 0:
        return

    if tile_size and tile_size > 0:
        centroids = vertices[faces].mean(axis=1)                 # (M, 3)
        idx = np.floor(centroids / tile_size).astype(np.int64)   # (M, 3), may be < 0
        order = np.lexsort((idx[:, 2], idx[:, 1], idx[:, 0]))
        s = idx[order]
        boundaries = np.flatnonzero(np.any(np.diff(s, axis=0) != 0, axis=1)) + 1
        groups = np.split(order, boundaries)
    else:
        idx = np.zeros((len(faces), 3), dtype=np.int64)
        groups = [np.arange(len(faces))]

    for grp in groups:
        ijk = tuple(int(v) for v in idx[grp[0]])
        fb = faces[grp]                              # (K, 3) global vertex indices
        used = np.unique(fb.reshape(-1))             # sorted global ids in this tile
        local_faces = np.searchsorted(used, fb).astype(np.int32)
        yield tile_name(ijk), vertices[used], local_faces, colors[used]


def _atomic_write(path: Path, write_fn) -> None:
    """Write via a temp file + ``os.replace`` so a reader never sees a partial
    file (the consumer polls these concurrently)."""
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    try:
        with open(tmp, "wb") as fh:
            write_fn(fh)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def write_tiles(
    out_dir,
    vertices: np.ndarray,
    faces: np.ndarray,
    colors: np.ndarray,
    tile_size: float = DEFAULT_TILE_SIZE_M,
    state: dict | None = None,
    rev: int = 0,
    fp_quant: float = DEFAULT_FP_QUANT_M,
) -> tuple[int, int, int]:
    """Write the mesh as per-tile ``.npz`` + a manifest, re-writing only CHANGED
    tiles.

    ``state`` (caller-owned, persists across calls) maps tile name -> fingerprint.
    Tiles whose fingerprint is unchanged are not rewritten; tiles that vanished
    are deleted. The manifest is written LAST and atomically, listing the full
    current tile set with fingerprints + a monotonic ``rev``, so a consumer that
    reads the manifest then loads the named tiles always finds complete files.
    Returns ``(n_tiles, n_written, n_removed)``."""
    if state is None:
        state = {}
    out_dir = Path(out_dir)
    tiles_dir = out_dir / TILES_SUBDIR
    tiles_dir.mkdir(parents=True, exist_ok=True)

    manifest_tiles: dict[str, list] = {}
    written = 0
    for name, lv, lf, lc in split_into_tiles(vertices, faces, colors, tile_size):
        fp = fingerprint(lv, len(lf), fp_quant)
        manifest_tiles[name] = list(fp)
        if state.get(name) == fp:
            continue
        _atomic_write(
            tiles_dir / f"{name}.npz",
            lambda fh, lv=lv, lf=lf, lc=lc: np.savez(fh, vertices=lv, faces=lf, colors=lc),
        )
        state[name] = fp
        written += 1

    removed = [n for n in list(state) if n not in manifest_tiles]
    for name in removed:
        (tiles_dir / f"{name}.npz").unlink(missing_ok=True)
        del state[name]

    manifest = {"rev": int(rev), "tile_size": float(tile_size), "tiles": manifest_tiles}
    _atomic_write(
        out_dir / MANIFEST_NAME,
        lambda fh: fh.write(json.dumps(manifest).encode("utf-8")),
    )
    return len(manifest_tiles), written, len(removed)


def load_manifest(out_dir) -> dict:
    """Read the manifest written by :func:`write_tiles`; ``{}`` if not present
    yet (the scan hasn't written a mesh)."""
    path = Path(out_dir) / MANIFEST_NAME
    if not path.exists():
        return {}
    with open(path, "rb") as fh:
        return json.loads(fh.read().decode("utf-8"))


def tile_npz_path(out_dir, name: str) -> Path:
    """Path to one tile's ``.npz`` (``vertices``/``faces``/``colors``)."""
    return Path(out_dir) / TILES_SUBDIR / f"{name}.npz"
