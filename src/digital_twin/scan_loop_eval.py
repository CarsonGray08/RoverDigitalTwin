"""Geometry metrics for replayed 360-degree nvblox scans.

The evaluator is intentionally file-based: run the mapper on a fixed
``.scanlog`` to produce ``nvblox_mesh.npz`` and ``trajectory.jsonl``, then run
this module on those artifacts. That makes ARKit, constrained-pose, ICP, and
cuVSLAM replay variants comparable with the same input.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ScanLoopMetrics:
    floor_plane_normal_error_deg: float
    floor_height_rms_m: float
    floor_height_p95_m: float
    start_end_overlap_nearest_surface_error_m: float | None
    start_end_overlap_floor_height_error_m: float | None
    largest_floor_height_discontinuity_m: float
    disconnected_floor_components: int
    disconnected_floor_area_m2: float
    floor_point_count: int
    mesh_vertex_count: int
    mesh_face_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "floor_plane_normal_error_deg": self.floor_plane_normal_error_deg,
            "floor_height_rms_m": self.floor_height_rms_m,
            "floor_height_p95_m": self.floor_height_p95_m,
            "start_end_overlap_nearest_surface_error_m": (
                self.start_end_overlap_nearest_surface_error_m
            ),
            "start_end_overlap_floor_height_error_m": (
                self.start_end_overlap_floor_height_error_m
            ),
            "largest_floor_height_discontinuity_m": (
                self.largest_floor_height_discontinuity_m
            ),
            "disconnected_floor_components": self.disconnected_floor_components,
            "disconnected_floor_area_m2": self.disconnected_floor_area_m2,
            "floor_point_count": self.floor_point_count,
            "mesh_vertex_count": self.mesh_vertex_count,
            "mesh_face_count": self.mesh_face_count,
        }


def load_mesh_npz(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    vertices = np.asarray(data["vertices"], dtype=np.float64)
    faces = np.asarray(data["faces"], dtype=np.int64) if "faces" in data else np.zeros((0, 3), dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"{path}: vertices must be shaped (N, 3)")
    if faces.ndim != 2 or (faces.size and faces.shape[1] != 3):
        raise ValueError(f"{path}: faces must be shaped (M, 3)")
    return vertices, faces


def load_trajectory_jsonl(path: str | Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    p = Path(path)
    if not p.exists():
        return []
    records: list[dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_surface_samples_npz(path: str | Path | None) -> tuple[np.ndarray, np.ndarray] | None:
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        return None
    data = np.load(p)
    points = np.asarray(data["points"], dtype=np.float64)
    frame_indices = np.asarray(data["frame_indices"], dtype=np.int64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"{p}: points must be shaped (N, 3)")
    if frame_indices.shape != (len(points),):
        raise ValueError(f"{p}: frame_indices must match points")
    return points, frame_indices


def evaluate_mesh(
    vertices: np.ndarray,
    faces: np.ndarray | None = None,
    trajectory: list[dict[str, Any]] | None = None,
    surface_samples: tuple[np.ndarray, np.ndarray] | None = None,
    *,
    up_axis: tuple[float, float, float] = (0.0, 0.0, 1.0),
    floor_band_m: float = 0.08,
    grid_cell_m: float = 0.05,
    overlap_radius_m: float = 0.75,
) -> ScanLoopMetrics:
    verts = np.asarray(vertices, dtype=np.float64)
    if verts.ndim != 2 or verts.shape[1] != 3 or len(verts) < 3:
        raise ValueError("at least three mesh vertices are required")
    f = np.asarray(faces if faces is not None else np.zeros((0, 3)), dtype=np.int64)
    up = _unit(np.asarray(up_axis, dtype=np.float64))

    normal, offset, floor_mask = fit_floor_plane(verts, up, floor_band_m=floor_band_m)
    signed = verts @ normal + offset
    floor_signed = signed[floor_mask]
    floor_abs = np.abs(floor_signed)
    normal_error = math.degrees(math.acos(float(np.clip(np.dot(normal, up), -1.0, 1.0))))
    rms = float(np.sqrt(np.mean(floor_abs * floor_abs))) if floor_abs.size else math.nan
    p95 = float(np.percentile(floor_abs, 95)) if floor_abs.size else math.nan

    cells = floor_height_cells(verts[floor_mask], normal, offset, grid_cell_m)
    largest_step = largest_neighbor_discontinuity(cells)
    component_count, disconnected_area = disconnected_floor_components(cells, grid_cell_m)
    seam_error = start_end_overlap_error(
        verts, trajectory or [], overlap_radius_m, surface_samples=surface_samples)
    seam_floor_height = start_end_overlap_floor_height_error(
        trajectory or [],
        overlap_radius_m,
        normal,
        offset,
        floor_band_m,
        surface_samples=surface_samples,
    )

    return ScanLoopMetrics(
        floor_plane_normal_error_deg=float(normal_error),
        floor_height_rms_m=rms,
        floor_height_p95_m=p95,
        start_end_overlap_nearest_surface_error_m=seam_error,
        start_end_overlap_floor_height_error_m=seam_floor_height,
        largest_floor_height_discontinuity_m=float(largest_step),
        disconnected_floor_components=int(component_count),
        disconnected_floor_area_m2=float(disconnected_area),
        floor_point_count=int(np.count_nonzero(floor_mask)),
        mesh_vertex_count=int(len(verts)),
        mesh_face_count=int(len(f)),
    )


def fit_floor_plane(
    vertices: np.ndarray,
    up: np.ndarray,
    *,
    floor_band_m: float = 0.08,
    max_iterations: int = 8,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Fit the dominant low plane and return ``(normal, offset, mask)``.

    The first seed uses the lower z-ish quantile. Iterations then keep points
    close to the plane and below the high side of the seed distribution, which
    lets duplicated floor bands show up as residuals instead of being averaged
    away.
    """
    verts = np.asarray(vertices, dtype=np.float64)
    up = _unit(np.asarray(up, dtype=np.float64))
    heights = verts @ up
    seed_limit = float(np.percentile(heights, 35.0))
    low_mask = heights <= seed_limit + max(0.02, floor_band_m)
    if np.count_nonzero(low_mask) < 3:
        low_mask = heights <= float(np.percentile(heights, 50.0))
    mask = low_mask.copy()
    normal = up.copy()
    offset = -float(np.median(heights[mask]))
    for _ in range(max_iterations):
        normal, offset = _fit_plane(verts[mask], up)
        signed = verts @ normal + offset
        low_side = heights <= seed_limit + floor_band_m * 2.5
        next_mask = (np.abs(signed) <= floor_band_m) & low_side
        if np.count_nonzero(next_mask) < 3:
            break
        if np.array_equal(next_mask, mask):
            mask = next_mask
            break
        mask = next_mask
    return normal, offset, mask


def floor_height_cells(
    floor_points: np.ndarray,
    normal: np.ndarray,
    offset: float,
    cell_m: float,
) -> dict[tuple[int, int], dict[str, float]]:
    if len(floor_points) == 0:
        return {}
    xy = np.floor(floor_points[:, :2] / float(cell_m)).astype(np.int64)
    residual = floor_points @ normal + offset
    grouped: dict[tuple[int, int], list[float]] = {}
    for key_arr, value in zip(xy, residual):
        key = (int(key_arr[0]), int(key_arr[1]))
        grouped.setdefault(key, []).append(float(value))
    cells: dict[tuple[int, int], dict[str, float]] = {}
    for key, values in grouped.items():
        arr = np.asarray(values, dtype=np.float64)
        cells[key] = {
            "median": float(np.median(arr)),
            "spread": float(np.max(arr) - np.min(arr)) if len(arr) else 0.0,
            "count": float(len(arr)),
        }
    return cells


def largest_neighbor_discontinuity(cells: dict[tuple[int, int], dict[str, float]]) -> float:
    largest = 0.0
    for (ix, iy), cell in cells.items():
        here = cell["median"]
        # Include within-cell spread so duplicated floor bands in the same XY
        # footprint are counted even when neighboring medians look smooth.
        largest = max(largest, cell["spread"])
        for neighbor in ((ix + 1, iy), (ix, iy + 1)):
            if neighbor in cells:
                largest = max(largest, abs(here - cells[neighbor]["median"]))
    return float(largest)


def disconnected_floor_components(
    cells: dict[tuple[int, int], dict[str, float]],
    cell_m: float,
) -> tuple[int, float]:
    remaining = set(cells)
    sizes: list[int] = []
    while remaining:
        start = remaining.pop()
        stack = [start]
        size = 1
        while stack:
            ix, iy = stack.pop()
            for n in ((ix + 1, iy), (ix - 1, iy), (ix, iy + 1), (ix, iy - 1)):
                if n in remaining:
                    remaining.remove(n)
                    stack.append(n)
                    size += 1
        sizes.append(size)
    if not sizes:
        return 0, 0.0
    sizes.sort(reverse=True)
    disconnected_cells = sum(sizes[1:])
    return len(sizes), float(disconnected_cells * cell_m * cell_m)


def start_end_overlap_error(
    vertices: np.ndarray,
    trajectory: list[dict[str, Any]],
    overlap_radius_m: float,
    *,
    surface_samples: tuple[np.ndarray, np.ndarray] | None = None,
) -> float | None:
    if surface_samples is not None:
        sampled = start_end_sample_overlap_error(
            surface_samples[0], surface_samples[1], trajectory, overlap_radius_m)
        if sampled is not None:
            return sampled
    poses = [
        np.asarray(record.get("pose"), dtype=np.float64).reshape(4, 4)
        for record in trajectory
        if "pose" in record
    ]
    if len(poses) < 2:
        return None
    start_xy = poses[0][:2, 3]
    end_xy = poses[-1][:2, 3]
    verts = np.asarray(vertices, dtype=np.float64)
    start = verts[np.linalg.norm(verts[:, :2] - start_xy, axis=1) <= overlap_radius_m]
    end = verts[np.linalg.norm(verts[:, :2] - end_xy, axis=1) <= overlap_radius_m]
    if len(start) < 3 or len(end) < 3:
        return None
    # Symmetric nearest-surface distance between the two local vertex clouds.
    a = _nearest_distances(start, end)
    b = _nearest_distances(end, start)
    if a.size == 0 or b.size == 0:
        return None
    return float(np.percentile(np.concatenate([a, b]), 95))


def start_end_sample_overlap_error(
    points: np.ndarray,
    frame_indices: np.ndarray,
    trajectory: list[dict[str, Any]],
    overlap_radius_m: float,
) -> float | None:
    pts = np.asarray(points, dtype=np.float64)
    frame_ids = np.asarray(frame_indices, dtype=np.int64)
    if len(pts) < 6 or len(frame_ids) != len(pts):
        return None
    min_frame = int(np.min(frame_ids))
    max_frame = int(np.max(frame_ids))
    if max_frame <= min_frame:
        return None
    span = max_frame - min_frame + 1
    early_limit = min_frame + max(1, span // 5)
    late_limit = max_frame - max(1, span // 5)
    early = pts[frame_ids <= early_limit]
    late = pts[frame_ids >= late_limit]
    if len(trajectory) >= 2:
        poses = [
            np.asarray(record.get("pose"), dtype=np.float64).reshape(4, 4)
            for record in trajectory
            if "pose" in record
        ]
        if len(poses) >= 2:
            start_xy = poses[0][:2, 3]
            end_xy = poses[-1][:2, 3]
            center = (start_xy + end_xy) * 0.5
            early = early[np.linalg.norm(early[:, :2] - start_xy, axis=1) <= overlap_radius_m]
            late = late[np.linalg.norm(late[:, :2] - end_xy, axis=1) <= overlap_radius_m]
            # Also keep samples near the midpoint; this catches cases where start
            # and end poses are close but not identical after drift correction.
            if len(early) < 3 or len(late) < 3:
                early = pts[(frame_ids <= early_limit) &
                            (np.linalg.norm(pts[:, :2] - center, axis=1) <= overlap_radius_m)]
                late = pts[(frame_ids >= late_limit) &
                           (np.linalg.norm(pts[:, :2] - center, axis=1) <= overlap_radius_m)]
    if len(early) < 3 or len(late) < 3:
        return None
    a = _nearest_distances(early, late)
    b = _nearest_distances(late, early)
    if a.size == 0 or b.size == 0:
        return None
    return float(np.percentile(np.concatenate([a, b]), 95))


def start_end_overlap_floor_height_error(
    trajectory: list[dict[str, Any]],
    overlap_radius_m: float,
    floor_normal: np.ndarray,
    floor_offset: float,
    floor_band_m: float,
    *,
    surface_samples: tuple[np.ndarray, np.ndarray] | None = None,
) -> float | None:
    if surface_samples is None:
        return None
    points, frame_indices = surface_samples
    pts = np.asarray(points, dtype=np.float64)
    frame_ids = np.asarray(frame_indices, dtype=np.int64)
    if len(pts) < 6 or len(frame_ids) != len(pts):
        return None
    poses = [
        np.asarray(record.get("pose"), dtype=np.float64).reshape(4, 4)
        for record in trajectory
        if "pose" in record
    ]
    if len(poses) < 2:
        return None
    min_frame = int(np.min(frame_ids))
    max_frame = int(np.max(frame_ids))
    if max_frame <= min_frame:
        return None
    span = max_frame - min_frame + 1
    early_limit = min_frame + max(1, span // 5)
    late_limit = max_frame - max(1, span // 5)
    start_xy = poses[0][:2, 3]
    end_xy = poses[-1][:2, 3]
    early = pts[
        (frame_ids <= early_limit)
        & (np.linalg.norm(pts[:, :2] - start_xy, axis=1) <= overlap_radius_m)
    ]
    late = pts[
        (frame_ids >= late_limit)
        & (np.linalg.norm(pts[:, :2] - end_xy, axis=1) <= overlap_radius_m)
    ]
    normal = np.asarray(floor_normal, dtype=np.float64)
    early_signed = early @ normal + float(floor_offset)
    late_signed = late @ normal + float(floor_offset)
    band = max(float(floor_band_m), 1e-6)
    early = early[np.abs(early_signed) <= band]
    early_signed = early_signed[np.abs(early_signed) <= band]
    late = late[np.abs(late_signed) <= band]
    late_signed = late_signed[np.abs(late_signed) <= band]
    if len(early) < 3 or len(late) < 3:
        return None
    a = _nearest_xy_height_deltas(early, early_signed, late, late_signed)
    b = _nearest_xy_height_deltas(late, late_signed, early, early_signed)
    if a.size == 0 or b.size == 0:
        return None
    return float(np.percentile(np.concatenate([a, b]), 95))


def make_heightmap(
    vertices: np.ndarray,
    normal: np.ndarray,
    offset: float,
    *,
    cell_m: float = 0.05,
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    cells = floor_height_cells(vertices, normal, offset, cell_m)
    if not cells:
        return np.zeros((0, 0), dtype=np.float32), (0.0, 0.0, 0.0, 0.0)
    xs = [k[0] for k in cells]
    ys = [k[1] for k in cells]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    arr = np.full((max_y - min_y + 1, max_x - min_x + 1), np.nan, dtype=np.float32)
    for (ix, iy), cell in cells.items():
        arr[iy - min_y, ix - min_x] = cell["median"]
    extent = (min_x * cell_m, (max_x + 1) * cell_m, min_y * cell_m, (max_y + 1) * cell_m)
    return arr, extent


def save_visual_outputs(
    output_dir: str | Path,
    vertices: np.ndarray,
    normal: np.ndarray,
    offset: float,
    trajectory: list[dict[str, Any]] | None = None,
    *,
    cell_m: float = 0.05,
) -> list[Path]:
    import matplotlib.pyplot as plt

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    heightmap, extent = make_heightmap(vertices, normal, offset, cell_m=cell_m)
    if heightmap.size:
        p = out / "heightmap.png"
        plt.figure(figsize=(8, 6))
        plt.imshow(heightmap, origin="lower", extent=extent, cmap="coolwarm")
        plt.colorbar(label="floor residual (m)")
        plt.xlabel("x (m)")
        plt.ylabel("y (m)")
        plt.tight_layout()
        plt.savefig(p, dpi=150)
        plt.close()
        paths.append(p)
    poses = [
        np.asarray(record.get("pose"), dtype=np.float64).reshape(4, 4)
        for record in (trajectory or [])
        if "pose" in record
    ]
    if poses:
        p = out / "trajectory.png"
        xy = np.asarray([pose[:2, 3] for pose in poses])
        plt.figure(figsize=(7, 7))
        plt.plot(xy[:, 0], xy[:, 1], "-k", linewidth=1.5)
        plt.scatter([xy[0, 0]], [xy[0, 1]], c="green", label="start")
        plt.scatter([xy[-1, 0]], [xy[-1, 1]], c="red", label="end")
        plt.axis("equal")
        plt.xlabel("x (m)")
        plt.ylabel("y (m)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(p, dpi=150)
        plt.close()
        paths.append(p)
    return paths


def _fit_plane(points: np.ndarray, up: np.ndarray) -> tuple[np.ndarray, float]:
    if len(points) < 3:
        raise ValueError("at least three points are needed to fit a plane")
    centroid = np.mean(points, axis=0)
    _, _, vh = np.linalg.svd(points - centroid, full_matrices=False)
    normal = _unit(vh[-1])
    if float(np.dot(normal, up)) < 0.0:
        normal = -normal
    offset = -float(np.dot(normal, centroid))
    return normal, offset


def _nearest_distances(source: np.ndarray, target: np.ndarray, chunk: int = 2048) -> np.ndarray:
    out: list[np.ndarray] = []
    for start in range(0, len(source), chunk):
        block = source[start:start + chunk]
        d2 = np.sum((block[:, None, :] - target[None, :, :]) ** 2, axis=2)
        out.append(np.sqrt(np.min(d2, axis=1)))
    return np.concatenate(out) if out else np.zeros((0,), dtype=np.float64)


def _nearest_xy_height_deltas(
    source: np.ndarray,
    source_height: np.ndarray,
    target: np.ndarray,
    target_height: np.ndarray,
    chunk: int = 2048,
) -> np.ndarray:
    out: list[np.ndarray] = []
    for start in range(0, len(source), chunk):
        block = source[start:start + chunk]
        d2 = np.sum((block[:, None, :2] - target[None, :, :2]) ** 2, axis=2)
        nearest = np.argmin(d2, axis=1)
        out.append(np.abs(source_height[start:start + len(block)] - target_height[nearest]))
    return np.concatenate(out) if out else np.zeros((0,), dtype=np.float64)


def _unit(v: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(v))
    if norm <= 0.0:
        raise ValueError("zero-length vector")
    return v / norm
