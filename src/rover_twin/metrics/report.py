"""Score a mapper run against its ground truth: the fidelity, pose and
latency table.

A run directory is reduced to the terrain physics gets (the elevation grid
as points), the per-frame surface samples, the rover's track and the latency
summaries, and each is scored against the recording's truth file.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from rover_twin import frames
from rover_twin.frames import Pose2D
from rover_twin.mapping.elevation import ElevationGrid
from rover_twin.metrics import fidelity
from rover_twin.metrics.latency import summarize
from rover_twin.phone import recording
from rover_twin.phone.protocol import RgbdFrame, decode

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

# What the twin aims for; latency targets are p95.
TARGETS = {
    "collider_live_p95_ms": 250.0,
    "floor_rms_m": 0.01,
    "seam_p95_m": 0.02,
    "box_error_m": 0.02,
    "floor_step_m": 0.01,
    "yaw_error_deg": 1.0,
    "drift_pct": 2.0,
}


@dataclass(frozen=True)
class Box:
    name: str
    center: tuple[float, float]
    size: tuple[float, float, float]


@dataclass(frozen=True)
class Truth:
    kind: str                    # spin, drive, loop or scene
    end: Pose2D                  # rover end pose in the twin, from floor marks
    boxes: tuple[Box, ...]

    @classmethod
    def load(cls, path: str | Path) -> "Truth":
        data = tomllib.loads(Path(path).read_text())
        end = data["end"]
        return cls(data["kind"], Pose2D(end["x_m"], end["y_m"], end["yaw_deg"]),
                   tuple(Box(b["name"], tuple(b["center_m"]), tuple(b["size_m"])) for b in data.get("boxes", [])))


@dataclass(frozen=True, eq=False)
class Run:
    terrain: np.ndarray          # (N, 3) surface points
    samples: np.ndarray          # (M, 3) per-frame surface samples ...
    sample_frames: np.ndarray    # ... and the fused-frame index of each
    track: np.ndarray            # (K, 3) rover x, y, yaw_deg in fused order
    latency_ms: dict[str, dict]  # stage -> summary


def load_run(path: str | Path) -> Run:
    path = Path(path)
    terrain = ElevationGrid.load(path / "elevation.npz").points()
    samples = np.load(path / "samples.npz")
    track = [json.loads(line)["rover"] for line in (path / "trajectory.jsonl").read_text().splitlines()]
    latency = {}
    for name in ("metrics.json", "twin_metrics.json"):
        if (path / name).exists():
            latency.update(json.loads((path / name).read_text())["latency_ms"])
    return Run(terrain, samples["points"], samples["frames"], np.asarray(track), latency)


def capture_jitter(scanlog: str | Path) -> dict:
    """Capture-to-arrival delay of a recording's RGB-D frames, relative to its
    minimum: phone and host clocks are not synchronised, so only the spread is
    known. This is the network's share of the latency budget."""
    delays = [offset - m.t_phone for offset, payload in recording.read(scanlog)
              if isinstance(m := decode(payload), RgbdFrame)]
    d = np.asarray(delays)
    return summarize(d - d.min()) if len(d) else {"n": 0}


def score(run: Run, truth: Truth) -> dict[str, float | None]:
    floor, mask = fidelity.fit_floor(run.terrain)
    out: dict[str, float | None] = {
        "floor_rms_m": fidelity.floor_rms(run.terrain, floor, mask),
        "floor_step_m": fidelity.floor_step(run.terrain, floor, mask),
        "seam_p95_m": fidelity.seam(run.samples, run.sample_frames) if truth.kind in ("spin", "loop") else None,
        "box_error_m": _box_error(run.terrain, floor, truth.boxes),
    }
    if len(run.track) and truth.kind != "scene":
        x, y, yaw = run.track[-1]
        position_error = math.hypot(x - truth.end.x, y - truth.end.y)
        distance = float(np.sum(np.linalg.norm(np.diff(run.track[:, :2], axis=0), axis=1)))
        out["yaw_error_deg"] = abs(frames.wrap_deg(yaw - truth.end.yaw_deg))
        out["position_error_m"] = position_error
        out["drift_pct"] = 100.0 * position_error / distance if distance > 0.5 else None
    for stage, summary in run.latency_ms.items():
        out[f"{stage}_p95_ms"] = summary.get("p95")
    return out


def _box_error(terrain, floor, boxes) -> float | None:
    """Largest dimension error over the truth boxes; inf if one is not found."""
    if not boxes:
        return None
    worst = 0.0
    for box in boxes:
        measured = fidelity.box_size(terrain, floor, box.center, search_m=0.5 * math.hypot(*box.size[:2]) + 0.15)
        if measured is None:
            return math.inf
        truth = (max(box.size[:2]), min(box.size[:2]), box.size[2])
        worst = max(worst, *(abs(m - t) for m, t in zip(measured, truth)))
    return worst


def table(scores: dict[str, dict]) -> str:
    """Markdown table, one column per run, with the targets."""
    names = list(scores)
    keys = sorted({k for s in scores.values() for k in s}, key=lambda k: (k not in TARGETS, k))
    lines = ["| metric | target | " + " | ".join(names) + " |", "|---|---|" + "---|" * len(names)]
    for k in keys:
        cells = [_fmt(scores[n].get(k)) for n in names]
        lines.append(f"| {k} | {_fmt(TARGETS.get(k))} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _fmt(v) -> str:
    if v is None:
        return "-"
    if math.isinf(v):
        return "missed"
    return f"{v:.4g}"
