"""Small, atomic session contract for the live nvblox twin.

The mapper owns this file alongside ``manifest.json`` and ``rover_pose.json``.
It deliberately contains only transport/session facts; it never turns inferred
support into map data.  Keeping this stdlib-only makes cold-start behaviour
testable without a phone, CUDA, or Isaac.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

SESSION_NAME = "session.json"


def new_session_id() -> str:
    return f"live-{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}-{uuid.uuid4().hex[:8]}"


def session_path(map_dir: str | Path) -> Path:
    return Path(map_dir) / SESSION_NAME


def new_session(session_id: str | None = None, *, resumed: bool = False) -> dict[str, Any]:
    return {
        "session_id": session_id or new_session_id(),
        "started_at_s": time.time(),
        "fresh": not resumed,
        "resumed": bool(resumed),
        "t_align": "absent",
        "tracking_normal": False,
    }


def write_session(map_dir: str | Path, payload: dict[str, Any]) -> None:
    """Atomically replace the session record; readers never see partial JSON."""
    path = session_path(map_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Pose publishing and tile writing can both update session metadata from
    # different threads. A fixed temp name lets one writer replace another
    # writer's temp file, which can crash the mapper with FileNotFoundError.
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, sort_keys=True)
    os.replace(tmp, path)


def read_session(map_dir: str | Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(session_path(map_dir).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def resolve_session_dir(path: str | Path) -> Path:
    """Resolve a map-dir argument to the active session directory.

    ``live_twin.sh`` writes each run to a timestamped ``session-*`` subdir and
    maintains a ``latest`` symlink to the active one, so an operator never has to
    copy the path. Accept any of: a direct session dir (has ``session.json``), the
    map root holding a ``latest`` pointer, or a root with ``session-*`` subdirs
    (newest wins as a fallback if the pointer is missing). Anything else is
    returned unchanged so a later read fails with a clear message."""
    path = Path(path)
    if (path / SESSION_NAME).exists():
        return path
    latest = path / "latest"
    if (latest / SESSION_NAME).exists():
        return latest.resolve()
    sessions = sorted(
        (p for p in path.glob("session-*") if (p / SESSION_NAME).exists()),
        key=lambda p: p.stat().st_mtime,
    )
    if sessions:
        return sessions[-1]
    return path


def update_session(map_dir: str | Path, **changes: Any) -> dict[str, Any]:
    """Apply mapper state changes while preserving the original session id."""
    payload = read_session(map_dir) or new_session()
    payload.update(changes)
    write_session(map_dir, payload)
    return payload


def derive_live_twin_status(
    *,
    session: dict[str, Any] | None,
    map_dir: str | Path | None,
    pose: dict[str, Any] | None,
    terrain_tiles: int,
    map_revision: int,
    baseline_min_terrain_tiles: int,
    coverage: dict[str, Any],
    footprint_ground_source: str,
    support_evidence_cells: int = 0,
) -> dict[str, Any]:
    """Derive the operator-facing phase without depending on Isaac APIs."""
    tracking_normal = bool(pose and pose.get("tracking_normal", False))
    t_align_present = bool(session and session.get("t_align") == "present")
    fresh_session = bool(session and session.get("fresh", False) and not session.get("resumed", False))
    # A resume is an explicit operator decision, but session.json still proves
    # that this mapper process owns the transport directory.  An arbitrary
    # directory without this record can never satisfy readiness.
    mapping_session_current = session is not None
    baseline_ready = (
        tracking_normal and t_align_present and mapping_session_current
        and terrain_tiles >= max(1, int(baseline_min_terrain_tiles))
        and (
            bool(coverage.get("complete", False))
            or not bool(coverage.get("enabled", True))
        )
    )
    phase = "awaiting_scan" if not (tracking_normal and t_align_present) else (
        "tracking" if baseline_ready else "baseline_mapping")
    return {
        "phase": phase,
        "session_id": session.get("session_id") if session else None,
        "map_dir": str(map_dir) if map_dir is not None else None,
        "fresh_session": fresh_session,
        "mapping_session_current": mapping_session_current,
        "tracking_normal": tracking_normal,
        "t_align": "present" if t_align_present else "absent",
        "alignment_reason": session.get("alignment_reason") if session else None,
        "align_stable_samples": session.get("align_stable_samples") if session else None,
        "terrain_tiles": int(terrain_tiles),
        "map_revision": int(map_revision),
        "baseline_ready": baseline_ready,
        "baseline_min_terrain_tiles": max(1, int(baseline_min_terrain_tiles)),
        "coverage": coverage,
        "dead_zone": {
            "status": "under_rover_exclusion" if footprint_ground_source != "observed" else "observed",
            "observed": footprint_ground_source == "observed",
            "inferred_support": footprint_ground_source == "inferred_support",
            "footprint_ground_source": footprint_ground_source,
            "support_evidence_cells": int(support_evidence_cells),
        },
    }
