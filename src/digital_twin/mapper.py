"""Supported mapper entry point with the field-tested configuration."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from digital_twin import _mapper_runtime


def _env_enabled(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def runtime_arguments() -> list[str]:
    project_root = Path(__file__).resolve().parents[2]
    map_dir = os.environ.get("DT_MAP_DIR")
    session_id = os.environ.get("DT_SESSION_ID")
    if not map_dir or not session_id:
        raise RuntimeError("mapper must be started through scripts/live_twin.sh")
    arguments = [
        "--port", os.environ.get("PHONE_PORT", "9099"),
        "--control-port", os.environ.get("CONTROL_PORT", "9100"),
        "--voxel-size", os.environ.get("VOXEL_SIZE_M", "0.02"),
        "--output", map_dir,
        "--tile-size", os.environ.get("TILE_SIZE_M", "4.0"),
        "--mesh-interval", os.environ.get("MAP_UPDATE_INTERVAL_S", "0.5"),
        "--color-every", "1",
        "--min-confidence", os.environ.get("MIN_DEPTH_CONFIDENCE", "2"),
        "--phone-mount-json", str(project_root / "config" / "phone_mount.json"),
        "--session-id", session_id,
        "--use-cuvslam",
        "--cuvslam-variant", "depth-gray",
        "--fill-mode", "delaunay",
        "--floor-stabilize-band", "0.04",
        "--resumable-scan",
    ]
    if _env_enabled("STABILIZE_FLOOR", default=True):
        arguments.append("--stabilize-floor")
    return arguments


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the supported iPhone-to-nvblox mapper configuration."
    )
    parser.parse_args(argv)
    previous_argv = sys.argv
    sys.argv = [previous_argv[0], *runtime_arguments()]
    try:
        return _mapper_runtime.main()
    finally:
        sys.argv = previous_argv


if __name__ == "__main__":
    raise SystemExit(main())
