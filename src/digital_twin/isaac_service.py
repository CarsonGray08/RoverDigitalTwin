"""Supported Isaac validator entry point."""

from __future__ import annotations

import argparse
import os
import sys

from digital_twin import _isaac_runtime


def runtime_arguments() -> list[str]:
    required = {
        name: os.environ.get(name)
        for name in ("DT_SCENE_PATH", "DT_MAP_DIR", "DT_MOUNT_PATH", "DT_NEWTON_KIT")
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise RuntimeError(
            "validator must be started through scripts/live_twin.sh; missing "
            + ", ".join(missing)
        )
    return [
        "--scene", required["DT_SCENE_PATH"],
        "--host", "127.0.0.1",
        "--port", os.environ.get("VALIDATOR_PORT", "8765"),
        "--rover-prim", "/World/RASSOR",
        "--pose-forward-axis", "neg_y",
        "--wheel-velocity-scale", "0.05",
        "--experience", required["DT_NEWTON_KIT"],
        "--gui",
        "--render",
        "--track-live-pose",
        "--load-terrain",
        "--map-dir", required["DT_MAP_DIR"],
        "--phone-mount-json", required["DT_MOUNT_PATH"],
        "--hide-static-scene-at-start",
        "--disable-static-scene-collision",
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the supported live-terrain Isaac validator configuration."
    )
    parser.parse_args(argv)
    previous_argv = sys.argv
    sys.argv = [previous_argv[0], *runtime_arguments()]
    try:
        return _isaac_runtime.main()
    finally:
        sys.argv = previous_argv


if __name__ == "__main__":
    raise SystemExit(main())
