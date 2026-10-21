"""Score runs against a recording's ground truth and print the table.

Each run is NAME=DIR, a mapper run directory; several are compared side by
side. ``--scanlog`` adds the recording's capture-to-arrival jitter, the part
of the latency budget the host does not control.

usage: python tools/evaluate.py TRUTH.toml NAME=RUN_DIR [NAME=RUN_DIR ...] [--scanlog S]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from rover_twin.metrics import report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("truth", type=Path)
    parser.add_argument("runs", nargs="+", help="NAME=DIR")
    parser.add_argument("--scanlog", type=Path)
    args = parser.parse_args()

    truth = report.Truth.load(args.truth)
    scores = {}
    for spec in args.runs:
        name, path = spec.split("=", 1)
        scores[name] = report.score(report.load_run(path), truth)
        if args.scanlog:
            scores[name]["capture_jitter_p95_ms"] = report.capture_jitter(args.scanlog).get("p95")
    print(report.table(scores))


if __name__ == "__main__":
    main()
