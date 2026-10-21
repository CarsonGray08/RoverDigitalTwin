"""Per-run timing: every process records into one of these and writes it on exit.

Latencies are measured from a frame's host arrival time, which travels with
everything derived from it (fused geometry, tiles, poses), so each stage's
number is "phone frame arrival to this point" and the last stage is the
headline. All processes on one host share ``time.monotonic``.
"""

from __future__ import annotations

import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


class Stopwatch:
    def __init__(self, clock=time.monotonic) -> None:
        self.clock = clock
        self._latency: dict[str, list[float]] = defaultdict(list)
        self._duration: dict[str, list[float]] = defaultdict(list)
        self.counts: Counter[str] = Counter()
        self.values: dict[str, float] = {}

    def since(self, stage: str, t_arrival: float) -> None:
        """Record that ``stage`` finished for a frame that arrived at ``t_arrival``."""
        self._latency[stage].append(self.clock() - t_arrival)

    def took(self, name: str, seconds: float) -> None:
        self._duration[name].append(seconds)

    def count(self, name: str, n: int = 1) -> None:
        self.counts[name] += n

    def summary(self) -> dict:
        return {
            "latency_ms": {k: summarize(v) for k, v in self._latency.items()},
            "duration_ms": {k: summarize(v) for k, v in self._duration.items()},
            "counts": dict(self.counts),
            "values": dict(self.values),
        }

    def write(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.summary(), indent=1, sort_keys=True))


def summarize(seconds) -> dict:
    ms = np.asarray(seconds, dtype=np.float64) * 1e3
    if ms.size == 0:
        return {"n": 0}
    p50, p95 = np.percentile(ms, [50, 95])
    return {"n": int(ms.size), "p50": round(float(p50), 2), "p95": round(float(p95), 2),
            "max": round(float(ms.max()), 2)}
