"""Calling the twin from other software: the map, rehearsing a plan, live state.

    from rover_twin.client import TwinClient
    twin = TwinClient("http://127.0.0.1:8765")
    grid = twin.map()
    plan = [twin.forward(1.0), twin.left(90), twin.forward(0.5)]
    result = twin.rehearse((0.0, 0.0, 0.0), plan)
    result["approved"], result["duration_s"]

Works the same against the lab twin (Isaac Sim) and the offline one
(``python -m rover_twin.twin.offline``). docs/twin-api.md is the contract:
frames, units, and what each answer means.
"""

from __future__ import annotations

import json
import math
import urllib.request
from dataclasses import dataclass

import numpy as np

UNKNOWN, OBSERVED, INFERRED = 0, 1, 2       # Map.state, as in the mapper's grid
OBSERVATION_TIMEOUT_S = 2.0                 # the twin answers it at once; a drive stops on a slow one


@dataclass(frozen=True)
class Map:
    """The twin's terrain at one moment. Arrays are indexed [i, j] with cell
    [i, j] covering x in origin_x + i*cell_m .. +cell_m, y likewise."""

    cell_m: float
    origin_m: tuple[float, float]
    height_m: np.ndarray        # float, NaN where unknown
    state: np.ndarray           # UNKNOWN, OBSERVED or INFERRED
    obstacle: np.ndarray        # bool: terrain the rover's body would hit

    def cell(self, x: float, y: float) -> tuple[int, int]:
        return (int(math.floor((x - self.origin_m[0]) / self.cell_m)),
                int(math.floor((y - self.origin_m[1]) / self.cell_m)))

    def centre(self, i: int, j: int) -> tuple[float, float]:
        return (self.origin_m[0] + (i + 0.5) * self.cell_m, self.origin_m[1] + (j + 0.5) * self.cell_m)


class TwinClient:
    def __init__(self, url: str = "http://127.0.0.1:8765", timeout_s: float = 600.0) -> None:
        self.url, self.timeout_s = url.rstrip("/"), timeout_s
        self.rover = self._get("/health")["rover"]       # units, limits, footprint

    def observation(self) -> dict:
        """The real rover's live pose as the twin last heard it, and the map's state."""
        return self._get("/observation", OBSERVATION_TIMEOUT_S)

    def map(self) -> Map:
        d = self._get("/map")
        shape = tuple(d["shape"])
        height = np.array([np.nan if h is None else h for row in d["height_m"] for h in row], float)
        return Map(d["cell_m"], tuple(d["origin_m"]), height.reshape(shape),
                   np.array(d["state"], np.uint8).reshape(shape), np.array(d["obstacle"], bool).reshape(shape))

    def rehearse(self, start, commands: list[dict], goal=None) -> dict:
        """Roll ``commands`` forward from ``start`` (x, y, yaw_deg) on the live
        map. ``goal`` defaults to where the commands would end on flat ground,
        so ``approved`` means: no collision, no unknown terrain, and it got there."""
        goal = goal if goal is not None else self.nominal_end(start, commands)
        body = {"start_pose": _pose(start), "goal": _pose(goal), "commands": commands}
        request = urllib.request.Request(self.url + "/validate_plan", json.dumps(body).encode(),
                                         {"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
            return json.loads(response.read())

    def forward(self, metres: float, speed: int = 30) -> dict:
        return {"command": "FWD", "speed": speed, "distance": round(metres * self.rover["distance_units_per_m"])}

    def reverse(self, metres: float, speed: int = 30) -> dict:
        return {"command": "REV", "speed": speed, "distance": round(metres * self.rover["distance_units_per_m"])}

    def left(self, degrees: float, speed: int = 30) -> dict:
        return {"command": "LEFT", "speed": speed, "distance": round(degrees)}

    def right(self, degrees: float, speed: int = 30) -> dict:
        return {"command": "RIGHT", "speed": speed, "distance": round(degrees)}

    def nominal_end(self, start, commands: list[dict]) -> tuple[float, float, float]:
        x, y, yaw = start
        for c in commands:
            if c["command"] in ("FWD", "REV"):
                d = c["distance"] / self.rover["distance_units_per_m"] * (1 if c["command"] == "FWD" else -1)
                x, y = x + d * math.cos(math.radians(yaw)), y + d * math.sin(math.radians(yaw))
            elif c["command"] in ("LEFT", "RIGHT"):
                yaw += c["distance"] if c["command"] == "LEFT" else -c["distance"]
        return x, y, (yaw + 180.0) % 360.0 - 180.0

    def _get(self, path: str, timeout_s: float | None = None) -> dict:
        with urllib.request.urlopen(self.url + path, timeout=timeout_s or self.timeout_s) as response:
            return json.loads(response.read())


def _pose(p) -> dict:
    x, y, yaw = p
    return {"x": float(x), "y": float(y), "yaw_deg": float(yaw)}
