"""Typed runtime configuration: one TOML file loaded into frozen dataclasses.

Every key is checked against its declared type at startup and unknown keys
are refused, so a typo fails loudly instead of silently using a default.
"""

from __future__ import annotations

import typing
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Literal

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib


@dataclass(frozen=True)
class Phone:
    host: str = "0.0.0.0"
    port: int = 9099


@dataclass(frozen=True)
class Depth:
    """Which depth pixels feed both map outputs."""

    min_confidence: int = 2     # ARKit confidence: 0 low, 1 medium, 2 high
    max_range_m: float = 5.0


@dataclass(frozen=True)
class Mount:
    """Phone camera pose in the rover body frame, in the phone's camera axes."""

    translation_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    quaternion_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)


@dataclass(frozen=True)
class Rover:
    """The physical rover, as the mount math, the twin and validation see it."""

    forward_axis: Literal["x", "-x", "y", "-y"] = "x"   # body axis that points the driving way
    footprint_half_length_m: float = 0.26                # along forward, with margin
    footprint_half_width_m: float = 0.30
    wheel_diameter_m: float = 0.25
    wheel_velocity_scale: float = 0.05                   # wheel rad/s per command speed unit
    distance_units_per_m: float = 23000.0                # linear command distance units
    # Observed ground this far above the floor is what the body hits (it fits
    # under nothing: a grid cell keeps its highest surface, so a table top and
    # a solid face read the same), and this far below is a drop it falls into.
    max_step_m: float = 0.12


@dataclass(frozen=True)
class Limits:
    """What a plan may ask for, and when it counts as done."""

    max_commands: int = 12
    min_speed: int = 1                   # of a moving command; STOP is speed 0
    max_speed: int = 60
    max_distance: int = 30000
    max_turn_deg: int = 360
    command_timeout_s: float = 60.0
    goal_tolerance_m: float = 0.15
    goal_tolerance_yaw_deg: float = 8.0
    lab_bounds_m: tuple[float, float, float, float] = (-5.0, 5.0, -5.0, 5.0)   # min x, max x, min y, max y
    min_known_fraction: float = 0.8      # of the footprint, along the path to the goal


@dataclass(frozen=True)
class Start:
    """Rover pose in the twin when the map is anchored."""

    x_m: float = 0.0
    y_m: float = 0.0
    yaw_deg: float = 0.0
    z_m: float = 0.0


@dataclass(frozen=True)
class Alignment:
    """Phone poses must hold still this well before the map is anchored."""

    stable_samples: int = 5
    max_translation_m: float = 0.03
    max_yaw_deg: float = 3.0


@dataclass(frozen=True)
class Odometry:
    max_speed_mps: float = 0.75          # faster motion is a tracking jump
    max_yaw_rate_deg_s: float = 180.0


@dataclass(frozen=True)
class Keyframes:
    """A frame joins the re-fuse buffer once the camera moves this far."""

    min_translation_m: float = 0.2
    min_rotation_deg: float = 10.0


@dataclass(frozen=True)
class Tsdf:
    voxel_m: float = 0.02
    mesh_interval_s: float = 1.0


@dataclass(frozen=True)
class Elevation:
    cell_m: float = 0.02
    tile_cells: int = 64
    min_height_m: float = -0.5           # points outside this band are ignored
    max_height_m: float = 1.0
    step_m: float = 0.05                 # larger height differences are another surface
    noise_per_m: float = 0.01            # depth noise (1 sigma) per metre of range
    attitude_noise_rad: float = 0.003    # camera tilt error (1 sigma)
    footprint_radius_m: float = 0.45     # unseen patch under the rover
    footprint_ring_m: float = 0.2

    def __post_init__(self) -> None:
        if self.cell_m <= 0 or self.tile_cells <= 0 or self.step_m <= 0:
            raise ValueError("elevation cell_m, tile_cells and step_m must be positive")


@dataclass(frozen=True)
class Link:
    host: str = "127.0.0.1"
    port: int = 9101


@dataclass(frozen=True)
class Twin:
    scene: str = "assets/rover_twin.usda"
    rover_prim: str = "/World/RASSOR"
    host: str = "127.0.0.1"           # HTTP API for validation and observation
    port: int = 8765
    gui: bool = True                  # also renders the operator's view at RENDER_HZ
    settle_s: float = 0.3             # after placing the rover, before a rollout (at rest by 0.23 s in Isaac)
    terrain_friction: float = 1.0


@dataclass(frozen=True)
class Run:
    root: str = "runs"
    record: bool = True          # tee the raw phone stream to <run>/phone.scanlog


@dataclass(frozen=True)
class Config:
    phone: Phone = field(default_factory=Phone)
    depth: Depth = field(default_factory=Depth)
    mount: Mount = field(default_factory=Mount)
    rover: Rover = field(default_factory=Rover)
    limits: Limits = field(default_factory=Limits)
    start: Start = field(default_factory=Start)
    alignment: Alignment = field(default_factory=Alignment)
    odometry: Odometry = field(default_factory=Odometry)
    keyframes: Keyframes = field(default_factory=Keyframes)
    tsdf: Tsdf = field(default_factory=Tsdf)
    elevation: Elevation = field(default_factory=Elevation)
    link: Link = field(default_factory=Link)
    twin: Twin = field(default_factory=Twin)
    run: Run = field(default_factory=Run)


def load(path: str | Path) -> Config:
    with open(path, "rb") as fh:
        return parse(tomllib.load(fh))


def parse(data: dict) -> Config:
    return _build(Config, data, "config")


def _build(cls, data, where: str):
    if not isinstance(data, dict):
        raise ValueError(f"{where} must be a table")
    hints = typing.get_type_hints(cls)
    unknown = set(data) - {f.name for f in fields(cls)}
    if unknown:
        raise ValueError(f"{where}: unknown keys {sorted(unknown)}")
    return cls(**{k: _coerce(hints[k], v, f"{where}.{k}") for k, v in data.items()})


def _coerce(tp, value, where: str):
    if is_dataclass(tp):
        return _build(tp, value, where)
    origin, args = typing.get_origin(tp), typing.get_args(tp)
    if origin is tuple:
        if not isinstance(value, list) or len(value) != len(args):
            raise ValueError(f"{where} must be a list of {len(args)} values")
        return tuple(_coerce(a, v, f"{where}[{i}]") for i, (a, v) in enumerate(zip(args, value)))
    if origin is Literal:
        if value not in args:
            raise ValueError(f"{where} must be one of {list(args)}, got {value!r}")
        return value
    if tp is float and isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, tp) and not (tp is int and isinstance(value, bool)):
        return value
    raise ValueError(f"{where} must be {tp.__name__}, got {value!r}")
