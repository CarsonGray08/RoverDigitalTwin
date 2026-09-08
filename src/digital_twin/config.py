from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from digital_twin.models import SafetyLimits, WorldModel


class GateMode(str, Enum):
    OFF = "off"
    DRY_RUN = "dry_run"
    SHADOW = "shadow"
    ENFORCE = "enforce"


@dataclass(frozen=True)
class PlannerSettings:
    mode: GateMode = GateMode.DRY_RUN
    max_attempts: int = 4
    validation_profile: str = "safe_lab_default"
    limits: SafetyLimits = field(default_factory=SafetyLimits)
    world: WorldModel = field(default_factory=WorldModel)
    acceptable_drift_m: float = 0.25
    acceptable_drift_yaw_deg: float = 12.0
    default_forward_speed: int = 40
    # 56 x the Isaac wheel-velocity scale (0.05) = 2.8 rad/s differential —
    # the measured sweet spot for skid-steer pivots (fast, ~0.8 mm drift per
    # degree). Slower turns (e.g. 30 -> 1.5 rad/s) drift more per degree and
    # burn more sim time per validation.
    default_turn_speed: int = 56

