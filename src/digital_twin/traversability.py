"""Physics-derived traversability verdict for terrain validation.

On real (lunar-analog) terrain, "is this command safe?" is not bbox clearance --
it is whether the rover can traverse the surface without tipping, bellying out,
or climbing a slope it cannot. As the validator rolls a command forward on the
collidable terrain, each sampled pose yields physical quantities (local slope,
body roll/pitch, ground/belly clearance); this module turns those into a verdict
against configurable limits.

The quantities themselves come from the sim (articulation orientation + downward
raycasts to the terrain), which is the validator's Isaac-side job. The
**classification** -- the research-relevant traversability criteria and limits --
is pure and lives here so it is unit-testable and easy to tune for a given rover.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TraversabilityLimits:
    """Traversability envelope for the rover. Defaults are conservative
    lunar-analog starting points; tune per rover and per study."""

    max_slope_deg: float = 20.0
    max_roll_deg: float = 15.0
    max_pitch_deg: float = 20.0
    min_ground_clearance_m: float = 0.05

    def __post_init__(self) -> None:
        for name in ("max_slope_deg", "max_roll_deg", "max_pitch_deg"):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive")
        if self.min_ground_clearance_m < 0.0:
            raise ValueError("min_ground_clearance_m must be >= 0")


@dataclass(frozen=True)
class TerrainSample:
    """Physical state of the rover at one sampled pose along a rollout.

    ``ground_clearance_m`` is the gap between the lowest point of the chassis
    (belly) and the terrain directly beneath it; negative means the belly is in
    the terrain (high-centred / bottomed out)."""

    slope_deg: float
    roll_deg: float
    pitch_deg: float
    ground_clearance_m: float


# Failure reasons (stable strings, mirroring the validator's failure_type style).
SLOPE_EXCEEDED = "slope_exceeded"
ROLL_EXCEEDED = "roll_exceeded"
PITCH_EXCEEDED = "pitch_exceeded"
INSUFFICIENT_CLEARANCE = "insufficient_clearance"


def classify_sample(sample: TerrainSample, limits: TraversabilityLimits) -> str | None:
    """Return a failure reason for this pose, or ``None`` if it is traversable.

    Order is by severity for a planetary rover: a tip-over (roll/pitch) or a
    belly-out (clearance) ends the mission, so they rank above an over-steep but
    still-upright slope."""
    if abs(sample.roll_deg) > limits.max_roll_deg:
        return ROLL_EXCEEDED
    if abs(sample.pitch_deg) > limits.max_pitch_deg:
        return PITCH_EXCEEDED
    if sample.ground_clearance_m < limits.min_ground_clearance_m:
        return INSUFFICIENT_CLEARANCE
    if sample.slope_deg > limits.max_slope_deg:
        return SLOPE_EXCEEDED
    return None


@dataclass(frozen=True)
class TraversabilityVerdict:
    traversable: bool
    failed_index: int | None = None
    reason: str | None = None
    # Worst observed values across the rollout, for reporting/telemetry.
    max_slope_deg: float = 0.0
    max_abs_roll_deg: float = 0.0
    max_abs_pitch_deg: float = 0.0
    min_ground_clearance_m: float = float("inf")


def evaluate(samples, limits: TraversabilityLimits) -> TraversabilityVerdict:
    """Evaluate a rollout's terrain samples in order, failing at the first
    sample that violates a limit and always reporting the worst values seen up to
    that point (the extremes are what a study cares about)."""
    max_slope = 0.0
    max_roll = 0.0
    max_pitch = 0.0
    min_clear = float("inf")
    for index, sample in enumerate(samples):
        max_slope = max(max_slope, sample.slope_deg)
        max_roll = max(max_roll, abs(sample.roll_deg))
        max_pitch = max(max_pitch, abs(sample.pitch_deg))
        min_clear = min(min_clear, sample.ground_clearance_m)
        reason = classify_sample(sample, limits)
        if reason is not None:
            return TraversabilityVerdict(
                traversable=False,
                failed_index=index,
                reason=reason,
                max_slope_deg=max_slope,
                max_abs_roll_deg=max_roll,
                max_abs_pitch_deg=max_pitch,
                min_ground_clearance_m=min_clear,
            )
    return TraversabilityVerdict(
        traversable=True,
        max_slope_deg=max_slope,
        max_abs_roll_deg=max_roll,
        max_abs_pitch_deg=max_pitch,
        min_ground_clearance_m=min_clear,
    )
