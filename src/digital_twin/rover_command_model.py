"""Real-rover command model + calibration.

Turns physical motion (metres / degrees) into the rover's wheel command
``distance`` field and back, and gives the deterministic command duration.

Grounded in the actual firmware/bridge (see the robot-command-interface memo):

- ``distance`` is **raw motor steps** (uint16), not metres; ``speed`` is ignored
  (firmware runs Mode B at a fixed RPM).
- Deployed firmware ``ezrassor_firmware.ino``: ``STEPS_PER_REV = 200`` (microstep
  1), ``MODE_B_RPM = 100``; the DRV8825 driver applies a 6/7 step-delay factor,
  so the real step rate is ``rpm * steps_per_rev / 60 * 7/6 ≈ 389 steps/s``.
- **Calibration source of truth is the DigitalTwin's own ``SafetyLimits``**,
  which already holds empirically-measured constants:
  ``distance_units_per_m = 23000`` (⇒ 23000 steps/m) and
  ``turn_degrees_per_distance_unit = 90/24000`` (⇒ 24000 steps/deg⁻¹ = ~266.7
  steps/deg). Use :meth:`RoverCommandCalibration.from_limits` to stay in sync.
  (The firmware bridge's ``REVERSE_STEPS=1070 ≈ 0.5 m`` comment implies ~2140
  steps/m and disagrees with the measured 23000 — it is stale / a different
  microstep config, so the measured value wins.)
- The bridge publishes ``command_status="Success"`` on serial write, *not* on
  motion completion, so the host must wait :meth:`duration_s` (deterministic
  stepper timing) before reading the final pose.
"""

from __future__ import annotations

from dataclasses import dataclass

from digital_twin.models import CommandName, CommandStep

# -- Firmware constants (ezrassor_firmware.ino + DRV8825.cpp) ----------------
STEPS_PER_REV = 200          # MICROSTEP_MODE 1
MODE_B_RPM = 100             # fixed Mode B speed
_STEP_DELAY_FACTOR = 6.0 / 7.0  # DRV8825.set_speed shortens the delay by 6/7
# Effective step rate: nominal steps/s divided by the 6/7 delay factor.
STEP_RATE_HZ = (MODE_B_RPM * STEPS_PER_REV / 60.0) / _STEP_DELAY_FACTOR  # ~388.9

# Measured on the real rover: a 90deg in-place turn (24000 steps) completes in
# ~30s, ~2x faster than the firmware-theoretical STEP_RATE_HZ above. Same
# "measured beats the datasheet" rule as steps_per_m (see header), so the
# deterministic wait uses this rate: 24000/800 + 0.2s latency = ~30.2s per 90deg.
_MEASURED_STEP_RATE_HZ = 800.0

# Firmware accepts uint16 steps; the command schema caps distance at 30000.
SCHEMA_MAX_DISTANCE = 30000

# Defaults mirror SafetyLimits (the measured calibration):
#   distance_units_per_m = 23000  -> steps_per_m
#   turn_degrees_per_distance_unit = 90/24000 -> steps_per_deg = 24000/90
_DEFAULT_STEPS_PER_M = 23000.0
_DEFAULT_STEPS_PER_DEG = 24000.0 / 90.0  # ~266.7


@dataclass(frozen=True)
class RoverCommandCalibration:
    """Maps motion <-> steps and gives command timing. Defaults mirror the
    measured ``SafetyLimits`` calibration; use :meth:`from_limits` to bind to a
    specific settings object, or :meth:`fit_linear`/:meth:`fit_turn` to refine
    from phone-pose measurements."""

    steps_per_m: float = _DEFAULT_STEPS_PER_M
    steps_per_deg: float = _DEFAULT_STEPS_PER_DEG
    step_rate_hz: float = _MEASURED_STEP_RATE_HZ
    # Serial + decel-ramp slack added to the pure stepping time.
    command_latency_s: float = 0.2
    # The defaults come from the measured SafetyLimits, so calibration is True.
    turn_calibrated: bool = True

    @classmethod
    def from_limits(cls, limits, **overrides) -> "RoverCommandCalibration":
        """Build from a SafetyLimits (the single source of truth for the
        measured steps/m and steps/deg)."""
        return cls(
            steps_per_m=float(limits.distance_units_per_m),
            steps_per_deg=1.0 / float(limits.turn_degrees_per_distance_unit),
            **overrides,
        )

    # -- metres / degrees <-> steps ----------------------------------------
    def distance_to_steps(self, meters: float) -> int:
        return _clamp_steps(round(abs(meters) * self.steps_per_m))

    def steps_to_distance(self, steps: int) -> float:
        return steps / self.steps_per_m

    def yaw_to_steps(self, degrees: float) -> int:
        return _clamp_steps(round(abs(degrees) * self.steps_per_deg))

    def steps_to_yaw(self, steps: int) -> float:
        return steps / self.steps_per_deg

    # -- timing ------------------------------------------------------------
    def duration_s(self, steps: int) -> float:
        """Deterministic time to execute ``steps`` (the real completion signal,
        since command_status does not report motion completion)."""
        return abs(steps) / self.step_rate_hz + self.command_latency_s

    # -- command builders --------------------------------------------------
    def drive_command(self, meters: float, speed: int = 1) -> CommandStep:
        """A straight drive of ``meters`` (negative => REV). ``speed`` is sent for
        schema/teleop compatibility but ignored by the firmware."""
        command = CommandName.FWD if meters >= 0 else CommandName.REV
        return CommandStep(command, speed, self.distance_to_steps(meters))

    def turn_command(self, degrees: float, speed: int = 1) -> CommandStep:
        """An in-place turn of ``degrees`` (positive => CCW => LEFT)."""
        command = CommandName.LEFT if degrees >= 0 else CommandName.RIGHT
        return CommandStep(command, speed, self.yaw_to_steps(degrees))

    # Calibration from measured travel and rotation.
    def fit_linear(self, measurements: list[tuple[int, float]]) -> "RoverCommandCalibration":
        """Refit ``steps_per_m`` from (steps, measured_metres) pairs."""
        steps_per_m = _fit_ratio(measurements)
        return dataclasses_replace(self, steps_per_m=steps_per_m)

    def fit_turn(self, measurements: list[tuple[int, float]]) -> "RoverCommandCalibration":
        """Refit ``steps_per_deg`` from (steps, measured_degrees) pairs."""
        steps_per_deg = _fit_ratio(measurements)
        return dataclasses_replace(self, steps_per_deg=steps_per_deg, turn_calibrated=True)

    def to_dict(self) -> dict:
        return {
            "steps_per_m": self.steps_per_m,
            "steps_per_deg": self.steps_per_deg,
            "step_rate_hz": self.step_rate_hz,
            "command_latency_s": self.command_latency_s,
            "turn_calibrated": self.turn_calibrated,
        }


def _clamp_steps(steps: float) -> int:
    return int(max(0, min(SCHEMA_MAX_DISTANCE, steps)))


def _fit_ratio(measurements: list[tuple[int, float]]) -> float:
    """Least-squares slope through the origin for steps = ratio * physical:
    ratio = sum(steps*phys) / sum(phys^2). Uses |physical| so sign conventions
    don't matter."""
    num = 0.0
    den = 0.0
    for steps, physical in measurements:
        p = abs(physical)
        num += steps * p
        den += p * p
    if den <= 0.0:
        raise ValueError("need at least one non-zero measurement to fit")
    return num / den


def dataclasses_replace(cal: RoverCommandCalibration, **changes) -> RoverCommandCalibration:
    # Small local helper to avoid importing dataclasses.replace at call sites.
    import dataclasses

    return dataclasses.replace(cal, **changes)
