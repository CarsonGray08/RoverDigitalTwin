"""Real-time safety + telemetry for the closed loop.

Three pieces the closed loop composes:

- :class:`TrackingGate` — decide whether it is safe to command the rover this
  cycle, from the observation's ``phone_pose`` block. Only execute when ARKit
  tracking is normal and the twin is aligned; otherwise WAIT (pause, retry).
- pose-jump detection — a measured pose that diverges from the predicted pose by
  more than a large threshold signals an ARKit jump/relocalization; HALT rather
  than trust it.
- :class:`ExecutionErrorLog` — record predicted-vs-measured execution error with
  timestamps, the loop's localization telemetry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from digital_twin.geometry import distance_m, yaw_error_deg
from digital_twin.models import Pose


class TrackingDecision(Enum):
    EXECUTE = "execute"   # safe to command the rover
    WAIT = "wait"         # tracking not usable yet; don't command, retry later
    HALT = "halt"         # unsafe (pose jump); stop the loop


@dataclass(frozen=True)
class TrackingGate:
    """Gate rover commands on phone tracking quality.

    Active only when configured into the loop. ``require_aligned`` insists the
    twin alignment is established so the measured pose is in the twin
    frame.
    """

    require_aligned: bool = True

    def evaluate(self, phone_pose_block: Any) -> TrackingDecision:
        if not isinstance(phone_pose_block, dict):
            return TrackingDecision.WAIT
        if not phone_pose_block.get("enabled", False):
            return TrackingDecision.WAIT
        if not phone_pose_block.get("have_pose", False):
            return TrackingDecision.WAIT
        if not phone_pose_block.get("tracking_normal", False):
            return TrackingDecision.WAIT
        if self.require_aligned and not phone_pose_block.get("rover_pose_aligned", False):
            return TrackingDecision.WAIT
        return TrackingDecision.EXECUTE


def pose_error(predicted: Pose, measured: Pose) -> tuple[float, float]:
    """(distance_error_m, yaw_error_deg) between predicted and measured pose."""
    return distance_m(predicted, measured), yaw_error_deg(measured.yaw_deg, predicted.yaw_deg)


def is_pose_jump(predicted: Pose, measured: Pose, max_jump_m: float, max_jump_deg: float) -> bool:
    """True when the measured pose diverges from the predicted pose by more than
    the jump thresholds — too large to be normal drift (ARKit jump)."""
    d, yaw = pose_error(predicted, measured)
    return d > max_jump_m or yaw > max_jump_deg


@dataclass(frozen=True)
class ExecutionErrorRecord:
    cycle: int
    predicted: Pose
    measured: Pose
    distance_error_m: float
    yaw_error_deg: float
    command_complete_host_s: float | None = None
    phone_timestamp_s: float | None = None
    tracking_normal: bool | None = None


@dataclass
class ExecutionErrorLog:
    """Predicted-vs-measured execution error, with timestamps."""

    records: list[ExecutionErrorRecord] = field(default_factory=list)

    def record(
        self,
        cycle: int,
        predicted: Pose,
        measured: Pose,
        *,
        command_complete_host_s: float | None = None,
        phone_timestamp_s: float | None = None,
        tracking_normal: bool | None = None,
    ) -> ExecutionErrorRecord:
        d, yaw = pose_error(predicted, measured)
        rec = ExecutionErrorRecord(
            cycle=cycle,
            predicted=predicted,
            measured=measured,
            distance_error_m=d,
            yaw_error_deg=yaw,
            command_complete_host_s=command_complete_host_s,
            phone_timestamp_s=phone_timestamp_s,
            tracking_normal=tracking_normal,
        )
        self.records.append(rec)
        return rec

    def summary(self) -> dict[str, Any]:
        if not self.records:
            return {"count": 0, "mean_distance_error_m": 0.0, "max_distance_error_m": 0.0,
                    "mean_yaw_error_deg": 0.0, "max_yaw_error_deg": 0.0}
        ds = [r.distance_error_m for r in self.records]
        ys = [r.yaw_error_deg for r in self.records]
        n = len(self.records)
        return {
            "count": n,
            "mean_distance_error_m": sum(ds) / n,
            "max_distance_error_m": max(ds),
            "mean_yaw_error_deg": sum(ys) / n,
            "max_yaw_error_deg": max(ys),
        }
