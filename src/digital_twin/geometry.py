from __future__ import annotations

import math

from digital_twin.models import Pose


def normalize_yaw_deg(yaw_deg: float) -> float:
    return ((yaw_deg + 180.0) % 360.0) - 180.0


def yaw_error_deg(actual: float, target: float) -> float:
    return abs(normalize_yaw_deg(target - actual))


def signed_yaw_delta_deg(current: float, target: float) -> float:
    return normalize_yaw_deg(target - current)


def distance_m(a: Pose, b: Pose) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def bearing_deg(start: Pose, goal: Pose) -> float:
    return math.degrees(math.atan2(goal.y - start.y, goal.x - start.x))


def advance_pose(pose: Pose, distance_meters: float) -> Pose:
    yaw_rad = math.radians(pose.yaw_deg)
    return Pose(
        x=pose.x + math.cos(yaw_rad) * distance_meters,
        y=pose.y + math.sin(yaw_rad) * distance_meters,
        yaw_deg=normalize_yaw_deg(pose.yaw_deg),
    )


def rotate_pose(pose: Pose, delta_yaw_deg: float) -> Pose:
    return Pose(x=pose.x, y=pose.y, yaw_deg=normalize_yaw_deg(pose.yaw_deg + delta_yaw_deg))

