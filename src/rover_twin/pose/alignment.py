"""Anchoring the ARKit world into the twin: ``T_W_A``."""

from __future__ import annotations

import math
from collections import deque

import numpy as np

from rover_twin import config, frames
from rover_twin.frames import M_AXIS, Mount, Pose2D
from rover_twin.phone.protocol import TRACKING_NORMAL, PhonePose


class Alignment:
    """Waits for the phone to hold still with normal tracking, then puts the
    rover it sees at the configured start pose. The rover is taken to stand
    level there, so the phone's tilt and roll on its mount are measured then
    (``mount``), not configured: they change a little every time it is mounted.

    ARKit can report normal tracking while the phone is still being handled,
    so ``stable_samples`` consecutive poses must agree within the configured
    span. :meth:`reset` (Reset Position) waits for a fresh stable window.
    """

    def __init__(self, cfg: config.Alignment, mount: Mount, start: Pose2D, start_z: float) -> None:
        self.cfg, self.start, self.start_z = cfg, start, start_z
        self.configured = self.mount = mount
        self.T_W_A: np.ndarray | None = None
        self._window: deque[np.ndarray] = deque(maxlen=cfg.stable_samples)

    def reset(self) -> None:
        self.T_W_A = None
        self._window.clear()

    def offer(self, pose: PhonePose) -> bool:
        """True when ``pose`` completes a stable window and anchors the map."""
        if self.T_W_A is not None or pose.tracking != TRACKING_NORMAL:
            return False
        self._window.append(pose.T_A_C)
        if len(self._window) < self.cfg.stable_samples:
            return False
        rovers = [frames.rover_pose(M_AXIS @ T, self.configured) for T in self._window]
        last = rovers[-1]
        moved = max(math.hypot(p.x - last.x, p.y - last.y) for p in rovers)
        turned = max(abs(frames.wrap_deg(p.yaw_deg - last.yaw_deg)) for p in rovers)
        if moved > self.cfg.max_translation_m or turned > self.cfg.max_yaw_deg:
            return False
        self.mount = self.configured.leveled(M_AXIS @ self._window[-1])
        self.T_W_A = frames.planar_alignment(self._window[-1], self.mount, self.start, self.start_z)
        return True
