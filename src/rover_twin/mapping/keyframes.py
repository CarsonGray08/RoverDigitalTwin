"""The keyframe buffer: what the map is re-fused from when poses change."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np

from rover_twin.config import Keyframes
from rover_twin.phone.protocol import RgbdFrame


@dataclass(frozen=True, eq=False)
class Keyframe:
    frame: RgbdFrame
    depth: np.ndarray        # confidence-masked, as fused
    T_A_C: np.ndarray        # odometry pose; corrections replace it, never the depth


RECENT = 3       # frames kept with a keyframe: enough for the map to see someone gone


class KeyframeBuffer:
    """Keeps a frame once the camera has moved ``min_translation_m`` or turned
    ``min_rotation_deg`` since the last keyframe. Both 0 keeps every frame,
    which covers more far floor but makes a re-fuse take seconds. A frame
    that is not a keyframe is kept with the last one as one of its RECENT
    frames, at most one per ``recent_s``: what its place looks like now.
    ``changed[k]`` marks a keyframe whose place has changed since (the map
    cleared something there), the only ones whose recent frames a re-fuse needs."""

    def __init__(self, cfg: Keyframes) -> None:
        self.cfg = cfg
        self.frames: list[Keyframe] = []
        self.recent: list[deque[Keyframe]] = []
        self.changed: list[bool] = []

    def offer(self, keyframe: Keyframe) -> bool:
        """True if ``keyframe`` is a new keyframe."""
        if self.frames:
            last = self.frames[-1].T_A_C
            moved = np.linalg.norm(keyframe.T_A_C[:3, 3] - last[:3, 3])
            cos = (np.trace(last[:3, :3].T @ keyframe.T_A_C[:3, :3]) - 1.0) / 2.0
            turned = math.degrees(math.acos(min(1.0, max(-1.0, cos))))
            if moved < self.cfg.min_translation_m and turned < self.cfg.min_rotation_deg:
                recent = self.recent[-1]
                since = (recent[-1] if recent else self.frames[-1]).frame.t_phone
                if self.cfg.recent_s > 0 and keyframe.frame.t_phone - since >= self.cfg.recent_s:
                    recent.append(keyframe)
                return False
        self.frames.append(keyframe)
        self.recent.append(deque(maxlen=RECENT))
        self.changed.append(False)
        return True

    def __iter__(self):
        return iter(self.frames)

    def __len__(self) -> int:
        return len(self.frames)
