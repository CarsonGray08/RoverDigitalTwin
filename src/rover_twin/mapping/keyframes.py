"""The keyframe buffer: what the map is re-fused from when poses change."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from rover_twin.config import Keyframes
from rover_twin.phone.protocol import RgbdFrame


@dataclass(frozen=True, eq=False)
class Keyframe:
    frame: RgbdFrame
    depth: np.ndarray        # confidence-masked, as fused
    T_A_C: np.ndarray        # odometry pose; corrections replace it, never the depth


class KeyframeBuffer:
    """Keeps a frame once the camera has moved ``min_translation_m`` or turned
    ``min_rotation_deg`` since the last keyframe. Both 0 keeps every frame,
    which covers more far floor but makes a re-fuse take seconds."""

    def __init__(self, cfg: Keyframes) -> None:
        self.cfg = cfg
        self.frames: list[Keyframe] = []

    def offer(self, keyframe: Keyframe) -> bool:
        if self.frames:
            last = self.frames[-1].T_A_C
            moved = np.linalg.norm(keyframe.T_A_C[:3, 3] - last[:3, 3])
            cos = (np.trace(last[:3, :3].T @ keyframe.T_A_C[:3, :3]) - 1.0) / 2.0
            turned = math.degrees(math.acos(min(1.0, max(-1.0, cos))))
            if moved < self.cfg.min_translation_m and turned < self.cfg.min_rotation_deg:
                return False
        self.frames.append(keyframe)
        return True

    def __iter__(self):
        return iter(self.frames)

    def __len__(self) -> int:
        return len(self.frames)
