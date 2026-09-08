from __future__ import annotations

import base64
import binascii
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib import error, request

from digital_twin.models import Pose


@dataclass(frozen=True)
class RoverObservation:
    pose: Pose | None
    payload: dict[str, Any]
    vision_rgb_base64: str | None = None
    chase_rgb_base64: str | None = None  # third-person view; rover visible in frame

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RoverObservation":
        pose = None
        raw_pose = payload.get("rover_pose")
        if isinstance(raw_pose, dict):
            pose = Pose.from_dict(raw_pose, "observation.rover_pose")

        def frame(key: str) -> str | None:
            vision = payload.get(key)
            if isinstance(vision, dict) and vision.get("status") == "ok" and isinstance(vision.get("rgb_base64"), str):
                return vision["rgb_base64"]
            return None

        return cls(
            pose=pose,
            payload=payload,
            vision_rgb_base64=frame("vision"),
            chase_rgb_base64=frame("vision_chase"),
        )

    # Fields from the raw Isaac observation that are useful to a planning model.
    # Everything else (USD prim paths, rover_z internals, wheel-joint paths,
    # lab_bounds) is noise that bloats the prompt and distracts the model.
    _CONTEXT_WHITELIST = (
        "rover_pose",
        "nearest_scene_geometry",
        "pose_control_prim_path",
        "pose_forward_axis",
    )

    def compact_context(self) -> dict[str, Any]:
        context: dict[str, Any] = {
            key: self.payload[key] for key in self._CONTEXT_WHITELIST if key in self.payload
        }
        for key in ("vision", "vision_chase"):
            vision = self.payload.get(key)
            if isinstance(vision, dict):
                vision = dict(vision)
                if "rgb_base64" in vision:
                    vision["rgb_base64"] = "<attached image>"
                context[key] = vision
        return context

    def save_vision_png(self, path: str | Path) -> Path:
        if not self.vision_rgb_base64:
            raise ValueError("Observation does not include an ok vision.rgb_base64 frame")
        try:
            image_bytes = base64.b64decode(self.vision_rgb_base64, validate=True)
        except binascii.Error as exc:
            raise ValueError("Observation vision.rgb_base64 is not valid base64") from exc
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(image_bytes)
        return output_path


@dataclass
class HttpObservationClient:
    url: str
    timeout_s: float = 30.0

    def observe(self) -> RoverObservation:
        http_request = request.Request(
            self.url,
            headers={"Accept": "application/json"},
            method="GET",
        )
        try:
            with request.urlopen(http_request, timeout=self.timeout_s) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Observation endpoint returned HTTP {exc.code}: {detail}") from exc
        except error.URLError as exc:
            raise RuntimeError(f"Could not reach observation endpoint at {self.url}: {exc.reason}") from exc
        except json.JSONDecodeError as exc:
            raise RuntimeError("Observation endpoint returned non-JSON response") from exc
        return RoverObservation.from_dict(payload)
