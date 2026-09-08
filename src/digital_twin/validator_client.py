from __future__ import annotations

import json
import http.client
from dataclasses import dataclass
from urllib import error, request

from digital_twin.models import Pose, ValidationRequest, ValidationResult
from digital_twin.validator import PlanValidator


@dataclass
class HttpPlanValidator(PlanValidator):
    url: str
    timeout_s: float = 300.0

    def set_goal_marker(self, goal: Pose) -> None:
        """Ask the Isaac service to place its visual goal marker (best effort).

        The marker is a rendering aid for the vision planner; older services
        without the endpoint, or transport errors, must not break the loop.
        """
        marker_url = self.url.rsplit("/", 1)[0] + "/goal_marker"
        body = json.dumps({"x": goal.x, "y": goal.y}).encode("utf-8")
        http_request = request.Request(
            marker_url,
            data=body,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(http_request, timeout=self.timeout_s):
                pass
        except (error.HTTPError, error.URLError, http.client.RemoteDisconnected):
            pass

    def validate(self, validation_request: ValidationRequest) -> ValidationResult:
        body = json.dumps(validation_request.to_dict()).encode("utf-8")
        http_request = request.Request(
            self.url,
            data=body,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(http_request, timeout=self.timeout_s) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Validator returned HTTP {exc.code}: {detail}") from exc
        except http.client.RemoteDisconnected as exc:
            raise RuntimeError(f"Validator at {self.url} closed the connection before returning a response") from exc
        except error.URLError as exc:
            raise RuntimeError(f"Could not reach validator at {self.url}: {exc.reason}") from exc
        return ValidationResult.from_dict(payload)
