"""LLM route advisor: the model picks waypoints, the host builds the commands.

This replaces both previous LLM paths (the text-only waypoint planner and the
vision planner that could only echo the deterministic plan). Single division
of labor:

- The model is a tactical advisor. It sees the pose, goal geometry, world
  model, validator/execution feedback, and the rover camera frame, and
  returns ``path_blocked`` plus 0-3 intermediate waypoints in lab-frame
  meters.
- The host converts waypoints to STOP/FWD/REV/LEFT/RIGHT deterministically
  (``HeuristicLowLevelPlanner.generate_route_plan``), so command units can
  never be wrong, and the full plan still goes through the validator gate.

Reliability rules learned from the local-model era:

- Every cycle falls back to the deterministic planner on any model failure;
  a parse failure never disables the model for subsequent cycles.
- Only repeated transport failures (endpoint unreachable/timeout
  ``max_transport_failures`` times in a row) stop further attempts, so a dead
  endpoint cannot add latency to every cycle.
- One compact log line per planning call records provider, latency, and the
  decision, so slow or useless model calls are visible immediately.
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass, field
from typing import Any

from digital_twin.llm_client import LLMClient, LLMClientError, LLMReply
from digital_twin.models import CommandPlan, Pose
from digital_twin.observation_client import RoverObservation
from digital_twin.planner import HeuristicLowLevelPlanner, LowLevelPlanner, PlannerContext


class LLMPlannerError(RuntimeError):
    """Raised when the model reply cannot be turned into a usable route."""


@dataclass(frozen=True)
class RouteAdvice:
    path_blocked: bool
    waypoints: tuple[Pose, ...]
    rationale: str


@dataclass
class LLMRouteAdvisorPlanner:
    """Ask an LLM for route waypoints; build and gate all commands locally."""

    client: LLMClient
    command_builder: HeuristicLowLevelPlanner = field(default_factory=HeuristicLowLevelPlanner)
    fallback_planner: LowLevelPlanner | None = field(default_factory=HeuristicLowLevelPlanner)
    latest_observation: RoverObservation | None = None
    verbose: bool = False
    max_transport_failures: int = 3
    _transport_failures: int = field(default=0, init=False)
    _dropped_waypoint_note: str | None = field(default=None, init=False)
    _last_proposed_waypoints: tuple[Pose, ...] = field(default=(), init=False)
    _known_failure_count: int = field(default=0, init=False)
    _failed_waypoint_proposals: list[tuple[Pose, ...]] = field(default_factory=list, init=False)

    def set_observation(self, observation: RoverObservation | None) -> None:
        self.latest_observation = observation

    def generate_plan(self, context: PlannerContext) -> CommandPlan:
        # A new validator rejection since our last proposal means that proposal
        # failed; remember its waypoints so the model is told not to repeat them.
        if len(context.previous_failures) > self._known_failure_count:
            self._known_failure_count = len(context.previous_failures)
            if self._last_proposed_waypoints:
                self._failed_waypoint_proposals.append(self._last_proposed_waypoints)
        self._last_proposed_waypoints = ()
        if self._transport_failures >= self.max_transport_failures:
            return self._fallback(context, "advisor disabled after repeated transport failures")
        try:
            advice, reply = self._request_advice(context)
        except LLMClientError as exc:
            self._transport_failures += 1
            return self._fallback(context, f"transport: {exc}")
        except LLMPlannerError as exc:
            # The endpoint answered but the content was unusable; keep trying
            # on later cycles instead of latching off.
            self._transport_failures = 0
            return self._fallback(context, f"parse: {exc}")

        self._transport_failures = 0
        waypoints = self._safe_waypoints(context, advice.waypoints)
        dropped = len(advice.waypoints) - len(waypoints)
        self._dropped_waypoint_note = self._describe_dropped(context, advice.waypoints, waypoints)
        self._last_proposed_waypoints = waypoints
        if waypoints:
            plan = self.command_builder.generate_route_plan(context, list(waypoints))
            outcome = f"route waypoints={len(waypoints)}"
        else:
            plan = self._deterministic_plan(context)
            outcome = "direct" if not advice.waypoints else "direct (all waypoints unsafe)"
        self._log(
            f"{reply.provider}/{reply.model} {reply.latency_s:.2f}s {outcome}"
            + (f" dropped={dropped}" if dropped else "")
            + (f" blocked={advice.path_blocked}" if advice.path_blocked else "")
            + (f" rationale={advice.rationale!r}" if advice.rationale else "")
        )
        return plan

    # ------------------------------------------------------------------
    # model round trip
    # ------------------------------------------------------------------

    def _request_advice(self, context: PlannerContext) -> tuple[RouteAdvice, LLMReply]:
        images: list[str] = []
        labels: list[str] = []
        if self.latest_observation is not None:
            if self.latest_observation.vision_rgb_base64:
                images.append(self.latest_observation.vision_rgb_base64)
                labels.append("onboard forward camera (first person, facing the drive direction)")
            if self.latest_observation.chase_rgb_base64:
                images.append(self.latest_observation.chase_rgb_base64)
                labels.append(
                    "third-person chase camera behind and above the rover, pitched down "
                    "(the rover is visible in frame; image up-direction = rover heading)"
                )
        # The validator drives every plan to its failure point and photographs
        # the scene there; show the model what the newest rejected plan hit.
        for failure in reversed(context.previous_failures):
            frames = failure.validation.failure_frames or {}
            front = (frames.get("vision") or {}).get("rgb_base64")
            chase = (frames.get("vision_chase") or {}).get("rgb_base64")
            if not front and not chase:
                continue
            if front:
                images.append(front)
                labels.append(
                    "forward camera AT THE FAILURE POINT of the newest rejected plan "
                    "(taken at its stopped_at_pose — this is what blocked the rover)"
                )
            if chase:
                images.append(chase)
                labels.append(
                    "chase camera AT THE FAILURE POINT of the newest rejected plan "
                    "(the rover is shown stopped where validation failed)"
                )
            break
        user_payload = self._user_payload(context, camera_labels=labels)
        if self.verbose:
            self._log(f"request:\n{json.dumps(user_payload, indent=2, sort_keys=True)}")
        reply = self.client.complete_json(
            system=_SYSTEM_PROMPT,
            user_text=json.dumps(user_payload, sort_keys=True),
            json_schema=_ADVICE_SCHEMA,
            images_png_base64=images or None,
        )
        if self.verbose:
            self._log(f"raw reply ({reply.latency_s:.2f}s): {reply.text}")
        return self._parse_advice(reply.text), reply

    def _user_payload(self, context: PlannerContext, camera_labels: list[str]) -> dict[str, Any]:
        world = context.settings.world
        clearance = context.settings.limits.min_clearance_m
        keep_out_zones = [
            {
                "name": obstacle.name,
                "x": obstacle.x,
                "y": obstacle.y,
                "keep_out_radius_m": round(obstacle.radius_m + clearance, 3),
            }
            for obstacle in world.obstacles
        ]
        # Every validated collision point is a confirmed object touch point;
        # surface it as a keep-out zone so the model gets coordinates to route
        # around even when the world model has no indexed obstacles.
        seen_points: set[tuple[float, float]] = set()
        for failure in context.previous_failures:
            point = failure.collision_point()
            if point is None:
                continue
            key = (round(point[0], 2), round(point[1], 2))
            if key in seen_points:
                continue
            seen_points.add(key)
            keep_out_zones.append(
                {
                    "name": f"validated_collision_{len(seen_points)}",
                    "x": point[0],
                    "y": point[1],
                    "keep_out_radius_m": round(clearance + 0.15, 3),
                }
            )
        payload: dict[str, Any] = {
            "task": "Decide whether the direct path to the goal is clear; if not, propose 1-3 intermediate waypoints.",
            "rover_pose": context.current_simulated_pose.to_dict(),
            "target_pose": context.target_pose.to_dict(),
            "goal_relative_to_rover": context.goal_relative(),
            "world": world.to_dict(),
            "waypoint_rules": {
                "min_clearance_m": clearance,
                "keep_out_zones": keep_out_zones,
                "note": (
                    "Waypoints closer than keep_out_radius_m to an obstacle center are "
                    "discarded, and the validated route between waypoints must also keep "
                    f"{clearance} m clearance, so leave extra margin."
                ),
            },
            "rejected_plan_feedback": [f.to_summary() for f in context.previous_failures[-4:]],
            "execution_feedback": [o.to_summary() for o in context.recent_executions[-4:]],
            "cameras": (
                [f"image {i + 1}: {label}" for i, label in enumerate(camera_labels)]
                if camera_labels
                else "unavailable"
            ),
        }
        open_directions = self._open_directions_from_observation()
        if open_directions is not None:
            payload["clear_directions_from_here"] = open_directions
        if self._failed_waypoint_proposals:
            payload["your_failed_waypoint_proposals"] = [
                [{"x": round(w.x, 2), "y": round(w.y, 2)} for w in proposal]
                for proposal in self._failed_waypoint_proposals[-4:]
            ]
        if self._dropped_waypoint_note:
            payload["previous_advice_feedback"] = self._dropped_waypoint_note
        return payload

    def _open_directions_from_observation(self) -> list[dict[str, Any]] | None:
        """Drivable headings from the rover's current pose (clearance sweep).

        Read from the live /observation payload so the model knows, before it
        proposes anything, which way out of the current spot is actually clear.
        """
        if self.latest_observation is None:
            return None
        sweep = self.latest_observation.payload.get("clearance_sweep")
        if not isinstance(sweep, list):
            return None
        return [
            {"heading_deg": d["heading_deg"], "free_m": d["free_m"], "clear_point": d["clear_point"]}
            for d in sweep
            if isinstance(d, dict) and d.get("open")
        ]

    @staticmethod
    def _describe_dropped(
        context: PlannerContext,
        proposed: tuple[Pose, ...],
        kept: tuple[Pose, ...],
    ) -> str | None:
        dropped = [w for w in proposed if w not in kept]
        if not dropped:
            return None
        coords = ", ".join(f"({w.x:.2f}, {w.y:.2f})" for w in dropped)
        return (
            f"Your previous waypoints {coords} were discarded for violating lab bounds "
            "or keep_out_zones. Propose waypoints with more margin from obstacles."
        )

    @staticmethod
    def _parse_advice(text: str) -> RouteAdvice:
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise LLMPlannerError("model did not return JSON") from exc
        if not isinstance(parsed, dict):
            raise LLMPlannerError("model JSON must be an object")
        raw_waypoints = parsed.get("intermediate_waypoints")
        if not isinstance(raw_waypoints, list):
            raise LLMPlannerError("model JSON must include intermediate_waypoints")
        waypoints: list[Pose] = []
        for index, item in enumerate(raw_waypoints[:3]):
            if not isinstance(item, dict):
                raise LLMPlannerError(f"waypoint {index} must be an object")
            x, y = item.get("x"), item.get("y")
            if isinstance(x, bool) or isinstance(y, bool) or not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
                raise LLMPlannerError(f"waypoint {index} must have numeric x and y")
            waypoints.append(Pose(float(x), float(y), 0.0))
        rationale = parsed.get("rationale")
        return RouteAdvice(
            path_blocked=bool(parsed.get("path_blocked", False)),
            waypoints=tuple(waypoints),
            rationale=rationale.strip() if isinstance(rationale, str) else "",
        )

    # ------------------------------------------------------------------
    # safety + fallback
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_waypoints(context: PlannerContext, waypoints: tuple[Pose, ...]) -> tuple[Pose, ...]:
        settings = context.settings
        collision_points = [
            point
            for point in (failure.collision_point() for failure in context.previous_failures)
            if point is not None
        ]
        keep_out_m = settings.limits.min_clearance_m + 0.15
        safe: list[Pose] = []
        for waypoint in waypoints:
            if not settings.world.lab_bounds.contains(waypoint):
                continue
            if any(zone.contains(waypoint) for zone in settings.world.forbidden_zones):
                continue
            if any(
                obstacle.clearance(waypoint) < settings.limits.min_clearance_m
                for obstacle in settings.world.obstacles
            ):
                continue
            if any(
                math.hypot(waypoint.x - px, waypoint.y - py) < keep_out_m
                for px, py in collision_points
            ):
                continue
            safe.append(waypoint)
        return tuple(safe)

    def _deterministic_plan(self, context: PlannerContext) -> CommandPlan:
        planner = self.fallback_planner or self.command_builder
        return planner.generate_plan(context)

    def _fallback(self, context: PlannerContext, reason: str) -> CommandPlan:
        if self.fallback_planner is None:
            raise LLMPlannerError(reason)
        self._log(f"fallback ({reason})")
        return self.fallback_planner.generate_plan(context)

    @staticmethod
    def _log(message: str) -> None:
        print(f"[llm-planner] {message}", file=sys.stderr)


_SYSTEM_PROMPT = """
You are the navigation advisor for a small rover in a safety-gated digital twin.
The host program computes all low-level wheel commands deterministically and
validates every plan in simulation before execution. You only choose the route.

Reply with JSON matching the schema:
- path_blocked=false and an empty intermediate_waypoints list when the direct
  straight line from the rover to the goal is clear.
- path_blocked=true and 1-3 intermediate waypoints (lab-frame x/y in meters)
  routing around the blockage when the camera image shows an obstacle on the
  direct path, or when rejected_plan_feedback / execution_feedback shows a
  direct plan already failed for collision, bounds, or forbidden-zone reasons.

Waypoints must stay inside world.lab_bounds, outside world.forbidden_zones,
and outside every waypoint_rules.keep_out_zones circle — waypoints inside a
keep-out radius are discarded. The straight segments between rover, waypoints,
and goal must also keep min_clearance_m from every obstacle, so place
waypoints well to the side of a blockage, not just past its edge.

Interpreting the cameras (they have NO depth; the "cameras" field labels the
attached images in order):
- The third-person chase view shows the rover in its surroundings — use it to
  judge the scene layout, free floor, and object heights around the rover.
- On the forward camera, only LOW obstacles on the floor, centered on the
  drive line, and visually near (large in frame, bottom half) can block the
  rover. Furniture, shelves, and walls in the background are usually FARTHER
  than the goal — compare with goal_relative_to_rover.range_m, which is often
  under a meter.
- A bright green marker may be visible in the images at the target_pose; it is
  virtual (cannot collide) and shows where the goal is. Use it to relate the
  scene to the goal direction.
- The validator physically drives every plan in simulation, so its feedback is
  ground truth and outranks your reading of the image: a rejection with
  failure_type "collision" or a stall means the route truly hit something
  (detour); "goal_not_reached" with route_was_driveable=true means the route
  WORKED and only stopped goal_error_m short — KEEP that route (same
  waypoints, or none if it was direct); never switch sides or re-route over a
  small shortfall. With no collision evidence, prefer path_blocked=false and
  no waypoints.

Learning from collision rejections (rejected_plan_feedback, oldest first):
- collision_near is the lab-frame point where the rover's footprint touched a
  REAL object; each one is also listed in keep_out_zones as a
  validated_collision circle. That point is one touch point, not the object's
  full extent — the object is almost always wider.
- When a plan is rejected, extra FAILURE-POINT images are attached: the
  validator drove the rover to the exact pose where it failed and photographed
  the scene. The blocking object is directly ahead in that forward view, and
  the chase view shows the rover stopped against it. Use these to judge the
  object's true width and which side has open floor.
- Route around the blockage on the open side. Place waypoints at least 0.5 m
  from collision_near, never just past its edge.
- your_failed_waypoint_proposals are detours you already tried that still
  failed. Do NOT propose a waypoint within 0.4 m of any of them; move the
  detour substantially further out, or switch to the other side of the
  blockage.

Free-space sweeps (USE THESE — they are measured, not guessed):
- clear_directions_from_here lists the drivable headings from the rover's
  CURRENT pose. Each entry has heading_deg (absolute lab frame: 0=+x, 90=+y),
  free_m (how far the footprint can travel that way before hitting something),
  and clear_point (a reachable lab-frame x/y along it). Only open directions
  are listed; a direction NOT listed is blocked.
- clear_directions_from_failure (inside rejected_plan_feedback) is the same
  sweep taken at the pose where that plan got stuck — the ways OUT of the spot
  it failed in.
- Your FIRST waypoint must lie along one of these open directions — prefer
  using a clear_point directly. Do not propose a first waypoint in a heading
  that the sweep omitted; that is exactly the move that keeps failing. Pick the
  open direction that best advances toward the goal, then route to the goal
  from there.
If the camera is unavailable, rely on the world model and feedback only.
Keep rationale to one short sentence.
""".strip()

_ADVICE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "path_blocked": {"type": "boolean"},
        "rationale": {"type": "string"},
        "intermediate_waypoints": {
            "type": "array",
            "maxItems": 3,
            "items": {
                "type": "object",
                "properties": {
                    "x": {"type": "number"},
                    "y": {"type": "number"},
                },
                "required": ["x", "y"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["path_blocked", "intermediate_waypoints"],
    "additionalProperties": False,
}
