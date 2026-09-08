#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import errno
import json
import math
import os
import queue
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from digital_twin.bootstrap import BootstrapGate
from digital_twin.config import PlannerSettings
from digital_twin.floor_estimate import FloorEstimator
from digital_twin.geometry import distance_m, normalize_yaw_deg, yaw_error_deg
from digital_twin.live_pose import read_pose as read_live_rover_pose
from digital_twin.live_twin_session import (
    derive_live_twin_status,
    read_session as read_live_session,
)
from digital_twin.mesh_tiles import MANIFEST_NAME, load_manifest, tile_npz_path
from digital_twin.nvblox_usd import author_nvblox_tiles
from digital_twin.terrain_heightmap import TerrainHeightmap
from digital_twin.traversability import (
    TerrainSample,
    TraversabilityLimits,
    evaluate as evaluate_traversability,
)
from digital_twin.live_scan import (
    MSG_FINALIZE,
    MSG_KEYFRAME_IMAGE,
    MSG_REALIGN,
    TRACKING_NORMAL,
    LiveScanReceiver,
    LiveScanState,
    PhonePoseStore,
    rasterize_xy_points,
)
from digital_twin.models import CommandName, Pose, ValidationError, ValidationRequest, ValidationResult
from digital_twin.occupancy import FloorPlane, OccupancyConfig, OccupancyGrid
from digital_twin.phone_mount import (
    Mat4,
    PhoneMountCalibration,
    compute_t_align,
    host_from_mat4,
    registration_delta,
    rover_pose_from_camera,
)


def _load_isaac(headless: bool, experience: str = ""):
    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": headless}, experience=experience)
    return simulation_app

@dataclass(frozen=True)
class SceneObstacle:
    path: str
    min_x: float
    max_x: float
    min_y: float
    max_y: float

    def clearance(self, pose: Pose) -> float:
        if self.min_x <= pose.x <= self.max_x and self.min_y <= pose.y <= self.max_y:
            return 0.0
        dx = max(self.min_x - pose.x, 0.0, pose.x - self.max_x)
        dy = max(self.min_y - pose.y, 0.0, pose.y - self.max_y)
        return math.hypot(dx, dy)


class IsaacSceneValidator:
    def __init__(
        self,
        simulation_app,
        scene_path: str,
        rover_prim_path: str,
        settings: PlannerSettings,
        render: bool,
        max_obstacle_bbox_area_m2: float,
        wheel_velocity_scale: float,
        wheel_max_force: float,
        left_wheel_sign: float,
        right_wheel_sign: float,
        pose_forward_axis: str,
        rover_start_z: float | None,
        rover_z_offset: float,
        settle_steps: int,
        wheel_diameter_m: float,
        camera_width: int,
        camera_height: int,
        disable_physx_overlap: bool,
        wheel_static_friction: float,
        wheel_dynamic_friction: float,
        solver_position_iterations: int,
        solver_velocity_iterations: int,
        center_com: bool,
        live_scan_enabled: bool = False,
        live_scan_host: str = "0.0.0.0",
        live_scan_port: int = 9099,
        live_scan_root: str = "/World/LiveScan",
        live_scan_rate_hz: float = 10.0,
        live_scan_grid_decay_s: float = 0.0,
        live_scan_transform_json: str | None = None,
        phone_mount_json: str | None = None,
        twin_start_pose: Pose | None = None,
        twin_start_z: float = 0.0,
        auto_twin_align: bool = False,
        support_ground_enabled: bool = False,
        support_ground_size_m: float = 6.0,
        support_ground_thickness_m: float = 0.1,
        support_ground_path: str = "/World/SupportGround",
        bootstrap_gate_enabled: bool = False,
        bootstrap_coverage_radius_m: float = 1.5,
        bootstrap_coverage_threshold: float = 0.6,
        baseline_exclusion_radius_m: float = 0.45,
        baseline_min_terrain_tiles: int = 1,
        occupancy_enabled: bool = False,
        hide_static_scan_after_live_scan: bool = False,
        hide_static_scene_at_start: bool = False,
        disable_static_scene_collision: bool = False,
        live_scan_light: bool = True,
        live_scan_light_scale: tuple[float, float, float] = (10.0, 10.0, 1.0),
        live_scan_light_height_padding_m: float = 0.5,
        live_scan_light_intensity: float = 5000.0,
        track_live_pose: bool = False,
        map_dir: Path | None = None,
        load_terrain: bool = False,
        terrain_root: str = "/World/NvbloxMap",
        terrain_static_friction: float = 1.0,
        terrain_dynamic_friction: float = 1.0,
        terrain_restitution: float = 0.0,
        terrain_traversability: bool = False,
        traversability_limits: TraversabilityLimits | None = None,
        scene_box: tuple[float, float, float] | None = None,
    ) -> None:
        self.simulation_app = simulation_app
        self.scene_path = scene_path
        self.rover_prim_path = rover_prim_path
        self.settings = settings
        self.render = render
        self.max_obstacle_bbox_area_m2 = max_obstacle_bbox_area_m2
        self.wheel_velocity_scale = wheel_velocity_scale
        self.wheel_max_force = wheel_max_force
        self.left_wheel_sign = left_wheel_sign
        self.right_wheel_sign = right_wheel_sign
        self.pose_forward_axis = pose_forward_axis
        self.rover_start_z = rover_start_z
        self.rover_z_offset = rover_z_offset
        self.settle_steps = settle_steps
        self.wheel_diameter_m = wheel_diameter_m
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.disable_physx_overlap = disable_physx_overlap
        self.wheel_static_friction = wheel_static_friction
        self.wheel_dynamic_friction = wheel_dynamic_friction
        self.solver_position_iterations = solver_position_iterations
        self.solver_velocity_iterations = solver_velocity_iterations
        self.center_com = center_com
        self.live_scan_enabled = live_scan_enabled
        self.live_scan_host = live_scan_host
        self.live_scan_port = live_scan_port
        self.live_scan_root = live_scan_root
        self.live_scan_rate_hz = live_scan_rate_hz
        self.live_scan_grid_decay_s = live_scan_grid_decay_s
        self.live_scan_transform_json = live_scan_transform_json
        self.phone_mount_json = phone_mount_json
        # Loaded eagerly so a bad calibration fails fast at startup, not mid-run.
        # None means "no mount calibration" -> reported rover pose falls back to
        # the raw axis-swapped camera pose (identity mount).
        self._phone_mount: PhoneMountCalibration | None = (
            PhoneMountCalibration.load(phone_mount_json) if phone_mount_json else None
        )
        # the single shared ARKit->twin alignment T_align. None until the
        # first normal ARKit pose initializes it. Feeds BOTH the rover pose path
        # and LiveScanState.arkit_to_twin (one transform, never two).
        self.twin_start_pose = twin_start_pose if twin_start_pose is not None else Pose(0.0, 0.0, 0.0)
        self.twin_start_z = twin_start_z
        self.auto_twin_align = auto_twin_align
        self._t_align: Mat4 | None = None
        # Set when the phone sends MSG_REALIGN (Reset Position) but no usable
        # normal-tracking pose is available yet; the next normal pose then
        # re-anchors the alignment to the phone's current (mounted) pose.
        self._pending_realign = False
        # SupportGround physics scaffold + gravity-frame floor estimate.
        # SupportGround is physics-only (a Cube, so it is never indexed as a
        # planning obstacle and never enters the traversability grid).
        self.support_ground_enabled = support_ground_enabled
        self.support_ground_size_m = support_ground_size_m
        self.support_ground_thickness_m = support_ground_thickness_m
        self.support_ground_path = support_ground_path
        # Keep the provisional slab just below the nominal scanned floor.  This
        # avoids a coplanar collider when a z=0 scan tile arrives while retaining
        # a stable, invisible fallback before any direct terrain exists.
        self._support_ground_height = twin_start_z - 0.005
        self._floor_estimator = FloorEstimator()
        self._floor_last_applied_height: float | None = None
        # nvblox-tile flow: drop SupportGround onto the scanned floor exactly once
        # (see _lower_support_ground_to_terrain). Re-authoring the collider every
        # rebuild re-cooks PhysX and keeps the rover in a fall/settle loop.
        self._support_ground_lowered = False
        # startup-360 numeric coverage gate. Navigation stays disabled
        # until enough cells within the radius are known.
        self.bootstrap_gate_enabled = bootstrap_gate_enabled
        self._bootstrap_gate = BootstrapGate(
            radius_m=bootstrap_coverage_radius_m,
            cell_m=self._POINT_GRID_CELL_M,
            coverage_threshold=bootstrap_coverage_threshold,
            exclusion_radius_m=baseline_exclusion_radius_m,
        )
        self.baseline_min_terrain_tiles = max(1, int(baseline_min_terrain_tiles))
        self._live_known_cell_keys: list[tuple[int, int]] = []
        # plane-relative occupancy/traversability. Classifies cells by
        # height above the fitted floor (not a fixed world-Z band).
        self.occupancy_enabled = occupancy_enabled
        self._occupancy_counts: dict[str, int] = {}
        self._obstacle_proxies: list = []
        self.hide_static_scan_after_live_scan = hide_static_scan_after_live_scan
        self.hide_static_scene_at_start = hide_static_scene_at_start
        self.disable_static_scene_collision = disable_static_scene_collision
        self.live_scan_light = live_scan_light
        self.live_scan_light_scale = live_scan_light_scale
        self.live_scan_light_height_padding_m = live_scan_light_height_padding_m
        self.live_scan_light_intensity = live_scan_light_intensity
        # Continuous localization ("Track" mode): slave the sim rover to the live
        # twin-frame rover pose the mapper publishes (rover_pose.json). The mapper
        # owns the phone socket, so the pose AND the terrain tiles arrive over the
        # filesystem boundary, both from this one map dir.
        self.track_live_pose = track_live_pose
        self.map_dir = map_dir
        self._live_pose_seq: int | None = None
        self._latest_live_pose_payload: dict[str, Any] | None = None
        # Live terrain ingest: author the nvblox tile transport (collidable) into
        # THIS stage, so the rover validates/drives on the live reconstructed
        # surface instead of a separate viewer stage.
        self.load_terrain = load_terrain
        self.terrain_root = terrain_root
        self.terrain_static_friction = terrain_static_friction
        self.terrain_dynamic_friction = terrain_dynamic_friction
        self.terrain_restitution = terrain_restitution
        self._terrain_tile_state: dict = {}
        self._terrain_last_mtime = -1.0
        # 2.5-D max-height grid over the live terrain, for belly-clearance lookup.
        self._terrain_heightmap: TerrainHeightmap | None = None
        self._terrain_tile_count = 0
        self._terrain_map_revision = 0
        # Physics-derived traversability verdict: once the rover settles on the
        # terrain colliders, its body tilt is the local slope, so a rollout's
        # slope/roll/pitch are checked against the rover's envelope (tip-over).
        self.terrain_traversability = terrain_traversability
        self.traversability_limits = traversability_limits or TraversabilityLimits()
        self.scene_box = scene_box
        self._collision_mesh_path: str | None = None
        self._pose_forward_axis_yaw_offset_deg = {
            "x": 0.0,
            "y": 90.0,
            "neg_x": 180.0,
            "neg_y": -90.0,
        }[pose_forward_axis]

        import omni.usd
        from pxr import UsdPhysics
        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleArticulation
        from isaacsim.core.utils.types import ArticulationAction
        from isaacsim.core.utils.stage import open_stage

        self._articulation_actuators_cls = None
        try:
            from isaacsim.core.utils.extensions import enable_extension

            for extension in (
                "omni.usd.schema.newton",
                "isaacsim.physics.newton",
                "isaacsim.physics.newton.tensors",
                "isaacsim.core.experimental.prims",
                "isaacsim.core.experimental.actuators",
            ):
                enable_extension(extension)
            for _ in range(20):
                self.simulation_app.update()
            from isaacsim.core.experimental.actuators import ArticulationActuators

            self._articulation_actuators_cls = ArticulationActuators
        except Exception as exc:
            print(f"Newton actuator support unavailable; falling back to legacy articulation drive: {exc}")

        self._omni_usd = omni.usd
        self._usd_physics = UsdPhysics
        self._world_cls = World
        self._single_articulation_cls = SingleArticulation
        self._articulation_action_cls = ArticulationAction
        self._open_stage = open_stage
        self._world = None
        self._articulation = None
        self._newton_actuators = None
        self._wheel_drive_backend = "unconfigured"
        self._pose_control_prim = None
        self._wheel_dof_indices: list[int] = []
        self._wheel_dof_sides: list[str] = []
        self._camera = None
        self._chase_camera = None
        self._cameras_parented = False
        self._front_camera_path = "/World/front_camera"
        self._chase_camera_path = "/World/chase_camera"
        self._body_half_extents: tuple[float, float, float] | None = None
        # Distance from the articulation root down to the rover's lowest collider,
        # measured once at init. Lets us sit the rover on terrain of any height:
        # root_z = terrain_z_under_footprint + this. None until measured.
        self._rover_root_to_bottom_m: float | None = None
        self._last_motion_failure: dict[str, Any] | None = None
        # Mid-validation /observation servicing: main() installs a callback
        # that drains queued observation requests; _step() invokes it while a
        # validation is running so the dashboard's live view keeps updating
        # instead of freezing for the whole rehearsal.
        self.observation_service: Any = None
        self._validation_active = False
        self._observation_service_busy = False
        self._observation_service_last_s = 0.0
        self._failure_frames: dict[str, Any] | None = None
        self._failure_clearance: list[dict[str, Any]] | None = None
        self._live_scan_queue: queue.Queue | None = None
        self._live_scan_receiver: LiveScanReceiver | None = None
        self._live_scan_state: LiveScanState | None = None
        self._phone_pose_store: PhonePoseStore | None = None
        self._live_scan_last_usd_update_s = 0.0
        self._live_scan_obstacle_cells = 0
        self._live_scan_known_cells = 0
        self._live_scan_has_replaced_grid = False
        self._live_scan_static_hidden = False
        self._static_scene_hidden_at_start = False
        # Texture bookkeeping for projected-keyframe coloring: image ids whose
        # JPEG has been written to disk + material authored, and the temp dir.
        self._live_scan_texture_dir: Path | None = None
        self._live_scan_texture_image_ids: set[str] = set()
        self._live_scan_texture_warned = False
        # Snapshot of the static-scan obstacle grid, captured the first time the
        # live grid is refreshed. The live grid is unioned with this baseline so
        # unswept regions keep their static obstacles (conservative) instead of
        # becoming free space the moment the first live frame arrives.
        self._static_obstacle_grid: dict[tuple[int, int], Any] | None = None
        self._load_stage()
        if self.support_ground_enabled:
            self._ensure_support_ground()
        elif self.load_terrain:
            self._disable_authored_support_ground()
        if self.live_scan_enabled:
            self._initialize_live_scan()

    def validate(self, request: ValidationRequest) -> ValidationResult:
        self._validation_active = True
        try:
            return self._validate_impl(request)
        finally:
            self._validation_active = False
            self._restore_live_pose_after_validation()

    def _validate_impl(self, request: ValidationRequest) -> ValidationResult:
        self._last_motion_failure: dict[str, Any] | None = None
        self._failure_frames = None
        self._failure_clearance = None
        # Skip STOP-only prime/sync requests where goal == start: they would
        # plant the marker on top of the rover and mislead the vision planner.
        if distance_m(request.start_pose, request.goal) > 1e-3:
            self._set_goal_marker(request.goal)
        self._reset_stage(request.start_pose)
        limits_result = self._validate_command_limits(request)
        if limits_result is not None:
            return limits_result
        unknown_result = self._reject_unready_or_unknown_terrain(request)
        if unknown_result is not None:
            return unknown_result

        pose = request.start_pose
        trajectory: list[Pose] = [pose]
        step_results: list[dict[str, Any]] = []
        expected_step_end_poses: list[Pose] = []
        min_clearance = float("inf")
        terrain_samples: list[TerrainSample] = []

        for command_index, command in enumerate(request.commands):
            samples = self._execute_command(pose, command)
            for actual_pose in samples[1:]:
                trajectory.append(actual_pose)
                safety = self._check_safety(actual_pose)
                min_clearance = min(min_clearance, safety["min_clearance_m"])
                if safety["failure_type"] is not None:
                    self._stop_wheels()
                    return ValidationResult(
                        approved=False,
                        failure_type=safety["failure_type"],
                        failed_command_index=command_index,
                        reason=safety["reason"],
                        final_pose=actual_pose,
                        min_clearance_m=min_clearance,
                        trajectory=tuple(trajectory),
                        step_results=tuple(step_results),
                        expected_step_end_poses=tuple(expected_step_end_poses),
                        # The rover is sitting at the failure pose right now;
                        # photograph what it was about to hit and scan the free
                        # directions out of it.
                        failure_frames=self._failure_frames or self._capture_vision(),
                        failure_clearance=self._failure_clearance or self._clearance_sweep(actual_pose),
                    )

            pose = samples[-1]
            if self._last_motion_failure is not None:
                return ValidationResult(
                    approved=False,
                    failure_type=self._last_motion_failure["failure_type"],
                    failed_command_index=command_index,
                    reason=self._last_motion_failure["reason"],
                    final_pose=pose,
                    min_clearance_m=0.0 if min_clearance == float("inf") else min_clearance,
                    trajectory=tuple(trajectory),
                    step_results=tuple(step_results),
                    expected_step_end_poses=tuple(expected_step_end_poses),
                    failure_frames=self._failure_frames or self._capture_vision(),
                    failure_clearance=self._failure_clearance or self._clearance_sweep(pose),
                )
            if self.terrain_traversability:
                slope, roll, pitch = self._articulation_tilt()
                terrain_samples.append(
                    TerrainSample(slope_deg=slope, roll_deg=roll, pitch_deg=pitch,
                                  ground_clearance_m=self._terrain_clearance(pose)))
            expected_step_end_poses.append(pose)
            step_results.append(
                {
                    "command_index": command_index,
                    "command": command.to_dict(),
                    "end_pose": pose.to_dict(),
                    "status": "completed",
                }
            )

        if self.terrain_traversability:
            verdict = evaluate_traversability(terrain_samples, self.traversability_limits)
            if not verdict.traversable:
                self._stop_wheels()
                failed = verdict.failed_index if verdict.failed_index is not None else len(request.commands) - 1
                return ValidationResult(
                    approved=False,
                    failure_type=verdict.reason,
                    failed_command_index=failed,
                    reason=(
                        f"Terrain not traversable ({verdict.reason}): "
                        f"max slope {verdict.max_slope_deg:.1f} deg, roll "
                        f"{verdict.max_abs_roll_deg:.1f} deg, pitch "
                        f"{verdict.max_abs_pitch_deg:.1f} deg, "
                        f"min clearance {verdict.min_ground_clearance_m:.3f} m"
                    ),
                    final_pose=expected_step_end_poses[failed] if expected_step_end_poses else pose,
                    min_clearance_m=0.0 if min_clearance == float("inf") else min_clearance,
                    trajectory=tuple(trajectory),
                    step_results=tuple(step_results),
                    expected_step_end_poses=tuple(expected_step_end_poses),
                )

        goal_error = distance_m(pose, request.goal)
        yaw_error = yaw_error_deg(pose.yaw_deg, request.goal.yaw_deg)
        approved = (
            goal_error <= self.settings.limits.goal_tolerance_m
            and yaw_error <= self.settings.limits.goal_tolerance_yaw_deg
        )
        return ValidationResult(
            approved=approved,
            failure_type=None if approved else "goal_not_reached",
            reason="Plan reached target with no collisions"
            if approved
            else f"Final pose error {goal_error:.3f} m, yaw error {yaw_error:.1f} deg",
            final_pose=pose,
            min_clearance_m=0.0 if min_clearance == float("inf") else min_clearance,
            goal_error_m=goal_error,
            goal_error_yaw_deg=yaw_error,
            trajectory=tuple(trajectory),
            step_results=tuple(step_results),
            expected_step_end_poses=tuple(expected_step_end_poses),
        )

    def observation_snapshot(self) -> dict[str, Any]:
        """Cheap observation served mid-validation: pose + camera frames only.

        The full ``observation()`` is unsafe here -- ``_capture_vision`` steps
        the sim 4 frames to flush the render pipeline, and the clearance sweep
        is slow. During a validation the sim is stepping (and rendering)
        continuously anyway, so the render products are already fresh: just
        follow the body with the chase camera and encode what's there."""
        pose = self._get_rover_pose(fallback=Pose(0.0, 0.0, 0.0))
        result: dict[str, Any] = {
            "validation_in_progress": True,
            "rover_pose": pose.to_dict(),
        }
        try:
            self._sync_camera_to_body()
            if self._camera is not None:
                result["vision"] = self._encode_camera(self._camera)
            if self._chase_camera is not None:
                result["vision_chase"] = self._encode_camera(self._chase_camera)
        except Exception as exc:
            result["vision"] = {"status": "error", "reason": str(exc)}
        return result

    def observation(self) -> dict[str, Any]:
        pose = self._get_rover_pose(fallback=Pose(0.0, 0.0, 0.0))
        live_twin = self._live_twin_status()
        clearances = sorted(
            (
                {
                    "prim_path": obstacle.path,
                    "clearance_m": obstacle.clearance(pose),
                    "bounds_xy": {
                        "min_x": obstacle.min_x,
                        "max_x": obstacle.max_x,
                        "min_y": obstacle.min_y,
                        "max_y": obstacle.max_y,
                    },
                }
                for obstacle in self._obstacles
            ),
            key=lambda item: item["clearance_m"],
        )
        return {
            # Kept flat for shell/operator polling; ``live_twin`` below repeats
            # the same block for clients that prefer a namespace.
            **live_twin,
            "scene_path": self.scene_path,
            "rover_prim_path": self.rover_prim_path,
            "pose_control_prim_path": str(self._pose_control_prim.GetPath()),
            "articulation_prim_path": str(self._articulation_prim.GetPath()),
            "pose_prim_path": str(self._pose_prim.GetPath()),
            "wheel_static_friction": self.wheel_static_friction,
            "wheel_dynamic_friction": self.wheel_dynamic_friction,
            "solver_position_iterations": self.solver_position_iterations,
            "solver_velocity_iterations": self.solver_velocity_iterations,
            "pose_forward_axis": self.pose_forward_axis,
            "wheel_joints": {
                "left": [str(prim.GetPath()) for prim in self._left_wheel_joints],
                "right": [str(prim.GetPath()) for prim in self._right_wheel_joints],
            },
            "rover_pose": pose.to_dict(),
            "rover_z": {
                "root_z": self._get_world_translation(self.rover_prim)[2],
                "articulation_root_z": self._get_world_translation(self._articulation_prim)[2],
                "pose_body_z": self._get_world_translation(self._pose_prim)[2],
                "configured_root_z": self._base_z,
                "configured_articulation_root_z": self._articulation_base_z,
                "initial_root_z": self._initial_base_z,
                "initial_articulation_root_z": self._initial_articulation_base_z,
                "z_offset": self.rover_z_offset,
                "live_root_z": self._articulation_live_root_z(),
            },
            "nearest_scene_geometry": clearances[:20],
            "clearance_sweep": self._clearance_sweep(pose),
            "lab_bounds": self.settings.world.lab_bounds.to_dict(),
            "live_scan": self._live_scan_status(),
            "phone_pose": self._phone_pose_status(),
            "floor": self._floor_status(),
            "bootstrap": self._bootstrap_status(),
            "occupancy": self._occupancy_status(),
            "camera_debug": self._camera_debug(),
            "live_twin": live_twin,
            **self._capture_vision(),
        }

    def _initialize_twin_alignment(self, camera_transform: tuple[float, ...]) -> None:
        """Establish the shared ARKit->twin alignment from the first normal pose.

        Computes T_align so the start camera pose maps to the configured twin
        start pose, then re-anchors any mesh already received and adopts the new
        registration. Pose and mesh thereafter share this one transform.
        """

        calibration = self._phone_mount or PhoneMountCalibration.identity()
        t_align = compute_t_align(camera_transform, calibration, self.twin_start_pose, self.twin_start_z)
        new_registration = host_from_mat4(t_align)
        if self._live_scan_state is not None:
            delta = registration_delta(self._live_scan_state.arkit_to_twin, t_align)
            self._live_scan_state.apply_registration_delta(delta, new_registration)
        self._t_align = t_align
        pose = self.twin_start_pose
        print(
            "Twin alignment initialized at first normal ARKit pose -> rover start "
            f"({pose.x:.3f}, {pose.y:.3f}, yaw {pose.yaw_deg:.1f}); shared T_align now "
            "drives both rover pose and live-scan mesh."
        )

    def apply_live_pose_tracking(self) -> bool:
        """Track mode: slave the sim rover to the latest published live rover pose.

        Reads ``rover_pose.json`` from the mapper; on a fresh, normal-tracking
        pose it teleports the articulation there and zeroes its motion so the sim
        rover holds at the real rover's measured pose. Returns True if it moved
        the rover. Called only from the single-threaded main loop that owns the
        Isaac World, so no locking is needed."""
        if not self.track_live_pose or self.map_dir is None:
            return False
        payload = read_live_rover_pose(self.map_dir)
        if not payload:
            return False
        self._latest_live_pose_payload = payload
        seq = payload.get("seq")
        if seq == self._live_pose_seq or not payload.get("tracking_normal", False):
            return False
        self._live_pose_seq = seq
        pose = Pose(float(payload["x"]), float(payload["y"]), float(payload["yaw_deg"]))
        self._apply_live_tracking_pose(pose)
        return True

    def _restore_live_pose_after_validation(self) -> None:
        """Put the visible rover back on the real pose after validation playback.

        ``/validate_plan`` uses the live articulation to simulate a proposed
        command. With ARKit pose streaming the idle Track loop usually snaps it
        back immediately because a new phone-pose seq arrives. Online cuVSLAM
        only publishes on RGB-D keyframes, so there may be no fresh seq before
        the caller prompts the operator. Restore from the latest pose payload
        explicitly so validation never leaves the visible rover at the predicted
        future pose.
        """
        if not self.track_live_pose or self.map_dir is None or self._articulation is None:
            return
        payload = read_live_rover_pose(self.map_dir)
        if not payload or not payload.get("tracking_normal", False):
            self._stop_wheels()
            return
        try:
            pose = Pose(float(payload["x"]), float(payload["y"]), float(payload["yaw_deg"]))
            self._apply_live_tracking_pose(pose)
            self._zero_articulation_motion()
            self._latest_live_pose_payload = payload
            self._live_pose_seq = payload.get("seq")
            for _ in range(2):
                self._step()
        except Exception as exc:  # noqa: BLE001 - validation result should still return
            print(f"[track-live-pose] restore after validation failed: {exc}")

    def load_terrain_tiles(self) -> int:
        """Author the live nvblox terrain (collidable) into THIS stage from the
        tile transport, so command validation runs on the live reconstructed
        surface. Throttled to the manifest mtime and O(changed tiles); returns the
        number of tiles authored this call. Single-threaded with validation jobs."""
        if not self.load_terrain or self.map_dir is None:
            return 0
        try:
            mtime = (Path(self.map_dir) / MANIFEST_NAME).stat().st_mtime
        except OSError:
            return 0
        if mtime == self._terrain_last_mtime:
            return 0
        self._terrain_last_mtime = mtime
        _, authored, _ = author_nvblox_tiles(
            self.stage, self.map_dir, self.terrain_root, self._terrain_tile_state,
            collision=True,
            terrain_static_friction=self.terrain_static_friction,
            terrain_dynamic_friction=self.terrain_dynamic_friction,
            terrain_restitution=self.terrain_restitution)
        self._rebuild_terrain_heightmap()
        return authored

    def _rebuild_terrain_heightmap(self) -> None:
        """Rebuild the belly-clearance heightmap from the current tile vertices
        (the same geometry just authored as colliders)."""
        try:
            manifest = load_manifest(self.map_dir)
            tiles = manifest.get("tiles", {})
            chunks = []
            for name in tiles:
                try:
                    chunks.append(np.load(tile_npz_path(self.map_dir, name))["vertices"])
                except (FileNotFoundError, OSError, ValueError):
                    continue
            if chunks:
                vertices = np.concatenate(chunks, axis=0)
                self._terrain_heightmap = TerrainHeightmap.from_points(vertices)
                self._live_known_cell_keys = list(
                    self._terrain_heightmap.observed_cells(self._POINT_GRID_CELL_M))
                self._lower_support_ground_to_terrain()
                # Feed the loaded terrain's wall-band points into the predictive
                # collision grid. With a static map and no live scan,
                # _refresh_live_obstacle_grid() returns early, so _obstacle_grid
                # would otherwise stay empty and _predict_linear_collision()
                # could not reject a drive into a reconstructed wall before the
                # physics stepped into hard contact (which can trip a
                # Newton/Warp GPU fault). Only override when there is no live
                # scan, so the live-scan grid path is unaffected.
                if not self._live_scan_state:
                    lo, hi = self._OBSTACLE_BAND_Z
                    band = vertices[(vertices[:, 2] > lo) & (vertices[:, 2] < hi)][:, :2]
                    # _footprint_hit indexes cells as numpy arrays (points[:, 0]),
                    # so materialize the rasterized lists into float arrays.
                    self._obstacle_grid = {
                        key: np.asarray(points, dtype=np.float64)
                        for key, points in rasterize_xy_points(
                            band, self._POINT_GRID_CELL_M).items()
                    }
            else:
                self._terrain_heightmap = None
                self._live_known_cell_keys = []
            self._terrain_tile_count = len(tiles)
            self._terrain_map_revision = int(manifest.get("rev", 0))
            self._update_bootstrap_gate()
        except Exception as exc:  # noqa: BLE001 - clearance is best-effort
            print(f"[terrain] heightmap rebuild failed: {exc}")

    def _terrain_clearance(self, pose: Pose) -> float:
        """Belly clearance at ``pose``; unknown terrain is never clear terrain."""
        hm = self._terrain_heightmap
        if hm is None or hm.is_empty or self._body_half_extents is None:
            return float("-inf")
        half_x, half_y, half_z = self._body_half_extents
        belly_z = self._get_world_translation(self._pose_prim)[2] - half_z
        terrain_z = hm.max_height_in_footprint(pose.x, pose.y, math.hypot(half_x, half_y))
        if terrain_z is None:
            return float("-inf")
        return belly_z - terrain_z

    def _footprint_radius_m(self) -> float:
        if self._body_half_extents is None:
            return 0.3
        return math.hypot(self._body_half_extents[0], self._body_half_extents[1])

    def _ground_support(self, x: float, y: float):
        hm = self._terrain_heightmap
        if hm is None or hm.is_empty:
            return None
        return hm.ground_support(x, y, self._footprint_radius_m())

    def _live_twin_status(self) -> dict[str, Any]:
        """Plain cold-start state sourced only from the nvblox tile boundary."""
        session = read_live_session(self.map_dir) if self.map_dir is not None else None
        pose = self._latest_live_pose_payload or (
            read_live_rover_pose(self.map_dir) if self.map_dir is not None else None)
        pose_xy = (float(pose["x"]), float(pose["y"])) if pose else (
            self.twin_start_pose.x, self.twin_start_pose.y)
        support = self._ground_support(*pose_xy)
        source = support.source if support is not None else "unknown"
        return derive_live_twin_status(
            session=session, map_dir=self.map_dir, pose=pose,
            terrain_tiles=self._terrain_tile_count, map_revision=self._terrain_map_revision,
            baseline_min_terrain_tiles=self.baseline_min_terrain_tiles,
            coverage=self._bootstrap_status(), footprint_ground_source=source,
            support_evidence_cells=support.evidence_cells if support else 0,
        )

    def _handle_realign_request(self) -> None:
        """Reset Position (MSG_REALIGN): re-anchor the shared ARKit->twin
        alignment so the phone's current pose maps to the rover start pose. If a
        usable normal-tracking pose is in hand we re-align immediately; otherwise
        we arm a pending realign that the next normal pose fulfils."""
        latest = self._phone_pose_store.latest if self._phone_pose_store is not None else None
        if latest is not None and latest.tracking_state == TRACKING_NORMAL:
            self._pending_realign = False
            print("  live-scan reset position -> re-anchoring twin alignment to current pose")
            self._initialize_twin_alignment(latest.camera_transform)
        else:
            self._pending_realign = True
            print("  live-scan reset position requested; waiting for a normal-tracking pose")

    def _update_bootstrap_gate(self) -> None:
        if not self.bootstrap_gate_enabled:
            return
        pose = self._get_rover_pose(fallback=self.twin_start_pose)
        self._bootstrap_gate.update(self._live_known_cell_keys, (pose.x, pose.y))

    def _bootstrap_status(self) -> dict[str, Any]:
        status = self._bootstrap_gate.status()
        status["enabled"] = self.bootstrap_gate_enabled
        # When the gate is off, navigation is not bootstrap-restricted.
        if not self.bootstrap_gate_enabled:
            status["navigation_enabled"] = True
        return status

    def _occupancy_status(self) -> dict[str, Any]:
        if not self.occupancy_enabled:
            return {"enabled": False}
        floor_height = (
            self._floor_estimator.height_m if self._floor_estimator.has_estimate else self.twin_start_z
        )
        return {
            "enabled": True,
            # Plane-relative classification: cells are floor/occupied by
            # height above the fitted floor, not a fixed world-Z band.
            "classification": "plane_relative",
            "floor_height_m": round(float(floor_height), 4),
            "cell_m": self._POINT_GRID_CELL_M,
            "counts": dict(self._occupancy_counts),
            "obstacle_proxy_count": len(self._obstacle_proxies),
            "obstacle_proxies": [
                {
                    "min_x": p.min_x, "max_x": p.max_x,
                    "min_y": p.min_y, "max_y": p.max_y,
                    "min_z": p.min_z, "max_z": p.max_z,
                }
                for p in self._obstacle_proxies[:50]
            ],
        }

    def _floor_status(self) -> dict[str, Any]:
        status = self._floor_estimator.status()
        status["support_ground_enabled"] = self.support_ground_enabled
        status["support_ground_top_z"] = self._support_ground_height
        support = self._ground_support(self.twin_start_pose.x, self.twin_start_pose.y)
        status["source"] = support.source if support is not None else "provisional_support_ground"
        # SupportGround is a physics scaffold only; it is NOT confirmed-traversable.
        status["traversable"] = False
        return status

    def _phone_pose_status(self) -> dict[str, Any]:
        if self._phone_pose_store is None:
            return {"enabled": False}
        status = self._phone_pose_store.status()
        status["mount_calibrated"] = self._phone_mount is not None
        status["auto_twin_align"] = self.auto_twin_align
        status["twin_aligned"] = self._t_align is not None
        # Derived rover-body pose from the latest camera pose + mount calibration,
        # using the shared T_align. Before alignment is established the pose is
        # correct only up to a rigid transform (rover_pose_aligned=False).
        latest = self._phone_pose_store.latest
        if latest is not None:
            calibration = self._phone_mount or PhoneMountCalibration.identity()
            rover_pose = rover_pose_from_camera(latest.camera_transform, calibration, self._t_align)
            status["rover_pose"] = rover_pose.to_dict()
            status["rover_pose_aligned"] = self._t_align is not None
            status["lever_arm_m"] = list(calibration.lever_arm_m)
            status["rover_forward_axis"] = calibration.rover_forward_axis
        return status

    def _live_scan_status(self) -> dict[str, Any]:
        if not self.live_scan_enabled or self._live_scan_state is None:
            return {"enabled": False}
        status = self._live_scan_state.status(
            receiver=self._live_scan_receiver,
            obstacle_cells=self._live_scan_obstacle_cells,
            known_cells=self._live_scan_known_cells,
        )
        if self._live_scan_queue is not None:
            status["queued_frames"] = self._live_scan_queue.qsize()
        status["host"] = self.live_scan_host
        status["port"] = self.live_scan_port
        status["root_prim_path"] = self.live_scan_root
        status["live_grid_merged_with_static"] = self._live_scan_has_replaced_grid
        status["keyframes"] = len(self._live_scan_state.keyframes)
        return status

    def _initialize_live_scan(self) -> None:
        transform = self._load_live_scan_transform()
        self._live_scan_queue = queue.Queue()
        self._live_scan_state = LiveScanState(arkit_to_twin=transform)
        self._phone_pose_store = PhonePoseStore()
        self._ensure_live_scan_root()
        if self.support_ground_enabled:
            self._ensure_support_ground()
        self._live_scan_receiver = LiveScanReceiver(
            self.live_scan_host,
            self.live_scan_port,
            self._live_scan_queue,
        )
        self._live_scan_receiver.start()
        print(
            f"Live scan receiver listening at tcp://{self.live_scan_host}:{self.live_scan_port} "
            f"(root {self.live_scan_root}, rate {self.live_scan_rate_hz:g} Hz)"
        )

    def _load_live_scan_transform(self) -> tuple[float, ...]:
        if not self.live_scan_transform_json:
            return (
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
            )
        path = Path(self.live_scan_transform_json)
        with path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
        if isinstance(payload, list):
            matrix = payload
        elif isinstance(payload, dict):
            matrix = payload.get("arkit_to_twin")
        else:
            matrix = None
        if not isinstance(matrix, list) or len(matrix) != 16:
            raise RuntimeError(
                f"{path} must contain a 16-number list or an object with an 'arkit_to_twin' list"
            )
        return tuple(float(value) for value in matrix)

    def _ensure_live_scan_root(self) -> None:
        from pxr import UsdGeom

        prim = self.stage.GetPrimAtPath(self.live_scan_root)
        if not prim.IsValid():
            UsdGeom.Xform.Define(self.stage, self.live_scan_root)

    def _ensure_support_ground(self) -> None:
        """Author the SupportGround physics scaffold: a large, thin static
        collision slab so the sim rover never falls through an empty scene.

        It is a Cube (never indexed as a planning obstacle — see
        _collect_scene_obstacles, which only indexes Mesh prims) and carries
        collision but no rigid body, so it is a static support surface that does
        not participate in traversability semantics.
        """
        from pxr import Gf, UsdGeom, UsdPhysics

        path = self.support_ground_path
        prim = self.stage.GetPrimAtPath(path)
        if not prim.IsValid():
            cube = UsdGeom.Cube.Define(self.stage, path)
            cube.CreateSizeAttr().Set(1.0)
            prim = cube.GetPrim()
            cube.CreateDisplayColorAttr().Set([Gf.Vec3f(0.25, 0.25, 0.28)])
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI.Apply(prim)
        xform = UsdGeom.Xformable(prim)
        ops = {op.GetOpName(): op for op in xform.GetOrderedXformOps()}
        translate_op = ops.get("xformOp:translate") or xform.AddTranslateOp()
        scale_op = ops.get("xformOp:scale") or xform.AddScaleOp()
        # Force this every launch/update, not only when the prim is first
        # created. A SupportGround authored in an existing stage may already be
        # visible, and reusing that prim must not leak the physics scaffold into
        # the rendered scene.
        self._make_prim_subtree_invisible(prim)
        # Unit cube spans [-0.5, 0.5]; scale to (size, size, thickness) and drop
        # the slab so its TOP surface sits at the support height.
        scale_z = self.support_ground_thickness_m
        scale_op.Set(
            Gf.Vec3d(self.support_ground_size_m, self.support_ground_size_m, scale_z)
        )
        translate_op.Set(
            Gf.Vec3d(self.twin_start_pose.x, self.twin_start_pose.y, self._support_ground_height - scale_z * 0.5)
        )

    def _disable_authored_support_ground(self) -> None:
        """Hide/disable a slab baked into the source USD when terrain replaces it."""
        from pxr import UsdPhysics

        prim = self.stage.GetPrimAtPath(self.support_ground_path)
        if not prim.IsValid():
            return
        self._make_prim_subtree_invisible(prim)
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr().Set(False)
        print(f"Disabled authored support ground under loaded terrain: {self.support_ground_path}")

    def _scene_box_floor_z(self, x: float, y: float, side_m: float) -> float:
        """Read local support height directly from the configured terrain tiles."""
        if self.load_terrain and self.map_dir is not None:
            try:
                manifest = load_manifest(self.map_dir)
                local_z = []
                radius = max(0.08, side_m * 0.35)
                for name in manifest.get("tiles", {}):
                    vertices = np.load(tile_npz_path(self.map_dir, name))["vertices"]
                    distance = np.hypot(vertices[:, 0] - x, vertices[:, 1] - y)
                    values = vertices[distance <= radius, 2]
                    if len(values):
                        local_z.append(values)
                if local_z:
                    return float(np.percentile(np.concatenate(local_z), 90.0))
            except (OSError, ValueError):
                pass
        return -0.14

    def _author_scene_box(self, box: tuple[float, float, float]) -> None:
        """Author a visible static cube collider before physics/obstacle indexing."""
        from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade

        bx, by, side_m = box
        floor_z = self._scene_box_floor_z(bx, by, side_m)
        cube = UsdGeom.Cube.Define(self.stage, "/World/PaperBox")
        cube.CreateDisplayColorAttr().Set([Gf.Vec3f(0.80, 0.38, 0.12)])
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        material = UsdShade.Material.Define(self.stage, "/World/PaperBoxMaterial")
        shader = UsdShade.Shader.Define(
            self.stage, "/World/PaperBoxMaterial/PreviewSurface"
        )
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput(
            "diffuseColor", Sdf.ValueTypeNames.Color3f
        ).Set(Gf.Vec3f(0.80, 0.20, 0.03))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.65)
        material.CreateSurfaceOutput().ConnectToSource(
            shader.ConnectableAPI(), "surface"
        )
        UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(material)
        xform = UsdGeom.Xformable(cube.GetPrim())
        xform.ClearXformOpOrder()
        xform.AddTranslateOp().Set(
            Gf.Vec3d(bx, by, floor_z + side_m / 2.0)
        )
        xform.AddScaleOp().Set(
            Gf.Vec3d(side_m / 2.0, side_m / 2.0, side_m / 2.0)
        )
        print(
            f"Authored scene box: center=({bx:.3f}, {by:.3f}), "
            f"side={side_m:.3f} m, floor_z={floor_z:.3f}"
        )

    def set_advisor_test_obstacles(self, obstacles: list[dict[str, Any]]) -> dict[str, Any]:
        """Replace the controlled advisor-battery obstacles in the live stage.

        Obstacles are vertical cylinders matching the circle geometry supplied
        to the route advisor.  They are authored on Isaac's main thread, given
        static collision, and immediately re-indexed by the gate.
        """
        from pxr import Gf, UsdGeom, UsdPhysics

        root_path = "/World/AdvisorTestObstacles"
        if self.stage.GetPrimAtPath(root_path).IsValid():
            self.stage.RemovePrim(root_path)
        UsdGeom.Xform.Define(self.stage, root_path)
        authored = []
        for index, obstacle in enumerate(obstacles):
            x = float(obstacle["x"])
            y = float(obstacle["y"])
            radius = float(obstacle["radius_m"])
            name = str(obstacle.get("name", f"obstacle_{index}"))
            height = max(0.35, radius * 2.0)
            floor_z = self._scene_box_floor_z(x, y, radius * 2.0)
            path = f"{root_path}/Obstacle_{index:02d}"
            cylinder = UsdGeom.Cylinder.Define(self.stage, path)
            cylinder.CreateAxisAttr().Set(UsdGeom.Tokens.z)
            cylinder.CreateRadiusAttr().Set(radius)
            cylinder.CreateHeightAttr().Set(height)
            cylinder.CreateDisplayColorAttr().Set([Gf.Vec3f(0.80, 0.30, 0.08)])
            UsdPhysics.CollisionAPI.Apply(cylinder.GetPrim())
            xform = UsdGeom.Xformable(cylinder.GetPrim())
            xform.ClearXformOpOrder()
            xform.AddTranslateOp().Set(Gf.Vec3d(x, y, floor_z + height / 2.0))
            authored.append({
                "name": name,
                "path": path,
                "x": x,
                "y": y,
                "radius_m": radius,
                "height_m": height,
                "floor_z": floor_z,
            })
        # Let USD/physics consume the stage edits before the next validation,
        # then rebuild the inexpensive bbox index used for clearance checks.
        for _ in range(2):
            self._step()
        self._obstacles = self._collect_scene_obstacles()
        return {
            "status": "ok",
            "authored": authored,
            "indexed_obstacle_count": len(self._obstacles),
        }

    def _make_prim_subtree_invisible(self, root_prim) -> None:
        """Hide a renderable helper prim and any authored children.

        Collision debug overlays can still draw colliders if enabled in Isaac's
        viewport, but normal render/Hydra visibility should not show this
        physics scaffold.
        """
        from pxr import Usd, UsdGeom

        for prim in Usd.PrimRange(root_prim):
            if prim.IsA(UsdGeom.Imageable):
                imageable = UsdGeom.Imageable(prim)
                imageable.MakeInvisible()
                imageable.CreateVisibilityAttr().Set(UsdGeom.Tokens.invisible)

    # SupportGround sits this far below the first scanned floor so the terrain
    # colliders are unambiguously the rover's support and the slab is a pure
    # safety net (catches holes) without z-fighting the scan surface.
    _SUPPORT_GROUND_FLOOR_MARGIN_M = 0.02

    def _lower_support_ground_to_terrain(self) -> None:
        """One-shot: drop the SupportGround backstop just below the scanned floor.

        SupportGround starts at ``twin_start_z`` (the rover's body-origin height),
        a safe floor before any terrain exists. But the scanned floor lands a
        body-height *below* that, so leaving SupportGround at twin_start_z makes
        it intercept the rover ~a body-height above the real terrain -- the rover
        floats on the scaffold instead of settling on the scan. The live-scan
        FloorEstimator handles this on the ARKit-mesh path; the nvblox-tile path
        has no estimator, so drive SupportGround from the terrain heightmap here.

        Done exactly ONCE, the first time tiles reveal a solid floor under the
        rover. Re-authoring the collider on every rebuild re-cooks PhysX and
        repeatedly pulls the floor out from under a physics-active rover (a
        fall/settle loop) -- both tank FPS during the scan. After this fires the
        terrain colliders are the real support; the slab just backstops holes."""
        if (not self.support_ground_enabled or self._terrain_heightmap is None
                or self._support_ground_lowered):
            return
        pose = self._latest_live_pose_payload or (
            read_live_rover_pose(self.map_dir) if self.map_dir is not None else None)
        x, y = (
            (float(pose["x"]), float(pose["y"])) if pose
            else (self.twin_start_pose.x, self.twin_start_pose.y)
        )
        support = self._ground_support(x, y)
        if support is None or support.height_m is None:
            return  # no solid floor yet -- wait (do NOT consume the one shot)
        height = float(support.height_m) - self._SUPPORT_GROUND_FLOOR_MARGIN_M
        if height >= self._support_ground_height - 1e-4:
            self._support_ground_lowered = True  # already at/below the floor
            return
        self._support_ground_height = height
        self._floor_last_applied_height = height
        self._support_ground_lowered = True
        self._ensure_support_ground()
        print(f"[support-ground] lowered to scanned floor z={height:.3f} m (one-shot)", flush=True)

    def _update_floor_estimate(self) -> None:
        """Feed nearby live-scan points to the floor estimator and nudge the
        SupportGround height toward the estimate (slowly, via the estimator's
        EMA). Planning is untouched: only the physics scaffold moves."""
        if self._live_scan_state is None:
            return
        pose = self._get_rover_pose(fallback=self.twin_start_pose)
        points = self._live_scan_state.points_xyz()
        height = self._floor_estimator.update(points, (pose.x, pose.y))
        if height is None or not self.support_ground_enabled:
            return
        # Re-author only on a meaningful change to avoid churning USD every tick.
        if self._floor_last_applied_height is None or abs(height - self._floor_last_applied_height) > 0.002:
            self._support_ground_height = height
            self._floor_last_applied_height = height
            self._ensure_support_ground()

    def process_live_scan_updates(self, max_updates: int = 8) -> int:
        if not self.live_scan_enabled or self._live_scan_queue is None or self._live_scan_state is None:
            return 0
        now = time.monotonic()
        # Throttle the expensive USD-mutation + grid-rebuild work to
        # live_scan_rate_hz. Frames stay queued between ticks (max_updates caps
        # per-tick work), so a fast _step() loop does not stall rewriting USD
        # every frame. rate_hz <= 0 disables throttling.
        if self.live_scan_rate_hz > 0.0:
            if now - self._live_scan_last_usd_update_s < 1.0 / self.live_scan_rate_hz:
                return 0
        processed = 0
        expired = self._live_scan_state.expire_stale(now, self.live_scan_grid_decay_s)
        for anchor_id in expired:
            self._remove_live_mesh_prim(anchor_id)
        while processed < max_updates:
            try:
                frame = self._live_scan_queue.get_nowait()
            except queue.Empty:
                break
            if frame.is_phone_pose:
                # Continuous rover-localization pose. Store latest; on the first
                # normal-tracking pose, initialize the shared twin alignment.
                if self._phone_pose_store is not None:
                    self._phone_pose_store.update(frame, received_monotonic_s=now)
                    if frame.tracking_state == TRACKING_NORMAL and (
                        self._pending_realign
                        or (self.auto_twin_align and self._t_align is None)
                    ):
                        # A pending Reset Position request re-anchors regardless
                        # of auto_twin_align (it's an explicit operator action).
                        self._pending_realign = False
                        self._initialize_twin_alignment(frame.anchor_transform)
                processed += 1
                continue
            if frame.message_type == MSG_REALIGN:
                self._handle_realign_request()
                processed += 1
                continue
            if frame.is_keyframe_image:
                # Store the posed keyframe (pose + intrinsics + jpeg) for the bake.
                self._live_scan_state.apply_image(frame)
                processed += 1
                continue
            if frame.message_type == MSG_FINALIZE:
                processed += 1
                continue
            anchor = self._live_scan_state.apply_frame(frame, now_s=now)
            if frame.removed:
                self._remove_live_mesh_prim(frame.anchor_id)
            elif anchor is not None:
                self._upsert_live_mesh_prim(anchor)
                if self.hide_static_scan_after_live_scan and not self._live_scan_static_hidden:
                    self._set_static_scan_visibility(visible=False)
                    self._live_scan_static_hidden = True
            processed += 1

        if processed or expired:
            self._live_scan_last_usd_update_s = now
            self._refresh_live_obstacle_grid()
            self._update_floor_estimate()
            self._update_bootstrap_gate()
            self._update_live_scan_top_light()
            self._gc_live_textures()
        return processed

    def _live_scan_prim_path(self, anchor_id: str) -> str:
        safe = "".join(ch if ch.isalnum() else "_" for ch in anchor_id)
        return f"{self.live_scan_root}/anchor_{safe}"

    def _upsert_live_mesh_prim(self, anchor) -> None:
        from pxr import Gf, Sdf, UsdGeom, Vt

        path = self._live_scan_prim_path(anchor.anchor_id)
        mesh = UsdGeom.Mesh.Define(self.stage, path)
        # CRITICAL for perf: a UsdGeom.Mesh defaults to Catmull-Clark, so Hydra
        # would subdivide this dense scanned mesh on the GPU (and re-subdivide on
        # every update) -- the dominant cause of viewport lag that scales with
        # scan size. Scanned triangle soup must render as a polygon mesh.
        mesh.CreateSubdivisionSchemeAttr().Set("none")

        # Hand USD contiguous numpy buffers via Vt.*.FromNumpy rather than
        # building a Gf.Vec object per vertex/index in Python. At building scale
        # the live preview re-authors many anchors per tick on the Isaac main
        # thread; the comprehension path stalls it. Fall back element-wise if a
        # USD build lacks FromNumpy (Set overwrites, so the retry is safe).
        points = np.asarray(anchor.vertices_twin, dtype=np.float32).reshape(-1, 3)
        faces = np.asarray(anchor.faces, dtype=np.int32).reshape(-1, 3)
        cols = anchor.colors or tuple((255, 255, 255) for _ in anchor.vertices_twin)
        colors = np.asarray(cols, dtype=np.float32).reshape(-1, 3) / 255.0
        # Per-vertex normals turn the faceted (flat-shaded) live mesh into a
        # smooth-shaded surface. Authored only when the sender supplied them
        # (length matches the points); otherwise Isaac falls back to computing
        # face normals, i.e. the previous flat look.
        normals = anchor.normals_twin
        have_normals = bool(normals) and len(normals) == len(anchor.vertices_twin)
        normals_np = (
            np.asarray(normals, dtype=np.float32).reshape(-1, 3) if have_normals else None
        )

        color_primvar = UsdGeom.PrimvarsAPI(mesh.GetPrim()).CreatePrimvar(
            "displayColor",
            Sdf.ValueTypeNames.Color3fArray,
            UsdGeom.Tokens.vertex,
        )
        try:
            mesh.CreatePointsAttr().Set(Vt.Vec3fArray.FromNumpy(points))
            mesh.CreateFaceVertexCountsAttr().Set(
                Vt.IntArray.FromNumpy(np.full(len(faces), 3, dtype=np.int32))
            )
            mesh.CreateFaceVertexIndicesAttr().Set(
                Vt.IntArray.FromNumpy(np.ascontiguousarray(faces.reshape(-1)))
            )
            color_primvar.Set(Vt.Vec3fArray.FromNumpy(colors))
            if normals_np is not None:
                mesh.CreateNormalsAttr().Set(Vt.Vec3fArray.FromNumpy(normals_np))
                mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
        except (AttributeError, TypeError) as exc:  # pragma: no cover - USD-version guard
            print(f"  (Vt.FromNumpy fast path unavailable, using slow author: {exc})")
            mesh.CreatePointsAttr().Set([Gf.Vec3f(*p) for p in points.tolist()])
            mesh.CreateFaceVertexCountsAttr().Set([3] * len(faces))
            mesh.CreateFaceVertexIndicesAttr().Set([int(i) for i in faces.reshape(-1).tolist()])
            color_primvar.Set([Gf.Vec3f(*c) for c in colors.tolist()])
            if normals_np is not None:
                mesh.CreateNormalsAttr().Set([Gf.Vec3f(*n) for n in normals_np.tolist()])
                mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)

        # Projected-keyframe texture: when the anchor is textured AND its keyframe
        # JPEG has arrived, author per-vertex UVs (primvars:st) and bind a shared
        # per-image material. Otherwise the displayColor above is the visible
        # color (no holes/regression for not-yet-textured anchors).
        material = None
        if (
            anchor.texture_image_id is not None
            and len(anchor.uvs) == len(anchor.vertices_twin)
            and self._live_scan_state is not None
            and anchor.texture_image_id in self._live_scan_state.images
        ):
            # Never let a texture/material authoring error crash live-scan
            # processing (that would also drop the socket): fall back to the
            # vertex-color path already authored above.
            try:
                material = self._ensure_texture_material(anchor.texture_image_id)
            except Exception as exc:  # noqa: BLE001 - degrade, never crash
                if not self._live_scan_texture_warned:
                    print(f"  (live-scan texture authoring failed, using vertex color: {exc})")
                    self._live_scan_texture_warned = True
                material = None
        if material is not None:
            st = UsdGeom.PrimvarsAPI(mesh.GetPrim()).CreatePrimvar(
                "st",
                Sdf.ValueTypeNames.TexCoord2fArray,
                UsdGeom.Tokens.vertex,
            )
            st.Set([Gf.Vec2f(float(u), float(v)) for u, v in anchor.uvs])
            from pxr import UsdShade

            UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
        else:
            # No (longer a) texture: ensure any prior binding is cleared so the
            # mesh falls back to displayColor.
            from pxr import UsdShade

            UsdShade.MaterialBindingAPI(mesh.GetPrim()).UnbindAllBindings()

        if len(points):
            lo = points.min(axis=0)
            hi = points.max(axis=0)
            mesh.CreateExtentAttr().Set(
                [
                    Gf.Vec3f(float(lo[0]), float(lo[1]), float(lo[2])),
                    Gf.Vec3f(float(hi[0]), float(hi[1]), float(hi[2])),
                ]
            )

        # Once every tile this anchor touches has been baked, hide it so the
        # textured tiles show through instead of the (blocky) live vertex color.
        self._apply_anchor_visibility(anchor)

    def _remove_live_mesh_prim(self, anchor_id: str) -> None:
        path = self._live_scan_prim_path(anchor_id)
        if self.stage.GetPrimAtPath(path).IsValid():
            self.stage.RemovePrim(path)

    def _live_scan_looks_path(self) -> str:
        return f"{self.live_scan_root}/Looks"

    def _texture_material_path(self, image_id: str) -> str:
        safe = "".join(ch if ch.isalnum() else "_" for ch in image_id)
        return f"{self._live_scan_looks_path()}/tex_{safe}"

    def _ensure_texture_material(self, image_id: str):
        """Write the keyframe JPEG once and author/reuse a UsdPreviewSurface
        material that samples it via primvars:st. Shared by every anchor that
        references this image. Returns the UsdShade.Material (or None on error)."""
        from pxr import Sdf, UsdGeom, UsdShade

        mat_path = self._texture_material_path(image_id)
        if image_id in self._live_scan_texture_image_ids:
            existing = self.stage.GetPrimAtPath(mat_path)
            if existing.IsValid():
                return UsdShade.Material(existing)

        # Write the JPEG to the texture temp dir (one file per image id).
        if self._live_scan_texture_dir is None:
            self._live_scan_texture_dir = Path(tempfile.mkdtemp(prefix="live_scan_tex_"))
        jpeg_path = self._live_scan_texture_dir / f"{image_id}.jpg"
        try:
            if not jpeg_path.exists():
                jpeg_path.write_bytes(self._live_scan_state.images[image_id])
        except OSError as exc:
            print(f"  (could not write live-scan texture {jpeg_path}: {exc})")
            return None

        UsdGeom.Scope.Define(self.stage, self._live_scan_looks_path())

        material = UsdShade.Material.Define(self.stage, mat_path)

        shader = UsdShade.Shader.Define(self.stage, f"{mat_path}/Surface")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(1.0)
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
        diffuse_input = shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f)
        surface_output = shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)

        st_reader = UsdShade.Shader.Define(self.stage, f"{mat_path}/stReader")
        st_reader.CreateIdAttr("UsdPrimvarReader_float2")
        st_reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        st_output = st_reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)

        tex = UsdShade.Shader.Define(self.stage, f"{mat_path}/diffuseTexture")
        tex.CreateIdAttr("UsdUVTexture")
        tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(str(jpeg_path))
        tex.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("clamp")
        tex.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("clamp")
        tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(st_output)
        tex_output = tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)

        # Modern single-argument ConnectToSource(UsdShadeOutput); the two-arg
        # (ConnectableAPI, name) form is removed in recent USD (Isaac Sim 6.0).
        diffuse_input.ConnectToSource(tex_output)
        material.CreateSurfaceOutput().ConnectToSource(surface_output)

        self._live_scan_texture_image_ids.add(image_id)
        return material

    def _gc_live_textures(self) -> None:
        """Drop texture materials + JPEGs whose image id is no longer referenced
        by any live anchor, so a long scan doesn't accumulate keyframes without
        bound."""
        if self._live_scan_state is None:
            return
        referenced = self._live_scan_state.referenced_image_ids()
        stale = self._live_scan_texture_image_ids - referenced
        for image_id in stale:
            mat_path = self._texture_material_path(image_id)
            if self.stage.GetPrimAtPath(mat_path).IsValid():
                self.stage.RemovePrim(mat_path)
            self._live_scan_state.images.pop(image_id, None)
            if self._live_scan_texture_dir is not None:
                try:
                    (self._live_scan_texture_dir / f"{image_id}.jpg").unlink(missing_ok=True)
                except OSError:
                    pass
            self._live_scan_texture_image_ids.discard(image_id)

    # ------------------------------------------------------------------
    # Progressive unwrap-cache warming (during the scan)
    # ------------------------------------------------------------------
    def send_live_scan_control(self, start: bool) -> tuple[int, dict]:
        """Push a start/stop-scan control message to the connected phone over the
        live-scan socket. STOP makes the phone stop streaming and send finalize,
        which flushes the remaining dirty tiles. Returns (http_status, payload)."""
        from digital_twin.live_scan import (
            MSG_CONTROL_START_SCAN,
            MSG_CONTROL_STOP_SCAN,
        )
        if not self.live_scan_enabled or self._live_scan_receiver is None:
            return 409, {"error": "live_scan_disabled"}
        msg = MSG_CONTROL_START_SCAN if start else MSG_CONTROL_STOP_SCAN
        ok = self._live_scan_receiver.send_control(msg)
        if not ok:
            return 409, {"error": "no_phone_connected",
                         "detail": self._live_scan_receiver.last_error}
        action = "start" if start else "stop"
        print(f"  live-scan control -> phone: {action} scan")
        return 200, {"status": "ok", "action": action}

    def _update_live_scan_top_light(self) -> None:
        if self._live_scan_state is None:
            return
        points = [
            point
            for anchor in self._live_scan_state.anchors.values()
            for point in anchor.vertices_twin
        ]
        self._author_scan_top_light(f"{self.live_scan_root}/scan_top_light", points)

    def _author_scan_top_light(self, light_path: str, points) -> None:
        """Create/update a rectangular area light centered over scan bounds.

        The authored transform intentionally has no rotation. With the default
        USD rect-light orientation, local -Z points downward, matching the
        manually-tested 0,0,0 orientation.
        """
        if not self.live_scan_light or len(points) == 0:
            return
        try:
            from pxr import Gf, UsdGeom, UsdLux

            pts = np.asarray(points, dtype=float).reshape(-1, 3)
            lo = pts.min(axis=0)
            hi = pts.max(axis=0)
            xs = (lo[0], hi[0])
            ys = (lo[1], hi[1])
            zs = (lo[2], hi[2])
            cx = 0.5 * (float(min(xs)) + float(max(xs)))
            cy = 0.5 * (float(min(ys)) + float(max(ys)))
            top_z = float(max(zs)) + float(self.live_scan_light_height_padding_m)
            sx, sy, sz = (float(v) for v in self.live_scan_light_scale)

            light = UsdLux.RectLight.Define(self.stage, light_path)
            # Keep intrinsic size at 1x1 and express the tested "10x10x1"
            # setup as transform scale, matching the Isaac UI convention.
            light.CreateWidthAttr().Set(1.0)
            light.CreateHeightAttr().Set(1.0)
            light.CreateIntensityAttr().Set(float(self.live_scan_light_intensity))

            xform = UsdGeom.Xformable(light.GetPrim())
            xform.ClearXformOpOrder()
            xform.AddTranslateOp().Set(Gf.Vec3d(cx, cy, top_z))
            xform.AddScaleOp().Set(Gf.Vec3f(sx, sy, sz))
        except Exception as exc:  # noqa: BLE001 - lighting must not break scan ingest
            print(f"  (live-scan top light authoring failed: {exc})")

    def _build_occupancy_grid(self, now: float) -> tuple[dict, dict]:
        """Plane-relative occupancy: classify cells by height above the fitted
        floor. Returns (obstacle_grid, known_grid) shaped like the rasterized
        grids the merge step expects (cell -> list of (x, y) points)."""
        floor_height = (
            self._floor_estimator.height_m if self._floor_estimator.has_estimate else self.twin_start_z
        )
        grid = OccupancyGrid(
            floor=FloorPlane.level(floor_height),
            config=OccupancyConfig(cell_m=self._POINT_GRID_CELL_M),
        )
        # Stale anchors are already expired at the top of process_live_scan_updates,
        # so no extra decay pass here (matches the fixed-band path).
        grid.ingest_points(self._live_scan_state.points_xyz(now_s=now, decay_s=0.0))
        cell = self._POINT_GRID_CELL_M
        obstacle_grid = {
            key: [((key[0] + 0.5) * cell, (key[1] + 0.5) * cell)] for key in grid.occupied_cells()
        }
        known_grid = {key: [] for key in grid.known_cells()}
        self._occupancy_counts = grid.counts()
        self._obstacle_proxies = grid.obstacle_proxies()
        return obstacle_grid, known_grid

    def _refresh_live_obstacle_grid(self) -> None:
        if self._live_scan_state is None:
            return
        now = time.monotonic()
        if self.occupancy_enabled:
            obstacle_grid, known_grid = self._build_occupancy_grid(now)
        else:
            obstacle_points = self._live_scan_state.obstacle_points_xy(
                self._OBSTACLE_BAND_Z,
                now_s=now,
                decay_s=0.0,
            )
            known_points = self._live_scan_state.known_points_xy(now_s=now, decay_s=0.0)
            obstacle_grid = rasterize_xy_points(obstacle_points, self._POINT_GRID_CELL_M)
            known_grid = rasterize_xy_points(known_points, self._POINT_GRID_CELL_M)

        # Capture the static grid once, then union static + live so unswept
        # regions stay conservatively blocked. NOTE: this means a static
        # obstacle that has physically been removed will not clear from the
        # grid (only live-added geometry decays). This static-grid union does
        # not clear obstacles using free-space rays.
        if self._static_obstacle_grid is None:
            if self.hide_static_scene_at_start:
                self._static_obstacle_grid = {}
            else:
                self._static_obstacle_grid = {
                    key: np.asarray(points, dtype=np.float64)
                    for key, points in self._obstacle_grid.items()
                }

        merged: dict[tuple[int, int], Any] = {
            key: np.asarray(points, dtype=np.float64)
            for key, points in self._static_obstacle_grid.items()
        }
        live_obstacle_cells = 0
        for key, points in obstacle_grid.items():
            live_cell = np.asarray(points, dtype=np.float64)
            live_obstacle_cells += 1
            if key in merged:
                merged[key] = np.vstack((merged[key], live_cell))
            else:
                merged[key] = live_cell

        self._obstacle_grid = merged
        self._live_scan_obstacle_cells = live_obstacle_cells
        self._live_scan_known_cells = len(known_grid)
        self._live_known_cell_keys = list(known_grid.keys())
        self._live_scan_has_replaced_grid = True

    def _set_static_scene_enabled(self, *, visible: bool, collision_enabled: bool | None = None) -> None:
        from pxr import UsdGeom, UsdPhysics

        rover_path = str(self.rover_prim.GetPath())
        live_root = self.live_scan_root
        mesh_prims = 0
        collision_prims = 0
        for prim in self.stage.Traverse():
            path = str(prim.GetPath())
            if path == rover_path or path.startswith(rover_path + "/"):
                continue
            if path == live_root or path.startswith(live_root + "/"):
                continue
            # Preserve PaperBox only when this invocation explicitly authored it
            # with --box.  Some saved stages contain a stale /World/PaperBox;
            # treating that as special would silently contaminate controlled
            # trials that otherwise disable the static scene.
            if path == "/World/PaperBox" and self.scene_box is not None:
                continue
            if path == self.support_ground_path and self.support_ground_enabled:
                continue
            if path in (
                "/World/goal_marker",
                self._front_camera_path,
                self._chase_camera_path,
            ):
                continue
            if prim.IsA(UsdGeom.Gprim):
                UsdGeom.Imageable(prim).MakeInvisible() if not visible else UsdGeom.Imageable(prim).MakeVisible()
                mesh_prims += 1
            if collision_enabled is not None and prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr().Set(bool(collision_enabled))
                collision_prims += 1
        print(
            f"Static scene update: visible={visible} mesh_prims={mesh_prims}, "
            f"collision_enabled={collision_enabled} collision_prims={collision_prims}"
        )

    def _set_static_scan_visibility(self, visible: bool) -> None:
        self._set_static_scene_enabled(visible=visible, collision_enabled=None)

    def _set_goal_marker(self, goal: Pose) -> None:
        """Author/update a bright green pole at the goal pose.

        Purely visual navigation aid for the vision planner: it carries no
        collision or physics APIs, so the rover drives through it and the
        safety checks never see it. It is created after startup, so it is
        also absent from the indexed scene obstacles.
        """
        from pxr import Gf, UsdGeom

        path = "/World/goal_marker"
        prim = self.stage.GetPrimAtPath(path)
        if not prim.IsValid():
            cyl = UsdGeom.Cylinder.Define(self.stage, path)
            cyl.CreateRadiusAttr().Set(0.06)
            cyl.CreateHeightAttr().Set(0.8)
            cyl.CreateExtentAttr().Set(
                [Gf.Vec3f(-0.06, -0.06, -0.4), Gf.Vec3f(0.06, 0.06, 0.4)]
            )
            cyl.CreateDisplayColorAttr().Set([Gf.Vec3f(0.0, 1.0, 0.1)])
            prim = cyl.GetPrim()
            UsdGeom.Xformable(prim).AddTranslateOp()
        # Pole center at z=0.5: bottom near the scan floor (~0.07), top at
        # 0.9 m so it stands out above low clutter in both camera views.
        prim.GetAttribute("xformOp:translate").Set(Gf.Vec3d(goal.x, goal.y, 0.5))

    def _camera_debug(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        if self._cameras_parented:
            # get_world_pose composes USD transforms, which Newton does not
            # write back to; for parented cameras these are the SPAWN poses,
            # not live ones. The rendered frames do follow the body link.
            out["note"] = "cameras parented to body link; USD world poses below are stale under Newton"
        for name, cam in (("front", self._camera), ("chase", self._chase_camera)):
            if cam is None:
                continue
            try:
                pos, quat = cam.get_world_pose(camera_axes="world")

                def as_list(value):
                    cpu = getattr(value, "cpu", None)
                    if callable(cpu):
                        value = cpu()
                    return [round(float(v), 4) for v in np.asarray(value).reshape(-1)]

                out[name] = {"position": as_list(pos), "quat_wxyz": as_list(quat)}
            except Exception as exc:
                out[name] = {"error": str(exc)}
        return out

    def _initialize_camera(self) -> None:
        self._chase_camera = None
        self._cameras_parented = False
        try:
            from isaacsim.sensors.camera import Camera
        except Exception as exc:
            print(f"Camera unavailable: {exc}")
            self._camera = None
            return
        try:
            # Mount the cameras under the rover body LINK so they ride along
            # with the robot: Newton updates link transforms (only the
            # articulation root Xform stays frozen), so children of a link
            # follow under physics. Fall back to free /World cameras synced
            # per capture (_sync_camera_to_body) if no body link was found.
            if self._pose_prim is not None and self._pose_prim != self._articulation_prim:
                base = str(self._pose_prim.GetPath())
                self._cameras_parented = True
            else:
                base = "/World"
            self._front_camera_path = f"{base}/front_camera"
            self._chase_camera_path = f"{base}/chase_camera"
            self._camera = Camera(
                prim_path=self._front_camera_path,
                resolution=(self.camera_width, self.camera_height),
            )
            self._camera.initialize()
            self._chase_camera = Camera(
                prim_path=self._chase_camera_path,
                resolution=(self.camera_width, self.camera_height),
            )
            self._chase_camera.initialize()
            if self._cameras_parented:
                # Author the body-frame rig pose (idempotent with the baked
                # prims; positions cameras created fresh on older USDs).
                self._write_camera_usd_pose(
                    self._front_camera_path, self._CAMERA_OFFSET_BODY, (0.0, -1.0, 0.0)
                )
                pitch = math.radians(self._CHASE_PITCH_DOWN_DEG)
                self._write_camera_usd_pose(
                    self._chase_camera_path,
                    self._CHASE_OFFSET_BODY,
                    (0.0, -math.cos(pitch), -math.sin(pitch)),
                )
            for _ in range(5):
                self._step()
            print(
                f"Cameras mounted: {self._front_camera_path}, {self._chase_camera_path}"
                f" (parented={self._cameras_parented})"
            )
        except Exception as exc:
            print(f"Camera init failed: {exc}")
            self._camera = None
            self._chase_camera = None

    # Camera rig in the body frame: 0.1 m ahead along the drive axis (-y),
    # 0.22 m up, looking forward and level (measured mount position on the rover).
    _CAMERA_OFFSET_BODY = (0.0, -0.10, 0.22)
    # Camera 'world' axes are +X forward, +Z up (isaacsim camera docs), so the
    # camera world yaw must equal the DRIVE heading = body_yaw - 90 (the drive
    # axis is body -y). If this looks "sideways" relative to the rover MODEL,
    # that is the model pointing sideways to its travel direction (wheel axle
    # axis authoring), not a camera error.
    _CAMERA_YAW_OFFSET_FROM_BODY_DEG = -90.0
    # Video-game style chase camera: behind and above the rover, pitched down.
    _CHASE_OFFSET_BODY = (0.0, 1.3, 1.25)  # +y body = behind (drive axis is -y)
    _CHASE_PITCH_DOWN_DEG = 38.0

    def _sync_camera_to_body(self) -> None:
        """Place the cameras from the measured body pose every capture.

        Only used for the legacy free-floating /World cameras. When the
        cameras are parented under the rover body link they follow the robot
        through the scene graph and this sync must not touch them (the xform
        ops it writes would be reinterpreted in the body's local frame).
        """
        if self._cameras_parented:
            return
        if self._articulation is None or self._newton_actuators is None:
            return
        try:
            positions, orientations = self._articulation.get_world_poses()
            p = np.asarray(positions.numpy()).reshape(-1)
            q = np.asarray(orientations.numpy()).reshape(-1)  # w, x, y, z
            w, x, y, z = (float(v) for v in q)
            body_yaw_rad = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
            cam_yaw = body_yaw_rad + math.radians(self._CAMERA_YAW_OFFSET_FROM_BODY_DEG)

            def world_offset(offset_body):
                c, s = math.cos(body_yaw_rad), math.sin(body_yaw_rad)
                ox, oy, oz = offset_body
                return np.array(
                    [p[0] + c * ox - s * oy, p[1] + s * ox + c * oy, p[2] + oz], dtype=float
                )

            # Write poses through raw USD xform ops. Camera.set_world_pose
            # silently stops working after world.reset() (stale internal
            # handle), while authored USD camera transforms always render.
            if self._camera is not None:
                view = (math.cos(cam_yaw), math.sin(cam_yaw), 0.0)
                self._write_camera_usd_pose(
                    self._front_camera_path, world_offset(self._CAMERA_OFFSET_BODY), view
                )
            if self._chase_camera is not None:
                pitch = math.radians(self._CHASE_PITCH_DOWN_DEG)
                view = (
                    math.cos(cam_yaw) * math.cos(pitch),
                    math.sin(cam_yaw) * math.cos(pitch),
                    -math.sin(pitch),
                )
                self._write_camera_usd_pose(
                    self._chase_camera_path, world_offset(self._CHASE_OFFSET_BODY), view
                )
        except Exception as exc:
            print(f"[camera] sync failed: {exc}")

    def _write_camera_usd_pose(self, prim_path: str, eye, view_dir) -> None:
        """Author a USD camera transform looking along ``view_dir`` with +z-up.

        USD cameras view down their local -Z with +Y up; build the rotation
        with columns (right, up, -view) and write translate+orient ops.
        """
        from pxr import Gf, UsdGeom

        prim = self.stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            return
        f = np.array(view_dir, dtype=float)
        f /= max(1e-9, np.linalg.norm(f))
        world_up = np.array([0.0, 0.0, 1.0])
        right = np.cross(f, world_up)
        norm = np.linalg.norm(right)
        if norm < 1e-6:
            right = np.array([1.0, 0.0, 0.0])
        else:
            right /= norm
        up = np.cross(right, f)
        m = np.column_stack([right, up, -f])  # camera X, Y, Z axes in world
        # rotation matrix -> quaternion (w, x, y, z)
        tr = m[0, 0] + m[1, 1] + m[2, 2]
        if tr > 0:
            s = math.sqrt(tr + 1.0) * 2.0
            qw, qx, qy, qz = 0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s
        elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
            qw, qx, qy, qz = (m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s
        elif m[1, 1] > m[2, 2]:
            s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
            qw, qx, qy, qz = (m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s
        else:
            s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
            qw, qx, qy, qz = (m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s

        xform = UsdGeom.Xformable(prim)
        ops = {op.GetOpName(): op for op in xform.GetOrderedXformOps()}
        translate_op = ops.get("xformOp:translate") or xform.AddTranslateOp()
        orient_op = ops.get("xformOp:orient") or xform.AddOrientOp(UsdGeom.XformOp.PrecisionDouble)
        translate_op.Set(Gf.Vec3d(float(eye[0]), float(eye[1]), float(eye[2])))
        orient_op.Set(Gf.Quatd(float(qw), Gf.Vec3d(float(qx), float(qy), float(qz))))

    def _capture_vision(self) -> dict[str, Any]:
        """Capture both cameras; returns {'vision': ..., 'vision_chase': ...}."""
        if self._camera is None:
            return {"vision": {"status": "not_initialized"}}
        try:
            self._sync_camera_to_body()
            # The render product lags the authored camera transform by a frame
            # or two; step enough that get_rgba reflects the synced pose.
            for _ in range(4):
                self._step()
            result = {"vision": self._encode_camera(self._camera)}
            if self._chase_camera is not None:
                result["vision_chase"] = self._encode_camera(self._chase_camera)
            return result
        except Exception as exc:
            return {"vision": {"status": "error", "reason": str(exc)}}

    @staticmethod
    def _encode_camera(camera) -> dict[str, Any]:
        rgba = camera.get_rgba()
        if rgba is None or rgba.size == 0:
            return {"status": "no_frame"}
        from PIL import Image

        img = Image.fromarray(np.asarray(rgba)[..., :3].astype(np.uint8))
        buf = BytesIO()
        img.save(buf, format="PNG")
        return {
            "status": "ok",
            "format": "png_base64",
            "resolution": [int(rgba.shape[1]), int(rgba.shape[0])],
            "rgb_base64": base64.b64encode(buf.getvalue()).decode("ascii"),
        }

    def capture_top_down(
        self,
        out_path: str,
        box=None,
        height: float = 6.0,
        span: float = 6.0,
        center: tuple[float, float] | None = None,
        rover_pose: Pose | None = None,
    ) -> None:
        """One-shot: render a top-down PNG using the chase camera's optics
        (optionally with a placed box) and return. No HTTP loop."""
        from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade
        from isaacsim.sensors.camera import Camera

        # Ingest terrain before choosing the camera and obstacle elevations.
        # The live map's floor is generally not at the provisional support-ground
        # height, and placing a box before this step can leave it visibly floating.
        for _ in range(6):
            try:
                self.load_terrain_tiles()
            except Exception:
                pass
            self._step()
        if rover_pose is not None:
            self._reset_stage(rover_pose)

        floor_z = getattr(self, "_support_ground_height", None)
        if floor_z is None:
            floor_z = -0.14
        if self._terrain_heightmap is not None:
            support = self._terrain_heightmap.ground_support(
                self.twin_start_pose.x, self.twin_start_pose.y, 0.2
            )
            if support.height_m is not None:
                floor_z = support.height_m

        if box is not None:
            bx, by, bs = box
            box_floor_z = floor_z
            if self._terrain_heightmap is not None:
                support = self._terrain_heightmap.ground_support(
                    bx, by, max(0.05, bs * 0.35)
                )
                if support.height_m is not None:
                    box_floor_z = support.height_m
            cube = UsdGeom.Cube.Define(self.stage, "/World/PaperBox")
            cube.CreateDisplayColorAttr().Set([Gf.Vec3f(0.80, 0.38, 0.12)])
            UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
            material = UsdShade.Material.Define(self.stage, "/World/PaperBoxMaterial")
            shader = UsdShade.Shader.Define(
                self.stage, "/World/PaperBoxMaterial/PreviewSurface"
            )
            shader.CreateIdAttr("UsdPreviewSurface")
            shader.CreateInput(
                "diffuseColor", Sdf.ValueTypeNames.Color3f
            ).Set(Gf.Vec3f(0.80, 0.20, 0.03))
            shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.65)
            material.CreateSurfaceOutput().ConnectToSource(
                shader.ConnectableAPI(), "surface"
            )
            UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(material)
            xf = UsdGeom.Xformable(cube.GetPrim())
            xf.ClearXformOpOrder()
            xf.AddTranslateOp().Set(Gf.Vec3d(bx, by, box_floor_z + bs / 2.0))
            xf.AddScaleOp().Set(Gf.Vec3d(bs / 2.0, bs / 2.0, bs / 2.0))  # Cube half-size=1
        cx, cy = center or (self.twin_start_pose.x, self.twin_start_pose.y)
        cam_path = "/World/PaperTopCam"
        cam = Camera(prim_path=cam_path, resolution=(1400, 1400))
        cam.initialize()
        ucam = UsdGeom.Camera(self.stage.GetPrimAtPath(cam_path))
        chase = UsdGeom.Camera(self.stage.GetPrimAtPath(self._chase_camera_path))
        if not chase:
            raise RuntimeError(f"chase camera not found: {self._chase_camera_path}")
        for name in (
            "projection",
            "horizontalAperture",
            "verticalAperture",
            "horizontalApertureOffset",
            "verticalApertureOffset",
            "focalLength",
            "clippingRange",
            "fStop",
            "focusDistance",
            "stereoRole",
            "shutter:open",
            "shutter:close",
            "exposure",
        ):
            source_attr = chase.GetPrim().GetAttribute(name)
            if not source_attr or source_attr.Get() is None:
                continue
            target_attr = ucam.GetPrim().CreateAttribute(
                name, source_attr.GetTypeName(), custom=False
            )
            target_attr.Set(source_attr.Get())

        focal_length = float(ucam.GetFocalLengthAttr().Get())
        aperture = float(ucam.GetHorizontalApertureAttr().Get())
        # With identical chase optics, move the camera high enough that the
        # requested floor span fills the horizontal frame:
        # projected_width = height * aperture / focal_length.
        camera_height = (
            float(span) * focal_length / aperture
            if span > 0.0
            else float(height)
        )
        xf = UsdGeom.Xformable(self.stage.GetPrimAtPath(cam_path)); xf.ClearXformOpOrder()
        xf.AddTranslateOp().Set(
            Gf.Vec3d(cx, cy, floor_z + camera_height)
        )  # identity looks down -Z
        print(
            f"[top-down] cloned chase optics from {self._chase_camera_path}; "
            f"camera height={camera_height:.2f} m"
        )
        for _ in range(16):
            self._step()

        rgba = cam.get_rgba()
        from PIL import Image
        if rgba is None or getattr(rgba, "size", 0) == 0:
            print(f"[top-down] no frame captured (rgba empty)")
            return
        arr = np.asarray(rgba)[..., :3].astype(np.uint8)
        Image.fromarray(arr).save(out_path)
        # also record the world->pixel mapping for overlaying routes
        meta = {"center": [cx, cy], "span_m": span, "resolution": list(arr.shape[:2]),
                "floor_z": floor_z, "box": box}
        Path(out_path).with_suffix(".json").write_text(json.dumps(meta))
        print(f"[top-down] wrote {out_path} ({arr.shape[1]}x{arr.shape[0]}), "
              f"mean_lum={arr.mean():.1f}")

    def _load_stage(self) -> None:
        self._open_stage(self.scene_path)
        for _ in range(20):
            self.simulation_app.update()

        self.stage = self._omni_usd.get_context().get_stage()
        self.rover_prim = self.stage.GetPrimAtPath(self.rover_prim_path)
        if not self.rover_prim.IsValid():
            raise RuntimeError(f"Rover prim not found: {self.rover_prim_path}")
        self._unload_hidden_static_payloads()
        self._pose_control_prim = self.rover_prim
        if self.scene_box is not None:
            self._author_scene_box(self.scene_box)

        self._articulation_prim = self._find_articulation_prim()
        # Prefer a rigid-body link as the pose/camera prim. The articulation
        # root Xform's USD/Fabric transform is NOT updated by Newton physics —
        # only the links move — so anything attached to the root (like the
        # camera) would stay frozen at the spawn pose forever.
        self._pose_prim = self.stage.GetPrimAtPath(str(self._articulation_prim.GetPath()) + "/body_link")
        if not self._pose_prim.IsValid():
            self._pose_prim = self._find_base_link_prim()
        if self._pose_prim is None or not self._pose_prim.IsValid():
            self._pose_prim = self._articulation_prim
        self._left_wheel_joints, self._right_wheel_joints = self._find_wheel_joints()
        if self.hide_static_scene_at_start or self.disable_static_scene_collision:
            self._set_static_scene_enabled(
                visible=not self.hide_static_scene_at_start,
                collision_enabled=not self.disable_static_scene_collision,
            )
            self._static_scene_hidden_at_start = self.hide_static_scene_at_start
        self._initial_base_z = self._get_world_translation(self.rover_prim)[2]
        self._initial_articulation_base_z = self._get_world_translation(self._articulation_prim)[2]
        self._base_z = self._resolve_start_z(self._initial_base_z)
        self._articulation_base_z = self._resolve_start_z(self._initial_articulation_base_z)
        self._obstacles = self._collect_scene_obstacles()
        self._build_obstacle_point_grid()
        self._world = self._world_cls(stage_units_in_meters=1.0)
        self._configure_articulation_drive()
        self._body_half_extents = self._compute_body_half_extents()
        self._measure_rover_ride_height()
        self._collision_mesh_path = self._resolve_collision_mesh_path()
        print(f"Collision query prim: {self._collision_mesh_path}")
        self._initialize_camera()

    def _unload_hidden_static_payloads(self) -> None:
        """Keep intentionally hidden static payloads out of the live-twin stage.

        Newton rejects a stage with any USD composition error.  In particular,
        an unavailable payload outside the rover would otherwise surface later
        as the much less useful "Failed to create simulation view" exception.
        The supported live-twin profile hides the authored static scene and
        replaces it with live terrain, so those payloads should not be loaded.
        Never unload a payload belonging to the rover itself.
        """
        if not self.hide_static_scene_at_start:
            return

        rover_path = str(self.rover_prim.GetPath())
        static_payload_paths = [
            path
            for path in self.stage.GetLoadSet()
            if not (
                str(path) == rover_path
                or str(path).startswith(rover_path + "/")
            )
        ]
        for path in static_payload_paths:
            self.stage.Unload(path)
        if static_payload_paths:
            print(
                "Static scene payloads unloaded: "
                + ", ".join(str(path) for path in static_payload_paths)
            )

    def _reset_stage(self, start_pose: Pose) -> None:
        self._stop_wheels()
        # Start the rover resting on the terrain under it (when --load-terrain),
        # so validation drives from a settled pose instead of dropping out of the air.
        resting_z = self._resting_root_z(start_pose.x, start_pose.y)
        self._set_articulation_default_pose(start_pose, z=resting_z)
        self._world.reset()
        self._apply_articulation_pose(start_pose, z=resting_z)
        self._stop_wheels()
        self._zero_articulation_motion()
        # Let the rover settle onto the collision floor under gravity; with a real
        # collision floor it rests on its wheels on its own.
        # settle_steps is calibrated in nominal 60 Hz frames; convert to real steps.
        settle_steps = max(1, round(self.settle_steps * self._physics_steps_per_second() / 60.0))
        for _ in range(settle_steps):
            self._stop_wheels()
            self._step()
        self._stop_wheels()
        self._zero_articulation_motion()
        self._log_pivot_geometry()

    def _log_pivot_geometry(self) -> None:
        """Log the offset between body_link (the reported pose point) and the
        wheel centroid (the point a skid-steer actually pivots about). If this
        offset is large, an in-place turn will make body_link sweep an arc."""
        try:
            from pxr import UsdGeom, UsdPhysics

            cache = UsdGeom.XformCache()
            body = cache.GetLocalToWorldTransform(self._pose_prim).ExtractTranslation()
            tokens = ("front_left", "front_right", "back_left", "back_right")
            pts = []
            for prim in self.stage.Traverse():
                path = str(prim.GetPath())
                if not (path == self.rover_prim_path or path.startswith(self.rover_prim_path + "/")):
                    continue
                if not any(token in path.lower() for token in tokens):
                    continue
                if not prim.HasAPI(UsdPhysics.CollisionAPI):
                    continue
                t = cache.GetLocalToWorldTransform(prim).ExtractTranslation()
                pts.append((float(t[0]), float(t[1])))
            if not pts:
                print("[pivot] no wheel colliders found to compute centroid")
                return
            cx = sum(p[0] for p in pts) / len(pts)
            cy = sum(p[1] for p in pts) / len(pts)
            offset = math.hypot(cx - float(body[0]), cy - float(body[1]))
            print(
                f"[pivot] body_link=({float(body[0]):.3f}, {float(body[1]):.3f}) "
                f"wheel_centroid=({cx:.3f}, {cy:.3f}) offset={offset:.3f} m ({len(pts)} wheels)"
            )
        except Exception as exc:
            print(f"[pivot] geometry log failed: {exc}")

    # ------------------------------------------------------------------
    # Footprint collision checking against the scan point cloud.
    #
    # The physics-engine query APIs (overlap, triggers, contact reporting) are
    # all stubbed/broken in this Isaac Newton build, so instead we keep a
    # virtual rover footprint box "slightly above the ground": scan points in
    # the obstacle band (above floor bumps, below rover top) go into a coarse
    # 2D grid, and any pose can be tested by transforming nearby points into
    # the footprint frame. This gives instant, predictive collision checks —
    # the planned path is swept BEFORE any sim time is spent driving it.
    # ------------------------------------------------------------------

    _FOOTPRINT_HALF_LONG_M = 0.26  # along drive direction, incl. margin
    _FOOTPRINT_HALF_LAT_M = 0.30   # across, incl. wheels + margin
    _OBSTACLE_BAND_Z = (0.12, 0.45)  # above floor noise, below rover top
    _POINT_GRID_CELL_M = 0.30
    # Distance the body coasts after the wheels are zeroed during the settle
    # steps of a closed-loop forward drive; the drive stops this far short of
    # the commanded distance so the coast lands it on target. Measured on the
    # scan floor at the planner's default speeds.
    _FORWARD_COAST_MARGIN_M = 0.005

    def _build_obstacle_point_grid(self) -> None:
        self._obstacle_grid: dict[tuple[int, int], Any] = {}
        if self.hide_static_scene_at_start:
            print("[footprint] static scene grid skipped (--hide-static-scene-at-start)")
            return
        try:
            from pxr import UsdGeom

            cache = UsdGeom.XformCache()
            # Prefer the denoised collision proxies (they are the collision
            # truth); fall back to the visual scan meshes.
            sources = [
                prim
                for prim in self.stage.Traverse()
                if prim.GetTypeName() == "Mesh"
                and "/CollisionProxy/" in str(prim.GetPath())
            ]
            if not sources:
                sources = [
                    prim
                    for prim in self.stage.Traverse()
                    if prim.GetTypeName() == "Mesh"
                    and not str(prim.GetPath()).startswith(self.rover_prim_path)
                ]
            collected = []
            for prim in sources:
                pts_attr = prim.GetAttribute("points")
                if not pts_attr or pts_attr.Get() is None:
                    continue
                pts = np.array(pts_attr.Get(), dtype=np.float64)
                m = cache.GetLocalToWorldTransform(prim)
                mat = np.array([[m[i][j] for j in range(4)] for i in range(4)])
                world = pts @ mat[:3, :3] + mat[3, :3]
                lo, hi = self._OBSTACLE_BAND_Z
                band = world[(world[:, 2] > lo) & (world[:, 2] < hi)][:, :2]
                if len(band):
                    collected.append(band)
            if not collected:
                print("[footprint] no obstacle-band points found; footprint check disabled")
                return
            points = np.vstack(collected)
            cell = self._POINT_GRID_CELL_M
            keys = np.floor(points / cell).astype(np.int64)
            for key, point in zip(map(tuple, keys), points):
                self._obstacle_grid.setdefault(key, []).append(point)
            self._obstacle_grid = {k: np.array(v) for k, v in self._obstacle_grid.items()}
            print(
                f"[footprint] obstacle point grid: {len(points)} points in "
                f"{len(self._obstacle_grid)} cells (z band {self._OBSTACLE_BAND_Z})"
            )
        except Exception as exc:
            print(f"[footprint] grid build failed ({exc}); footprint check disabled")
            self._obstacle_grid = {}

    def _footprint_hit(self, pose: Pose) -> tuple[float, float] | None:
        """Return an offending obstacle point if the rover footprint at `pose`
        overlaps the scan obstacle band, else None."""
        if not self._obstacle_grid:
            return None
        cell = self._POINT_GRID_CELL_M
        reach = math.hypot(self._FOOTPRINT_HALF_LONG_M, self._FOOTPRINT_HALF_LAT_M)
        span = int(math.ceil(reach / cell)) + 1
        cx, cy = int(math.floor(pose.x / cell)), int(math.floor(pose.y / cell))
        yaw = math.radians(pose.yaw_deg)
        c, s = math.cos(yaw), math.sin(yaw)
        for ix in range(cx - span, cx + span + 1):
            for iy in range(cy - span, cy + span + 1):
                points = self._obstacle_grid.get((ix, iy))
                if points is None:
                    continue
                dx = points[:, 0] - pose.x
                dy = points[:, 1] - pose.y
                u = dx * c + dy * s       # along drive direction
                v = -dx * s + dy * c      # lateral
                inside = (np.abs(u) < self._FOOTPRINT_HALF_LONG_M) & (
                    np.abs(v) < self._FOOTPRINT_HALF_LAT_M
                )
                if inside.any():
                    hit = points[inside][0]
                    return float(hit[0]), float(hit[1])
        return None

    def _predict_linear_collision(self, start_pose: Pose, command) -> tuple[str, Pose] | None:
        """Sweep the footprint along the commanded straight segment.

        Returns (reason, last_safe_pose) on a predicted hit. last_safe_pose is
        the furthest probe that was still collision-free, so the caller can
        drive the rover right up to the blockage and photograph it.
        """
        direction = 1.0 if command.command == CommandName.FWD else -1.0
        total_m = direction * command.distance / self.settings.limits.distance_units_per_m
        yaw = math.radians(start_pose.yaw_deg)
        # ceil (not int) so the last <5 cm of the sweep can't skip a wall.
        n = max(1, math.ceil(abs(total_m) / 0.05))
        last_safe = start_pose
        for i in range(n + 1):
            d = total_m * i / n
            probe = Pose(
                start_pose.x + math.cos(yaw) * d,
                start_pose.y + math.sin(yaw) * d,
                start_pose.yaw_deg,
            )
            hit = self._footprint_hit(probe)
            if hit is not None:
                return (
                    f"Predicted collision: rover footprint would overlap scan geometry "
                    f"near ({hit[0]:.2f}, {hit[1]:.2f}) after {abs(d):.2f} m of "
                    f"{command.command.value}",
                    last_safe,
                )
            last_safe = probe
        return None

    def _predict_turn_collision(self, start_pose: Pose, command) -> tuple[str, Pose] | None:
        """Sweep the footprint through the commanded in-place rotation.

        Returns (reason, last_safe_pose) on a predicted hit, where
        last_safe_pose is the furthest collision-free heading reached.
        """
        turn_sign = 1.0 if command.command == CommandName.LEFT else -1.0
        total_deg = turn_sign * float(command.distance)
        n = max(1, int(abs(total_deg) / 10.0))
        last_safe = start_pose
        for i in range(n + 1):
            probe = Pose(
                start_pose.x,
                start_pose.y,
                normalize_yaw_deg(start_pose.yaw_deg + total_deg * i / n),
            )
            hit = self._footprint_hit(probe)
            if hit is not None:
                return (
                    f"Predicted collision: rover footprint would overlap scan geometry "
                    f"near ({hit[0]:.2f}, {hit[1]:.2f}) while turning",
                    last_safe,
                )
            last_safe = probe
        return None

    # --- local free-space sweep -------------------------------------------------
    # A cheap "where can I go from here" scan: orient the rover footprint to each
    # heading and march it forward over the scan point grid until it would hit
    # something. Pure geometry against the same grid the collision check uses, so
    # no rendering or sim time. Headings are ABSOLUTE lab-frame drive directions
    # (degrees, math convention: 0=+x, 90=+y) to match how the planner reasons in
    # lab-frame waypoints.
    _SWEEP_MAX_RANGE_M = 2.0
    _SWEEP_STEP_DEG = 30
    _SWEEP_PROBE_STEP_M = 0.05
    # Below this free distance a heading is "blocked" — the rover is ~0.5 m long,
    # so anything under that is not a usable escape direction.
    _SWEEP_OPEN_THRESHOLD_M = 0.45

    def _free_distance(self, pose: Pose, heading_deg: float) -> float:
        """Metres the footprint can travel along heading_deg before a hit."""
        if not self._obstacle_grid:
            return self._SWEEP_MAX_RANGE_M
        yaw = math.radians(heading_deg)
        cos_y, sin_y = math.cos(yaw), math.sin(yaw)
        steps = max(1, int(self._SWEEP_MAX_RANGE_M / self._SWEEP_PROBE_STEP_M))
        free = 0.0
        for i in range(1, steps + 1):
            d = i * self._SWEEP_PROBE_STEP_M
            probe = Pose(pose.x + cos_y * d, pose.y + sin_y * d, heading_deg)
            if self._footprint_hit(probe) is not None:
                break
            free = d
        return free

    def _clearance_sweep(self, pose: Pose) -> list[dict[str, Any]]:
        """Free distance per heading from `pose`, ordered most-open first.

        Each entry gives an absolute heading, the clear distance, whether it is
        open enough to drive, and a concrete reachable clear point (80% of the
        free distance) the planner can use directly as a waypoint.
        """
        sweep: list[dict[str, Any]] = []
        for heading in range(0, 360, self._SWEEP_STEP_DEG):
            free_m = self._free_distance(pose, float(heading))
            reach = free_m * 0.8
            yaw = math.radians(heading)
            sweep.append(
                {
                    "heading_deg": heading,
                    "free_m": round(free_m, 2),
                    "open": free_m >= self._SWEEP_OPEN_THRESHOLD_M,
                    "clear_point": {
                        "x": round(pose.x + math.cos(yaw) * reach, 2),
                        "y": round(pose.y + math.sin(yaw) * reach, 2),
                    },
                }
            )
        sweep.sort(key=lambda item: item["free_m"], reverse=True)
        return sweep

    def _step(self) -> None:
        self.process_live_scan_updates(max_updates=2)
        self._world.step(render=self.render)
        self._service_observations_mid_validation()

    def _service_observations_mid_validation(self) -> None:
        """Answer queued /observation requests between validation physics steps.

        Only active while a validation runs (idle observations are served by
        main()'s loop with the full payload). The busy guard breaks recursion:
        the service path must never re-enter _step() (observation_snapshot does
        not step, but _capture_vision on failure paths does)."""
        if (self.observation_service is None or not self._validation_active
                or self._observation_service_busy):
            return
        now = time.monotonic()
        if now - self._observation_service_last_s < 0.2:
            return
        self._observation_service_last_s = now
        self._observation_service_busy = True
        try:
            self.observation_service()
        finally:
            self._observation_service_busy = False

    def _physics_steps_per_second(self) -> float:
        """Steps of `_step()` per second of simulated time.

        The Newton solver runs physics at its own rate (1/world.get_physics_dt(),
        500 Hz by default), NOT the render/timeline 60 Hz. Counting steps as if
        each were 1/60 s makes every wheel-drive command run ~8x too short in
        sim time, which is where the historical distance/velocity-scale
        calibration confusion came from.

        BUT: with --render, `world.step(render=True)` advances one RENDER frame
        per call (rendering_dt, default 1/60 s = ~8 physics substeps), so each
        `_step()` is a render frame, not a physics step. Counting physics steps
        there makes commands run ~8x too LONG instead."""
        try:
            if self.render:
                dt = float(self._world.get_rendering_dt())
            else:
                dt = float(self._world.get_physics_dt())
            if dt > 0.0:
                return 1.0 / dt
        except Exception:
            pass
        return 60.0

    def _resolve_start_z(self, initial_z: float) -> float:
        if self.rover_start_z is not None:
            return self.rover_start_z
        return initial_z + self.rover_z_offset

    def _configure_articulation_drive(self) -> None:
        if self._has_newton_actuators() and self._articulation_actuators_cls is not None:
            self._configure_newton_actuator_drive()
            return

        self._articulation = self._world.scene.add(
            self._single_articulation_cls(
                str(self._articulation_prim.GetPath()),
                name="team1_rover",
                reset_xform_properties=False,
            )
        )
        # Tune the physics before the sim is reset so the values are baked in:
        # solver iterations on the articulation + a friction material on the
        # wheel colliders. This is what makes the wheel-driven turn pivot
        # cleanly instead of sliding.
        self._apply_physics_tuning()
        self._world.reset()
        self._wheel_dof_indices = []
        self._wheel_dof_sides = []
        for name in self._articulation.dof_names:
            lowered = name.lower()
            if "front_left" in lowered or "back_left" in lowered:
                self._wheel_dof_indices.append(self._articulation.get_dof_index(name))
                self._wheel_dof_sides.append("left")
            elif "front_right" in lowered or "back_right" in lowered:
                self._wheel_dof_indices.append(self._articulation.get_dof_index(name))
                self._wheel_dof_sides.append("right")
        if not self._wheel_dof_indices:
            raise RuntimeError(f"No wheel DOFs found on articulation {self._articulation_prim.GetPath()}")
        self._wheel_drive_backend = "legacy_articulation_action"
        print(f"Wheel-drive backend: {self._wheel_drive_backend}")

    def _configure_newton_actuator_drive(self) -> None:
        self._world.reset()
        self._newton_actuators = self._articulation_actuators_cls(str(self._articulation_prim.GetPath()))
        self._articulation = self._newton_actuators.articulation
        self._wheel_dof_indices = []
        self._wheel_dof_sides = []
        for index, name in enumerate(self._articulation.dof_names):
            lowered = name.lower()
            if "front_left" in lowered or "back_left" in lowered:
                self._wheel_dof_indices.append(index)
                self._wheel_dof_sides.append("left")
            elif "front_right" in lowered or "back_right" in lowered:
                self._wheel_dof_indices.append(index)
                self._wheel_dof_sides.append("right")
        if not self._wheel_dof_indices:
            self._newton_actuators.close()
            self._newton_actuators = None
            raise RuntimeError(f"No wheel DOFs found on articulation {self._articulation_prim.GetPath()}")
        self._wheel_drive_backend = "newton_actuators"
        print(f"Wheel-drive backend: {self._wheel_drive_backend}")
        print(f"Newton actuator DOFs: {list(zip(self._wheel_dof_indices, self._wheel_dof_sides))}")

    def _has_newton_actuators(self) -> bool:
        articulation_path = str(self._articulation_prim.GetPath())
        for prim in self.stage.Traverse():
            path = str(prim.GetPath())
            if not (path == articulation_path or path.startswith(articulation_path + "/")):
                continue
            if prim.GetTypeName() == "NewtonActuator":
                return True
        return False

    def _set_articulation_default_pose(self, pose: Pose, z: float | None = None) -> None:
        position = np.array([pose.x, pose.y, self._articulation_base_z if z is None else z], dtype=float)
        orientation = self._yaw_to_quaternion(pose.yaw_deg)
        if self._newton_actuators is not None:
            self._articulation.set_default_state(
                positions=position,
                orientations=orientation,
                linear_velocities=np.zeros(3, dtype=float),
                angular_velocities=np.zeros(3, dtype=float),
                dof_velocities=np.zeros(self._articulation.num_dofs, dtype=float),
            )
            return
        self._articulation.set_default_state(position=position, orientation=orientation)

    def _apply_articulation_pose(self, pose: Pose, zero_motion: bool = True, z: float | None = None) -> None:
        position = np.array([pose.x, pose.y, self._articulation_base_z if z is None else z], dtype=float)
        orientation = self._yaw_to_quaternion(pose.yaw_deg)
        if self._newton_actuators is not None:
            self._articulation.set_world_poses(positions=position, orientations=orientation)
        else:
            self._articulation.set_world_pose(position=position, orientation=orientation)
        if zero_motion:
            self._zero_articulation_motion()

    def _apply_live_tracking_pose(self, target: Pose) -> None:
        """Track live XY/yaw without pinning the rover to a synthetic Z.

        The mapper publishes planar rover pose from the mounted phone. It does
        not know the articulation root height. Recomputing and writing Z on each
        phone frame fights gravity/contact and, with a noisy live heightmap, makes
        the rover hop during the scan. Preserve the current physics Z and avoid
        zeroing vertical velocity so the rover can settle on SupportGround or
        observed terrain.
        """
        current = self._get_rover_pose(fallback=target)
        current_z = self._articulation_live_root_z()
        if current_z is None or not math.isfinite(current_z):
            current_z = self._get_world_translation(self._articulation_prim)[2]
        if current_z is None or not math.isfinite(current_z):
            current_z = self._resting_root_z(target.x, target.y)
        if (
            math.hypot(target.x - current.x, target.y - current.y) < 0.005
            and yaw_error_deg(current.yaw_deg, target.yaw_deg) < 0.5
        ):
            return
        self._apply_articulation_pose(target, zero_motion=False, z=float(current_z))

    def _drive_to_failure_preview(self, start: Pose, last_safe: Pose) -> None:
        """Photograph the blockage that triggered a predicted collision.

        The predictive footprint sweep stops the plan before the rover moves at
        all, so a capture at the start pose shows the obstacle far away. Move
        the body to the furthest collision-free pose along the path (proven
        clear by the same sweep), capture both cameras there so the blocking
        object fills the forward view, then restore the start pose so this
        preview never leaks into the reported pose or a later attempt.
        """
        self._failure_frames = None
        # The clearance sweep is pure geometry, independent of the body actually
        # being moved, so compute it from last_safe even if the camera capture
        # below is skipped or fails.
        self._failure_clearance = self._clearance_sweep(last_safe)
        if self._articulation is None:
            return
        try:
            self._apply_articulation_pose(last_safe, zero_motion=True)
            # let the rendered camera transform catch up to the new body pose
            for _ in range(3):
                self._step()
            self._failure_frames = self._capture_vision()
        except Exception as exc:
            print(f"[failure-preview] capture failed: {exc}")
            self._failure_frames = None
        finally:
            try:
                self._apply_articulation_pose(start, zero_motion=True)
                for _ in range(2):
                    self._step()
            except Exception as exc:
                print(f"[failure-preview] restore failed: {exc}")

    def _apply_physics_tuning(self) -> None:
        self._set_solver_iterations()
        self._apply_wheel_friction()
        self._center_com()

    def _center_com(self) -> None:
        """Move body_link's center of mass to the wheel centroid so the rover
        pivots about its geometric center instead of an off-center COM. A
        skid-steer rotates about its COM; if the COM is offset from the body
        center, an in-place turn makes the body swing in an arc (drift that is
        independent of friction)."""
        if not self.center_com:
            return
        try:
            from pxr import Gf, UsdGeom, UsdPhysics

            cache = UsdGeom.XformCache()
            tokens = ("front_left", "front_right", "back_left", "back_right")
            pts = []
            for prim in self.stage.Traverse():
                path = str(prim.GetPath())
                if not (path == self.rover_prim_path or path.startswith(self.rover_prim_path + "/")):
                    continue
                if not any(token in path.lower() for token in tokens):
                    continue
                if not prim.HasAPI(UsdPhysics.CollisionAPI):
                    continue
                t = cache.GetLocalToWorldTransform(prim).ExtractTranslation()
                pts.append(Gf.Vec3d(float(t[0]), float(t[1]), float(t[2])))
            if not pts:
                print("[com] no wheel colliders found; COM not centered")
                return
            centroid = Gf.Vec3d(
                sum(p[0] for p in pts) / len(pts),
                sum(p[1] for p in pts) / len(pts),
                sum(p[2] for p in pts) / len(pts),
            )
            # Express the centroid in body_link's local frame (where COM lives).
            body_world = cache.GetLocalToWorldTransform(self._pose_prim)
            local = body_world.GetInverse().Transform(centroid)
            mass_api = UsdPhysics.MassAPI.Apply(self._pose_prim)
            mass_api.CreateCenterOfMassAttr().Set(Gf.Vec3f(float(local[0]), float(local[1]), float(local[2])))
            print(
                f"[com] body_link center of mass set to wheel centroid "
                f"(local offset = {float(local[0]):.3f}, {float(local[1]):.3f}, {float(local[2]):.3f})"
            )
        except Exception as exc:
            print(f"[com] could not center COM: {exc}")

    def _set_solver_iterations(self) -> None:
        """Raise the articulation solver iteration counts so wheel-ground
        contacts and the pivot resolve cleanly (low counts let the rover slide
        instead of rotating)."""
        if self.solver_position_iterations <= 0 and self.solver_velocity_iterations <= 0:
            return
        try:
            from pxr import PhysxSchema

            api = PhysxSchema.PhysxArticulationAPI.Apply(self._articulation_prim)
            if self.solver_position_iterations > 0:
                api.CreateSolverPositionIterationCountAttr().Set(int(self.solver_position_iterations))
            if self.solver_velocity_iterations > 0:
                api.CreateSolverVelocityIterationCountAttr().Set(int(self.solver_velocity_iterations))
            print(
                f"Solver iterations: pos={self.solver_position_iterations} "
                f"vel={self.solver_velocity_iterations}"
            )
        except Exception as exc:
            print(f"Could not set solver iterations: {exc}")

    def _apply_wheel_friction(self) -> None:
        """Set the friction on whatever physics material is already bound to each
        wheel collider (e.g. RobotMaterial), rather than binding a new one. The
        wheels already have a material with higher binding priority, so a new
        binding would not win; tuning the bound material is guaranteed to take
        effect. Falls back to creating+binding a material only for colliders that
        have none. Negative friction values disable this (use the USD defaults)."""
        if self.wheel_static_friction < 0.0 and self.wheel_dynamic_friction < 0.0:
            return
        try:
            from pxr import UsdPhysics, UsdShade

            # Wheel links are named by corner (e.g. back_right_link/collision),
            # not "wheel" - match the same tokens the DOF finder uses.
            wheel_tokens = ("front_left", "front_right", "back_left", "back_right")
            colliders = []
            tuned_materials: set[str] = set()
            fallback_material = None
            for prim in self.stage.Traverse():
                path = str(prim.GetPath())
                if not (path == self.rover_prim_path or path.startswith(self.rover_prim_path + "/")):
                    continue
                lowered = path.lower()
                if not any(token in lowered for token in wheel_tokens):
                    continue
                if not prim.HasAPI(UsdPhysics.CollisionAPI):
                    continue
                colliders.append(path)
                # Find the physics material actually resolved for this collider.
                bound_material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial("physics")
                mat_prim = bound_material.GetPrim() if bound_material else None
                if mat_prim is not None and mat_prim.IsValid():
                    mat_path = str(mat_prim.GetPath())
                    if mat_path not in tuned_materials:
                        self._tune_physics_material(mat_prim)
                        tuned_materials.add(mat_path)
                else:
                    # No material bound: create one and bind it (stronger than
                    # descendants so it actually wins).
                    if fallback_material is None:
                        fallback_material = self._create_wheel_material()
                    binding = UsdShade.MaterialBindingAPI.Apply(prim)
                    binding.Bind(
                        fallback_material,
                        bindingStrength="strongerThanDescendants",
                        materialPurpose="physics",
                    )
            print(
                f"Wheel friction static={self.wheel_static_friction} "
                f"dynamic={self.wheel_dynamic_friction}: {len(colliders)} collider(s), "
                f"tuned materials={sorted(tuned_materials) or 'none (bound fallback)'}"
            )
            if not colliders:
                print(
                    "  WARNING: no wheel colliders matched (path contains a "
                    "front_left/front_right/back_left/back_right token + CollisionAPI). "
                    "Friction not applied; check the wheel collider prim names."
                )
        except Exception as exc:
            print(f"Could not apply wheel friction: {exc}")

    def _tune_physics_material(self, mat_prim) -> None:
        """Set static/dynamic friction + combine mode on an existing physics
        material prim."""
        from pxr import UsdPhysics

        phys_mat = UsdPhysics.MaterialAPI.Apply(mat_prim)
        if self.wheel_static_friction >= 0.0:
            phys_mat.CreateStaticFrictionAttr().Set(float(self.wheel_static_friction))
        if self.wheel_dynamic_friction >= 0.0:
            phys_mat.CreateDynamicFrictionAttr().Set(float(self.wheel_dynamic_friction))
        # Contact friction combines BOTH materials; "max" makes this (high) wheel
        # friction dominate the wheel-floor contact regardless of the floor's
        # material.
        try:
            from pxr import PhysxSchema

            px_mat = PhysxSchema.PhysxMaterialAPI.Apply(mat_prim)
            px_mat.CreateFrictionCombineModeAttr().Set("max")
        except Exception as exc:
            print(f"  (could not set friction combine mode on {mat_prim.GetPath()}: {exc})")

    def _create_wheel_material(self):
        from pxr import Sdf, UsdShade

        mat_path = self.rover_prim_path + "/wheel_physics_material"
        mat_prim = self.stage.GetPrimAtPath(mat_path)
        if not mat_prim.IsValid():
            mat_prim = UsdShade.Material.Define(self.stage, Sdf.Path(mat_path)).GetPrim()
        self._tune_physics_material(mat_prim)
        return UsdShade.Material(mat_prim)

    def _zero_articulation_motion(self) -> None:
        if self._articulation is None:
            return
        if self._newton_actuators is not None:
            zeros3 = np.zeros(3, dtype=float)
            zeros_dof = np.zeros(self._articulation.num_dofs, dtype=float)
            try:
                self._articulation.set_velocities(linear_velocities=zeros3, angular_velocities=zeros3)
                self._articulation.set_dof_velocities(zeros_dof)
                self._articulation.set_dof_velocity_targets(zeros_dof)
                self._newton_actuators.set_dof_feedforward_effort_targets(zeros_dof)
            except Exception as exc:
                print(f"Could not zero Newton articulation motion: {exc}")
            return
        self._articulation.set_world_velocity(np.zeros(6, dtype=float))
        try:
            self._articulation.set_joint_velocities(np.zeros(self._articulation.num_dof, dtype=float))
        except Exception:
            if self._wheel_dof_indices:
                self._articulation.set_joint_velocities(
                    np.zeros(len(self._wheel_dof_indices), dtype=float),
                    joint_indices=np.array(self._wheel_dof_indices, dtype=int),
                )

    def _yaw_to_quaternion(self, yaw_deg: float) -> np.ndarray:
        yaw_rad = math.radians(self._body_yaw_from_pose_yaw(yaw_deg))
        return np.array([math.cos(yaw_rad / 2.0), 0.0, 0.0, math.sin(yaw_rad / 2.0)], dtype=float)

    def _body_yaw_from_pose_yaw(self, yaw_deg: float) -> float:
        return normalize_yaw_deg(yaw_deg - self._pose_forward_axis_yaw_offset_deg)

    def _get_rover_pose(self, fallback: Pose) -> Pose:
        try:
            # Under the Newton engine, live physics state is only visible through
            # the articulation tensor view; raw USD transforms are NOT updated
            # while the sim runs, so reading them here would freeze the measured
            # pose (turns "reach 0.0 deg", goals never advance).
            if self._articulation is not None:
                tensor_pose = self._articulation_tensor_pose()
                if tensor_pose is not None:
                    return tensor_pose
            prim = self._pose_prim
            translation = self._get_world_translation(prim)
            yaw = self._get_world_yaw_deg(prim)
            return Pose(float(translation[0]), float(translation[1]), yaw)
        except Exception:
            return fallback

    def _articulation_tensor_pose(self) -> Pose | None:
        try:
            if self._newton_actuators is not None:
                positions, orientations = self._articulation.get_world_poses()
                position = np.asarray(positions.numpy()).reshape(-1)
                quat = np.asarray(orientations.numpy()).reshape(-1)  # w, x, y, z
            else:
                position, quat = self._articulation.get_world_pose()
                position = np.asarray(position).reshape(-1)
                quat = np.asarray(quat).reshape(-1)
        except Exception:
            return None
        if position.size < 2 or quat.size != 4:
            return None
        w, x, y, z = (float(v) for v in quat)
        body_yaw = math.degrees(math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))
        pose_yaw = normalize_yaw_deg(body_yaw + self._pose_forward_axis_yaw_offset_deg)
        return Pose(float(position[0]), float(position[1]), pose_yaw)

    def _articulation_live_root_z(self) -> float | None:
        """Live physics z of the articulation root via the tensor view (the USD
        xform is stale under Newton, so _get_world_translation lies during sim)."""
        try:
            if self._newton_actuators is not None:
                positions, _ = self._articulation.get_world_poses()
                return float(np.asarray(positions.numpy()).reshape(-1)[2])
            position, _ = self._articulation.get_world_pose()
            return float(np.asarray(position).reshape(-1)[2])
        except Exception:
            return None

    def _articulation_tilt(self) -> tuple[float, float, float]:
        """(slope_deg, roll_deg, pitch_deg) of the rover body from its orientation.

        Once the rover settles on the terrain colliders, its tilt IS the local
        surface incline -- the physics-derived traversability signal. ``slope`` is
        the angle of the body up-axis from world up; ``roll``/``pitch`` are the
        Tait-Bryan tilts about the body axes. Returns zeros if the orientation is
        unavailable (e.g. flat-floor / no terrain loaded), which reads as level."""
        try:
            if self._newton_actuators is not None:
                _, orientations = self._articulation.get_world_poses()
                quat = np.asarray(orientations.numpy()).reshape(-1)
            else:
                _, quat = self._articulation.get_world_pose()
                quat = np.asarray(quat).reshape(-1)
        except Exception:
            return 0.0, 0.0, 0.0
        if quat.size != 4:
            return 0.0, 0.0, 0.0
        w, x, y, z = (float(v) for v in quat)
        up_z = 1.0 - 2.0 * (x * x + y * y)  # z of the body up-axis rotated to world
        slope = math.degrees(math.acos(max(-1.0, min(1.0, up_z))))
        roll = math.degrees(math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y)))
        pitch = math.degrees(math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x)))))
        return slope, roll, pitch

    def _find_articulation_prim(self):
        for prim in self.stage.Traverse():
            path = str(prim.GetPath())
            if path == self.rover_prim_path or path.startswith(self.rover_prim_path + "/"):
                if prim.HasAPI(self._usd_physics.ArticulationRootAPI):
                    return prim
        return self.rover_prim

    def _find_base_link_prim(self):
        """First rigid-body child of the articulation that is not a wheel link."""
        wheel_tokens = ("front_left", "front_right", "back_left", "back_right")
        articulation_path = str(self._articulation_prim.GetPath())
        for prim in self._articulation_prim.GetChildren():
            if not prim.HasAPI(self._usd_physics.RigidBodyAPI):
                continue
            if any(token in prim.GetName().lower() for token in wheel_tokens):
                continue
            print(f"Pose/camera link: {prim.GetPath()} (rigid body under {articulation_path})")
            return prim
        return None

    def _find_wheel_joints(self):
        left = []
        right = []
        for prim in self.stage.Traverse():
            path = str(prim.GetPath())
            if not (path == self.rover_prim_path or path.startswith(self.rover_prim_path + "/")):
                continue
            name = prim.GetName().lower()
            if not ("wheel" in name or "joint" in name):
                continue
            target_velocity = prim.GetAttribute("drive:angular:physics:targetVelocity")
            if not target_velocity.IsValid():
                continue
            prim.GetAttribute("drive:angular:physics:maxForce").Set(self.wheel_max_force)
            if "left" in name:
                left.append(prim)
            elif "right" in name:
                right.append(prim)
        return left, right

    def _get_world_translation(self, prim) -> tuple[float, float, float]:
        from pxr import UsdGeom

        matrix = UsdGeom.XformCache().GetLocalToWorldTransform(prim)
        translation = matrix.ExtractTranslation()
        return float(translation[0]), float(translation[1]), float(translation[2])

    def _get_world_yaw_deg(self, prim) -> float:
        from pxr import UsdGeom

        matrix = UsdGeom.XformCache().GetLocalToWorldTransform(prim)
        x_axis = matrix.TransformDir((1.0, 0.0, 0.0))
        body_yaw = math.degrees(math.atan2(float(x_axis[1]), float(x_axis[0])))
        return normalize_yaw_deg(body_yaw + self._pose_forward_axis_yaw_offset_deg)

    def _validate_command_limits(self, request: ValidationRequest) -> ValidationResult | None:
        if len(request.commands) > self.settings.limits.max_commands:
            return self._rejected("too_many_commands", None, "Plan has too many commands", request.start_pose)
        for index, command in enumerate(request.commands):
            if command.speed < self.settings.limits.min_speed or command.speed > self.settings.limits.max_speed:
                return self._rejected("speed_limit", index, "Command speed is outside configured limits", request.start_pose)
            if command.distance < 0 or command.distance > self.settings.limits.max_distance:
                return self._rejected("distance_limit", index, "Command distance is outside configured limits", request.start_pose)
            if command.command in (CommandName.LEFT, CommandName.RIGHT) and command.distance > self.settings.limits.max_turn_degrees:
                return self._rejected(
                    "turn_distance_limit",
                    index,
                    (
                        f"Turn distance {command.distance} deg exceeds "
                        f"{self.settings.limits.max_turn_degrees} deg"
                    ),
                    request.start_pose,
                )
            if command.command == CommandName.STOP and (command.speed != 0 or command.distance != 0):
                return self._rejected("invalid_stop", index, "STOP command must use speed 0 and distance 0", request.start_pose)
        return None

    @staticmethod
    def _rejected(failure_type: str, index: int | None, reason: str, pose: Pose) -> ValidationResult:
        return ValidationResult(
            approved=False,
            failure_type=failure_type,
            failed_command_index=index,
            reason=reason,
            final_pose=pose,
            min_clearance_m=0.0,
        )

    def _execute_command(self, pose: Pose, command) -> list[Pose]:
        return self._execute_command_with_articulation_drive(pose, command)

    def _execute_command_with_articulation_drive(self, pose: Pose, command) -> list[Pose]:
        if command.command == CommandName.STOP:
            self._set_articulation_wheel_velocity(0.0, 0.0)
            self._step()
            return [self._get_rover_pose(fallback=pose)]
        if command.command in (CommandName.LEFT, CommandName.RIGHT):
            return self._execute_turn_closed_loop(pose, command)
        left_velocity, right_velocity = self._wheel_velocity_for_command(command)
        sps = self._physics_steps_per_second()
        samples = [self._get_rover_pose(fallback=pose)]
        # Instant predictive check: sweep the virtual footprint box along the
        # commanded segment against the scan point cloud before spending any
        # sim time driving it.
        predicted = self._predict_linear_collision(samples[0], command)
        if predicted is not None:
            reason, last_safe = predicted
            self._drive_to_failure_preview(samples[0], last_safe)
            self._last_motion_failure = {"failure_type": "collision", "reason": reason}
            return samples
        # Closed-loop on MEASURED forward progress (not an open-loop duration):
        # drive until the body has actually covered the commanded distance, so
        # wheel slip and spin-up ramp are compensated automatically instead of
        # leaving a consistent ~5% undershoot. Distance is projected onto the
        # start heading so lateral wander does not inflate the reading.
        target_m = command.distance / self.settings.limits.distance_units_per_m
        # Stop this far short so the brief coast during the settle steps lands
        # the rover on target rather than just past it (measured ~0.5 cm coast
        # at the planner's default speeds).
        coast_margin_m = self._FORWARD_COAST_MARGIN_M
        stop_at_m = max(0.0, target_m - coast_margin_m)
        start_x, start_y = samples[0].x, samples[0].y
        start_yaw = samples[0].yaw_deg
        head = math.radians(start_yaw)
        cos_h, sin_h = math.cos(head), math.sin(head)
        # Cap the run so a slipping/blocked drive cannot spin forever; the
        # ideal-rolling duration plus generous margin is the ceiling.
        omega = max(0.01, command.speed * self.wheel_velocity_scale)
        expected_speed_m_s = omega * (self.wheel_diameter_m / 2.0)
        ideal_duration_s = target_m / max(1e-3, expected_speed_m_s)
        steps = max(1, min(round(15.0 * sps), round(2.0 * ideal_duration_s * sps) + round(0.5 * sps)))
        # Contact/stall watchdog: wheels commanded but the body stops making
        # forward progress means the rover is pushing against scene geometry.
        stall_window_steps = max(1, round(0.5 * sps))
        # Below 60% of commanded ground speed for half a second = pushing against
        # something. Free driving on the (smoothed) floor measures ~95% of the
        # rolling model, so 0.6 leaves a wide margin while catching square-on
        # contact that only partially slows the rover.
        stall_progress_ratio = 0.6
        deflection_limit_deg = 15.0  # a straight drive should never yaw; contact deflects it
        window_anchor = samples[0]
        steps_since_check = 0
        for _ in range(steps):
            self._set_articulation_wheel_velocity(left_velocity, right_velocity)
            self._step()
            samples.append(self._get_rover_pose(fallback=samples[-1]))
            current = samples[-1]
            forward_m = abs((current.x - start_x) * cos_h + (current.y - start_y) * sin_h)
            if forward_m >= stop_at_m:
                break
            deflection = abs(normalize_yaw_deg(current.yaw_deg - start_yaw))
            if deflection > deflection_limit_deg:
                self._last_motion_failure = {
                    "failure_type": "collision",
                    "reason": (
                        f"Drive deflected {deflection:.1f} deg off heading during a straight "
                        "command — glancing contact with scene geometry"
                    ),
                }
                break
            steps_since_check += 1
            if steps_since_check >= stall_window_steps:
                moved_m = math.hypot(current.x - window_anchor.x, current.y - window_anchor.y)
                expected_m = expected_speed_m_s * (stall_window_steps / sps)
                if moved_m < stall_progress_ratio * expected_m:
                    self._last_motion_failure = {
                        "failure_type": "collision",
                        "reason": (
                            f"Drive stalled: moved {moved_m:.3f} m in the last "
                            f"{stall_window_steps / sps:.1f} s vs {expected_m:.3f} m expected — "
                            "likely contact with scene geometry"
                        ),
                    }
                    break
                hit = self._footprint_hit(current)
                if hit is not None:
                    self._last_motion_failure = {
                        "failure_type": "collision",
                        "reason": (
                            f"Rover footprint overlaps scan geometry near "
                            f"({hit[0]:.2f}, {hit[1]:.2f}) during drive"
                        ),
                    }
                    break
                window_anchor = current
                steps_since_check = 0
        self._set_articulation_wheel_velocity(0.0, 0.0)
        for _ in range(10):
            self._step()
        return samples

    def _execute_turn_closed_loop(self, pose: Pose, command) -> list[Pose]:
        # Normal wheel-driven turn: spin the wheels in opposite directions and
        # let physics rotate the rover, reading yaw until the requested angle is
        # reached. Drift/cleanliness of the pivot is controlled by the wheel
        # friction and solver tuning, not by locking the body.
        left_velocity, right_velocity = self._wheel_velocity_for_command(command)
        target_deg = float(command.distance)
        samples = [self._get_rover_pose(fallback=pose)]
        predicted = self._predict_turn_collision(samples[0], command)
        if predicted is not None:
            reason, last_safe = predicted
            self._drive_to_failure_preview(samples[0], last_safe)
            self._last_motion_failure = {"failure_type": "collision", "reason": reason}
            return samples
        last_yaw = samples[0].yaw_deg
        achieved_deg = 0.0
        stall_steps = 0
        sps = self._physics_steps_per_second()
        # Stall = yaw rate below 1.2 deg/s sustained for 2 s, expressed per-step
        # so the thresholds stay correct at any physics rate.
        stall_yaw_per_step = 1.2 / sps
        stall_step_limit = max(1, round(2.0 * sps))
        timeout_steps = min(round(30.0 * sps), max(1, int(self.settings.limits.command_timeout_s * sps)))
        for _ in range(timeout_steps):
            self._set_articulation_wheel_velocity(left_velocity, right_velocity)
            self._step()
            current_pose = self._get_rover_pose(fallback=samples[-1])
            samples.append(current_pose)
            delta = normalize_yaw_deg(current_pose.yaw_deg - last_yaw)
            abs_delta = abs(delta)
            achieved_deg += abs_delta
            last_yaw = current_pose.yaw_deg
            if achieved_deg >= target_deg:
                break
            if abs_delta < stall_yaw_per_step:
                stall_steps += 1
            else:
                stall_steps = 0
            if stall_steps >= stall_step_limit:
                break
        self._set_articulation_wheel_velocity(0.0, 0.0)
        for _ in range(10):
            self._step()
            samples.append(self._get_rover_pose(fallback=samples[-1]))
        if achieved_deg < target_deg:
            self._last_motion_failure = {
                "failure_type": "turn_not_completed",
                "reason": f"Turn reached {achieved_deg:.1f} deg of requested {target_deg:.1f} deg",
            }
        return samples

    def _wheel_velocity_for_command(self, command) -> tuple[float, float]:
        velocity = command.speed * self.wheel_velocity_scale
        if command.command == CommandName.FWD:
            return self.left_wheel_sign * velocity, self.right_wheel_sign * velocity
        if command.command == CommandName.REV:
            return -self.left_wheel_sign * velocity, -self.right_wheel_sign * velocity
        if command.command == CommandName.LEFT:
            return -self.left_wheel_sign * velocity, self.right_wheel_sign * velocity
        if command.command == CommandName.RIGHT:
            return self.left_wheel_sign * velocity, -self.right_wheel_sign * velocity
        return 0.0, 0.0

    def _set_wheel_velocity(self, left_velocity: float, right_velocity: float) -> None:
        for prim in self._left_wheel_joints:
            prim.GetAttribute("drive:angular:physics:targetVelocity").Set(float(left_velocity))
        for prim in self._right_wheel_joints:
            prim.GetAttribute("drive:angular:physics:targetVelocity").Set(float(right_velocity))

    def _stop_wheels(self) -> None:
        self._set_wheel_velocity(0.0, 0.0)
        self._set_articulation_wheel_velocity(0.0, 0.0)

    def _set_articulation_wheel_velocity(self, left_velocity: float, right_velocity: float) -> None:
        if self._articulation is None or not self._wheel_dof_indices:
            return
        velocities = np.array(
            [left_velocity if side == "left" else right_velocity for side in self._wheel_dof_sides],
            dtype=float,
        )
        if self._newton_actuators is not None:
            self._articulation.set_dof_velocity_targets(
                velocities,
                dof_indices=np.array(self._wheel_dof_indices, dtype=int),
            )
            return
        self._articulation.apply_action(
            self._articulation_action_cls(
                joint_velocities=velocities,
                joint_indices=np.array(self._wheel_dof_indices, dtype=int),
            )
        )

    def _check_safety(self, pose: Pose) -> dict[str, Any]:
        if not self.settings.world.lab_bounds.contains(pose):
            return {
                "failure_type": "out_of_bounds",
                "reason": "Rover left configured lab bounds",
                "min_clearance_m": 0.0,
            }

        physx_hits = self._physx_overlap_check(pose)
        if physx_hits:
            return {
                "failure_type": "collision",
                "reason": f"PhysX overlap with {physx_hits[0]}",
                "min_clearance_m": 0.0,
            }

        clearances = [obstacle.clearance(pose) for obstacle in self._obstacles]
        min_clearance = min(clearances, default=999.0)
        if min_clearance < self.settings.limits.min_clearance_m:
            return {
                "failure_type": "collision",
                "reason": f"Rover clearance to scene geometry is {min_clearance:.3f} m",
                "min_clearance_m": min_clearance,
            }

        return {"failure_type": None, "reason": "", "min_clearance_m": min_clearance}

    def _resolve_collision_mesh_path(self) -> str:
        body_path = str(self._pose_prim.GetPath())
        # Prefer an explicit ExclusionMesh child if the user added one.
        candidate = body_path + "/ExclusionMesh"
        if self.stage.GetPrimAtPath(candidate).IsValid():
            return candidate
        for child in self._pose_prim.GetChildren():
            if "exclusion" in child.GetName().lower():
                return str(child.GetPath())
        return body_path

    def _measure_rover_ride_height(self) -> None:
        """Measure the root->lowest-collider distance over the whole rover, once,
        from its initial resting pose. Used to place the rover ON terrain of any
        height (root_z = terrain_z + this) instead of a fixed reset height."""
        try:
            from pxr import Gf, Usd, UsdGeom

            cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
            bbox = Gf.Range3d()
            for prim in Usd.PrimRange(self._articulation_prim):
                if prim.IsA(UsdGeom.Boundable):
                    bbox.UnionWith(cache.ComputeWorldBound(prim).ComputeAlignedBox())
            if bbox.IsEmpty():
                return
            root_z = self._get_world_translation(self._articulation_prim)[2]
            self._rover_root_to_bottom_m = float(root_z - bbox.GetMin()[2])
            print(f"Rover ride height (root->bottom): {self._rover_root_to_bottom_m:.3f} m")
        except Exception as exc:
            print(f"Ride-height measure failed: {exc}")

    def _resting_root_z(self, x: float, y: float) -> float:
        """Articulation-root z that sits the rover on the terrain under (x, y).

        Uses the live terrain heightmap (highest terrain point under the rover
        footprint) plus the measured ride height. Falls back to the configured
        base z when no terrain is loaded/known there -- preserving the old
        flat-floor behavior when --load-terrain is off."""
        base = self._articulation_base_z
        if self._rover_root_to_bottom_m is None:
            return base
        support = self._ground_support(x, y)
        if support is None or support.height_m is None:
            return base
        # Inferred support affects only this tracked resting pose.  It is not
        # authored as terrain and it never participates in readiness or plans.
        return support.height_m + self._rover_root_to_bottom_m

    def _reject_unready_or_unknown_terrain(self, request: ValidationRequest) -> ValidationResult | None:
        """Reject planned motion without direct live terrain evidence.

        Track mode remains usable during mapping; this guard applies solely to
        validation.  A straight conservative sweep is enough to reject an
        unknown corridor before Isaac can manufacture a misleading clearance.
        """
        if not self.load_terrain:
            return None
        status = self._live_twin_status()
        if not status["baseline_ready"]:
            return ValidationResult(
                approved=False, failure_type="unknown_not_ready",
                reason="Live terrain baseline is not ready; mapping/coverage is incomplete.",
                final_pose=request.start_pose, min_clearance_m=0.0,
            )
        dx, dy = request.goal.x - request.start_pose.x, request.goal.y - request.start_pose.y
        steps = max(1, int(math.ceil(math.hypot(dx, dy) / 0.10)))
        for index in range(steps + 1):
            t = index / steps
            support = self._ground_support(request.start_pose.x + dx * t, request.start_pose.y + dy * t)
            if support is None or support.source != "observed":
                return ValidationResult(
                    approved=False, failure_type="unknown_not_ready",
                    reason="Planned path crosses terrain without direct observed support.",
                    final_pose=request.start_pose, min_clearance_m=0.0,
                )
        return None

    def _compute_body_half_extents(self) -> tuple[float, float, float] | None:
        try:
            from pxr import Gf, Usd, UsdGeom

            cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
            bbox = Gf.Range3d()
            pose_path = str(self._pose_prim.GetPath())
            for prim in Usd.PrimRange(self._pose_prim):
                path = str(prim.GetPath())
                if path == pose_path + "/Actuators" or path.startswith(pose_path + "/Actuators/"):
                    continue
                if not prim.IsA(UsdGeom.Boundable):
                    continue
                bbox.UnionWith(cache.ComputeWorldBound(prim).ComputeAlignedBox())
            if bbox.IsEmpty():
                return None
            size = bbox.GetSize()
            return (float(size[0]) / 2.0, float(size[1]) / 2.0, float(size[2]) / 2.0)
        except Exception as exc:
            print(f"Body bbox compute failed: {exc}")
            return None

    def _physx_overlap_check(self, pose: Pose) -> list[str] | None:
        if self.disable_physx_overlap:
            return None
        try:
            from omni.physx import get_physx_scene_query_interface
            from pxr import PhysicsSchemaTools
        except Exception:
            return None

        query_path = self._collision_mesh_path or str(self._pose_prim.GetPath())
        try:
            encoded_path = PhysicsSchemaTools.sdfPathToInt(query_path)
        except Exception as exc:
            print(f"PhysX path encode failed: {exc}")
            return None

        rover_path = self.rover_prim_path
        hits: list[str] = []

        def on_hit(hit) -> bool:
            path = str(getattr(hit, "collision", "") or getattr(hit, "rigid_body", ""))
            if not path:
                return True
            if path == query_path or path == rover_path or path.startswith(rover_path + "/"):
                return True
            hits.append(path)
            return True

        try:
            get_physx_scene_query_interface().overlap_mesh(encoded_path, on_hit, False)
        except Exception as exc:
            print(f"PhysX overlap_mesh failed: {exc}")
            return None
        return hits

    def _collect_scene_obstacles(self) -> list[SceneObstacle]:
        from pxr import Usd, UsdGeom

        if self.hide_static_scene_at_start:
            suffix = " except explicitly authored /World/PaperBox" if self.scene_box is not None else ""
            print(f"Static scene obstacles skipped{suffix}")
        bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
        rover_path = str(self.rover_prim.GetPath())
        terrain_path = self.terrain_root
        obstacles: list[SceneObstacle] = []
        for prim in self.stage.Traverse():
            path = str(prim.GetPath())
            if (
                self.hide_static_scene_at_start
                and not (self.scene_box is not None and path == "/World/PaperBox")
                and not path.startswith("/World/AdvisorTestObstacles/")
            ):
                continue
            if path == rover_path or path.startswith(rover_path + "/"):
                continue
            # The live nvblox terrain is the drive SURFACE, not a footprint
            # obstacle: collision with it is handled by physics + the
            # traversability verdict, never the bbox sweep.
            if path == terrain_path or path.startswith(terrain_path + "/"):
                continue
            if prim.GetTypeName() not in ("Mesh", "Cube", "Cylinder"):
                continue
            if not prim.IsA(UsdGeom.Boundable):
                continue
            try:
                bounds = bbox_cache.ComputeWorldBound(prim).ComputeAlignedBox()
                if bounds.IsEmpty():
                    continue
                minimum = bounds.GetMin()
                maximum = bounds.GetMax()
            except Exception:
                continue
            if not all(math.isfinite(float(value)) for value in (minimum[0], minimum[1], maximum[0], maximum[1])):
                continue
            if abs(float(maximum[0]) - float(minimum[0])) < 0.01 and abs(float(maximum[1]) - float(minimum[1])) < 0.01:
                continue
            area = abs(float(maximum[0]) - float(minimum[0])) * abs(float(maximum[1]) - float(minimum[1]))
            if area > self.max_obstacle_bbox_area_m2:
                continue
            obstacles.append(
                SceneObstacle(
                    path=path,
                    min_x=float(minimum[0]),
                    max_x=float(maximum[0]),
                    min_y=float(minimum[1]),
                    max_y=float(maximum[1]),
                )
            )
        return obstacles


class IsaacRequestHandler(BaseHTTPRequestHandler):
    work_queue: queue.Queue
    # Observations ride a dedicated queue so they can also be answered from
    # INSIDE a running validation (via the _step hook) instead of queueing
    # behind it on work_queue -- that ordering froze the dashboard's twin
    # view for the whole rehearsal.
    observation_queue: queue.Queue

    def do_GET(self) -> None:
        if self.path == "/health":
            self._write_json(200, {"status": "ok"})
            return
        if self.path == "/observation":
            self._submit({"type": "observation"}, target_queue=self.observation_queue)
            return
        self._write_json(404, {"error": "not_found"})

    def do_POST(self) -> None:
        if self.path in ("/live_scan/start", "/live_scan/stop"):
            self._submit({"type": "live_scan_control",
                          "start": self.path.endswith("/start")})
            return
        if self.path == "/goal_marker":
            try:
                content_length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(content_length).decode("utf-8"))
                x, y = float(payload["x"]), float(payload["y"])
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                self._write_json(400, {"error": "invalid_goal_marker", "reason": str(exc)})
                return
            self._submit({"type": "goal_marker", "x": x, "y": y})
            return
        if self.path == "/advisor_obstacles":
            try:
                content_length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(content_length).decode("utf-8"))
                raw = payload["obstacles"]
                if not isinstance(raw, list) or len(raw) > 12:
                    raise ValueError("obstacles must be a list with at most 12 entries")
                obstacles = []
                for index, item in enumerate(raw):
                    if not isinstance(item, dict):
                        raise ValueError(f"obstacles[{index}] must be an object")
                    x = float(item["x"])
                    y = float(item["y"])
                    radius = float(item["radius_m"])
                    if not all(math.isfinite(v) for v in (x, y, radius)) or radius <= 0:
                        raise ValueError(f"obstacles[{index}] has invalid geometry")
                    obstacles.append({
                        "name": str(item.get("name", f"obstacle_{index}")),
                        "x": x,
                        "y": y,
                        "radius_m": radius,
                    })
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                self._write_json(400, {"error": "invalid_advisor_obstacles", "reason": str(exc)})
                return
            self._submit({"type": "advisor_obstacles", "obstacles": obstacles})
            return
        if self.path != "/validate_plan":
            self._write_json(404, {"error": "not_found"})
            return
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(content_length).decode("utf-8"))
            validation_request = ValidationRequest.from_dict(payload)
        except json.JSONDecodeError as exc:
            self._write_json(400, {"error": "invalid_json", "reason": str(exc)})
            return
        except ValidationError as exc:
            self._write_json(400, {"error": "schema_error", "reason": str(exc)})
            return
        self._submit({"type": "validate", "request": validation_request})

    def log_message(self, format: str, *args) -> None:
        return

    def _submit(self, job: dict[str, Any], target_queue: queue.Queue | None = None) -> None:
        response_queue: queue.Queue = queue.Queue(maxsize=1)
        job["response_queue"] = response_queue
        (target_queue if target_queue is not None else self.work_queue).put(job)
        status, payload = response_queue.get()
        self._write_json(status, payload)

    def _write_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Isaac Sim-backed /validate_plan service")
    parser.add_argument("--scene", default="/home/ca152656/Downloads/NewtonModel.usd")
    parser.add_argument("--rover-prim", default="/World/RASSOR")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--gui", action="store_true", help="Open the Isaac Sim GUI so validation playback is visible")
    parser.add_argument("--render", action="store_true")
    parser.add_argument(
        "--experience",
        default="",
        help="Kit experience file for SimulationApp (e.g. a Newton-physics app kit). "
        "Empty selects the stock isaacsim.exp.base.python.kit, which runs PhysX "
        "until the Newton extension switches engines at runtime.",
    )
    parser.add_argument("--wheel-velocity-scale", type=float, default=0.1)
    parser.add_argument("--wheel-diameter-m", type=float, default=0.25)
    parser.add_argument("--camera-width", type=int, default=320)
    parser.add_argument("--camera-height", type=int, default=240)
    parser.add_argument(
        "--disable-physx-overlap",
        action="store_true",
        default=True,
        help="Skip the PhysX overlap collision check; fall back to bbox clearance only.",
    )
    parser.add_argument(
        "--enable-physx-overlap",
        action="store_false",
        dest="disable_physx_overlap",
        help="Enable PhysX overlap collision checks. Disabled by default because this Isaac build rejects the overlap_mesh call signature.",
    )
    parser.add_argument("--wheel-max-force", type=float, default=100000.0)
    # Current NewtonModel wheel joints share the same positive rolling direction.
    parser.add_argument("--left-wheel-sign", type=float, default=1.0)
    parser.add_argument("--right-wheel-sign", type=float, default=1.0)
    parser.add_argument(
        "--wheel-static-friction",
        type=float,
        default=1.0,
        help="Static friction for the wheel contact material (higher = more grip, cleaner pivot). Negative to use USD defaults.",
    )
    parser.add_argument(
        "--wheel-dynamic-friction",
        type=float,
        default=1.0,
        help="Dynamic friction for the wheel contact material. Negative to use USD defaults.",
    )
    parser.add_argument(
        "--solver-position-iterations",
        type=int,
        default=32,
        help="PhysX articulation position-solver iterations (higher = stiffer contacts, less sliding). 0 to leave unchanged.",
    )
    parser.add_argument(
        "--solver-velocity-iterations",
        type=int,
        default=4,
        help="PhysX articulation velocity-solver iterations. 0 to leave unchanged.",
    )
    parser.add_argument(
        "--no-center-com",
        action="store_false",
        dest="center_com",
        default=True,
        help=(
            "Do not move the body center of mass to the wheel centroid. By default the COM is "
            "centered so in-place turns pivot about the rover's center instead of swinging in an arc."
        ),
    )
    parser.add_argument(
        "--rover-start-z",
        type=float,
        help="Override the rover/articulation root Z used when resetting validation start poses.",
    )
    parser.add_argument(
        "--rover-z-offset",
        type=float,
        default=0.0,
        help="Add this many meters to the USD's initial rover/articulation root Z when resetting start poses.",
    )
    parser.add_argument(
        "--settle-steps",
        type=int,
        default=90,
        help="Physics frames to wait after resetting the rover pose before executing the first command.",
    )
    parser.add_argument(
        "--pose-forward-axis",
        choices=["x", "y", "neg_x", "neg_y"],
        default="x",
        help="Which local body axis should be treated as rover forward for reported/commanded yaw.",
    )
    parser.add_argument("--lab-min-x", type=float, default=-5.0)
    parser.add_argument("--lab-max-x", type=float, default=5.0)
    parser.add_argument("--lab-min-y", type=float, default=-5.0)
    parser.add_argument("--lab-max-y", type=float, default=5.0)
    parser.add_argument("--min-clearance-m", type=float, default=0.25)
    parser.add_argument(
        "--max-obstacle-bbox-area-m2",
        type=float,
        default=16.0,
        help=(
            "Ignore room-scale meshes with larger XY bbox area; "
            "the scanned room mesh is too large for bbox collision."
        ),
    )
    parser.add_argument(
        "--live-scan",
        action="store_true",
        help="Enable the iPhone LiDAR live mesh receiver and derive planning obstacles from incoming mesh anchors.",
    )
    parser.add_argument("--live-scan-host", default="0.0.0.0")
    parser.add_argument("--live-scan-port", type=int, default=9099)
    parser.add_argument("--live-scan-root", default="/World/LiveScan")
    parser.add_argument(
        "--live-scan-rate-hz",
        type=float,
        default=10.0,
        help="Maximum USD visual mesh mutation rate for live scan updates.",
    )
    parser.add_argument(
        "--live-scan-grid-decay-s",
        type=float,
        default=0.0,
        help=(
            "Seconds after which an unrefreshed live mesh anchor is removed from the live mesh and "
            "planning grid. Default 0 disables expiry so room scans accumulate while you walk."
        ),
    )
    parser.add_argument(
        "--live-scan-transform-json",
        help="Optional JSON file containing a 16-number arkit_to_twin registration matrix.",
    )
    parser.add_argument(
        "--phone-mount-json",
        help=(
            "Optional JSON file with the phone-mount calibration (camera pose in the "
            "rover body frame, including the lever arm). See config/phone_mount.json. "
            "When set, /observation reports a derived rover_pose under phone_pose."
        ),
    )
    parser.add_argument("--twin-start-x", type=float, default=0.0, help="Twin-frame rover start X.")
    parser.add_argument("--twin-start-y", type=float, default=0.0, help="Twin-frame rover start Y.")
    parser.add_argument(
        "--twin-start-yaw-deg", type=float, default=0.0, help="Twin-frame rover start yaw in degrees."
    )
    parser.add_argument("--twin-start-z", type=float, default=0.0, help="Twin-frame rover start Z height.")
    parser.add_argument(
        "--no-auto-twin-align",
        action="store_true",
        help=(
            "Disable automatic twin alignment at the first normal ARKit pose. "
            "Auto-align is on by default when --phone-mount-json is set."
        ),
    )
    parser.add_argument(
        "--enable-support-ground",
        action="store_true",
        help=(
            "Enable the flat SupportGround physics scaffold without requiring "
            "a phone-mount calibration. Useful for controlled simulation-only tests."
        ),
    )
    parser.add_argument(
        "--no-support-ground",
        action="store_true",
        help=(
            "Disable the SupportGround physics scaffold + floor estimate. "
            "SupportGround is on by default when --phone-mount-json is set."
        ),
    )
    parser.add_argument("--support-ground-size-m", type=float, default=6.0, help="SupportGround slab side length (m).")
    parser.add_argument(
        "--support-ground-thickness-m", type=float, default=0.1, help="SupportGround slab thickness (m)."
    )
    parser.add_argument(
        "--no-bootstrap-gate",
        action="store_true",
        help=(
            "Disable the startup-360 coverage gate (navigation stays enabled). "
            "The gate is on by default when --phone-mount-json is set."
        ),
    )
    parser.add_argument(
        "--bootstrap-coverage-radius-m", type=float, default=1.5, help="Radius for the startup coverage gate (m)."
    )
    parser.add_argument(
        "--bootstrap-coverage-threshold",
        type=float,
        default=0.6,
        help="Known-cell fraction within the radius required to enable navigation.",
    )
    parser.add_argument(
        "--baseline-min-terrain-tiles", type=int, default=1,
        help="minimum directly observed nvblox tiles required before baseline can be ready",
    )
    parser.add_argument(
        "--baseline-exclusion-radius-m", type=float, default=0.45,
        help="rover footprint plus safety margin excluded from annular baseline coverage",
    )
    parser.add_argument(
        "--no-occupancy",
        action="store_true",
        help=(
            "Disable the plane-relative occupancy grid and fall back to the fixed "
            "world-Z obstacle band. Occupancy is on by default when --phone-mount-json is set."
        ),
    )
    parser.add_argument(
        "--hide-static-scan-after-live-scan",
        action="store_true",
        help="Hide existing static scan meshes after the first live mesh arrives. Off by default for visual safety.",
    )
    parser.add_argument(
        "--hide-static-scene-at-start",
        action="store_true",
        help=(
            "Hide non-rover static Mesh prims at startup and exclude them from "
            "the static planning grid. Use this for mounted live-scan tests where "
            "the old lab scan should not be visible or mixed with the new map."
        ),
    )
    parser.add_argument(
        "--disable-static-scene-collision",
        action="store_true",
        help=(
            "Disable CollisionAPI on non-rover static scene prims at startup. "
            "Use with care; keep SupportGround enabled if the rover needs a floor."
        ),
    )
    parser.add_argument(
        "--no-live-scan-light",
        action="store_false",
        dest="live_scan_light",
        default=True,
        help="Do not create the rectangular light centered above the live/baked scan.",
    )
    parser.add_argument(
        "--live-scan-light-scale",
        type=float,
        nargs=3,
        metavar=("SX", "SY", "SZ"),
        default=(10.0, 10.0, 1.0),
        help="Scale for the scan-top RectLight. Default matches the tested 10x10x1 setup.",
    )
    parser.add_argument(
        "--live-scan-light-height-padding-m",
        type=float,
        default=0.5,
        help="Meters above the scan's max Z to place the scan-top light.",
    )
    parser.add_argument(
        "--live-scan-light-intensity",
        type=float,
        default=5000.0,
        help="Intensity for the scan-top RectLight.",
    )
    parser.add_argument(
        "--track-live-pose",
        action="store_true",
        help="Continuous localization: slave the sim rover to the live twin-frame "
             "rover pose the mapper publishes (rover_pose.json in --map-dir), so the "
             "sim rover moves with the real rover.",
    )
    parser.add_argument(
        "--load-terrain",
        action="store_true",
        help="Author the live nvblox terrain (collidable) from --map-dir into this "
             "stage, so the rover validates/drives on the live surface. Usually paired "
             "with --hide-static-scene-at-start so the old lab scan isn't also present.",
    )
    parser.add_argument(
        "--map-dir",
        type=Path,
        default=Path("/tmp/nvblox_map"),
        help="The mapper's output dir (its --output): rover_pose.json + manifest.json "
             "+ tiles/. Source for both --track-live-pose and --load-terrain.",
    )
    parser.add_argument("--capture-top-down", default=None,
                        help="one-shot: render a chase-optics top-down PNG here and exit")
    parser.add_argument("--box", default=None,
                        help="author a persistent static box collider: 'x,y,size' in metres")
    parser.add_argument("--td-height", type=float, default=6.0,
                        help="fallback top-down camera height when --td-span is nonpositive")
    parser.add_argument("--td-span", type=float, default=6.0,
                        help="top-down floor width (m), framed using the chase camera optics")
    parser.add_argument("--td-center", default=None,
                        help="optional top-down center 'x,y' in world metres")
    parser.add_argument("--td-rover-pose", default=None,
                        help="optional rendered rover pose 'x,y,yaw_deg'")
    parser.add_argument("--terrain-root", default="/World/NvbloxMap",
                        help="prim path the live terrain is authored under")
    parser.add_argument("--terrain-static-friction", type=float, default=1.0,
                        help="static friction of the regolith material on the terrain colliders")
    parser.add_argument("--terrain-dynamic-friction", type=float, default=1.0,
                        help="dynamic friction of the regolith material on the terrain colliders")
    parser.add_argument("--terrain-restitution", type=float, default=0.0,
                        help="restitution of the regolith material on the terrain colliders")
    parser.add_argument("--terrain-traversability", action="store_true",
                        help="reject a plan if the rover's tilt on the terrain exceeds the "
                             "traversability envelope (slope/roll/pitch). Meaningful with "
                             "--load-terrain; the verdict is physics-derived from the rover's "
                             "settled orientation on the live surface.")
    parser.add_argument("--max-slope-deg", type=float, default=20.0,
                        help="max traversable terrain slope (deg)")
    parser.add_argument("--max-roll-deg", type=float, default=15.0,
                        help="max body roll before tip-over is called (deg)")
    parser.add_argument("--max-pitch-deg", type=float, default=20.0,
                        help="max body pitch before tip-over is called (deg)")
    parser.add_argument("--min-ground-clearance-m", type=float, default=0.05,
                        help="min belly clearance (m); enforced once a terrain raycast feeds it")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scene_box = None
    if args.box:
        bx, by, bs = (float(v) for v in args.box.split(","))
        if bs <= 0.0:
            raise ValueError("--box size must be positive")
        scene_box = (bx, by, bs)

    from digital_twin.models import LabBounds, SafetyLimits, WorldModel

    settings = PlannerSettings(
        limits=SafetyLimits(min_clearance_m=args.min_clearance_m),
        world=WorldModel(lab_bounds=LabBounds(args.lab_min_x, args.lab_max_x, args.lab_min_y, args.lab_max_y)),
    )

    work_queue: queue.Queue = queue.Queue()
    IsaacRequestHandler.work_queue = work_queue
    observation_queue: queue.Queue = queue.Queue()
    IsaacRequestHandler.observation_queue = observation_queue
    try:
        server = ThreadingHTTPServer((args.host, args.port), IsaacRequestHandler)
    except OSError as exc:
        if exc.errno == errno.EADDRINUSE:
            print(
                f"Isaac validator port {args.host}:{args.port} is already in use. "
                "Use the existing validator or stop it before launching another one.",
                file=sys.stderr,
            )
            return 2
        raise

    simulation_app = _load_isaac(headless=not args.gui, experience=args.experience)
    try:
        validator = IsaacSceneValidator(
            simulation_app=simulation_app,
            scene_path=args.scene,
            rover_prim_path=args.rover_prim,
            settings=settings,
            render=args.render,
            max_obstacle_bbox_area_m2=args.max_obstacle_bbox_area_m2,
            wheel_velocity_scale=args.wheel_velocity_scale,
            wheel_max_force=args.wheel_max_force,
            left_wheel_sign=args.left_wheel_sign,
            right_wheel_sign=args.right_wheel_sign,
            pose_forward_axis=args.pose_forward_axis,
            rover_start_z=args.rover_start_z,
            rover_z_offset=args.rover_z_offset,
            settle_steps=args.settle_steps,
            wheel_diameter_m=args.wheel_diameter_m,
            camera_width=args.camera_width,
            camera_height=args.camera_height,
            disable_physx_overlap=args.disable_physx_overlap,
            wheel_static_friction=args.wheel_static_friction,
            wheel_dynamic_friction=args.wheel_dynamic_friction,
            solver_position_iterations=args.solver_position_iterations,
            solver_velocity_iterations=args.solver_velocity_iterations,
            center_com=args.center_com,
            live_scan_enabled=args.live_scan,
            live_scan_host=args.live_scan_host,
            live_scan_port=args.live_scan_port,
            live_scan_root=args.live_scan_root,
            live_scan_rate_hz=args.live_scan_rate_hz,
            live_scan_grid_decay_s=args.live_scan_grid_decay_s,
            live_scan_transform_json=args.live_scan_transform_json,
            phone_mount_json=args.phone_mount_json,
            twin_start_pose=Pose(args.twin_start_x, args.twin_start_y, args.twin_start_yaw_deg),
            twin_start_z=args.twin_start_z,
            auto_twin_align=bool(args.phone_mount_json) and not args.no_auto_twin_align,
            support_ground_enabled=(bool(args.phone_mount_json) or args.enable_support_ground)
            and not args.no_support_ground,
            support_ground_size_m=args.support_ground_size_m,
            support_ground_thickness_m=args.support_ground_thickness_m,
            bootstrap_gate_enabled=bool(args.phone_mount_json) and not args.no_bootstrap_gate,
            bootstrap_coverage_radius_m=args.bootstrap_coverage_radius_m,
            bootstrap_coverage_threshold=args.bootstrap_coverage_threshold,
            baseline_exclusion_radius_m=args.baseline_exclusion_radius_m,
            baseline_min_terrain_tiles=args.baseline_min_terrain_tiles,
            occupancy_enabled=bool(args.phone_mount_json) and not args.no_occupancy,
            hide_static_scan_after_live_scan=args.hide_static_scan_after_live_scan,
            hide_static_scene_at_start=args.hide_static_scene_at_start,
            disable_static_scene_collision=args.disable_static_scene_collision,
            live_scan_light=args.live_scan_light,
            live_scan_light_scale=tuple(args.live_scan_light_scale),
            live_scan_light_height_padding_m=args.live_scan_light_height_padding_m,
            live_scan_light_intensity=args.live_scan_light_intensity,
            track_live_pose=args.track_live_pose,
            map_dir=args.map_dir,
            load_terrain=args.load_terrain,
            terrain_root=args.terrain_root,
            terrain_static_friction=args.terrain_static_friction,
            terrain_dynamic_friction=args.terrain_dynamic_friction,
            terrain_restitution=args.terrain_restitution,
            terrain_traversability=args.terrain_traversability,
            traversability_limits=TraversabilityLimits(
                max_slope_deg=args.max_slope_deg,
                max_roll_deg=args.max_roll_deg,
                max_pitch_deg=args.max_pitch_deg,
                min_ground_clearance_m=args.min_ground_clearance_m,
            ),
            scene_box=scene_box,
        )
    except Exception:
        server.server_close()
        raise

    if args.capture_top_down:
        td_center = None
        if args.td_center:
            center_x, center_y = (float(v) for v in args.td_center.split(","))
            td_center = (center_x, center_y)
        td_rover_pose = None
        if args.td_rover_pose:
            rover_x, rover_y, rover_yaw = (
                float(v) for v in args.td_rover_pose.split(",")
            )
            td_rover_pose = Pose(rover_x, rover_y, rover_yaw)
        try:
            validator.capture_top_down(args.capture_top_down, box=scene_box,
                                       height=args.td_height, span=args.td_span,
                                       center=td_center, rover_pose=td_rover_pose)
        finally:
            server.server_close()
            simulation_app.close()
        return 0

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    print(f"Isaac validator listening at http://{args.host}:{args.port}/validate_plan")
    print(f"Scene: {args.scene}")
    print(f"Rover: {args.rover_prim}")
    print(f"Wheel-drive backend: {validator._wheel_drive_backend}")
    print(f"Pose forward axis: {args.pose_forward_axis}")
    print(f"Wheel friction: static={args.wheel_static_friction} dynamic={args.wheel_dynamic_friction}")
    print(f"Solver iterations: pos={args.solver_position_iterations} vel={args.solver_velocity_iterations}")
    print(f"Center COM on wheel centroid: {args.center_com}")
    print(f"Camera resolution: {args.camera_width}x{args.camera_height}")
    print(f"Reset settle steps: {args.settle_steps}")
    print(f"Live scan: {args.live_scan} ({args.live_scan_host}:{args.live_scan_port})")
    print(f"Track live pose: {args.track_live_pose} (from {args.map_dir}/rover_pose.json)")
    print(f"Load live terrain: {args.load_terrain} (from {args.map_dir} -> {args.terrain_root})")
    print(f"Terrain traversability: {args.terrain_traversability} "
          f"(slope<={args.max_slope_deg} roll<={args.max_roll_deg} pitch<={args.max_pitch_deg} deg)")
    print(f"Indexed scene obstacles: {len(validator._obstacles)}")
    print(f"Wheel joints: left={len(validator._left_wheel_joints)} right={len(validator._right_wheel_joints)}")

    def service_observation_requests() -> None:
        """Answer queued /observation requests on the sim thread.

        Called from two places: every main-loop iteration (idle -> full
        observation payload) and from the _step hook while a validation runs
        (-> lightweight observation_snapshot, so the dashboard's twin view
        shows the rehearsal instead of freezing until it finishes)."""
        while True:
            try:
                job = observation_queue.get_nowait()
            except queue.Empty:
                return
            try:
                payload = (validator.observation_snapshot()
                           if validator._validation_active
                           else validator.observation())
                job["response_queue"].put((200, payload))
            except Exception as exc:
                job["response_queue"].put(
                    (500, {"error": "isaac_validator_error", "reason": str(exc)}))

    validator.observation_service = service_observation_requests

    try:
        while True:
            service_observation_requests()
            try:
                job = work_queue.get(timeout=0.02)
            except queue.Empty:
                # Guard the idle background work: an exception here used to unwind
                # the whole loop into the `finally: server.shutdown()`, so the
                # service printed "listening" and then silently stopped serving.
                # Log and continue instead, which also surfaces the real fault.
                try:
                    validator.process_live_scan_updates(max_updates=8)
                    # Ingest the latest live terrain (collidable) into this stage
                    # so the rover drives/validates on the real reconstructed
                    # surface.
                    if validator.load_terrain:
                        validator.load_terrain_tiles()
                    if validator.track_live_pose:
                        # Track mode: snap the sim rover onto the real rover's
                        # live pose, then step so the teleport settles on the
                        # terrain and the measured pose updates. Single-threaded
                        # with validation jobs (same loop), so the two never
                        # fight over the World.
                        validator.apply_live_pose_tracking()
                        validator._step()
                    else:
                        simulation_app.update()
                except Exception as exc:
                    print(f"Isaac background update failed: {exc!r}",
                          file=sys.stderr, flush=True)
                    time.sleep(0.1)
                continue

            try:
                if job["type"] == "validate":
                    result = validator.validate(job["request"])
                    job["response_queue"].put((200, result.to_dict()))
                elif job["type"] == "goal_marker":
                    validator._set_goal_marker(Pose(job["x"], job["y"], 0.0))
                    job["response_queue"].put((200, {"status": "ok"}))
                elif job["type"] == "advisor_obstacles":
                    payload = validator.set_advisor_test_obstacles(job["obstacles"])
                    job["response_queue"].put((200, payload))
                elif job["type"] == "live_scan_control":
                    status, payload = validator.send_live_scan_control(job["start"])
                    job["response_queue"].put((status, payload))
                else:
                    job["response_queue"].put((400, {"error": "unknown_job"}))
            except Exception as exc:
                job["response_queue"].put((500, {"error": "isaac_validator_error", "reason": str(exc)}))
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        if validator._live_scan_receiver is not None:
            validator._live_scan_receiver.stop()
        if validator._newton_actuators is not None:
            validator._newton_actuators.close()
        simulation_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
