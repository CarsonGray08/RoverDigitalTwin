#!/usr/bin/env python3
"""Live nvblox mapping from the iPhone depth stream.

Stands up the live-scan TCP receiver, feeds every MSG_DEPTH_FRAME into an
nvblox TSDF via NvbloxFeeder, and periodically saves the incremental color mesh.
Connect the LiveScanner app to this receiver to build a map from its RGB-D stream.

The map is built in the **twin frame**: the driver computes the same rigid
ARKit->twin alignment (``T_align``) the Isaac service uses -- from the first
normal-tracking phone pose via ``compute_t_align`` with the phone-mount
calibration -- and the feeder applies it to every frame. So the saved mesh
co-registers with the rover's twin start pose (it drops into the twin where the
rover actually is), and a Reset Position (MSG_REALIGN) re-anchors it to the
current pose. ``--no-align`` uses identity alignment (map at the
ARKit origin), e.g. for pure geometry-quality checks.

Launch through ``./scripts/live_twin.sh`` to configure mapping and validation
for the same session.
"""

from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import queue
import signal
import sys
import time
from pathlib import Path

# Make sibling package imports available when this file is executed directly.
_SRC = Path(__file__).resolve().parents[1]
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from digital_twin.live_scan import (
    MSG_CONTROL_DISABLE_IMU,
    MSG_CONTROL_ENABLE_IMU,
    MSG_CONTROL_START_SCAN,
    MSG_CONTROL_STOP_SCAN,
    MSG_DEPTH_FRAME,
    MSG_FINALIZE,
    MSG_IMU,
    MSG_KEYFRAME_IMAGE,
    MSG_KEYFRAME_RGBD,
    MSG_PHONE_POSE,
    MSG_REALIGN,
    MSG_REMOVE_ANCHOR,
    MSG_UPSERT_ANCHOR,
    TRACKING_NORMAL,
    LiveScanReceiver,
)
from digital_twin.mesh_tiles import load_manifest, write_tiles
from digital_twin.models import Pose
from digital_twin.geometry import distance_m, yaw_error_deg
from digital_twin.phone_mount import (
    AlignmentStabilityGate,
    PhoneMountCalibration,
    compute_t_align,
    rover_pose_from_camera,
    rover_pose_from_twin_camera_matrix,
)
from digital_twin.live_pose import write_pose as write_rover_pose
from digital_twin.live_twin_session import read_session, new_session, update_session, write_session
from digital_twin.terrain_denoise import denoise_terrain_mesh

# Human labels for the diagnostic tally of what the phone actually sends.
_MSG_LABELS = {
    MSG_UPSERT_ANCHOR: "mesh_anchor",
    MSG_REMOVE_ANCHOR: "remove_anchor",
    MSG_KEYFRAME_IMAGE: "keyframe",
    MSG_KEYFRAME_RGBD: "keyframe_rgbd",
    MSG_FINALIZE: "finalize",
    MSG_PHONE_POSE: "phone_pose",
    MSG_REALIGN: "realign",
    MSG_DEPTH_FRAME: "depth_frame",
    MSG_IMU: "imu",
}


def _depth_preview_png(depth_bytes: bytes, width: int, height: int,
                       near_m: float = 0.25, far_m: float = 4.0) -> bytes:
    """Colorize a raw float32 depth map into a grayscale PNG (near = bright).

    The phone only sends color keyframes while a scan is running AND the
    camera has moved enough (a phone-side motion gate), so a color-only live
    preview goes dark whenever the rig is still. Depth frames stream
    continuously during a scan, so this is the always-on fallback the camera
    endpoint serves between color keyframes. Stdlib zlib PNG encode -- no
    imaging library needed in the mapper process.
    """
    import struct as _struct
    import zlib

    import numpy as np

    depth = np.frombuffer(depth_bytes, dtype=np.float32, count=width * height)
    valid = np.isfinite(depth) & (depth > 0.0)
    norm = np.clip((far_m - depth) / (far_m - near_m), 0.0, 1.0)
    gray = np.where(valid, norm * 255.0, 0.0).astype(np.uint8).reshape(height, width)
    raw = b"".join(b"\x00" + gray[y].tobytes() for y in range(height))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (_struct.pack(">I", len(data)) + tag + data
                + _struct.pack(">I", zlib.crc32(tag + data)))

    ihdr = _struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
            + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


class MapperControlHandler(BaseHTTPRequestHandler):
    """Loopback control plane for the process that owns the phone TCP socket."""

    receiver: LiveScanReceiver | None = None
    # Set by main(): a {"pending": bool} flag the main loop polls to run a
    # one-shot under-rover fill + final save while the mapper keeps running.
    fill_request: dict | None = None
    # Set by main(): while paused, incoming frames may update localization but
    # are not fused into the TSDF/tile map.
    integration_control: dict | None = None
    # Set by main(): the most recent phone camera data, updated by the main
    # loop as frames arrive (regardless of alignment/pause state) so a
    # dashboard preview works even before the scan is aligned. Keys:
    #   jpeg / jpeg_received_s   -- latest color keyframe + host receive time
    #   depth / depth_received_s -- (bytes, w, h) latest raw depth + time
    # Times are HOST wall clock (time.time()), not the phone's ARKit clock
    # (device uptime), so consumers can compute a meaningful frame age.
    camera_frame: dict | None = None

    def do_GET(self) -> None:
        if self.path != "/camera/latest":
            self._reply(404, b'{"error":"not_found"}')
            return
        frame = self.camera_frame
        if frame is None:
            self._reply(404, b'{"error":"camera_unavailable"}')
            return
        jpeg = frame.get("jpeg")
        jpeg_t = frame.get("jpeg_received_s") or 0.0
        depth = frame.get("depth")
        depth_t = frame.get("depth_received_s") or 0.0
        # Color is the better image, so it wins unless the depth stream is
        # meaningfully fresher (the phone's keyframe motion gate paused color).
        if jpeg is not None and (depth is None or jpeg_t + 2.0 >= depth_t):
            body, content_type, source, stamp = jpeg, "image/jpeg", "color", jpeg_t
        elif depth is not None:
            depth_bytes, width, height = depth
            try:
                body = _depth_preview_png(depth_bytes, width, height)
            except Exception as exc:  # noqa: BLE001 - preview must never kill control
                self._reply(500, json.dumps(
                    {"error": "depth_preview_failed", "reason": str(exc)}).encode())
                return
            content_type, source, stamp = "image/png", "depth", depth_t
        else:
            self._reply(404, b'{"error":"no_frame_yet"}')
            return
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Frame-Timestamp", str(stamp))
        self.send_header("X-Frame-Source", source)
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        if self.path == "/map/fill_under_rover":
            if self.fill_request is None:
                self._reply(409, b'{"error":"fill_unavailable"}')
                return
            self.fill_request["pending"] = True
            self._reply(202, b'{"status":"queued"}')
            return
        if self.path in ("/map/integration/pause", "/map/integration/resume"):
            if self.integration_control is None:
                self._reply(409, b'{"error":"integration_control_unavailable"}')
                return
            paused = self.path.endswith("/pause")
            self.integration_control["paused"] = paused
            body = b'{"status":"paused"}' if paused else b'{"status":"resumed"}'
            self._reply(200, body)
            return
        message = {
            "/scan/start": MSG_CONTROL_START_SCAN,
            "/scan/stop": MSG_CONTROL_STOP_SCAN,
        }.get(self.path)
        if message is None:
            self._reply(404, b'{"error":"not_found"}')
            return
        receiver = self.receiver
        if receiver is None or not receiver.send_control(message):
            self._reply(409, b'{"error":"phone_not_connected"}')
            return
        self._reply(200, b'{"status":"ok"}')

    def _reply(self, status: int, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, _format: str, *_args) -> None:
        return


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", default="0.0.0.0", help="bind address for the receiver")
    p.add_argument("--port", type=int, default=9099, help="live-scan TCP port")
    p.add_argument("--control-host", default="127.0.0.1",
                   help="bind address for the local HTTP scan control service")
    p.add_argument("--control-port", type=int, default=9100,
                   help="local HTTP control port; 0 disables /scan/start and /scan/stop")
    p.add_argument("--voxel-size", type=float, default=0.02,
                   help="nvblox TSDF voxel size (m). 0.02 = chosen operating point "
                        "(benchmarked: ~1 ms integrate, ~1.16 GB/room on the 5070).")
    p.add_argument("--max-range", type=float, default=5.0,
                   help="max depth integration distance (m)")
    p.add_argument("--min-confidence", type=int, default=1,
                   help="drop ARKit depth below this confidence (0/1/2)")
    p.add_argument("--no-color", action="store_true",
                   help="geometry only; ignore texture keyframes")
    p.add_argument("--output", type=Path, default=Path("/tmp/nvblox_map"),
                   help="output dir for the mesh (.ply) and nvblox map")
    p.add_argument("--record", type=Path, default=None,
                   help="tee the raw phone wire stream to this .scanlog while mapping, "
                        "so this 360 scan can be replayed offline later. No effect on the "
                        "live run; just captures bytes.")
    p.add_argument("--record-imu", action="store_true",
                   help="with --record, tell the phone to include MSG_IMU samples in "
                        "the raw scanlog for cuVSLAM visual-inertial tests. IMU is "
                        "still skipped by the live nvblox mapper.")
    p.add_argument("--replay", type=Path, default=None,
                   help="replay a .scanlog (from --record) through the full mapper instead "
                        "of listening for a phone -- no hardware, deterministic. Frames are "
                        "injected as fast as the GPU consumes them (original timing ignored). "
                        "All mapping flags (--voxel-size, --loop-closure, ...) apply, so this "
                        "is the iterate/fix loop for scan issues. Mutually exclusive with --record.")
    p.add_argument("--session-id", default=None,
                   help="identifier written to session.json; normally supplied by live_twin.sh")
    p.add_argument("--resume-session", action="store_true",
                   help="explicitly mark this output directory as a resumed session")
    p.add_argument("--mesh-interval", type=float, default=5.0,
                   help="seconds between incremental mesh saves (the visual refresh "
                        "cadence; depth keeps integrating at full rate regardless)")
    p.add_argument("--tile-size", type=float, default=2.0,
                   help="spatial tile size (m) for the incremental sink: each save "
                        "rewrites only the tiles whose geometry changed, so the Isaac "
                        "twin tracks O(new geometry) not O(whole map). 0 = single tile.")
    p.add_argument("--max-backlog", type=int, default=128,
                   help="if more than this many frames are queued (host falling "
                        "behind the live stream), drop disposable telemetry frames to "
                        "stay real-time. RGB-D geometry is preserved. 0 = never drop.")
    p.add_argument("--integrate-fps", type=float, default=0.0,
                   help="max high-rate depth/RGB-D integration cadence. 0 = integrate "
                        "every frame received from the phone.")
    p.add_argument("--color-every", type=int, default=3,
                   help="color every Nth integrated RGB-D frame in ARKit mode. "
                        "1 = color every frame, 0 = no RGB color fusion.")
    p.add_argument("--start-scan", action="store_true",
                   help="send START to the phone once it connects")
    p.add_argument("--resumable-scan", action="store_true",
                   help="treat Stop Scan (MSG_FINALIZE) as a PAUSE, not a shutdown: "
                        "finalize and save the current map but keep the process and the "
                        "nvblox TSDF alive, so a later Start Scan resumes integrating into "
                        "the SAME map (ARKit's world frame is continuous across the phone's "
                        "stop/start, so the resumed scan lands in the same frame). Only "
                        "Ctrl-C (SIGINT) shuts the mapper down. Off by default keeps the "
                        "legacy one-shot behavior where finalize ends the run.")
    p.add_argument("--save-map", dest="save_map", action="store_true", default=True,
                   help="dump the nvblox voxel map (.nvblx) on finalize/exit so "
                        "--resume-session can restore the TSDF and append new geometry. "
                        "Enabled by default.")
    p.add_argument("--no-save-map", dest="save_map", action="store_false",
                   help="disable .nvblx map persistence; resumed sessions with only "
                        "tile files are served read-only to avoid clobbering old maps.")
    p.add_argument("--eval-sample-stride", type=int, default=16,
                   help="pixel stride for decimated per-frame surface samples written "
                        "to eval_surface_samples.npz for loop-closure seam metrics. "
                        "0 disables sample export.")
    p.add_argument("--eval-max-samples", type=int, default=250000,
                   help="cap for eval_surface_samples.npz points; older samples are "
                        "kept by decimation once this cap is reached.")
    # Twin alignment (co-register the map with the rover pose).
    p.add_argument("--phone-mount-json", default=None,
                   help="phone-mount calibration JSON (same file the Isaac service uses). "
                        "Omit for an identity mount.")
    p.add_argument("--twin-start-x", type=float, default=0.0)
    p.add_argument("--twin-start-y", type=float, default=0.0)
    p.add_argument("--twin-start-yaw-deg", type=float, default=0.0,
                   help="rover start pose in the twin that the first phone pose maps to")
    p.add_argument("--twin-start-z", type=float, default=0.0)
    p.add_argument("--align-stable-samples", type=int, default=5,
                   help="normal tracking phone poses that must be stable before auto-align/reset anchors")
    p.add_argument("--align-stable-max-translation-m", type=float, default=0.03,
                   help="max XY span across the alignment stability window")
    p.add_argument("--align-stable-max-yaw-deg", type=float, default=3.0,
                   help="max yaw span across the alignment stability window")
    p.add_argument("--no-align", action="store_true",
                   help="skip T_align; build the map at the ARKit origin")
    p.add_argument("--no-localize", action="store_true",
                   help="do not publish the live twin-frame rover pose (rover_pose.json). "
                        "By default, once aligned, every normal phone pose is converted to a "
                        "twin-frame rover pose and published for the Isaac twin to track.")
    p.add_argument("--pose-gate-max-speed-mps", type=float, default=0.75,
                   help="reject live rover poses that imply faster XY motion than this. "
                        "Protects the twin from ARKit/cuVSLAM jumps. 0 disables speed gating.")
    p.add_argument("--pose-gate-max-yaw-rate-deg-s", type=float, default=180.0,
                   help="reject live rover poses that imply faster yaw motion than this. "
                        "0 disables yaw-rate gating.")
    p.add_argument("--pose-gate-min-dt-s", type=float, default=0.05,
                   help="minimum dt used by the pose sanity gate to avoid tiny timestamp "
                        "deltas exaggerating normal noise")
    p.add_argument("--turn-in-place-lock", action="store_true",
                   help="startup-360 mode: ignore ARKit translation drift while integrating "
                        "depth. The rover body origin is held at twin-start x/y/z, ARKit yaw "
                        "is kept, and the calibrated phone lever arm reconstructs the camera "
                        "orbit. Use only for a commanded rotate-in-place scan.")
    p.add_argument("--loop-closure", action="store_true",
                   help="ARKit path: periodically ICP the live depth against the built "
                        "map and fold the correction into the pose, so revisited "
                        "surfaces snap onto themselves instead of doubling from drift.")
    p.add_argument("--icp-gravity-lock", action="store_true",
                   help="with --loop-closure: strip roll/pitch from each ICP correction "
                        "(apply yaw + translation only). Prevents floor tilt on "
                        "turn-in-place scans where point-to-point ICP is under-constrained "
                        "against the dominant ground plane.")
    p.add_argument("--replay-imu-yaw-closure", action="store_true",
                   help="replay only: replace ARKit yaw with integrated scanlog gyro yaw "
                        "and distribute start/end ARKit translation drift across the loop. "
                        "Use for fixed 360 benchmark logs where ARKit under-reports rotation.")
    p.add_argument("--replay-imu-yaw-axis", choices=("x", "y", "z"), default="x",
                   help="gyro axis used by --replay-imu-yaw-closure")
    p.add_argument("--replay-imu-target-yaw-deg", type=float, default=360.0,
                   help="normalize integrated IMU yaw to this final loop angle. "
                        "0 keeps raw integrated gyro yaw.")
    p.add_argument("--replay-imu-attitude-mode", choices=("planar", "arkit"),
                   default="planar",
                   help="with --replay-imu-yaw-closure: 'planar' uses fixed mount "
                        "attitude; 'arkit' preserves ARKit roll/pitch while replacing "
                        "yaw/translation.")
    p.add_argument("--replay-imu-translation-scale", type=float, default=1.0,
                   help="with --replay-imu-yaw-closure: scale loop-closed ARKit XY "
                        "translation around the start pose. 1 keeps ARKit local motion; "
                        "0 locks the rover origin for turn-in-place diagnostics.")
    p.add_argument("--use-cuvslam", action="store_true",
                   help="use cuVSLAM RGB-D odometry online for RGB-D keyframes. "
                        "Works for live scans and replay without precomputing poses.")
    p.add_argument("--cuvslam-variant",
                   choices=("depth-gray", "depth-rgb", "full-gray", "full-rgb"),
                   default="depth-gray",
                   help="image/depth convention fed to online cuVSLAM")
    p.add_argument("--cuvslam-depth-scale-factor", type=float, default=1000.0,
                   help="online cuVSLAM RGB-D depth scale denominator. iPhone depth "
                        "is converted to uint16 millimeters, so 1000 means meters.")
    p.add_argument("--cuvslam-use-gpu",
                   action=argparse.BooleanOptionalAction, default=True,
                   help="use GPU inside online cuVSLAM")
    p.add_argument("--replay-cuvslam", action="store_true",
                   help="replay only: replace ARKit poses with cuVSLAM RGB-D odometry "
                        "poses computed from the scanlog. The first cuVSLAM camera "
                        "frame is anchored into the normal twin alignment.")
    p.add_argument("--replay-cuvslam-variant",
                   choices=("depth-gray", "depth-rgb", "full-gray", "full-rgb"),
                   default="depth-gray",
                   help="image/depth convention fed to cuVSLAM during --replay-cuvslam")
    p.add_argument("--replay-cuvslam-depth-scale-factor", type=float, default=1000.0,
                   help="cuVSLAM RGB-D depth scale denominator. iPhone depth is "
                        "converted to uint16 millimeters, so 1000 means meters.")
    p.add_argument("--replay-cuvslam-use-gpu",
                   action=argparse.BooleanOptionalAction, default=True,
                   help="use GPU inside cuVSLAM for --replay-cuvslam")
    p.add_argument("--replay-cuvslam-close-translation", action="store_true",
                   help="with --replay-cuvslam: distribute final cuVSLAM translation "
                        "drift back across the loop while keeping cuVSLAM rotations")
    p.add_argument("--replay-cuvslam-translation-scale", type=float, default=1.0,
                   help="with --replay-cuvslam: scale cuVSLAM translation after optional "
                        "drift closure. 1 keeps it, 0 locks the camera at the first-frame "
                        "origin except for mount/orientation.")
    p.add_argument("--replay-cuvslam-async-sba",
                   action=argparse.BooleanOptionalAction, default=False,
                   help="with --replay-cuvslam: enable cuVSLAM asynchronous bundle "
                        "adjustment when computing the replay trajectory")
    p.add_argument("--replay-cuvslam-use-denoising",
                   action=argparse.BooleanOptionalAction, default=False,
                   help="with --replay-cuvslam: enable cuVSLAM input denoising when "
                        "computing the replay trajectory")
    p.add_argument("--denoise", action="store_true",
                   help="feature-preserving denoise of the terrain mesh before tiling, "
                        "so reconstruction noise doesn't become phantom collider hazards. "
                        "Off by default so you can first observe the raw noise.")
    p.add_argument("--denoise-iterations", type=int, default=10,
                   help="bounded-Laplacian smoothing passes (with --denoise)")
    p.add_argument("--denoise-step", type=float, default=0.5,
                   help="per-pass blend toward the neighbour centroid, 0..1 (with --denoise)")
    p.add_argument("--denoise-max-displacement", type=float, default=0.01,
                   help="cap (m) on how far any vertex may move from its original position: "
                        "the noise-amplitude dial. Below the wheel-scale hazard size so real "
                        "rocks/steps survive; ~ a fraction of the voxel size (with --denoise).")
    p.add_argument("--stabilize-floor", action="store_true",
                   help="final mesh output: snap the dominant floor band onto one fitted "
                        "support plane. This removes residual duplicate floor sheets after "
                        "loop closure without moving non-floor obstacles.")
    p.add_argument("--floor-stabilize-band", type=float, default=0.08,
                   help="floor residual band (m) affected by --stabilize-floor")
    p.add_argument("--fill-under-rover", action="store_true",
                   help="final mesh output: close the rover's occlusion disc (the "
                        "empty circle a turn-in-place 360 leaves under the rover) "
                        "with a floor-flush patch, so the rover sits on real "
                        "geometry. Authored as observed terrain. Also exposed live "
                        "via the POST /map/fill_under_rover control endpoint.")
    p.add_argument("--fill-radius-m", type=float, default=1.0,
                   help="max radius (m) cap for the under-rover fill. The hole is "
                        "found adaptively (flood-fill out to the surrounding floor); "
                        "this only bounds a runaway fill if the floor ring has a gap. "
                        "Set comfortably above the rover footprint.")
    p.add_argument("--fill-cell-m", type=float, default=0.05,
                   help="triangle grid resolution (m) of the fill patch")
    p.add_argument("--fill-ring-margin-m", type=float, default=0.35,
                   help="width (m) of the floor ring outside the disc used to fit "
                        "the fill height/plane and colour")
    p.add_argument("--fill-seam-overlap-cells", type=int, default=2,
                   help="fill patch overlap into the detected surrounding floor, "
                        "in --fill-cell-m grid cells. The retained floor renders "
                        "over this band, burying the patch's grid boundary. "
                        "Default 2 = 10 cm at the default 5 cm cell size.")
    p.add_argument("--fill-mode", choices=("patch", "delaunay"), default="patch",
                   help="under-rover fill method. 'patch': robust grid patch tucked "
                        "under the floor (buried seam, separate mesh). 'delaunay': "
                        "weld the mesh and Delaunay-remesh the footprint into one "
                        "continuous shared-vertex surface (true watertight fill in "
                        "the hole; needs scipy). Default patch.")
    p.add_argument("--fill-weld-tol-m", type=float, default=0.0005,
                   help="vertex weld tolerance (m) for --fill-mode delaunay. nvblox "
                        "emits unwelded duplicate vertices; merging within this "
                        "recovers a connected surface.")
    p.add_argument("--fill-seam-drop-m", type=float, default=0.002,
                   help="sit the fill patch this far (m) below the fitted floor so "
                        "the retained surrounding floor wins the overlap and hides "
                        "the patch seam (no T-junction crack, no z-fight). Small "
                        "enough to be near-invisible; raise toward 0.004 if z-fight "
                        "speckle appears, 0 keeps the patch coplanar.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.record and args.replay:
        print("ERROR: --record and --replay are mutually exclusive.", file=sys.stderr)
        return 2
    if args.record_imu and not args.record:
        print("ERROR: --record-imu requires --record so the IMU samples are saved.",
              file=sys.stderr)
        return 2
    if args.replay_imu_yaw_closure and not args.replay:
        print("ERROR: --replay-imu-yaw-closure requires --replay.", file=sys.stderr)
        return 2
    if args.replay_cuvslam and not args.replay:
        print("ERROR: --replay-cuvslam requires --replay.", file=sys.stderr)
        return 2
    if args.use_cuvslam and args.replay_cuvslam:
        print("ERROR: --use-cuvslam and --replay-cuvslam are mutually exclusive.",
              file=sys.stderr)
        return 2
    if args.use_cuvslam and args.replay_imu_yaw_closure:
        print("ERROR: --use-cuvslam and --replay-imu-yaw-closure are mutually exclusive.",
              file=sys.stderr)
        return 2
    if args.replay_cuvslam and args.replay_imu_yaw_closure:
        print("ERROR: --replay-cuvslam and --replay-imu-yaw-closure are mutually exclusive.",
              file=sys.stderr)
        return 2

    calibration = (
        PhoneMountCalibration.load(args.phone_mount_json)
        if args.phone_mount_json else PhoneMountCalibration.identity()
    )
    twin_start_pose = Pose(args.twin_start_x, args.twin_start_y, args.twin_start_yaw_deg)
    precomputed_replay_cuvslam = None
    if args.replay_cuvslam:
        # cuVSLAM and nvblox both load CUDA solver/BLAS libraries. Import and run
        # cuVSLAM before nvblox initializes CUDA to avoid symbol-resolution
        # conflicts from the process-global dynamic loader state.
        from digital_twin.cuvslam_replay import ReplayCuvslamTrajectory
        try:
            precomputed_replay_cuvslam = ReplayCuvslamTrajectory.from_scanlog(
                args.replay,
                calibration,
                twin_start_pose,
                args.twin_start_z,
                variant=args.replay_cuvslam_variant,
                min_confidence=args.min_confidence,
                depth_scale_factor=args.replay_cuvslam_depth_scale_factor,
                use_gpu=args.replay_cuvslam_use_gpu,
                async_sba=args.replay_cuvslam_async_sba,
                use_denoising=args.replay_cuvslam_use_denoising,
                close_translation=args.replay_cuvslam_close_translation,
                translation_scale=args.replay_cuvslam_translation_scale,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"ERROR: --replay-cuvslam failed: {exc}", file=sys.stderr)
            return 2
        print(
            "replay cuVSLAM precomputed: "
            f"variant={args.replay_cuvslam_variant} "
            f"valid={precomputed_replay_cuvslam.valid_poses}/"
            f"{precomputed_replay_cuvslam.total_frames} "
            f"yaw_unwrapped={precomputed_replay_cuvslam.final_yaw_y_unwrapped_deg:.1f}deg "
            f"final_translation={precomputed_replay_cuvslam.final_translation_norm_m:.3f}m "
            f"close_translation={args.replay_cuvslam_close_translation} "
            f"translation_scale={args.replay_cuvslam_translation_scale:.2f}",
            flush=True,
        )
    live_cuvslam_tracker = None
    if args.use_cuvslam:
        # Import/construct cuVSLAM before nvblox initializes CUDA, matching the
        # replay precompute path and avoiding CUDA solver/BLAS loader conflicts.
        from digital_twin.cuvslam_tracker import LiveCuvslamRgbdTracker
        try:
            live_cuvslam_tracker = LiveCuvslamRgbdTracker(
                variant=args.cuvslam_variant,
                min_confidence=args.min_confidence,
                depth_scale_factor=args.cuvslam_depth_scale_factor,
                use_gpu=args.cuvslam_use_gpu,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"ERROR: --use-cuvslam failed to initialize: {exc}", file=sys.stderr)
            return 2
        print(
            "online cuVSLAM enabled: "
            f"variant={args.cuvslam_variant} "
            f"depth_scale={args.cuvslam_depth_scale_factor:g} "
            f"use_gpu={args.cuvslam_use_gpu}",
            flush=True,
        )

    # Imported lazily so --help works on a box without CUDA/nvblox_torch.
    from digital_twin.nvblox_bridge import NvbloxMapperBridge, nvblox_available
    from digital_twin.nvblox_feeder import NvbloxFeeder

    if not nvblox_available():
        print("ERROR: nvblox_torch + a CUDA device are required.", file=sys.stderr)
        return 2

    import os
    import threading

    import numpy as np
    from digital_twin.nvblox_feeder import decode_jpeg_rgb

    args.output.mkdir(parents=True, exist_ok=True)
    existing_manifest = load_manifest(args.output) if args.resume_session else {}
    existing_tiles = existing_manifest.get("tiles", {}) if isinstance(existing_manifest, dict) else {}
    resume_readonly = bool(args.resume_session and existing_tiles)

    # This record is created before the phone connects.  It makes an empty map
    # observable and lets the Isaac process distinguish a fresh launch from an
    # arbitrary old directory without looking at stale tile files.  On disk
    # resume, preserve the old readiness metadata: the mapper process cannot
    # reload nvblox's in-memory TSDF from the tile transport, so existing tiles
    # are treated as read-only terrain until a new full scan is started.
    session = read_session(args.output) if args.resume_session else None
    if session is None:
        session = new_session(args.session_id, resumed=args.resume_session)
    session.update(
        session_id=args.session_id or session.get("session_id"),
        fresh=False if args.resume_session else session.get("fresh", True),
        resumed=bool(args.resume_session),
    )
    if resume_readonly:
        session.update(
            terrain_tiles=len(existing_tiles),
            map_revision=int(existing_manifest.get("rev", 0)),
            resume_readonly=True,
        )
    write_session(args.output, session)
    mesh_path = args.output / "nvblox_mesh.ply"
    npz_path = args.output / "nvblox_mesh.npz"
    map_path = args.output / "nvblox_map.nvblx"
    trajectory_path = args.output / "trajectory.jsonl"
    eval_samples_path = args.output / "eval_surface_samples.npz"
    rover_pose_history_path = args.output / "rover_pose_history.jsonl"

    # Disk writes must not stall frame integration: a slow save lets the receive
    # queue (and the phone over TCP) back up. The live sink consumes an
    # INCREMENTAL per-tile transport (manifest.json + tiles/*.npz): each save
    # extracts the mesh on this (CUDA) thread, then a background writer tiles it
    # and rewrites ONLY the tiles whose geometry changed -- so the cross-process
    # payload and the Isaac authoring are both O(new geometry), not O(whole map),
    # which is what keeps the twin tracking the scan as the room grows.
    # save_lock is acquired *before* a writer starts and released when it ends, so
    # only one write is ever in flight and ticks during a write are skipped. The
    # whole-mesh .npz/.ply are written once, on the final save, for offline
    # viewers -- nothing in the live path reads them.
    save_lock = threading.Lock()
    tile_state: dict = {
        name: tuple(fp) for name, fp in existing_tiles.items()
    }                                      # tile name -> fingerprint, across saves
    save_rev = {"n": int(existing_manifest.get("rev", 0))}  # monotonic manifest revision
    write_stats = {"tiles": len(existing_tiles), "written": 0, "removed": 0}
    # Profiling: where the integrate loop's wall time goes (see the periodic log).
    # save_timing is filled by _save_outputs; integ_timer accumulates the per-frame
    # CPU-side cost (decode + ICP + GPU launch) between saves.
    save_timing = {"extract_ms": 0.0, "settarget_ms": 0.0, "total_ms": 0.0, "verts": 0}
    integ_timer = {"ms": 0.0, "n": 0}
    under_rover_fill = {"enabled": False, "patch": None}

    def _output_mesh_arrays(mesh, fill: bool = False):
        """Mesh arrays to tile/author after optional terrain cleanup."""
        vertices = mesh.vertices
        faces = mesh.faces
        colors = mesh.colors
        if args.denoise:
            vertices = denoise_terrain_mesh(
                vertices, mesh.faces,
                iterations=args.denoise_iterations,
                step=args.denoise_step,
                max_displacement_m=args.denoise_max_displacement)
        if args.stabilize_floor:
            from digital_twin.floor_stabilize import stabilize_floor_mesh
            vertices, faces, colors = stabilize_floor_mesh(
                vertices, faces, colors, floor_band_m=args.floor_stabilize_band)
        if fill:
            before_v = len(vertices)
            before_f = len(faces)
            if args.fill_mode == "delaunay":
                from digital_twin.floor_fill import fill_floor_disc_delaunay
                vertices, faces, colors = fill_floor_disc_delaunay(
                    vertices, faces, colors,
                    center=(args.twin_start_x, args.twin_start_y),
                    max_radius_m=args.fill_radius_m,
                    cell_m=args.fill_cell_m,
                    ring_margin_m=args.fill_ring_margin_m,
                    weld_tol_m=args.fill_weld_tol_m)
                delta_v = len(vertices) - before_v
                delta_f = len(faces) - before_f
                print(f"  fill-under-rover ({args.fill_mode}): net vertices={delta_v:+d} "
                      f"faces={delta_f:+d} at "
                      f"({args.twin_start_x:.2f},{args.twin_start_y:.2f}) "
                      f"max_r={args.fill_radius_m:.2f}m",
                      flush=True)
            else:
                from digital_twin.floor_fill import (
                    apply_floor_disc_patch,
                    build_floor_disc_patch,
                )
                patch = under_rover_fill["patch"]
                if patch is None:
                    patch = build_floor_disc_patch(
                        vertices, faces, colors,
                        center=(args.twin_start_x, args.twin_start_y),
                        max_radius_m=args.fill_radius_m,
                        cell_m=args.fill_cell_m,
                        ring_margin_m=args.fill_ring_margin_m,
                        preserve_obstacles=False,
                        seam_overlap_cells=args.fill_seam_overlap_cells,
                        seam_drop_m=args.fill_seam_drop_m)
                    under_rover_fill["patch"] = patch
                    if patch is not None:
                        print(
                            f"  fill-under-rover ({args.fill_mode}): saved overlay "
                            f"vertices={len(patch.vertices)} faces={len(patch.faces)} at "
                            f"({args.twin_start_x:.2f},{args.twin_start_y:.2f}) "
                            f"max_r={args.fill_radius_m:.2f}m",
                            flush=True,
                        )
                if patch is not None:
                    vertices, faces, colors = apply_floor_disc_patch(
                        vertices, faces, colors, patch, remove_original_hole=True)
        return vertices, faces, colors

    def _final_fill_enabled(force_fill: bool) -> bool:
        return bool(force_fill or args.fill_under_rover)

    def _under_rover_filled() -> bool:
        return bool(
            under_rover_fill["enabled"]
            and (
                args.fill_mode == "delaunay"
                or under_rover_fill["patch"] is not None
            )
        )

    def _write_tiles_bg(mesh, rev: int) -> None:
        try:
            # Denoise in this background thread so it never stalls integration.
            vertices, faces, colors = _output_mesh_arrays(
                mesh, fill=under_rover_fill["enabled"])
            n, w, r = write_tiles(
                args.output, vertices, faces, colors,
                tile_size=args.tile_size, state=tile_state, rev=rev)
            write_stats.update(tiles=n, written=w, removed=r)
            update_session(
                args.output, terrain_tiles=n, map_revision=rev,
                under_rover_filled=_under_rover_filled())
        finally:
            save_lock.release()

    def _write_whole_mesh(mesh, fill: bool = False, arrays=None) -> None:
        # Whole-mesh outputs for offline viewers (final save only). Write the npz
        # through a file handle: np.savez(<str>) silently appends ".npz".
        # ``arrays`` lets the caller reuse the cleanup output it already computed
        # for tiling, so the (final-only) denoise/stabilize/fill passes and their
        # logs don't run twice.
        vertices, faces, colors = (
            arrays if arrays is not None else _output_mesh_arrays(mesh, fill=fill))
        tmp_path = npz_path.with_name(npz_path.name + ".tmp")
        with open(tmp_path, "wb") as fh:
            np.savez(fh, vertices=vertices, faces=faces, colors=colors)
        os.replace(tmp_path, npz_path)
        if args.stabilize_floor or args.denoise or fill:
            _write_mesh_ply(mesh_path, vertices, faces, colors)
        else:
            bridge.save_mesh(str(mesh_path))  # PLY once, at the end

    def _write_mesh_ply(path: Path, vertices, faces, colors) -> None:
        verts = np.asarray(vertices, dtype=np.float32)
        f = np.asarray(faces, dtype=np.int64)
        c = np.asarray(colors)
        has_color = c.ndim == 2 and len(c) == len(verts) and c.shape[1] >= 3
        if has_color:
            c = c[:, :3]
            if np.issubdtype(c.dtype, np.floating) and float(np.nanmax(c)) <= 1.0:
                c = np.clip(c * 255.0, 0, 255).astype(np.uint8)
            else:
                c = np.clip(c, 0, 255).astype(np.uint8)
        tmp = path.with_name(path.name + ".tmp")
        with tmp.open("w", encoding="ascii") as fh:
            fh.write("ply\nformat ascii 1.0\n")
            fh.write(f"element vertex {len(verts)}\n")
            fh.write("property float x\nproperty float y\nproperty float z\n")
            if has_color:
                fh.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
            fh.write(f"element face {len(f)}\n")
            fh.write("property list uchar int vertex_indices\nend_header\n")
            if has_color:
                for pnt, col in zip(verts, c):
                    fh.write(
                        f"{pnt[0]} {pnt[1]} {pnt[2]} "
                        f"{int(col[0])} {int(col[1])} {int(col[2])}\n")
            else:
                for pnt in verts:
                    fh.write(f"{pnt[0]} {pnt[1]} {pnt[2]}\n")
            for tri in f:
                fh.write(f"3 {int(tri[0])} {int(tri[1])} {int(tri[2])}\n")
        os.replace(tmp, path)

    def _save_outputs(final: bool = False, force_fill: bool = False) -> None:
        if resume_readonly:
            if final or force_fill:
                print(
                    "  resume-session read-only: keeping existing map tiles; "
                    "not writing a new partial mesh.",
                    flush=True,
                )
            return
        t0 = time.perf_counter()
        mesh = bridge.mesh_arrays()  # update + extract on the main (CUDA) thread
        t_extract = time.perf_counter()
        if corrector is not None:
            corrector.set_target(mesh.vertices)  # refresh the ICP target from the map
        t_settarget = time.perf_counter()
        save_timing.update(
            extract_ms=(t_extract - t0) * 1e3,
            settarget_ms=(t_settarget - t_extract) * 1e3,
            total_ms=(t_settarget - t0) * 1e3,
            verts=len(mesh.vertices),
        )
        save_rev["n"] += 1
        rev = save_rev["n"]
        if final:
            do_fill = _final_fill_enabled(force_fill)
            if do_fill:
                under_rover_fill["enabled"] = True
            save_lock.acquire()  # wait out any in-flight write, then write synchronously
            try:
                vertices, faces, colors = _output_mesh_arrays(
                    mesh, fill=under_rover_fill["enabled"])
                n, w, r = write_tiles(
                    args.output, vertices, faces, colors,
                    tile_size=args.tile_size, state=tile_state, rev=rev)
                write_stats.update(tiles=n, written=w, removed=r)
                update_session(args.output, terrain_tiles=n, map_revision=rev,
                               under_rover_filled=_under_rover_filled())
                _write_whole_mesh(
                    mesh, fill=under_rover_fill["enabled"],
                    arrays=(vertices, faces, colors))
            finally:
                save_lock.release()
        elif save_lock.acquire(blocking=False):
            # Background the IO; if a write is still going the lock is taken and we skip.
            threading.Thread(target=_write_tiles_bg, args=(mesh, rev), daemon=True).start()

    def _save_nvblox_map(reason: str) -> None:
        if resume_readonly or not args.save_map:
            return
        try:
            bridge.save_map(str(map_path))
            update_session(args.output, nvblox_map=str(map_path))
            print(f"  saved nvblox map ({reason}) -> {map_path}", flush=True)
        except Exception as exc:  # noqa: BLE001 - map save is best-effort
            print(f"  (map save failed during {reason}: {exc})", flush=True)

    queue_size = 4096 if args.max_backlog <= 0 else max(256, args.max_backlog * 4)
    frames: queue.Queue = queue.Queue(maxsize=queue_size)
    recorder = None
    if args.replay:
        # Offline replay: a scanlog stands in for the phone, decoded straight into
        # the frame queue. Everything downstream is the production mapper.
        from digital_twin.scan_recording import ScanLogReplaySource
        if not args.replay.exists():
            print(f"ERROR: --replay file not found: {args.replay}", file=sys.stderr)
            return 2
        receiver = ScanLogReplaySource(args.replay, frames)
        print(f"replay: feeding {args.replay} through the mapper (no phone)", flush=True)
    else:
        if args.record:
            from digital_twin.scan_recording import ScanRecorder
            args.record.parent.mkdir(parents=True, exist_ok=True)
            recorder = ScanRecorder(args.record)
            print(f"recording raw phone stream -> {args.record}", flush=True)
        receiver = LiveScanReceiver(
            args.host, args.port, frames,
            max_backlog_frames=args.max_backlog,
            recorder=recorder,
            initial_control_messages=(
                [MSG_CONTROL_ENABLE_IMU] if args.record_imu else None
            ),
        )
        if args.record_imu:
            print("recording IMU stream -> enabled on phone connect", flush=True)
    print(f"nvblox live map: building bridge (voxel {args.voxel_size} m)...", flush=True)
    bridge = NvbloxMapperBridge(
        voxel_size_m=args.voxel_size, max_integration_distance_m=args.max_range
    )
    if args.resume_session and map_path.exists():
        try:
            bridge.load_map(str(map_path))
            resume_readonly = False
            update_session(args.output, resume_readonly=False, nvblox_map=str(map_path))
            print(f"resume-session: restored nvblox TSDF from {map_path}", flush=True)
        except Exception as exc:  # noqa: BLE001 - fall back to non-destructive tiles
            resume_readonly = bool(existing_tiles)
            update_session(args.output, resume_readonly=resume_readonly)
            print(f"  (resume map load failed: {exc})", flush=True)
    if resume_readonly:
        print(
            "resume-session: existing tile map found; serving it read-only. "
            "Phone reconnect will update rover pose but will not append/overwrite "
            "geometry because the nvblox TSDF was not restored.",
            flush=True,
        )
    feeder = NvbloxFeeder(bridge, min_confidence=args.min_confidence)

    # ICP loop closure (ARKit path): snap revisited geometry onto the map.
    corrector = None
    if args.loop_closure:
        try:
            import open3d  # noqa: F401
        except Exception as exc:  # noqa: BLE001
            print(f"ERROR: --loop-closure needs Open3D installed: {exc}",
                  file=sys.stderr)
            return 2
        from digital_twin.icp_correction import IcpCorrector
        corrector = IcpCorrector(gravity_locked=args.icp_gravity_lock)
        lock_note = " (gravity-locked: yaw+translation only)" if args.icp_gravity_lock else ""
        print(f"ICP loop closure enabled (ARKit path){lock_note}.", flush=True)

    # Twin alignment: mirror the Isaac service (compute_t_align from the first
    # normal phone pose). Until it's established, hold off integrating so nothing
    # lands in the wrong frame. --no-align keeps the old identity behavior.
    align_enabled = not args.no_align
    aligned = not align_enabled  # identity path is "aligned" immediately
    align_gate = AlignmentStabilityGate(
        samples_required=max(1, int(args.align_stable_samples)),
        max_translation_m=max(0.0, float(args.align_stable_max_translation_m)),
        max_yaw_deg=max(0.0, float(args.align_stable_max_yaw_deg)),
    )
    # Continuous localization: once aligned, publish the twin-frame rover pose so
    # the Isaac twin can slave the sim rover to the real rover. Meaningless at the
    # ARKit origin, so it requires alignment.
    localize_enabled = align_enabled and not args.no_localize
    # Optional diagnostic mode for a true pivot-in-place 360: ARKit yaw is used,
    # but rover-body translation is locked to the start pose. Keep this as
    # runtime state so the lock can be released after the 360 finalizes and
    # follow-on drive scans can use true translation again.
    turn_lock_active = bool(args.turn_in_place_lock)
    pose_seq = {"n": 0}
    pose_gate = {
        "last_pose": None,
        "last_phone_timestamp_s": None,
        "last_host_timestamp_s": None,
        "last_reject_log_s": 0.0,
        "rejects": 0,
    }
    pose_history_fh = rover_pose_history_path.open(
        "a" if args.resume_session else "w", encoding="utf-8")
    eval_trajectory: list[dict] = []
    eval_sample_points: list[np.ndarray] = []
    eval_sample_frame_indices: list[np.ndarray] = []
    eval_sample_count = {"n": 0}

    if args.turn_in_place_lock and not args.phone_mount_json:
        receiver.stop()
        print("ERROR: --turn-in-place-lock requires --phone-mount-json",
              file=sys.stderr)
        return 2
    replay_imu_closure = None
    replay_cuvslam = precomputed_replay_cuvslam
    if args.replay_imu_yaw_closure:
        from digital_twin.replay_imu_closure import ReplayImuYawClosure
        try:
            replay_imu_closure = ReplayImuYawClosure.from_scanlog(
                args.replay,
                calibration,
                twin_start_pose,
                args.twin_start_z,
                gyro_axis=args.replay_imu_yaw_axis,
                target_yaw_degrees=args.replay_imu_target_yaw_deg,
                attitude_mode=args.replay_imu_attitude_mode,
                translation_scale=args.replay_imu_translation_scale,
            )
        except Exception as exc:  # noqa: BLE001
            receiver.stop()
            print(f"ERROR: --replay-imu-yaw-closure failed: {exc}",
                  file=sys.stderr)
            return 2
        dx, dy = replay_imu_closure.translation_drift_m
        print(
            "replay IMU yaw closure enabled: "
            f"raw_yaw={replay_imu_closure.raw_final_yaw_deg:.1f}deg "
            f"applied_yaw={replay_imu_closure.applied_final_yaw_deg:.1f}deg "
            f"translation_drift=({dx:.3f},{dy:.3f})m",
            flush=True,
        )

    receiver.start()
    fill_request = {"pending": False}
    integration_control = {"paused": False}
    camera_frame = {"jpeg": None, "jpeg_received_s": None,
                    "depth": None, "depth_received_s": None}
    control_server = None
    if args.control_port:
        MapperControlHandler.receiver = receiver
        MapperControlHandler.fill_request = fill_request
        MapperControlHandler.integration_control = integration_control
        MapperControlHandler.camera_frame = camera_frame
        try:
            control_server = ThreadingHTTPServer(
                (args.control_host, args.control_port), MapperControlHandler)
        except OSError as exc:
            receiver.stop()
            print(f"ERROR: control port {args.control_host}:{args.control_port}: {exc}",
                  file=sys.stderr)
            return 2
        threading.Thread(target=control_server.serve_forever, daemon=True).start()
        print(f"scan control: http://{args.control_host}:{args.control_port}/scan/{{start,stop}}"
              " (+ POST /map/fill_under_rover, GET /camera/latest)", flush=True)
    print(f"listening on {args.host}:{args.port}; waiting for the phone...", flush=True)
    if args.start_scan:
        # Best-effort: only works once a phone is connected.
        receiver.send_control(MSG_CONTROL_START_SCAN)

    stop = {"flag": False}

    def _handle_sigint(_sig, _frm):
        stop["flag"] = True
    signal.signal(signal.SIGINT, _handle_sigint)

    integrated = 0
    colored = 0
    dropped = 0
    # Resumable-scan pause state: set when a Stop Scan finalizes the map while the
    # process stays alive, cleared again when the next frame resumes integration.
    scan_paused = False
    warned_dropping = False
    warned_integration_paused = False
    warned_resume_readonly_geometry = False
    last_mesh_save = time.monotonic()
    last_logged_integrated = 0
    last_bytes = 0
    last_integrated_phone_ts = -1e9
    min_integrate_interval = 0.0 if args.integrate_fps <= 0 else 1.0 / args.integrate_fps
    seen: dict[int, int] = {}

    def _should_integrate_high_rate(frame) -> bool:
        nonlocal dropped, warned_dropping, last_integrated_phone_ts
        if min_integrate_interval <= 0.0:
            last_integrated_phone_ts = frame.timestamp_s
            return True
        if frame.timestamp_s - last_integrated_phone_ts >= min_integrate_interval:
            last_integrated_phone_ts = frame.timestamp_s
            return True
        dropped += 1
        if not warned_dropping:
            warned_dropping = True
            print("  (throttling high-rate RGB-D/depth frames to stay real-time; "
                  "raise --integrate-fps for denser fusion)", flush=True)
        return False

    def _log_icp_if_attempted() -> None:
        if corrector is None or not corrector.last_attempted:
            return
        if corrector.last_applied:
            print(f"  ICP applied #{corrector.correction_count}: "
                  f"fitness={corrector.last_fitness:.2f} "
                  f"shift={corrector.last_translation_m * 100.0:.1f} cm "
                  f"rot={corrector.last_rotation_degrees:.1f} deg",
                  flush=True)
        else:
            print(f"  ICP checked: fitness={corrector.last_fitness:.2f} "
                  f"shift={corrector.last_translation_m * 100.0:.1f} cm "
                  f"rot={corrector.last_rotation_degrees:.1f} deg "
                  f"({corrector.last_reject_reason or 'rejected'})", flush=True)

    def _tally_str() -> str:
        if not seen:
            return "(nothing received)"
        return ", ".join(
            f"{_MSG_LABELS.get(mt, f'type{mt}')}={n}" for mt, n in sorted(seen.items())
        )

    def _establish_alignment(camera_transform, reason: str) -> None:
        """Compute T_align from a phone pose and apply it. If geometry was already
        integrated in a different frame (e.g. a mid-scan Reset Position), clear it
        -- nvblox can't re-transform a built TSDF, so the old map is now stale."""
        nonlocal aligned, integrated, colored
        t_align = compute_t_align(
            camera_transform, calibration, twin_start_pose, args.twin_start_z)
        note = ""
        if integrated:
            bridge.clear()
            integrated = 0
            colored = 0
            note = " (cleared prior geometry: alignment changed)"
        feeder.set_alignment(t_align)
        aligned = True
        update_session(
            args.output,
            t_align="present",
            alignment_reason=reason,
            align_stable_samples=align_gate.samples_required,
        )
        print(f"  twin alignment {reason} -> map in twin frame{note}", flush=True)

    def _pose_gate_check(
        pose: Pose,
        phone_timestamp_s: float | None,
        *,
        source: str,
        update: bool,
    ) -> tuple[bool, str | None]:
        """Reject impossible rover motion before it reaches the live twin/map."""
        now = time.time()
        last = pose_gate["last_pose"]
        if last is None:
            if update:
                pose_gate.update(
                    last_pose=pose,
                    last_phone_timestamp_s=phone_timestamp_s,
                    last_host_timestamp_s=now,
                )
            return True, None

        last_phone_ts = pose_gate["last_phone_timestamp_s"]
        dt = None
        if phone_timestamp_s is not None and last_phone_ts is not None:
            dt = float(phone_timestamp_s) - float(last_phone_ts)
        if dt is None or dt <= 0.0:
            last_host_ts = pose_gate["last_host_timestamp_s"] or now
            dt = now - float(last_host_ts)
        dt = max(float(args.pose_gate_min_dt_s), float(dt))
        dist = distance_m(last, pose)
        yaw_delta = yaw_error_deg(pose.yaw_deg, last.yaw_deg)
        max_speed = float(args.pose_gate_max_speed_mps)
        max_yaw_rate = float(args.pose_gate_max_yaw_rate_deg_s)
        if max_speed > 0.0 and dist / dt > max_speed:
            return (
                False,
                f"{source} pose jump {dist:.3f} m in {dt:.3f}s "
                f"({dist / dt:.2f} m/s > {max_speed:.2f})",
            )
        if max_yaw_rate > 0.0 and yaw_delta / dt > max_yaw_rate:
            return (
                False,
                f"{source} yaw jump {yaw_delta:.1f} deg in {dt:.3f}s "
                f"({yaw_delta / dt:.1f} deg/s > {max_yaw_rate:.1f})",
            )
        if update:
            pose_gate.update(
                last_pose=pose,
                last_phone_timestamp_s=phone_timestamp_s,
                last_host_timestamp_s=now,
            )
        return True, None

    def _record_pose_history(
        *,
        source: str,
        phone_timestamp_s: float | None,
        observed: Pose,
        published: Pose,
        accepted: bool,
        reject_reason: str | None = None,
    ) -> None:
        pose_history_fh.write(json.dumps({
            "seq": pose_seq["n"],
            "timestamp_s": time.time(),
            "phone_timestamp_s": phone_timestamp_s,
            "source": source,
            "observed_pose": observed.to_dict(),
            "published_pose": published.to_dict(),
            "turn_lock_active": bool(turn_lock_active),
            "accepted": bool(accepted),
            "reject_reason": reject_reason,
        }, separators=(",", ":")) + "\n")
        pose_history_fh.flush()

    def _publish_rover_pose(
        observed: Pose,
        *,
        source: str,
        phone_timestamp_s: float | None = None,
    ) -> bool:
        """Publish a twin-frame rover pose for Isaac, after sanity gating."""
        if not (localize_enabled and aligned):
            return False
        pose = (
            Pose(twin_start_pose.x, twin_start_pose.y, observed.yaw_deg)
            if turn_lock_active else observed
        )
        accepted, reason = _pose_gate_check(
            pose, phone_timestamp_s, source=source, update=True)
        if not accepted:
            pose_gate["rejects"] = int(pose_gate["rejects"]) + 1
            _record_pose_history(
                source=source,
                phone_timestamp_s=phone_timestamp_s,
                observed=observed,
                published=pose,
                accepted=False,
                reject_reason=reason,
            )
            now_mono = time.monotonic()
            if now_mono - float(pose_gate["last_reject_log_s"]) > 2.0:
                pose_gate["last_reject_log_s"] = now_mono
                print(f"  (live pose rejected: {reason})", flush=True)
            update_session(args.output, tracking_normal=False, pose_reject_reason=reason)
            return False
        pose_seq["n"] += 1
        write_rover_pose(args.output, pose, seq=pose_seq["n"], timestamp_s=time.time())
        _record_pose_history(
            source=source,
            phone_timestamp_s=phone_timestamp_s,
            observed=observed,
            published=pose,
            accepted=True,
        )
        update_session(args.output, tracking_normal=True, last_pose_seq=pose_seq["n"])
        return True

    def _publish_rover_pose_from_arkit(
        camera_transform, phone_timestamp_s: float | None = None
    ) -> bool:
        if not (localize_enabled and aligned):
            return False
        try:
            observed = rover_pose_from_camera(camera_transform, calibration, feeder.t_align)
        except Exception as exc:  # noqa: BLE001 - a bad pose frame must not kill mapping
            print(f"  (rover pose publish skipped: {exc})", flush=True)
            return False
        return _publish_rover_pose(
            observed, source="arkit_phone_pose", phone_timestamp_s=phone_timestamp_s)

    def _publish_rover_pose_from_twin_camera_pose(
        camera_to_twin, phone_timestamp_s: float | None = None
    ) -> bool:
        if not (localize_enabled and aligned):
            return False
        try:
            observed = rover_pose_from_twin_camera_matrix(camera_to_twin, calibration)
        except Exception as exc:  # noqa: BLE001
            print(f"  (cuVSLAM rover pose publish skipped: {exc})", flush=True)
            return False
        return _publish_rover_pose(
            observed, source="cuvslam_keyframe", phone_timestamp_s=phone_timestamp_s)

    def _arkit_geometry_pose_healthy(
        camera_transform, phone_timestamp_s: float | None, source: str
    ) -> bool:
        if not (localize_enabled and aligned):
            return True
        try:
            observed = rover_pose_from_camera(camera_transform, calibration, feeder.t_align)
            pose = (
                Pose(twin_start_pose.x, twin_start_pose.y, observed.yaw_deg)
                if turn_lock_active else observed
            )
            accepted, reason = _pose_gate_check(
                pose, phone_timestamp_s, source=source, update=False)
        except Exception as exc:  # noqa: BLE001
            reason = f"{source} pose check failed: {exc}"
            accepted = False
        if accepted:
            return True
        pose_gate["rejects"] = int(pose_gate["rejects"]) + 1
        now_mono = time.monotonic()
        if now_mono - float(pose_gate["last_reject_log_s"]) > 2.0:
            pose_gate["last_reject_log_s"] = now_mono
            print(f"  (geometry frame skipped: {reason})", flush=True)
        update_session(args.output, tracking_normal=False, pose_reject_reason=reason)
        return False

    def _backproject_sample(depth_m, intr, pose, stride: int) -> np.ndarray:
        depth_sample = depth_m[::stride, ::stride]
        if depth_sample.size == 0:
            return np.zeros((0, 3), dtype=np.float32)
        vv, uu = np.nonzero(np.isfinite(depth_sample) & (depth_sample > 0.0))
        if len(vv) == 0:
            return np.zeros((0, 3), dtype=np.float32)
        z = depth_sample[vv, uu].astype(np.float64)
        u_full = uu.astype(np.float64) * stride
        v_full = vv.astype(np.float64) * stride
        x = (u_full - intr.cx) / intr.fx * z
        y = (v_full - intr.cy) / intr.fy * z
        cam = np.stack([x, y, z], axis=1)
        pose64 = np.asarray(pose, dtype=np.float64).reshape(4, 4)
        world = cam @ pose64[:3, :3].T + pose64[:3, 3]
        return world.astype(np.float32)

    def _record_eval_frame(frame, pose, depth_m, intr) -> None:
        frame_index = integrated + 1
        eval_trajectory.append({
            "frame_index": frame_index,
            "message_type": int(frame.message_type),
            "phone_timestamp_s": float(frame.timestamp_s),
            "pose": np.asarray(pose, dtype=float).reshape(4, 4).tolist(),
        })
        stride = max(0, int(args.eval_sample_stride))
        if stride <= 0 or args.eval_max_samples <= 0:
            return
        pts = _backproject_sample(depth_m, intr, pose, stride)
        if len(pts) == 0:
            return
        max_samples = max(1, int(args.eval_max_samples))
        if eval_sample_count["n"] + len(pts) > max_samples:
            keep_every = int(np.ceil((eval_sample_count["n"] + len(pts)) / max_samples))
            pts = pts[::keep_every]
        if len(pts) == 0:
            return
        eval_sample_points.append(pts)
        eval_sample_frame_indices.append(
            np.full((len(pts),), frame_index, dtype=np.int32))
        eval_sample_count["n"] += len(pts)

    def _write_eval_artifacts() -> None:
        if eval_trajectory:
            tmp = trajectory_path.with_name(trajectory_path.name + ".tmp")
            with tmp.open("w", encoding="utf-8") as fh:
                for record in eval_trajectory:
                    fh.write(json.dumps(record, separators=(",", ":")) + "\n")
            os.replace(tmp, trajectory_path)
        if eval_sample_points:
            points = np.concatenate(eval_sample_points, axis=0)
            frame_indices = np.concatenate(eval_sample_frame_indices, axis=0)
            if len(points) > int(args.eval_max_samples) > 0:
                step = int(np.ceil(len(points) / int(args.eval_max_samples)))
                points = points[::step]
                frame_indices = frame_indices[::step]
            tmp = eval_samples_path.with_name(eval_samples_path.name + ".tmp")
            with open(tmp, "wb") as fh:
                np.savez(fh, points=points, frame_indices=frame_indices)
            os.replace(tmp, eval_samples_path)

    def _warn_resume_readonly_geometry() -> None:
        nonlocal warned_resume_readonly_geometry
        if warned_resume_readonly_geometry:
            return
        warned_resume_readonly_geometry = True
        print(
            "  resume-session read-only: ignoring incoming geometry because no "
            "nvblox TSDF was restored; rover pose updates still publish.",
            flush=True,
        )

    def _warn_integration_paused() -> None:
        nonlocal warned_integration_paused
        if warned_integration_paused:
            return
        warned_integration_paused = True
        print(
            "  map integration paused: using incoming frames for localization only; "
            "TSDF/tiles will not be updated.",
            flush=True,
        )

    try:
        while not stop["flag"]:
            if fill_request["pending"]:
                # One-shot under-rover fill requested over the control plane
                # (the 360 driver calls this after Stop Scan). Run it when the
                # stream is idle so it closes the final, settled map.
                fill_request["pending"] = False
                if integrated:
                    print("  fill-under-rover requested; closing the dead zone.",
                          flush=True)
                    _save_outputs(final=True, force_fill=True)
                    _save_nvblox_map("fill-under-rover")
                else:
                    print("  fill-under-rover requested but nothing integrated yet; "
                          "ignored.", flush=True)
            try:
                frame = frames.get(timeout=0.25)
            except queue.Empty:
                # In replay, once the scanlog is drained there's no more input --
                # stop instead of waiting forever for a phone that isn't there.
                if getattr(receiver, "replay_exhausted", False):
                    print("  replay exhausted; finishing.", flush=True)
                    break
                continue
            seen[frame.message_type] = seen.get(frame.message_type, 0) + 1
            # Camera-preview cache: unconditional (not gated on `aligned`/pause)
            # so a dashboard preview works before the scan is aligned or while
            # integration is paused -- it's just the latest phone data.
            if frame.is_keyframe_image and frame.image_bytes:
                if camera_frame["jpeg"] is None:
                    print(f"  first phone camera keyframe received "
                          f"({len(frame.image_bytes)} bytes, type={frame.message_type})",
                          flush=True)
                camera_frame["jpeg"] = frame.image_bytes
                camera_frame["jpeg_received_s"] = time.time()
            if frame.depth_bytes and frame.depth_width and frame.depth_height:
                camera_frame["depth"] = (
                    frame.depth_bytes, int(frame.depth_width), int(frame.depth_height))
                camera_frame["depth_received_s"] = time.time()
            t_proc0 = time.perf_counter()
            integrated_before = integrated
            if frame.is_phone_pose:
                # Rover-localization pose; the first normal one fixes T_align, and
                # every subsequent normal one updates the twin-frame rover pose the
                # Isaac twin tracks.
                if frame.tracking_state == TRACKING_NORMAL:
                    if align_enabled and not aligned:
                        ready, why = align_gate.add(frame.anchor_transform, calibration)
                        if ready:
                            _establish_alignment(
                                frame.anchor_transform, f"initialized after {why}")
                        elif pose_seq["n"] == 0:
                            print(f"  alignment pending: {why}", flush=True)
                    if live_cuvslam_tracker is None:
                        _publish_rover_pose_from_arkit(
                            frame.anchor_transform, frame.timestamp_s)
            elif frame.is_realign:
                # Reset Position: re-anchor on a fresh stable normal pose, not a
                # possibly stale pose received before the operator hit Reset.
                if align_enabled:
                    aligned = False
                    align_gate.reset()
                    print("  reset position: waiting for a fresh stable normal pose to re-anchor",
                          flush=True)
            elif frame.is_depth_frame:
                # Hold off until aligned so geometry doesn't land in the wrong frame.
                if not aligned:
                    pass
                elif resume_readonly:
                    _warn_resume_readonly_geometry()
                elif integration_control["paused"]:
                    dropped += 1
                    _warn_integration_paused()
                elif not _arkit_geometry_pose_healthy(
                    frame.anchor_transform, frame.timestamp_s, "arkit_depth_frame"
                ):
                    dropped += 1
                elif not _should_integrate_high_rate(frame):
                    pass
                elif args.max_backlog and frames.qsize() > args.max_backlog:
                    # Depth-only streams are older phone builds. For RGB-D keyframes
                    # we preserve geometry by default; this branch only protects
                    # legacy high-rate depth-only streams from unbounded backlog.
                    dropped += 1
                    if not warned_dropping:
                        warned_dropping = True
                        print("  (host behind live stream: dropping legacy depth-only "
                              "frames; RGB-D geometry frames are preserved)", flush=True)
                elif (
                    corrector is not None
                    or turn_lock_active
                    or replay_imu_closure is not None
                    or replay_cuvslam is not None
                ):
                    try:
                        from digital_twin.nvblox_bridge import CameraIntrinsics
                        from digital_twin.nvblox_feeder import (
                            apply_confidence_mask, arkit_camera_pose_to_nvblox,
                            decode_depth_map, turn_in_place_camera_pose_to_nvblox,
                        )
                        depth = apply_confidence_mask(
                            decode_depth_map(frame.depth_bytes, frame.depth_width,
                                             frame.depth_height),
                            frame.confidence_bytes, args.min_confidence)
                        depth_intr = CameraIntrinsics(
                            fx=frame.intrinsics[0], fy=frame.intrinsics[1],
                            cx=frame.intrinsics[2], cy=frame.intrinsics[3],
                            width=int(frame.depth_width), height=int(frame.depth_height))
                        base_pose = None
                        if replay_cuvslam is not None:
                            base_pose = replay_cuvslam.pose_for_frame(frame)
                        if base_pose is None and replay_imu_closure is not None:
                            base_pose = replay_imu_closure.pose_for_frame(frame)
                        if base_pose is None:
                            base_pose = (
                                turn_in_place_camera_pose_to_nvblox(
                                    frame.anchor_transform, calibration, feeder.t_align,
                                    twin_start_pose, args.twin_start_z)
                                if turn_lock_active else
                                arkit_camera_pose_to_nvblox(
                                    frame.anchor_transform, feeder.t_align)
                            )
                        corrected = (
                            corrector.process(
                                depth, depth_intr, base_pose, time.monotonic())
                            if corrector is not None else base_pose
                        )
                        if corrector is not None:
                            _log_icp_if_attempted()
                        feeder.integrate_at_pose(depth, depth_intr, corrected)
                        _record_eval_frame(frame, corrected, depth, depth_intr)
                        integrated += 1
                    except Exception as exc:  # noqa: BLE001 - bad frame shouldn't kill the map
                        print(f"  (depth frame skipped: {exc})", flush=True)
                else:
                    try:
                        from digital_twin.nvblox_bridge import CameraIntrinsics
                        from digital_twin.nvblox_feeder import (
                            apply_confidence_mask, arkit_camera_pose_to_nvblox,
                            decode_depth_map,
                        )
                        depth = apply_confidence_mask(
                            decode_depth_map(frame.depth_bytes, frame.depth_width,
                                             frame.depth_height),
                            frame.confidence_bytes, args.min_confidence)
                        depth_intr = CameraIntrinsics(
                            fx=frame.intrinsics[0], fy=frame.intrinsics[1],
                            cx=frame.intrinsics[2], cy=frame.intrinsics[3],
                            width=int(frame.depth_width), height=int(frame.depth_height))
                        pose = arkit_camera_pose_to_nvblox(
                            frame.anchor_transform, feeder.t_align)
                        feeder.integrate_at_pose(depth, depth_intr, pose)
                        _record_eval_frame(frame, pose, depth, depth_intr)
                        integrated += 1
                    except Exception as exc:  # noqa: BLE001
                        print(f"  (depth frame skipped: {exc})", flush=True)
            elif frame.is_keyframe_image and aligned:
                # ARKit path. The continuous-RGB-D app streams MSG_KEYFRAME_RGBD
                # (no separate MSG_DEPTH_FRAME), so integrate its DEPTH as geometry
                # here (+ optional color); a plain image keyframe is color-only.
                if resume_readonly:
                    _warn_resume_readonly_geometry()
                    continue
                if integration_control["paused"]:
                    if (
                        live_cuvslam_tracker is not None
                        and frame.message_type == MSG_KEYFRAME_RGBD
                        and frame.depth_bytes
                    ):
                        try:
                            base_pose = live_cuvslam_tracker.track_frame(
                                frame, feeder.t_align)
                            if base_pose is not None and localize_enabled:
                                _publish_rover_pose_from_twin_camera_pose(
                                    base_pose, frame.timestamp_s)
                        except Exception as exc:  # noqa: BLE001
                            print(f"  (localization-only keyframe skipped: {exc})",
                                  flush=True)
                    dropped += 1
                    _warn_integration_paused()
                    continue
                if live_cuvslam_tracker is None and not _arkit_geometry_pose_healthy(
                    frame.anchor_transform, frame.timestamp_s, "arkit_keyframe"
                ):
                    dropped += 1
                    continue
                try:
                    want_color = not args.no_color
                    if not _should_integrate_high_rate(frame):
                        continue
                    fuse_color = (
                        want_color and args.color_every > 0
                        and (integrated % args.color_every == 0)
                    )
                    rgb = (decode_jpeg_rgb(frame.image_bytes)
                           if fuse_color and frame.image_bytes else None)
                    if frame.message_type == MSG_KEYFRAME_RGBD and frame.depth_bytes:
                        if (
                            corrector is not None
                            or turn_lock_active
                            or replay_imu_closure is not None
                            or replay_cuvslam is not None
                            or live_cuvslam_tracker is not None
                        ):
                            from digital_twin.nvblox_bridge import CameraIntrinsics
                            from digital_twin.nvblox_feeder import (
                                apply_confidence_mask, arkit_camera_pose_to_nvblox,
                                decode_depth_map, scale_intrinsics_to_depth,
                                turn_in_place_camera_pose_to_nvblox,
                            )
                            depth = apply_confidence_mask(
                                decode_depth_map(frame.depth_bytes, frame.depth_width,
                                                 frame.depth_height),
                                frame.confidence_bytes, args.min_confidence)
                            depth_intr = scale_intrinsics_to_depth(
                                frame.intrinsics, frame.image_width, frame.image_height,
                                frame.depth_width, frame.depth_height)
                            base_pose = None
                            if live_cuvslam_tracker is not None:
                                base_pose = live_cuvslam_tracker.track_frame(
                                    frame, feeder.t_align)
                                if base_pose is None:
                                    print("  (keyframe skipped: cuVSLAM did not return a pose)",
                                          flush=True)
                                    continue
                                if localize_enabled and not _publish_rover_pose_from_twin_camera_pose(
                                    base_pose, frame.timestamp_s
                                ):
                                    dropped += 1
                                    continue
                            if base_pose is None and replay_cuvslam is not None:
                                base_pose = replay_cuvslam.pose_for_frame(frame)
                            if base_pose is None and replay_imu_closure is not None:
                                base_pose = replay_imu_closure.pose_for_frame(frame)
                            if base_pose is None:
                                base_pose = (
                                    turn_in_place_camera_pose_to_nvblox(
                                        frame.anchor_transform, calibration, feeder.t_align,
                                        twin_start_pose, args.twin_start_z)
                                        if turn_lock_active else
                                    arkit_camera_pose_to_nvblox(
                                        frame.anchor_transform, feeder.t_align)
                                )
                            corrected = (
                                corrector.process(
                                    depth, depth_intr, base_pose, time.monotonic())
                                if corrector is not None else base_pose
                            )
                            if corrector is not None:
                                _log_icp_if_attempted()
                            cintr = None if rgb is None else CameraIntrinsics(
                                fx=frame.intrinsics[0], fy=frame.intrinsics[1],
                                cx=frame.intrinsics[2], cy=frame.intrinsics[3],
                                width=int(frame.image_width), height=int(frame.image_height))
                            feeder.integrate_at_pose(
                                depth, depth_intr, corrected, rgb=rgb, color_intr=cintr)
                            _record_eval_frame(frame, corrected, depth, depth_intr)
                            integrated += 1
                            if rgb is not None:
                                colored += 1
                        else:
                            from digital_twin.nvblox_bridge import CameraIntrinsics
                            from digital_twin.nvblox_feeder import (
                                apply_confidence_mask, arkit_camera_pose_to_nvblox,
                                decode_depth_map, scale_intrinsics_to_depth,
                            )
                            depth = apply_confidence_mask(
                                decode_depth_map(frame.depth_bytes, frame.depth_width,
                                                 frame.depth_height),
                                frame.confidence_bytes, args.min_confidence)
                            depth_intr = scale_intrinsics_to_depth(
                                frame.intrinsics, frame.image_width, frame.image_height,
                                frame.depth_width, frame.depth_height)
                            pose = arkit_camera_pose_to_nvblox(
                                frame.anchor_transform, feeder.t_align)
                            cintr = None if rgb is None else CameraIntrinsics(
                                fx=frame.intrinsics[0], fy=frame.intrinsics[1],
                                cx=frame.intrinsics[2], cy=frame.intrinsics[3],
                                width=int(frame.image_width), height=int(frame.image_height))
                            feeder.integrate_at_pose(
                                depth, depth_intr, pose, rgb=rgb, color_intr=cintr)
                            _record_eval_frame(frame, pose, depth, depth_intr)
                            integrated += 1
                            if rgb is not None:
                                colored += 1
                    elif rgb is not None and feeder.feed_keyframe_color(frame, rgb=rgb):
                        colored += 1
                except Exception as exc:  # noqa: BLE001 - bad frame shouldn't kill the map
                    print(f"  (keyframe skipped: {exc})", flush=True)
            elif frame.is_finalize:
                # Stop Scan. Don't exit if we never got depth -- keep listening so
                # a re-scan (or a newer build) can still feed the map.
                print(f"  finalize received (so far: {_tally_str()})", flush=True)
                if integrated and not args.resumable_scan:
                    break
                if integrated and args.resumable_scan and not scan_paused:
                    # Resumable mode: finalize + save the current map but stay alive
                    # with the TSDF intact. A later Start Scan (the phone resumes
                    # streaming in the same ARKit world frame) integrates straight
                    # into this same map -- the scan picks up where it left off.
                    _save_outputs(final=True)
                    _save_nvblox_map("scan pause")
                    _write_eval_artifacts()
                    scan_paused = True
                    if turn_lock_active:
                        turn_lock_active = False
                        print(
                            "  turn-in-place lock released; subsequent resumed scans "
                            "will use live rover translation.",
                            flush=True,
                        )
                    update_session(args.output, scanning=False)
                    print("  scan paused; map finalized and saved. Send Start Scan "
                          "(or POST /scan/start) to resume into the SAME map.",
                          flush=True)

            if scan_paused and integrated > integrated_before:
                # First frame after a resumable-scan pause integrated into the
                # retained TSDF: the operator restarted the scan and it picked up
                # in the same map/frame.
                scan_paused = False
                update_session(args.output, scanning=True)
                print("  scan resumed into the existing map (continuing where it "
                      "left off).", flush=True)

            integ_timer["ms"] += (time.perf_counter() - t_proc0) * 1e3
            integ_timer["n"] += 1

            now = time.monotonic()
            # While paused (resumable-scan, post-finalize) the phone still streams
            # localization poses, but no new geometry integrates -- skip the
            # incremental re-extract/save so we don't spin the GPU on a static map.
            if integrated and not scan_paused and now - last_mesh_save >= args.mesh_interval:
                interval_s = now - last_mesh_save
                fps = (integrated - last_logged_integrated) / interval_s if interval_s else 0.0
                busy_s = integ_timer["ms"] / 1e3   # per-frame CPU work this interval
                per_msg = integ_timer["ms"] / max(1, integ_timer["n"])
                in_bytes = getattr(receiver, "bytes_received", 0)
                in_mbps = (in_bytes - last_bytes) / interval_s / 1e6 if interval_s else 0.0
                _save_outputs()
                total_dropped = dropped + getattr(receiver, "frames_dropped", 0)
                drop_note = f" dropped={total_dropped}" if total_dropped else ""
                # Triage line. in= is inbound wire bandwidth: if it's flat across
                # runs and the fps tracks it, we're link/phone-bound (the host is
                # idle -- busy_s<<interval_s, qsize~0). If qsize pegs and busy_s ~=
                # interval_s instead, the host is the limiter.
                print(f"  depth={integrated} color={colored}{drop_note} -> "
                      f"{write_stats['written']}/{write_stats['tiles']} tiles | "
                      f"{fps:.1f} integ fps in={in_mbps:.2f}MB/s qsize={frames.qsize()} | "
                      f"proc={per_msg:.1f}ms/msg (busy {busy_s:.1f}/{interval_s:.1f}s) | "
                      f"save={save_timing['total_ms']:.0f}ms (extract="
                      f"{save_timing['extract_ms']:.0f} settarget="
                      f"{save_timing['settarget_ms']:.0f}, {save_timing['verts'] // 1000}k v)",
                      flush=True)
                last_mesh_save = now
                last_logged_integrated = integrated
                last_bytes = in_bytes
                integ_timer["ms"] = 0.0
                integ_timer["n"] = 0
    finally:
        if control_server is not None:
            control_server.shutdown()
            control_server.server_close()
        MapperControlHandler.receiver = None
        MapperControlHandler.fill_request = None
        MapperControlHandler.integration_control = None
        MapperControlHandler.camera_frame = None
        total_dropped = dropped + getattr(receiver, "frames_dropped", 0)
        drop_note = f"; dropped stale/high-rate={total_dropped}" if total_dropped else ""
        print(f"received from phone: {_tally_str()}{drop_note}", flush=True)
        # Inbound bandwidth breakdown: which streams owned the link. Anything the
        # nvblox path discards (e.g. mesh_anchor) here is pure waste to cut.
        by_type = getattr(receiver, "bytes_by_type", {})
        if by_type:
            total_mb = sum(by_type.values()) / 1e6
            parts = ", ".join(
                f"{_MSG_LABELS.get(mt, f'type{mt}')}={mb / 1e6:.1f}MB ({mb / sum(by_type.values()) * 100:.0f}%)"
                for mt, mb in sorted(by_type.items(), key=lambda kv: -kv[1])
            )
            print(f"inbound bytes: {total_mb:.1f}MB total -> {parts}", flush=True)
        if live_cuvslam_tracker is not None:
            stats = live_cuvslam_tracker.stats
            print(
                "cuVSLAM online: "
                f"valid={stats.valid_poses}/{stats.frames} "
                f"dropped={stats.dropped_poses} "
                f"yaw_unwrapped={stats.final_yaw_y_unwrapped_deg:.1f}deg "
                f"translation={stats.final_translation_norm_m:.3f}m",
                flush=True,
            )
        if int(pose_gate["rejects"]):
            print(f"live pose gate: rejected {int(pose_gate['rejects'])} frame(s)",
                  flush=True)
        if integrated:
            _save_outputs(final=True)
            _write_eval_artifacts()
            _save_nvblox_map("exit")
            print(f"done: depth={integrated} color={colored} dropped={dropped} -> "
                  f"{write_stats['tiles']} tiles in {args.output} "
                  f"(+ {mesh_path.name}/.npz, trajectory.jsonl, "
                  f"eval_surface_samples.npz)", flush=True)
        elif seen.get(MSG_DEPTH_FRAME) and align_enabled and not aligned:
            print("depth frames arrived but T_align was never established (no "
                  "normal-tracking phone pose). Move the phone to recover tracking "
                  "and tap Reset Position, or rerun with --no-align.", flush=True)
        elif seen:
            print("no DEPTH frames arrived, but other messages did -> the phone is "
                  "connected but NOT on a depth-streaming build. Sideload the latest "
                  "ios/LiveScannerTheos/packages/*.ipa (continuous RGB-D).", flush=True)
        else:
            print("nothing received (phone never connected / not scanning).", flush=True)
        if args.start_scan:
            receiver.send_control(MSG_CONTROL_STOP_SCAN)
        if getattr(args, "record_imu", False) and not args.replay:
            receiver.send_control(MSG_CONTROL_DISABLE_IMU)
        receiver.stop()
        if recorder is not None:
            recorder.close()
            print(f"recorded {recorder.records_written} frames "
                  f"({recorder.bytes_written / 1e6:.1f}MB) -> {recorder.path}", flush=True)
        pose_history_fh.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
