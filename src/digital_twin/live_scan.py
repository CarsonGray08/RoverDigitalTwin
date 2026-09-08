from __future__ import annotations

import queue
import socket
import struct
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Iterable


PROTOCOL_MAGIC = 0x44544C53  # "DTLS" / Digital Twin Live Scan
PROTOCOL_VERSION = 1
MSG_UPSERT_ANCHOR = 1
MSG_REMOVE_ANCHOR = 2
MSG_HEARTBEAT = 3
# Carries one posed JPEG keyframe for the offline texture bake. The header's
# 16-byte anchor-id field is reused as the image id; vertex/face counts are 0
# and the header transform carries the camera-to-world pose. Payload after the
# header: float32 fx, fy, cx, cy (intrinsics in JPEG pixels), uint32 width,
# uint32 height, uint32 jpeg_len, then jpeg_len bytes.
MSG_KEYFRAME_IMAGE = 4
# Sent from the app's Stop Scan button. Zero-length body; tells the mapper to
# finalize and save the map.
MSG_FINALIZE = 5
# Like MSG_KEYFRAME_IMAGE, but appends a contiguous float32 depth map and
# optional uint8 confidence map after the JPEG. The mapper uses these frames
# for cuVSLAM localization and nvblox fusion; the resulting terrain mesh is
# used for planning and collision checks.
MSG_KEYFRAME_RGBD = 6
# Continuous rover-localization pose stream from the phone. The header transform
# carries T_A_C (camera-to-ARKit-world, row-major, as the phone sends camera
# transforms), the header timestamp is the phone clock (CACurrentMediaTime
# seconds), and vertex/face counts are 0. Payload after the header:
# uint32 tracking_state (see TRACKING_*), uint32 sequence. The host stores
# the latest sample and converts it into a
# rover pose downstream.
MSG_PHONE_POSE = 7
# Host -> phone scan control messages. Zero-length body, header only, like
# MSG_FINALIZE. IMU control messages below also travel host -> phone.
# The phone reads these on the same socket it streams over and toggles capture:
#   START: begin/resume mesh + keyframe streaming.
#   STOP:  stop streaming and emit a MSG_FINALIZE (the remote equivalent of the
#          on-screen Stop Scan button), so the mapper saves the final map.
MSG_CONTROL_START_SCAN = 8
MSG_CONTROL_STOP_SCAN = 9
# Phone -> host "reset position": re-anchor the shared ARKit->twin alignment so
# the phone's *current* pose becomes the rover's twin start pose. Sent from the
# app's Reset Position button after the phone is mounted on the rover, so the
# operator can zero the origin in the mounted pose. Zero-length body, header
# only, like MSG_FINALIZE. The host re-runs compute_t_align from the current
# pose and re-registers any mesh already received.
MSG_REALIGN = 10
# Phone -> host continuous depth frame for the nvblox mapper.
# Unlike MSG_KEYFRAME_RGBD this carries no JPEG: it is
# depth-only so the phone can stream LiDAR depth at its native cadence (~10-30Hz)
# for dense TSDF fusion, decoupled from the heavier, motion-gated texture
# keyframes. Header transform = camera-to-ARKit-world pose; the body is depth-
# resolution intrinsics + a float32 depth map (meters) + optional uint8 ARKit
# confidence (0 low / 1 medium / 2 high).
MSG_DEPTH_FRAME = 11
# IMU sample for cuVSLAM inertial odometry: body is 6 float32
# (accel x/y/z m/s^2 incl. gravity, gyro x/y/z rad/s); header timestamp = sample time.
MSG_IMU = 12
# Host -> phone control messages for explicit visual-inertial test capture.
# Normal production scans leave IMU off; RECORD_IMU/--record-imu sends ENABLE
# when collecting a .scanlog for cuVSLAM experiments, then DISABLE on shutdown.
MSG_CONTROL_ENABLE_IMU = 13
MSG_CONTROL_DISABLE_IMU = 14
# flags bitfield (header field at offset 6).
#   FLAG_HAS_NORMALS: payload carries a per-vertex float32 normal block
#     (vertex_count * 3 floats) appended after the color block.
#   FLAG_TEXTURED: the anchor is textured from a keyframe image. After the
#     color (and normal) blocks the payload appends a 16-byte image id followed
#     by per-vertex float32 UVs (vertex_count * 2), already in texture space
#     (s = u/W, t = 1 - v/H). Colors are still sent as a fallback.
# Older senders leave flags 0 and omit both blocks.
FLAG_HAS_NORMALS = 0x0001
FLAG_TEXTURED = 0x0002
MAX_MESSAGE_BYTES = 64 * 1024 * 1024

# ARKit ARCamera.TrackingState, flattened. Only `normal` is trusted for
# high-confidence localization (plan: Drift And Tracking Safety).
TRACKING_NOT_AVAILABLE = 0
TRACKING_LIMITED = 1
TRACKING_NORMAL = 2

_LENGTH_STRUCT = struct.Struct("<I")
_HEADER_STRUCT = struct.Struct("<IBBH16sd16fII")
# fx, fy, cx, cy (JPEG-pixel intrinsics), width, height, jpeg_len
_IMAGE_HEADER_STRUCT = struct.Struct("<ffffIII")
# depth_width, depth_height, depth_byte_len, confidence_byte_len
_RGBD_HEADER_STRUCT = struct.Struct("<IIII")
# MSG_DEPTH_FRAME body: fx, fy, cx, cy (DEPTH-pixel intrinsics), depth_width,
# depth_height, depth_byte_len, confidence_byte_len.
_DEPTH_FRAME_STRUCT = struct.Struct("<ffffIIII")
# tracking_state, sequence (MSG_PHONE_POSE payload)
_PHONE_POSE_STRUCT = struct.Struct("<II")
# accel x/y/z, gyro x/y/z (MSG_IMU payload)
_IMU_STRUCT = struct.Struct("<ffffff")

UV2 = tuple[float, float]

Matrix4 = tuple[float, ...]
Point3 = tuple[float, float, float]
Color3ub = tuple[int, int, int]
Face3 = tuple[int, int, int]


class LiveScanProtocolError(ValueError):
    """Raised when a live-scan socket message is malformed."""


@dataclass(frozen=True)
class LiveMeshAnchorFrame:
    anchor_id: str
    timestamp_s: float
    anchor_transform: Matrix4 = field(default_factory=lambda: identity_matrix4())
    vertices: tuple[Point3, ...] = ()
    faces: tuple[Face3, ...] = ()
    colors: tuple[Color3ub, ...] = ()
    normals: tuple[Point3, ...] = ()
    # Texture projection (FLAG_TEXTURED). texture_image_id references a keyframe
    # delivered by a MSG_KEYFRAME_IMAGE message; uvs are per-vertex.
    texture_image_id: str | None = None
    uvs: tuple[UV2, ...] = ()
    # Only populated for MSG_KEYFRAME_IMAGE frames; anchor_id holds the image id,
    # anchor_transform holds the camera-to-world pose, and intrinsics holds
    # (fx, fy, cx, cy) in JPEG-pixel units.
    image_width: int = 0
    image_height: int = 0
    image_bytes: bytes = b""
    intrinsics: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    depth_width: int = 0
    depth_height: int = 0
    depth_bytes: bytes = b""
    confidence_bytes: bytes = b""
    # Only populated for MSG_PHONE_POSE frames. anchor_transform holds T_A_C
    # (camera-to-ARKit-world); tracking_state is one of TRACKING_*; sequence is a
    # monotonically increasing counter used to detect dropped/reordered poses.
    tracking_state: int = TRACKING_NORMAL
    sequence: int = 0
    # Only populated for MSG_IMU frames: (ax, ay, az) m/s^2, (gx, gy, gz) rad/s.
    imu_accel: tuple[float, float, float] = (0.0, 0.0, 0.0)
    imu_gyro: tuple[float, float, float] = (0.0, 0.0, 0.0)
    message_type: int = MSG_UPSERT_ANCHOR

    @property
    def removed(self) -> bool:
        return self.message_type == MSG_REMOVE_ANCHOR

    @property
    def is_keyframe_image(self) -> bool:
        return self.message_type in (MSG_KEYFRAME_IMAGE, MSG_KEYFRAME_RGBD)

    @property
    def is_finalize(self) -> bool:
        return self.message_type == MSG_FINALIZE

    @property
    def is_phone_pose(self) -> bool:
        return self.message_type == MSG_PHONE_POSE

    @property
    def is_realign(self) -> bool:
        return self.message_type == MSG_REALIGN

    @property
    def is_depth_frame(self) -> bool:
        return self.message_type == MSG_DEPTH_FRAME

    @property
    def is_imu(self) -> bool:
        return self.message_type == MSG_IMU


@dataclass
class LiveScanAnchor:
    anchor_id: str
    timestamp_s: float
    received_monotonic_s: float
    vertices_twin: tuple[Point3, ...]
    faces: tuple[Face3, ...]
    colors: tuple[Color3ub, ...]
    normals_twin: tuple[Point3, ...] = ()
    texture_image_id: str | None = None
    uvs: tuple[UV2, ...] = ()


@dataclass
class LiveKeyframe:
    """A posed keyframe retained for the offline texture bake."""

    image_id: str
    jpeg: bytes
    # Camera-to-world pose (ARKit world frame), 16 floats as the phone sends them.
    camera_transform: Matrix4
    # (fx, fy, cx, cy) in JPEG-pixel units.
    intrinsics: tuple[float, float, float, float]
    width: int
    height: int
    # Optional RGB-D payload. depth_bytes is contiguous little-endian float32
    # depth_width*depth_height values in meters. confidence_bytes is optional
    # uint8 depth_width*depth_height ARKit confidence values.
    depth_width: int = 0
    depth_height: int = 0
    depth_bytes: bytes = b""
    confidence_bytes: bytes = b""
    # Sender timestamp. Used by the bake to prefer recent views (ARKit world
    # drift makes older keyframes land a few cm off the final refined mesh).
    timestamp_s: float = 0.0


@dataclass(frozen=True)
class PhonePose:
    """One sample of the phone's camera pose in the ARKit world frame."""

    # T_A_C: camera-to-ARKit-world, 16 floats row-major, as the phone sends it.
    camera_transform: Matrix4
    # Phone clock (CACurrentMediaTime seconds).
    timestamp_s: float
    tracking_state: int
    sequence: int
    # Host monotonic clock at receipt; used for staleness and clock alignment.
    received_monotonic_s: float

    @property
    def tracking_normal(self) -> bool:
        return self.tracking_state == TRACKING_NORMAL


class PhonePoseStore:
    """Holds the latest phone camera pose and the phone/host clock offset.

    Pure-Python so it is unit-testable without Isaac. The validator owns one
    instance and feeds it MSG_PHONE_POSE frames from the live-scan queue. The
    clock offset is estimated once, on the first sample, against the host
    *monotonic* clock so that ``host_time_for(phone_ts)`` lines up with the
    monotonic times the closed loop uses for command-complete association.
    """

    def __init__(self) -> None:
        self.latest: PhonePose | None = None
        # host_monotonic - phone_timestamp at the first received sample.
        self.clock_offset_s: float | None = None
        self.frames_received = 0
        self.last_sequence: int | None = None
        # Count of out-of-order/dropped sequence transitions, for diagnostics.
        self.sequence_gaps = 0

    def update(
        self,
        frame: LiveMeshAnchorFrame,
        received_monotonic_s: float | None = None,
    ) -> PhonePose:
        if not frame.is_phone_pose:
            raise ValueError("PhonePoseStore.update requires a MSG_PHONE_POSE frame")
        now_m = time.monotonic() if received_monotonic_s is None else received_monotonic_s
        if self.clock_offset_s is None:
            self.clock_offset_s = now_m - frame.timestamp_s
        if self.last_sequence is not None and frame.sequence != self.last_sequence + 1:
            self.sequence_gaps += 1
        self.last_sequence = frame.sequence
        pose = PhonePose(
            camera_transform=frame.anchor_transform,
            timestamp_s=frame.timestamp_s,
            tracking_state=frame.tracking_state,
            sequence=frame.sequence,
            received_monotonic_s=now_m,
        )
        self.latest = pose
        self.frames_received += 1
        return pose

    def host_time_for(self, phone_timestamp_s: float) -> float | None:
        """Map a phone timestamp into the host monotonic clock, or None if no
        offset has been established yet."""
        if self.clock_offset_s is None:
            return None
        return phone_timestamp_s + self.clock_offset_s

    def status(self, now_s: float | None = None) -> dict:
        now = time.monotonic() if now_s is None else now_s
        if self.latest is None:
            return {
                "enabled": True,
                "have_pose": False,
                "frames_received": self.frames_received,
                "clock_offset_s": self.clock_offset_s,
            }
        latest = self.latest
        return {
            "enabled": True,
            "have_pose": True,
            "frames_received": self.frames_received,
            "clock_offset_s": self.clock_offset_s,
            "sequence": latest.sequence,
            "sequence_gaps": self.sequence_gaps,
            "tracking_state": latest.tracking_state,
            "tracking_normal": latest.tracking_normal,
            "phone_timestamp_s": latest.timestamp_s,
            "age_s": round(max(0.0, now - latest.received_monotonic_s), 3),
            "camera_transform": list(latest.camera_transform),
        }


def identity_matrix4() -> Matrix4:
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


def transform_point_row_major(point: Point3, matrix: Matrix4) -> Point3:
    """Transform a point by a row-major, row-vector 4x4 matrix.

    This matches the convention already used by the Isaac validator when it
    converts USD matrices to NumPy arrays: translation lives in row 3.
    """

    if len(matrix) != 16:
        raise ValueError("matrix must contain 16 floats")
    x, y, z = point
    return (
        x * matrix[0] + y * matrix[4] + z * matrix[8] + matrix[12],
        x * matrix[1] + y * matrix[5] + z * matrix[9] + matrix[13],
        x * matrix[2] + y * matrix[6] + z * matrix[10] + matrix[14],
    )


def transform_direction_row_major(direction: Point3, matrix: Matrix4) -> Point3:
    """Rotate a direction by a row-major, row-vector 4x4 matrix.

    Same as :func:`transform_point_row_major` but without the translation row
    (indices 12-14), so normals/directions are only rotated, not translated.
    """

    if len(matrix) != 16:
        raise ValueError("matrix must contain 16 floats")
    x, y, z = direction
    return (
        x * matrix[0] + y * matrix[4] + z * matrix[8],
        x * matrix[1] + y * matrix[5] + z * matrix[9],
        x * matrix[2] + y * matrix[6] + z * matrix[10],
    )


def arkit_world_to_usd_point(point: Point3) -> Point3:
    """Convert ARKit Y-up world axes to this USD stage's Z-up axes.

    ARKit world: +X right, +Y up, camera looks toward -Z.
    USD/twin: +Z up. We preserve ARKit X, turn ARKit forward/back into USD Y,
    and map ARKit up into USD Z.
    """

    x, y, z = point
    return (x, -z, y)


def transform_vertices_to_twin(
    vertices: Iterable[Point3],
    anchor_transform: Matrix4,
    arkit_to_twin: Matrix4 | None = None,
) -> tuple[Point3, ...]:
    registration = arkit_to_twin or identity_matrix4()
    return tuple(
        transform_point_row_major(
            arkit_world_to_usd_point(transform_point_row_major(vertex, anchor_transform)),
            registration,
        )
        for vertex in vertices
    )


def transform_normals_to_twin(
    normals: Iterable[Point3],
    anchor_transform: Matrix4,
    arkit_to_twin: Matrix4 | None = None,
) -> tuple[Point3, ...]:
    """Map per-vertex normals from the ARMeshAnchor local frame into the twin.

    Normals follow the same pipeline as vertices (anchor transform -> ARKit
    Y-up to USD Z-up axis swap -> registration) but as *directions*: the
    rotation parts only, no translation. The ARKit anchor transform and the
    registration are rigid, so the rotation itself maps normals correctly; we
    renormalize to absorb numerical drift. The axis swap (x, y, z) -> (x, -z, y)
    is a pure rotation, so it applies unchanged to directions.
    """

    registration = arkit_to_twin or identity_matrix4()
    out: list[Point3] = []
    for normal in normals:
        rotated = transform_direction_row_major(
            arkit_world_to_usd_point(transform_direction_row_major(normal, anchor_transform)),
            registration,
        )
        nx, ny, nz = rotated
        magnitude = (nx * nx + ny * ny + nz * nz) ** 0.5
        if magnitude > 0.0:
            out.append((nx / magnitude, ny / magnitude, nz / magnitude))
        else:
            out.append((0.0, 0.0, 0.0))
    return tuple(out)


def encode_frame(frame: LiveMeshAnchorFrame) -> bytes:
    """Encode a frame as one length-prefixed TCP message.

    This helper is used by tests and by the synthetic sender script. The iOS
    app should write the exact same binary layout.
    """

    anchor_uuid = uuid.UUID(frame.anchor_id)
    vertices = frame.vertices
    faces = frame.faces
    colors = frame.colors or tuple((255, 255, 255) for _ in vertices)
    if len(colors) != len(vertices):
        raise LiveScanProtocolError("colors length must match vertices length")
    if len(frame.anchor_transform) != 16:
        raise LiveScanProtocolError("anchor transform must contain 16 floats")

    normals = frame.normals
    has_normals = bool(normals)
    if has_normals and len(normals) != len(vertices):
        raise LiveScanProtocolError("normals length must match vertices length")

    textured = frame.texture_image_id is not None
    if textured and len(frame.uvs) != len(vertices):
        raise LiveScanProtocolError("uvs length must match vertices length")

    flags = 0
    if has_normals:
        flags |= FLAG_HAS_NORMALS
    if textured:
        flags |= FLAG_TEXTURED

    header = _HEADER_STRUCT.pack(
        PROTOCOL_MAGIC,
        PROTOCOL_VERSION,
        int(frame.message_type),
        flags,
        anchor_uuid.bytes,
        float(frame.timestamp_s),
        *(float(value) for value in frame.anchor_transform),
        len(vertices),
        len(faces),
    )
    vertex_payload = b"".join(struct.pack("<fff", *vertex) for vertex in vertices)
    face_payload = b"".join(struct.pack("<III", *face) for face in faces)
    color_payload = b"".join(struct.pack("<BBB", *color) for color in colors)
    normal_payload = b"".join(struct.pack("<fff", *normal) for normal in normals) if has_normals else b""
    if textured:
        texture_payload = uuid.UUID(frame.texture_image_id).bytes + b"".join(
            struct.pack("<ff", float(u), float(v)) for u, v in frame.uvs
        )
    else:
        texture_payload = b""
    payload = header + vertex_payload + face_payload + color_payload + normal_payload + texture_payload
    return _LENGTH_STRUCT.pack(len(payload)) + payload


def encode_keyframe_image(
    image_id: str,
    width: int,
    height: int,
    jpeg_bytes: bytes,
    timestamp_s: float = 0.0,
    camera_transform: Matrix4 | None = None,
    intrinsics: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
) -> bytes:
    """Encode one MSG_KEYFRAME_IMAGE message.

    The image id goes in the anchor-id slot and the camera-to-world pose in the
    header transform; intrinsics (fx, fy, cx, cy) are in JPEG-pixel units.
    """

    transform = camera_transform or identity_matrix4()
    if len(transform) != 16:
        raise LiveScanProtocolError("camera transform must contain 16 floats")
    fx, fy, cx, cy = intrinsics
    header = _HEADER_STRUCT.pack(
        PROTOCOL_MAGIC,
        PROTOCOL_VERSION,
        MSG_KEYFRAME_IMAGE,
        0,
        uuid.UUID(image_id).bytes,
        float(timestamp_s),
        *(float(value) for value in transform),
        0,
        0,
    )
    payload = header + _IMAGE_HEADER_STRUCT.pack(
        float(fx), float(fy), float(cx), float(cy), int(width), int(height), len(jpeg_bytes)
    ) + bytes(jpeg_bytes)
    return _LENGTH_STRUCT.pack(len(payload)) + payload


def encode_keyframe_rgbd(
    image_id: str,
    width: int,
    height: int,
    jpeg_bytes: bytes,
    depth_width: int,
    depth_height: int,
    depth_bytes: bytes,
    confidence_bytes: bytes = b"",
    timestamp_s: float = 0.0,
    camera_transform: Matrix4 | None = None,
    intrinsics: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
) -> bytes:
    """Encode one MSG_KEYFRAME_RGBD message.

    The RGB fields match :func:`encode_keyframe_image`; the trailing depth block
    is contiguous little-endian float32 depth values plus optional uint8
    confidence values. Depth resolution is allowed to differ from JPEG
    resolution; the bake scales projected RGB pixels into depth pixels.
    """

    expected_depth_len = int(depth_width) * int(depth_height) * 4
    if len(depth_bytes) != expected_depth_len:
        raise LiveScanProtocolError(
            f"depth bytes length must be {expected_depth_len}, got {len(depth_bytes)}"
        )
    expected_confidence_len = int(depth_width) * int(depth_height)
    if confidence_bytes and len(confidence_bytes) != expected_confidence_len:
        raise LiveScanProtocolError(
            f"confidence bytes length must be {expected_confidence_len}, got {len(confidence_bytes)}"
        )
    transform = camera_transform or identity_matrix4()
    if len(transform) != 16:
        raise LiveScanProtocolError("camera transform must contain 16 floats")
    fx, fy, cx, cy = intrinsics
    header = _HEADER_STRUCT.pack(
        PROTOCOL_MAGIC,
        PROTOCOL_VERSION,
        MSG_KEYFRAME_RGBD,
        0,
        uuid.UUID(image_id).bytes,
        float(timestamp_s),
        *(float(value) for value in transform),
        0,
        0,
    )
    payload = (
        header
        + _IMAGE_HEADER_STRUCT.pack(
            float(fx), float(fy), float(cx), float(cy), int(width), int(height), len(jpeg_bytes)
        )
        + bytes(jpeg_bytes)
        + _RGBD_HEADER_STRUCT.pack(
            int(depth_width), int(depth_height), len(depth_bytes), len(confidence_bytes)
        )
        + bytes(depth_bytes)
        + bytes(confidence_bytes)
    )
    return _LENGTH_STRUCT.pack(len(payload)) + payload


def encode_depth_frame(
    width: int,
    height: int,
    depth_bytes: bytes,
    confidence_bytes: bytes = b"",
    timestamp_s: float = 0.0,
    camera_transform: Matrix4 | None = None,
    intrinsics: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
) -> bytes:
    """Encode one MSG_DEPTH_FRAME (continuous depth, no JPEG) for nvblox fusion.

    ``intrinsics`` (fx, fy, cx, cy) are in DEPTH-pixel units -- i.e. they already
    correspond to ``width`` x ``height`` -- so the host needs no rescaling.
    ``depth_bytes`` is contiguous little-endian float32 meters; ``confidence_bytes``
    is optional uint8 ARKit confidence, one per depth pixel.
    """
    expected_depth_len = int(width) * int(height) * 4
    if len(depth_bytes) != expected_depth_len:
        raise LiveScanProtocolError(
            f"depth bytes length must be {expected_depth_len}, got {len(depth_bytes)}"
        )
    if confidence_bytes and len(confidence_bytes) != int(width) * int(height):
        raise LiveScanProtocolError(
            f"confidence bytes length must be {int(width) * int(height)}, got {len(confidence_bytes)}"
        )
    transform = camera_transform or identity_matrix4()
    if len(transform) != 16:
        raise LiveScanProtocolError("camera transform must contain 16 floats")
    fx, fy, cx, cy = intrinsics
    header = _HEADER_STRUCT.pack(
        PROTOCOL_MAGIC,
        PROTOCOL_VERSION,
        MSG_DEPTH_FRAME,
        0,
        uuid.UUID(int=0).bytes,
        float(timestamp_s),
        *(float(value) for value in transform),
        0,
        0,
    )
    payload = (
        header
        + _DEPTH_FRAME_STRUCT.pack(
            float(fx), float(fy), float(cx), float(cy),
            int(width), int(height), len(depth_bytes), len(confidence_bytes),
        )
        + bytes(depth_bytes)
        + bytes(confidence_bytes)
    )
    return _LENGTH_STRUCT.pack(len(payload)) + payload


def encode_finalize(timestamp_s: float = 0.0) -> bytes:
    """Encode one MSG_FINALIZE message (zero-length body)."""
    return _encode_header_only(MSG_FINALIZE, timestamp_s)


def encode_realign(timestamp_s: float = 0.0) -> bytes:
    """Encode one MSG_REALIGN message (zero-length body); the phone's Reset
    Position button re-anchors the twin alignment to the current pose."""
    return _encode_header_only(MSG_REALIGN, timestamp_s)


def encode_control(message_type: int, timestamp_s: float = 0.0) -> bytes:
    """Encode one host->phone control message. Header-only, same framing as
    MSG_FINALIZE."""
    if message_type not in (
        MSG_CONTROL_START_SCAN,
        MSG_CONTROL_STOP_SCAN,
        MSG_CONTROL_ENABLE_IMU,
        MSG_CONTROL_DISABLE_IMU,
    ):
        raise ValueError(f"not a control message type: {message_type}")
    return _encode_header_only(message_type, timestamp_s)


def _encode_header_only(message_type: int, timestamp_s: float) -> bytes:
    """A header-only frame (zero body): magic/version/type + zeroed fields."""
    header = _HEADER_STRUCT.pack(
        PROTOCOL_MAGIC,
        PROTOCOL_VERSION,
        message_type,
        0,
        uuid.UUID(int=0).bytes,
        float(timestamp_s),
        *identity_matrix4(),
        0,
        0,
    )
    return _LENGTH_STRUCT.pack(len(header)) + header


def encode_phone_pose(
    camera_transform: Matrix4,
    *,
    tracking_state: int = TRACKING_NORMAL,
    sequence: int = 0,
    timestamp_s: float = 0.0,
) -> bytes:
    """Encode one MSG_PHONE_POSE message.

    camera_transform is T_A_C (camera-to-ARKit-world), 16 floats row-major.
    """

    if len(camera_transform) != 16:
        raise ValueError("camera_transform must contain 16 floats")
    header = _HEADER_STRUCT.pack(
        PROTOCOL_MAGIC,
        PROTOCOL_VERSION,
        MSG_PHONE_POSE,
        0,
        uuid.UUID(int=0).bytes,
        float(timestamp_s),
        *(float(value) for value in camera_transform),
        0,
        0,
    )
    body = _PHONE_POSE_STRUCT.pack(int(tracking_state), int(sequence))
    payload = header + body
    return _LENGTH_STRUCT.pack(len(payload)) + payload


def encode_imu(accel, gyro, *, timestamp_s: float = 0.0) -> bytes:
    """Encode one MSG_IMU message: accel (m/s^2, incl. gravity) + gyro (rad/s),
    both 3-tuples in the IMU frame; header timestamp is the sample time."""
    header = _HEADER_STRUCT.pack(
        PROTOCOL_MAGIC, PROTOCOL_VERSION, MSG_IMU, 0,
        uuid.UUID(int=0).bytes, float(timestamp_s),
        *(0.0 for _ in range(16)), 0, 0,
    )
    body = _IMU_STRUCT.pack(
        float(accel[0]), float(accel[1]), float(accel[2]),
        float(gyro[0]), float(gyro[1]), float(gyro[2]))
    payload = header + body
    return _LENGTH_STRUCT.pack(len(payload)) + payload


def decode_payload(payload: bytes) -> LiveMeshAnchorFrame:
    if len(payload) < _HEADER_STRUCT.size:
        raise LiveScanProtocolError("payload is shorter than the live-scan header")
    (
        magic,
        version,
        message_type,
        flags,
        anchor_bytes,
        timestamp_s,
        *tail,
    ) = _HEADER_STRUCT.unpack_from(payload)
    if magic != PROTOCOL_MAGIC:
        raise LiveScanProtocolError(f"bad live-scan magic: 0x{magic:08x}")
    if version != PROTOCOL_VERSION:
        raise LiveScanProtocolError(f"unsupported live-scan version: {version}")
    if message_type not in (
        MSG_UPSERT_ANCHOR,
        MSG_REMOVE_ANCHOR,
        MSG_HEARTBEAT,
        MSG_KEYFRAME_IMAGE,
        MSG_KEYFRAME_RGBD,
        MSG_FINALIZE,
        MSG_PHONE_POSE,
        MSG_CONTROL_START_SCAN,
        MSG_CONTROL_STOP_SCAN,
        MSG_REALIGN,
        MSG_DEPTH_FRAME,
        MSG_IMU,
        MSG_CONTROL_ENABLE_IMU,
        MSG_CONTROL_DISABLE_IMU,
    ):
        raise LiveScanProtocolError(f"unsupported live-scan message type: {message_type}")

    offset = _HEADER_STRUCT.size

    if message_type == MSG_IMU:
        if len(payload) < offset + _IMU_STRUCT.size:
            raise LiveScanProtocolError("imu payload is too short")
        ax, ay, az, gx, gy, gz = _IMU_STRUCT.unpack_from(payload, offset)
        return LiveMeshAnchorFrame(
            anchor_id=str(uuid.UUID(bytes=anchor_bytes)),
            timestamp_s=float(timestamp_s),
            imu_accel=(float(ax), float(ay), float(az)),
            imu_gyro=(float(gx), float(gy), float(gz)),
            message_type=message_type,
        )

    if message_type in (
        MSG_FINALIZE,
        MSG_CONTROL_START_SCAN,
        MSG_CONTROL_STOP_SCAN,
        MSG_REALIGN,
        MSG_CONTROL_ENABLE_IMU,
        MSG_CONTROL_DISABLE_IMU,
    ):
        return LiveMeshAnchorFrame(
            anchor_id=str(uuid.UUID(bytes=anchor_bytes)),
            timestamp_s=float(timestamp_s),
            message_type=message_type,
        )

    if message_type == MSG_PHONE_POSE:
        expected = offset + _PHONE_POSE_STRUCT.size
        if len(payload) != expected:
            raise LiveScanProtocolError(
                f"phone-pose payload length {len(payload)} does not match expected {expected}"
            )
        tracking_state, sequence = _PHONE_POSE_STRUCT.unpack_from(payload, offset)
        return LiveMeshAnchorFrame(
            anchor_id=str(uuid.UUID(bytes=anchor_bytes)),
            timestamp_s=float(timestamp_s),
            anchor_transform=tuple(float(value) for value in tail[:16]),
            tracking_state=int(tracking_state),
            sequence=int(sequence),
            message_type=message_type,
        )

    if message_type == MSG_DEPTH_FRAME:
        if len(payload) < offset + _DEPTH_FRAME_STRUCT.size:
            raise LiveScanProtocolError("depth-frame payload is too short")
        fx, fy, cx, cy, dw, dh, depth_len, conf_len = _DEPTH_FRAME_STRUCT.unpack_from(payload, offset)
        offset += _DEPTH_FRAME_STRUCT.size
        if int(depth_len) != int(dw) * int(dh) * 4:
            raise LiveScanProtocolError(
                f"depth length {depth_len} does not match {dw}x{dh} float32"
            )
        if int(conf_len) not in (0, int(dw) * int(dh)):
            raise LiveScanProtocolError(
                f"confidence length {conf_len} does not match depth map dimensions"
            )
        expected = offset + int(depth_len) + int(conf_len)
        if len(payload) != expected:
            raise LiveScanProtocolError(
                f"depth-frame payload length {len(payload)} does not match expected {expected}"
            )
        depth_bytes = bytes(payload[offset:offset + int(depth_len)])
        offset += int(depth_len)
        confidence_bytes = bytes(payload[offset:offset + int(conf_len)])
        return LiveMeshAnchorFrame(
            anchor_id=str(uuid.UUID(bytes=anchor_bytes)),
            timestamp_s=float(timestamp_s),
            anchor_transform=tuple(float(value) for value in tail[:16]),
            intrinsics=(float(fx), float(fy), float(cx), float(cy)),
            depth_width=int(dw),
            depth_height=int(dh),
            depth_bytes=depth_bytes,
            confidence_bytes=confidence_bytes,
            message_type=message_type,
        )

    if message_type in (MSG_KEYFRAME_IMAGE, MSG_KEYFRAME_RGBD):
        if len(payload) < offset + _IMAGE_HEADER_STRUCT.size:
            raise LiveScanProtocolError("keyframe image payload is too short")
        fx, fy, cx, cy, width, height, jpeg_len = _IMAGE_HEADER_STRUCT.unpack_from(payload, offset)
        offset += _IMAGE_HEADER_STRUCT.size
        expected = offset + int(jpeg_len)
        if len(payload) < expected:
            raise LiveScanProtocolError(f"payload length {len(payload)} is shorter than expected JPEG end {expected}")
        jpeg = bytes(payload[offset:expected])
        offset = expected
        depth_width = depth_height = 0
        depth_bytes = b""
        confidence_bytes = b""
        if message_type == MSG_KEYFRAME_RGBD:
            if len(payload) < offset + _RGBD_HEADER_STRUCT.size:
                raise LiveScanProtocolError("RGB-D keyframe payload is too short")
            depth_width, depth_height, depth_len, confidence_len = _RGBD_HEADER_STRUCT.unpack_from(payload, offset)
            offset += _RGBD_HEADER_STRUCT.size
            expected_depth_len = int(depth_width) * int(depth_height) * 4
            if int(depth_len) != expected_depth_len:
                raise LiveScanProtocolError(
                    f"depth length {depth_len} does not match expected {expected_depth_len}"
                )
            if int(confidence_len) not in (0, int(depth_width) * int(depth_height)):
                raise LiveScanProtocolError(
                    f"confidence length {confidence_len} does not match depth map dimensions"
                )
            expected = offset + int(depth_len) + int(confidence_len)
        if len(payload) != expected:
            raise LiveScanProtocolError(f"payload length {len(payload)} does not match expected {expected}")
        if message_type == MSG_KEYFRAME_RGBD:
            depth_bytes = bytes(payload[offset:offset + int(depth_len)])
            offset += int(depth_len)
            confidence_bytes = bytes(payload[offset:offset + int(confidence_len)])
        return LiveMeshAnchorFrame(
            anchor_id=str(uuid.UUID(bytes=anchor_bytes)),
            timestamp_s=float(timestamp_s),
            anchor_transform=tuple(float(value) for value in tail[:16]),
            image_width=int(width),
            image_height=int(height),
            image_bytes=jpeg,
            intrinsics=(float(fx), float(fy), float(cx), float(cy)),
            depth_width=int(depth_width),
            depth_height=int(depth_height),
            depth_bytes=depth_bytes,
            confidence_bytes=confidence_bytes,
            message_type=message_type,
        )

    has_normals = bool(flags & FLAG_HAS_NORMALS)
    textured = bool(flags & FLAG_TEXTURED)
    transform = tuple(float(value) for value in tail[:16])
    vertex_count = int(tail[16])
    face_count = int(tail[17])
    expected = offset + vertex_count * 12 + face_count * 12 + vertex_count * 3
    if has_normals:
        expected += vertex_count * 12
    if textured:
        expected += 16 + vertex_count * 8
    if len(payload) != expected:
        raise LiveScanProtocolError(f"payload length {len(payload)} does not match expected {expected}")

    vertices: list[Point3] = []
    for _ in range(vertex_count):
        vertices.append(struct.unpack_from("<fff", payload, offset))
        offset += 12

    faces: list[Face3] = []
    for _ in range(face_count):
        face = struct.unpack_from("<III", payload, offset)
        if any(index >= vertex_count for index in face):
            raise LiveScanProtocolError(f"face index out of range: {face}")
        faces.append(face)
        offset += 12

    colors: list[Color3ub] = []
    for _ in range(vertex_count):
        colors.append(struct.unpack_from("<BBB", payload, offset))
        offset += 3

    normals: list[Point3] = []
    if has_normals:
        for _ in range(vertex_count):
            normals.append(struct.unpack_from("<fff", payload, offset))
            offset += 12

    texture_image_id: str | None = None
    uvs: list[UV2] = []
    if textured:
        texture_image_id = str(uuid.UUID(bytes=bytes(payload[offset:offset + 16])))
        offset += 16
        for _ in range(vertex_count):
            uvs.append(struct.unpack_from("<ff", payload, offset))
            offset += 8

    return LiveMeshAnchorFrame(
        anchor_id=str(uuid.UUID(bytes=anchor_bytes)),
        timestamp_s=float(timestamp_s),
        anchor_transform=transform,
        vertices=tuple(vertices),
        faces=tuple(faces),
        colors=tuple(colors),
        normals=tuple(normals),
        texture_image_id=texture_image_id,
        uvs=tuple(uvs),
        message_type=message_type,
    )


def read_length_prefixed_message(
    sock: socket.socket,
    max_bytes: int = MAX_MESSAGE_BYTES,
    stop_event: threading.Event | None = None,
) -> bytes | None:
    header = _read_exact(sock, _LENGTH_STRUCT.size, stop_event)
    if header is None:
        return None
    (length,) = _LENGTH_STRUCT.unpack(header)
    if length <= 0 or length > max_bytes:
        raise LiveScanProtocolError(f"invalid live-scan message length: {length}")
    payload = _read_exact(sock, length, stop_event)
    if payload is None:
        return None
    return payload


def _read_exact(
    sock: socket.socket,
    size: int,
    stop_event: threading.Event | None = None,
) -> bytes | None:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        try:
            chunk = sock.recv(remaining)
        except socket.timeout:
            # The per-recv timeout only exists so the read loop can periodically
            # notice a stop request -- it does NOT mean the peer is gone. Keep
            # waiting for the rest of the data, whether the gap is an idle pause
            # between small messages or a mid-message stall while a large depth
            # frame streams over a congested link. Only a real stop, EOF, or
            # socket error ends the read.
            if stop_event is not None and stop_event.is_set():
                return None
            continue
        if not chunk:
            return None
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


class LiveScanReceiver:
    """Small TCP server that decodes live-scan frames into a queue."""

    def __init__(
        self,
        host: str,
        port: int,
        out_queue: queue.Queue[LiveMeshAnchorFrame],
        max_message_bytes: int = MAX_MESSAGE_BYTES,
        drop_when_backlogged: set[int] | None = None,
        max_backlog_frames: int = 0,
        recorder=None,
        initial_control_messages: Iterable[int] | None = None,
        read_timeout_s: float = 2.0,
    ) -> None:
        self.host = host
        self.port = port
        self.out_queue = out_queue
        self.max_message_bytes = max_message_bytes
        # Per-recv timeout, used only to wake the read loop periodically so it can
        # notice a stop request. A timeout while idle between messages does NOT
        # drop the phone -- the connection is kept open across scan pauses.
        self.read_timeout_s = float(read_timeout_s)
        self.drop_when_backlogged = set(drop_when_backlogged or ())
        self.max_backlog_frames = int(max_backlog_frames)
        # Optional ScanRecorder: tee every raw wire payload to a .scanlog so a
        # real scan can be replayed back through the mapper offline.
        self.recorder = recorder
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._server_socket: socket.socket | None = None
        self.connected = False
        self.frames_received = 0
        self.frames_dropped = 0
        # Inbound bandwidth accounting (diagnostics): total wire bytes and a
        # per-message-type breakdown, so the driver can show whether the link is
        # saturated and which streams (RGB-D vs discarded mesh anchors) own it.
        self.bytes_received = 0
        self.bytes_by_type: dict[int, int] = {}
        self.last_error: str | None = None
        self.last_frame_monotonic_s: float | None = None
        # Active phone connection + a lock, so the host can push control frames
        # (start/stop scan) back to the phone from another thread while the
        # receiver loop reads on the same socket (TCP is full-duplex).
        self._active_conn: socket.socket | None = None
        self._send_lock = threading.Lock()
        self.initial_control_messages = tuple(initial_control_messages or ())

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="live-scan-receiver", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._server_socket is not None:
            try:
                self._server_socket.close()
            except OSError:
                pass
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def send_control(self, message_type: int) -> bool:
        """Push a host->phone control frame to the connected phone. Returns False
        if no phone is connected or the send fails (the connection is then dropped
        so the receiver loop re-accepts)."""
        data = encode_control(message_type)
        with self._send_lock:
            conn = self._active_conn
            if conn is None:
                return False
            try:
                conn.sendall(data)
                return True
            except OSError as exc:
                self.last_error = f"control send failed: {exc}"
                try:
                    conn.close()  # forces the read loop out; receiver re-accepts
                except OSError:
                    pass
                return False

    def _run(self) -> None:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
                self._server_socket = server
                server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                server.bind((self.host, self.port))
                server.listen(1)
                server.settimeout(0.25)
                while not self._stop.is_set():
                    try:
                        conn, _addr = server.accept()
                    except socket.timeout:
                        continue
                    except OSError as exc:
                        if not self._stop.is_set():
                            self.last_error = str(exc)
                        break
                    with conn:
                        self.connected = True
                        conn.settimeout(self.read_timeout_s)
                        with self._send_lock:
                            self._active_conn = conn
                        for message_type in self.initial_control_messages:
                            self.send_control(message_type)
                        try:
                            self._serve_connection(conn)
                        finally:
                            with self._send_lock:
                                self._active_conn = None
                            self.connected = False
        except OSError as exc:
            self.last_error = str(exc)
            self.connected = False
        finally:
            self._server_socket = None

    def _serve_connection(self, conn: socket.socket) -> None:
        while not self._stop.is_set():
            try:
                payload = read_length_prefixed_message(
                    conn, self.max_message_bytes, self._stop)
            except (OSError, LiveScanProtocolError) as exc:
                self.last_error = str(exc)
                return
            if payload is None:
                return
            if self.recorder is not None:
                # Tee the literal wire bytes before decode/drop logic, so the
                # recording is exactly what the phone sent.
                self.recorder.write(payload)
            try:
                frame = decode_payload(payload)
            except LiveScanProtocolError as exc:
                self.last_error = str(exc)
                return
            # Count wire bytes per type BEFORE any drop logic: a dropped frame
            # still consumed link bandwidth, which is what we're measuring.
            self.bytes_received += len(payload) + _LENGTH_STRUCT.size
            self.bytes_by_type[frame.message_type] = (
                self.bytes_by_type.get(frame.message_type, 0) + len(payload))
            if frame.is_imu:
                # The raw recorder above has already captured IMU for offline
                # cuVSLAM inspection. The production ARKit/nvblox mapper does
                # not consume it, so do not push 100 Hz inertial samples into the
                # live frame queue.
                continue
            if frame.message_type != MSG_HEARTBEAT:
                if frame.is_finalize:
                    self.frames_dropped += self._drop_backlogged_frames()
                elif self._should_drop_frame(frame):
                    self.frames_dropped += 1
                    continue
                self.out_queue.put(frame)
                self.frames_received += 1
                self.last_frame_monotonic_s = time.monotonic()

    def _should_drop_frame(self, frame: LiveMeshAnchorFrame) -> bool:
        if frame.message_type not in self.drop_when_backlogged:
            return False
        if self.max_backlog_frames <= 0:
            return False
        return self.out_queue.qsize() >= self.max_backlog_frames

    def _drop_backlogged_frames(self) -> int:
        """Drop stale high-rate frames before enqueueing a finalize/control event.

        The live mapper prioritizes Stop Scan responsiveness over draining old
        camera frames that were captured before the user stopped the scan.
        """
        if not self.drop_when_backlogged:
            return 0
        kept = []
        dropped = 0
        while True:
            try:
                queued = self.out_queue.get_nowait()
            except queue.Empty:
                break
            if queued.message_type in self.drop_when_backlogged:
                dropped += 1
            else:
                kept.append(queued)
        for queued in kept:
            self.out_queue.put(queued)
        return dropped


class LiveScanState:
    def __init__(self, arkit_to_twin: Matrix4 | None = None) -> None:
        self.arkit_to_twin = arkit_to_twin or identity_matrix4()
        self.anchors: dict[str, LiveScanAnchor] = {}
        # JPEG keyframe bytes by image id, delivered by MSG_KEYFRAME_IMAGE.
        self.images: dict[str, bytes] = {}
        # Posed keyframes (pose + intrinsics + jpeg) retained for the offline bake.
        self.keyframes: dict[str, LiveKeyframe] = {}
        self.frames_applied = 0
        self.last_update_monotonic_s: float | None = None

    def apply_image(self, frame: LiveMeshAnchorFrame) -> str:
        """Store a posed keyframe (anchor_id holds the image id). Returns the id."""
        self.images[frame.anchor_id] = frame.image_bytes
        self.keyframes[frame.anchor_id] = LiveKeyframe(
            image_id=frame.anchor_id,
            jpeg=frame.image_bytes,
            camera_transform=frame.anchor_transform,
            intrinsics=frame.intrinsics,
            width=frame.image_width,
            height=frame.image_height,
            depth_width=frame.depth_width,
            depth_height=frame.depth_height,
            depth_bytes=frame.depth_bytes,
            confidence_bytes=frame.confidence_bytes,
            timestamp_s=frame.timestamp_s,
        )
        self.frames_applied += 1
        return frame.anchor_id

    def referenced_image_ids(self) -> set[str]:
        return {
            anchor.texture_image_id
            for anchor in self.anchors.values()
            if anchor.texture_image_id is not None
        }

    def apply_frame(self, frame: LiveMeshAnchorFrame, now_s: float | None = None) -> LiveScanAnchor | None:
        now = time.monotonic() if now_s is None else now_s
        if frame.removed:
            self.anchors.pop(frame.anchor_id, None)
            self.frames_applied += 1
            self.last_update_monotonic_s = now
            return None
        vertices_twin = transform_vertices_to_twin(
            frame.vertices,
            frame.anchor_transform,
            self.arkit_to_twin,
        )
        normals_twin = (
            transform_normals_to_twin(frame.normals, frame.anchor_transform, self.arkit_to_twin)
            if frame.normals
            else ()
        )
        anchor = LiveScanAnchor(
            anchor_id=frame.anchor_id,
            timestamp_s=frame.timestamp_s,
            received_monotonic_s=now,
            vertices_twin=vertices_twin,
            faces=frame.faces,
            colors=frame.colors,
            normals_twin=normals_twin,
            texture_image_id=frame.texture_image_id,
            uvs=frame.uvs,
        )
        self.anchors[frame.anchor_id] = anchor
        self.frames_applied += 1
        self.last_update_monotonic_s = now
        return anchor

    def apply_registration_delta(self, delta: Matrix4, new_registration: Matrix4) -> None:
        """Re-anchor every stored anchor by ``delta`` and adopt ``new_registration``.

        Used when the shared ARKit->twin alignment changes (init or a
        relocalization) so the accumulated mesh and the rover pose stay
        registered to one transform. ``delta`` is the row-major transform that
        maps old twin points to new twin points (see
        ``phone_mount.registration_delta``); ``new_registration`` becomes the
        registration applied to future frames.
        """

        for anchor in self.anchors.values():
            anchor.vertices_twin = tuple(
                transform_point_row_major(vertex, delta) for vertex in anchor.vertices_twin
            )
            if anchor.normals_twin:
                renormalized: list[Point3] = []
                for normal in anchor.normals_twin:
                    nx, ny, nz = transform_direction_row_major(normal, delta)
                    magnitude = (nx * nx + ny * ny + nz * nz) ** 0.5
                    if magnitude > 0.0:
                        renormalized.append((nx / magnitude, ny / magnitude, nz / magnitude))
                    else:
                        renormalized.append((0.0, 0.0, 0.0))
                anchor.normals_twin = tuple(renormalized)
        self.arkit_to_twin = new_registration

    def expire_stale(self, now_s: float, decay_s: float) -> list[str]:
        if decay_s <= 0:
            return []
        expired = [
            anchor_id
            for anchor_id, anchor in self.anchors.items()
            if now_s - anchor.received_monotonic_s > decay_s
        ]
        for anchor_id in expired:
            del self.anchors[anchor_id]
        return expired

    def obstacle_points_xy(
        self,
        z_band: tuple[float, float],
        now_s: float | None = None,
        decay_s: float = 0.0,
    ) -> tuple[tuple[float, float], ...]:
        now = time.monotonic() if now_s is None else now_s
        if decay_s > 0:
            self.expire_stale(now, decay_s)
        lo, hi = z_band
        points: list[tuple[float, float]] = []
        for anchor in self.anchors.values():
            for x, y, z in anchor.vertices_twin:
                if lo < z < hi:
                    points.append((x, y))
        return tuple(points)

    def known_points_xy(
        self,
        now_s: float | None = None,
        decay_s: float = 0.0,
    ) -> tuple[tuple[float, float], ...]:
        now = time.monotonic() if now_s is None else now_s
        if decay_s > 0:
            self.expire_stale(now, decay_s)
        return tuple((x, y) for anchor in self.anchors.values() for x, y, _z in anchor.vertices_twin)

    def points_xyz(
        self,
        now_s: float | None = None,
        decay_s: float = 0.0,
    ) -> tuple[Point3, ...]:
        """All accumulated twin-frame points (x, y, z). Used by the floor
        estimator, which needs the height the XY-only accessors drop."""
        now = time.monotonic() if now_s is None else now_s
        if decay_s > 0:
            self.expire_stale(now, decay_s)
        return tuple(point for anchor in self.anchors.values() for point in anchor.vertices_twin)

    def status(
        self,
        now_s: float | None = None,
        receiver: LiveScanReceiver | None = None,
        obstacle_cells: int = 0,
        known_cells: int = 0,
    ) -> dict:
        now = time.monotonic() if now_s is None else now_s
        vertex_count = sum(len(anchor.vertices_twin) for anchor in self.anchors.values())
        triangle_count = sum(len(anchor.faces) for anchor in self.anchors.values())
        last_age = None if self.last_update_monotonic_s is None else max(0.0, now - self.last_update_monotonic_s)
        data = {
            "enabled": True,
            "anchor_count": len(self.anchors),
            "vertex_count": vertex_count,
            "triangle_count": triangle_count,
            "frames_applied": self.frames_applied,
            "last_update_age_s": None if last_age is None else round(last_age, 3),
            "registered": self.arkit_to_twin != identity_matrix4(),
            "obstacle_cells": obstacle_cells,
            "known_cells": known_cells,
        }
        if receiver is not None:
            data.update(
                {
                    "connected": receiver.connected,
                    "frames_received": receiver.frames_received,
                    "receiver_error": receiver.last_error,
                }
            )
        return data


def rasterize_xy_points(points: Iterable[tuple[float, float]], cell_m: float) -> dict[tuple[int, int], list[tuple[float, float]]]:
    if cell_m <= 0.0:
        raise ValueError("cell_m must be positive")
    grid: dict[tuple[int, int], list[tuple[float, float]]] = {}
    for x, y in points:
        key = (int(x // cell_m), int(y // cell_m))
        grid.setdefault(key, []).append((float(x), float(y)))
    return grid
