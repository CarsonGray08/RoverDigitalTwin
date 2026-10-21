"""LiveScanner wire protocol, shared by the phone and the mapper.

Every message is a little-endian ``u32`` length and a payload. The payload
starts with a fixed header; the body depends on the message type::

    header  magic "DTLS" u32 | version u8 | type u8 | flags u16 | id 16s |
            t_phone f64 | transform 16 x f32 (simd column-major) | 2 x u32

    RGBD        fx fy cx cy f32 (JPEG pixels) | width height jpeg_len u32 | jpeg |
                depth_w depth_h depth_len conf_len u32 | depth f32[] | conf u8[]
    PHONE_POSE  tracking_state u32 | sequence u32
    IMU         accel xyz f32 (m/s^2, incl. gravity) | gyro xyz f32 (rad/s)
    others      header only

``t_phone`` is the phone's capture clock (CACurrentMediaTime) for every type.
"""

from __future__ import annotations

import socket
import struct
import threading
from dataclasses import dataclass
from enum import IntEnum
from typing import NamedTuple, Union

import numpy as np

from rover_twin.frames import PHONE_T_CV, Intrinsics, from_wire, to_wire

MAGIC = 0x44544C53
VERSION = 1
MAX_MESSAGE_BYTES = 64 * 1024 * 1024

TRACKING_NORMAL = 2  # ARKit trackingState: 0 not available, 1 limited, 2 normal

LENGTH = struct.Struct("<I")
_HEADER = struct.Struct("<IBBH16sd16fII")
_IMAGE = struct.Struct("<ffffIII")
_DEPTH = struct.Struct("<IIII")
_POSE = struct.Struct("<II")
_IMU = struct.Struct("<6f")


class Msg(IntEnum):
    HEARTBEAT = 3
    FINALIZE = 5      # phone: Stop Scan pressed
    RGBD = 6
    PHONE_POSE = 7
    START_SCAN = 8    # host -> phone
    STOP_SCAN = 9     # host -> phone
    REALIGN = 10      # phone: Reset Position pressed
    IMU = 12
    ENABLE_IMU = 13   # host -> phone
    DISABLE_IMU = 14  # host -> phone


class ProtocolError(ValueError):
    pass


class Arrival(NamedTuple):
    """A raw payload and the host monotonic time it arrived."""

    t: float
    payload: bytes


@dataclass(frozen=True, eq=False)
class RgbdFrame:
    t_phone: float
    T_A_C: np.ndarray               # camera pose in ARKit world, CV axes
    depth: np.ndarray               # (H, W) float32 metres, 0 = no return
    confidence: np.ndarray | None   # (H, W) uint8, 0 low .. 2 high
    jpeg: bytes
    color_intrinsics: Intrinsics    # in JPEG pixels

    @property
    def depth_intrinsics(self) -> Intrinsics:
        h, w = self.depth.shape
        return self.color_intrinsics.scaled(w, h)


@dataclass(frozen=True, eq=False)
class PhonePose:
    t_phone: float
    T_A_C: np.ndarray
    tracking: int


@dataclass(frozen=True)
class Imu:
    t_phone: float
    accel: tuple[float, float, float]
    gyro: tuple[float, float, float]


@dataclass(frozen=True)
class Event:
    """Any header-only message: phone events and host controls."""

    kind: Msg
    t_phone: float = 0.0


Message = Union[RgbdFrame, PhonePose, Imu, Event]


def decode(payload: bytes) -> Message | None:
    """Decode one payload; ``None`` for message types the host does not use."""
    if len(payload) < _HEADER.size:
        raise ProtocolError("payload shorter than the header")
    magic, version, kind, _flags, _id, t, *rest = _HEADER.unpack_from(payload)
    if magic != MAGIC or version != VERSION:
        raise ProtocolError(f"bad magic 0x{magic:08x} or version {version}")
    try:
        kind = Msg(kind)
    except ValueError:
        return None
    body = memoryview(payload)[_HEADER.size:]
    if kind == Msg.RGBD:
        return _decode_rgbd(t, rest[:16], body)
    if kind == Msg.PHONE_POSE:
        _expect(len(body) == _POSE.size, "phone pose")
        tracking, _seq = _POSE.unpack(body)
        return PhonePose(t, from_wire(rest[:16]) @ PHONE_T_CV, tracking)
    if kind == Msg.IMU:
        _expect(len(body) == _IMU.size, "imu")
        v = _IMU.unpack(body)
        return Imu(t, v[:3], v[3:])
    if kind == Msg.HEARTBEAT:
        return None
    return Event(kind, t)


def _decode_rgbd(t: float, transform, body: memoryview) -> RgbdFrame:
    _expect(len(body) >= _IMAGE.size, "rgbd image header")
    fx, fy, cx, cy, width, height, jpeg_len = _IMAGE.unpack_from(body)
    at = _IMAGE.size + jpeg_len
    _expect(len(body) >= at + _DEPTH.size, "rgbd jpeg")
    jpeg = bytes(body[_IMAGE.size:at])
    dw, dh, depth_len, conf_len = _DEPTH.unpack_from(body, at)
    at += _DEPTH.size
    _expect(depth_len == dw * dh * 4 and conf_len in (0, dw * dh), "rgbd depth sizes")
    _expect(len(body) == at + depth_len + conf_len, "rgbd length")
    depth = np.frombuffer(body, "<f4", dw * dh, at).reshape(dh, dw)
    confidence = np.frombuffer(body, np.uint8, conf_len, at + depth_len).reshape(dh, dw) if conf_len else None
    return RgbdFrame(t, from_wire(transform) @ PHONE_T_CV, depth, confidence, jpeg,
                     Intrinsics(fx, fy, cx, cy, width, height))


def _expect(ok: bool, what: str) -> None:
    if not ok:
        raise ProtocolError(f"malformed {what}")


def encode(message: Message) -> bytes:
    """Length-prefixed wire bytes for ``message``."""
    T, body = None, b""
    if isinstance(message, RgbdFrame):
        kind, T, body = Msg.RGBD, message.T_A_C, _encode_rgbd(message)
    elif isinstance(message, PhonePose):
        kind, T, body = Msg.PHONE_POSE, message.T_A_C, _POSE.pack(message.tracking, 0)
    elif isinstance(message, Imu):
        kind, body = Msg.IMU, _IMU.pack(*message.accel, *message.gyro)
    else:
        kind = message.kind
    wire = to_wire(np.eye(4) if T is None else T @ PHONE_T_CV)
    payload = _HEADER.pack(MAGIC, VERSION, kind, 0, bytes(16), message.t_phone, *wire, 0, 0) + body
    return LENGTH.pack(len(payload)) + payload


def read_frame(sock: socket.socket, stop: threading.Event) -> bytes | None:
    """Read one length-prefixed payload; ``None`` on EOF or ``stop``.

    The socket's timeout only lets the loop notice ``stop``; a pause on the
    link, even mid-message, is not a disconnect.
    """
    header = _read_exact(sock, LENGTH.size, stop)
    if header is None:
        return None
    (length,) = LENGTH.unpack(header)
    if not 0 < length <= MAX_MESSAGE_BYTES:
        raise ProtocolError(f"invalid message length {length}")
    return _read_exact(sock, length, stop)


def _read_exact(sock: socket.socket, size: int, stop: threading.Event) -> bytes | None:
    buf = bytearray()
    while len(buf) < size:
        try:
            chunk = sock.recv(size - len(buf))
        except socket.timeout:
            if stop.is_set():
                return None
            continue
        if not chunk:
            return None
        buf += chunk
    return bytes(buf)


def _encode_rgbd(f: RgbdFrame) -> bytes:
    k = f.color_intrinsics
    dh, dw = f.depth.shape
    depth = np.ascontiguousarray(f.depth, "<f4").tobytes()
    conf = b"" if f.confidence is None else np.ascontiguousarray(f.confidence, np.uint8).tobytes()
    return (_IMAGE.pack(k.fx, k.fy, k.cx, k.cy, k.width, k.height, len(f.jpeg)) + f.jpeg
            + _DEPTH.pack(dw, dh, len(depth), len(conf)) + depth + conf)
