"""Mapper -> twin messages on one ordered stream.

Same length-prefixed framing as the phone protocol. Order is the contract:
the mapper sends the tiles a pose depends on before the pose, so a twin that
applies what it has received, tiles first, is never ahead of its terrain::

    header  magic "RTLK" | version u8 | type u8
    TILES   rev u64 | t_arrival f64 | reset u8 | cell_m f64 | n u32 | count u32 |
            count x (tx ty i32 | height f32[(n+1)^2] | state u8[(n+1)^2])
    POSE    rev u64 | t_arrival f64 | t_phone f64 | x y yaw_deg f64
    MESH    t_arrival f64 | nv nf u32 | vertices f32[nv,3] | faces i32[nf,3] | colors u8[nv,3]

``rev`` counts terrain updates; a pose carries the revision it was made on.
``t_arrival`` is the host arrival time of the newest phone frame behind the
message, for end-to-end latency. A tile patch is the tile's n x n cells plus
the first row and column of its +x / +y neighbours (see ``ElevationGrid.patch``).
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from enum import IntEnum
from typing import Union

import numpy as np

from rover_twin.frames import Pose2D
from rover_twin.mapping.tsdf import Mesh
from rover_twin.phone.protocol import LENGTH, ProtocolError

_HEADER = struct.Struct("<4sBB")
_TILES = struct.Struct("<QdBdII")
_KEY = struct.Struct("<ii")
_POSE = struct.Struct("<Qdd4d")
_MESH = struct.Struct("<dII")
MAGIC, VERSION = b"RTLK", 1


class Kind(IntEnum):
    TILES = 1
    POSE = 2
    MESH = 3


@dataclass(frozen=True, eq=False)
class Patch:
    key: tuple[int, int]
    height: np.ndarray       # (n+1, n+1) float32, NaN where unknown
    state: np.ndarray        # (n+1, n+1) uint8: elevation.UNKNOWN / OBSERVED / INFERRED


@dataclass(frozen=True, eq=False)
class Tiles:
    rev: int
    t_arrival: float
    reset: bool              # these patches replace the whole terrain
    cell_m: float
    n: int
    patches: tuple[Patch, ...]


@dataclass(frozen=True)
class LivePose:
    rev: int
    t_arrival: float
    t_phone: float
    pose: Pose2D
    z_m: float = 0.0        # the ground under the rover, from the phone's pose


@dataclass(frozen=True, eq=False)
class VisualMesh:
    t_arrival: float
    mesh: Mesh


Message = Union[Tiles, LivePose, VisualMesh]


def encode(message: Message) -> bytes:
    if isinstance(message, Tiles):
        parts = [_TILES.pack(message.rev, message.t_arrival, message.reset, message.cell_m, message.n,
                             len(message.patches))]
        for p in message.patches:
            parts += [_KEY.pack(*p.key), p.height.astype("<f4").tobytes(), p.state.astype(np.uint8).tobytes()]
        kind, body = Kind.TILES, b"".join(parts)
    elif isinstance(message, LivePose):
        p = message.pose
        kind, body = Kind.POSE, _POSE.pack(message.rev, message.t_arrival, message.t_phone, p.x, p.y, p.yaw_deg,
                                           message.z_m)
    else:
        m = message.mesh
        kind = Kind.MESH
        body = (_MESH.pack(message.t_arrival, len(m.vertices), len(m.faces)) + m.vertices.astype("<f4").tobytes()
                + m.faces.astype("<i4").tobytes() + m.colors.astype(np.uint8).tobytes())
    payload = _HEADER.pack(MAGIC, VERSION, kind) + body
    return LENGTH.pack(len(payload)) + payload


def decode(payload: bytes) -> Message:
    magic, version, kind = _HEADER.unpack_from(payload)
    if magic != MAGIC or version != VERSION:
        raise ProtocolError("not a link message")
    at = _HEADER.size
    if kind == Kind.POSE:
        rev, t, t_phone, x, y, yaw, z = _POSE.unpack_from(payload, at)
        return LivePose(rev, t, t_phone, Pose2D(x, y, yaw), z)
    if kind == Kind.TILES:
        rev, t, reset, cell_m, n, count = _TILES.unpack_from(payload, at)
        at += _TILES.size
        k = (n + 1) * (n + 1)
        patches = []
        for _ in range(count):
            key = _KEY.unpack_from(payload, at)
            at += _KEY.size
            height = np.frombuffer(payload, "<f4", k, at).reshape(n + 1, n + 1)
            state = np.frombuffer(payload, np.uint8, k, at + 4 * k).reshape(n + 1, n + 1)
            patches.append(Patch(key, height, state))
            at += 5 * k
        return Tiles(rev, t, bool(reset), cell_m, n, tuple(patches))
    if kind == Kind.MESH:
        t, nv, nf = _MESH.unpack_from(payload, at)
        at += _MESH.size
        vertices = np.frombuffer(payload, "<f4", 3 * nv, at).reshape(nv, 3)
        faces = np.frombuffer(payload, "<i4", 3 * nf, at + 12 * nv).reshape(nf, 3)
        colors = np.frombuffer(payload, np.uint8, 3 * nv, at + 12 * nv + 12 * nf).reshape(nv, 3)
        return VisualMesh(t, Mesh(vertices, faces, colors))
    raise ProtocolError(f"unknown link message type {kind}")
