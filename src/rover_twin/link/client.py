"""Twin end of the link."""

from __future__ import annotations

import socket
import struct
import threading
from dataclasses import dataclass, field

from rover_twin.link.messages import LivePose, Patch, Tiles, VisualMesh, decode
from rover_twin.phone.protocol import read_frame


@dataclass
class Update:
    """Everything received since the last :meth:`LinkClient.take`, coalesced."""

    reset: bool = False                       # drop the current terrain first
    patches: dict[tuple[int, int], Patch] = field(default_factory=dict)   # newest per tile
    cell_m: float = 0.0
    n: int = 0
    tile_arrivals: list[float] = field(default_factory=list)   # t_arrival of each tile message
    pose: LivePose | None = None
    mesh: VisualMesh | None = None

    def merge(self, message) -> None:
        if isinstance(message, Tiles):
            if message.reset:
                self.reset, self.patches = True, {}
            self.patches.update((p.key, p) for p in message.patches)
            self.cell_m, self.n = message.cell_m, message.n
            self.tile_arrivals.append(message.t_arrival)
        elif isinstance(message, LivePose):
            self.pose = message
        elif isinstance(message, VisualMesh):
            self.mesh = message


class LinkClient:
    """Connects to the mapper (retrying), and keeps only what the twin still
    has to apply: every tile not yet applied, the latest pose, the latest mesh.
    Taking them all at once keeps the stream's order: no pose is handed over
    without the tiles that were sent before it."""

    def __init__(self, host: str, port: int, retry_s: float = 0.5) -> None:
        self._address, self._retry_s = (host, port), retry_s
        self._lock = threading.Lock()
        self._pending = Update()
        self._closed = threading.Event()
        self.connected = threading.Event()
        self._thread = threading.Thread(target=self._run, name="link-client", daemon=True)
        self._thread.start()

    def take(self) -> Update | None:
        with self._lock:
            update, self._pending = self._pending, Update()
        if update.reset or update.patches or update.pose or update.mesh:
            return update
        return None

    def close(self) -> None:
        self._closed.set()
        self._thread.join(timeout=1.0)

    def _run(self) -> None:
        while not self._closed.is_set():
            try:
                with socket.create_connection(self._address, timeout=self._retry_s) as conn:
                    conn.settimeout(0.25)
                    self.connected.set()
                    while (payload := read_frame(conn, self._closed)) is not None:
                        message = decode(payload)
                        with self._lock:
                            self._pending.merge(message)
            except (OSError, ValueError, struct.error):
                pass       # reconnect; the mapper answers with a fresh snapshot
            self.connected.clear()
            self._closed.wait(self._retry_s)
