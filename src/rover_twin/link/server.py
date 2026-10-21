"""Mapper end of the link: serves one twin at a time."""

from __future__ import annotations

import socket
import threading

from rover_twin.link.messages import Message, encode


class LinkServer:
    """Sends on the mapper's own thread, so messages leave in the order made.

    A twin that connects first needs the whole map: its connection stays
    pending until the mapper calls :meth:`snapshot_requested`, gets True and
    sends a snapshot, so nothing incremental can reach it before that. A twin
    that stops reading for ``send_timeout_s`` is dropped; when it reconnects
    it gets a fresh snapshot.
    """

    def __init__(self, host: str, port: int, send_timeout_s: float = 1.0) -> None:
        self._server = socket.create_server((host, port))
        self._server.settimeout(0.25)
        self.port = self._server.getsockname()[1]
        self._timeout = send_timeout_s
        self._lock = threading.Lock()
        self._active: socket.socket | None = None
        self._pending: socket.socket | None = None
        self._closed = threading.Event()
        self._thread = threading.Thread(target=self._accept, name="link-server", daemon=True)
        self._thread.start()

    @property
    def connected(self) -> bool:
        """A twin is connected (or waiting for its snapshot)."""
        return self._active is not None or self._pending is not None

    def snapshot_requested(self) -> bool:
        with self._lock:
            if self._pending is None:
                return False
            if self._active is not None:
                self._active.close()
            self._active, self._pending = self._pending, None
            return True

    def send(self, message: Message) -> None:
        with self._lock:
            if self._active is None:
                return
            try:
                self._active.sendall(encode(message))
            except OSError:
                self._active.close()
                self._active = None

    def close(self) -> None:
        self._closed.set()
        self._server.close()
        self._thread.join(timeout=1.0)
        with self._lock:
            for conn in (self._active, self._pending):
                if conn is not None:
                    conn.close()

    def _accept(self) -> None:
        while not self._closed.is_set():
            try:
                conn, _ = self._server.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            conn.settimeout(self._timeout)
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            with self._lock:
                if self._pending is not None:
                    self._pending.close()
                self._pending = conn
