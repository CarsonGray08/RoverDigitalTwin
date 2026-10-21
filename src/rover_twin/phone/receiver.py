"""TCP server for the phone stream."""

from __future__ import annotations

import queue
import socket
import threading
import time
from typing import Iterator

from rover_twin.phone.protocol import Arrival, Event, Msg, ProtocolError, encode, read_frame
from rover_twin.phone.recording import Recorder


class PhoneLink:
    """Serves one phone at a time and yields what it sends, as arrivals.

    A reader thread stamps each payload with its arrival time and puts it on a
    bounded queue. When the mapper falls behind, the queue fills, the reader
    blocks and TCP pushes back on the phone, whose capture gate then skips
    frames instead of letting a backlog grow. IMU streaming is switched on for
    every connection.
    """

    def __init__(self, host: str, port: int, recorder: Recorder | None = None, queue_size: int = 32) -> None:
        self._server = socket.create_server((host, port))
        self._server.settimeout(0.25)
        self.port = self._server.getsockname()[1]
        self._queue: queue.Queue[Arrival] = queue.Queue(queue_size)
        self._recorder = recorder
        self._closed = threading.Event()
        self._conn: socket.socket | None = None
        self._send_lock = threading.Lock()
        self._thread = threading.Thread(target=self._serve, name="phone-link", daemon=True)
        self._thread.start()

    def __iter__(self) -> Iterator[Arrival]:
        while not self._closed.is_set():
            try:
                yield self._queue.get(timeout=0.25)
            except queue.Empty:
                continue

    def send(self, kind: Msg) -> bool:
        """Send a host -> phone control; False when no phone is connected."""
        with self._send_lock:
            if self._conn is None:
                return False
            try:
                self._conn.sendall(encode(Event(kind)))
                return True
            except OSError:
                return False

    def close(self) -> None:
        self._closed.set()
        self._server.close()
        self._thread.join(timeout=1.0)

    def _serve(self) -> None:
        while not self._closed.is_set():
            try:
                conn, _ = self._server.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            with conn:
                conn.settimeout(0.25)
                with self._send_lock:
                    self._conn = conn
                self.send(Msg.ENABLE_IMU)
                try:
                    self._read(conn)
                except (OSError, ProtocolError):
                    pass  # the phone reconnects; nothing half-read is queued
                finally:
                    with self._send_lock:
                        self._conn = None

    def _read(self, conn: socket.socket) -> None:
        while (payload := read_frame(conn, self._closed)) is not None:
            arrival = Arrival(time.monotonic(), payload)
            if self._recorder is not None:
                self._recorder.write(arrival)
            while not self._closed.is_set():
                try:
                    self._queue.put(arrival, timeout=0.25)
                    break
                except queue.Full:
                    continue
