"""Record/replay of the raw iPhone live-scan wire stream.

The recording boundary is the wire: every frame the phone sends arrives in
:meth:`LiveScanReceiver._serve_connection` as a length-prefixed ``payload``
(bytes), then ``decode_payload`` turns it into a frame on the queue. Teeing those
raw payloads to a file captures a 360 scan byte-exactly; replaying them back
through ``decode_payload`` into the same frame queue drives the *entire* host
mapper (feeder, T_align, ICP, nvblox integration, tiling, rover-pose publishing)
deterministically and offline -- no phone, no GPU on the capture side. That makes
a real scan a repeatable fixture you can iterate the mapper against.

File format (little-endian, append-only stream)::

    magic   : 8 bytes  = b"DTSCAN01"
    record* : <d offset_s><I payload_len><payload bytes>

``offset_s`` is monotonic seconds since the first recorded frame (kept so a
faithful, timing-preserving replayer could pace frames; the direct-injection
replayer here ignores it and runs as fast as the GPU allows).
"""

from __future__ import annotations

import queue
import struct
import threading
import time
import warnings
from pathlib import Path
from typing import Iterator

from digital_twin.live_scan import (
    LiveMeshAnchorFrame,
    LiveScanProtocolError,
    _LENGTH_STRUCT,
    decode_payload,
    encode_finalize,
)

MAGIC = b"DTSCAN01"
_RECORD_HEADER = struct.Struct("<dI")  # offset_s, payload_len
# A ready-made MSG_FINALIZE frame, built through the wire codec so it matches a
# real one exactly (no coupling to the dataclass constructor's required fields).
_FINALIZE_FRAME = decode_payload(encode_finalize()[_LENGTH_STRUCT.size:])


class ScanRecorder:
    """Append raw wire payloads to a ``.scanlog`` file, thread-safe.

    The receiver calls :meth:`write` for every payload it reads off the socket,
    before decode/drop logic, so the recording is the literal wire -- including
    frames a newer decoder might handle differently.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._fh = open(self.path, "wb")
        self._fh.write(MAGIC)
        self._lock = threading.Lock()
        self._t0: float | None = None
        self.records_written = 0
        self.bytes_written = 0

    def write(self, payload: bytes) -> None:
        now = time.monotonic()
        with self._lock:
            if self._fh is None:
                return
            if self._t0 is None:
                self._t0 = now
            self._fh.write(_RECORD_HEADER.pack(now - self._t0, len(payload)))
            self._fh.write(payload)
            self.records_written += 1
            self.bytes_written += len(payload)

    def close(self) -> None:
        with self._lock:
            if self._fh is not None:
                self._fh.flush()
                self._fh.close()
                self._fh = None


def iter_records(
    path: str | Path, *, strict: bool = False
) -> Iterator[tuple[float, bytes]]:
    """Yield ``(offset_s, payload)`` for each recorded frame, in order.

    A recorder killed mid-write (Ctrl-C during a scan) leaves a truncated final
    record. A short read can only happen at physical end-of-file, so there is
    never valid data after it -- all complete records before it are intact. By
    default such a truncated tail is treated as a clean end of stream (with a
    warning) so the captured scan is still usable. Pass ``strict=True`` to raise
    on truncation instead, e.g. for integrity validation.
    """
    with open(path, "rb") as fh:
        magic = fh.read(len(MAGIC))
        if magic != MAGIC:
            raise ValueError(
                f"{path}: not a scan recording (bad magic {magic!r})")
        record_index = 0
        while True:
            header = fh.read(_RECORD_HEADER.size)
            if not header:
                return
            if len(header) != _RECORD_HEADER.size:
                if strict:
                    raise ValueError(f"{path}: truncated record header")
                warnings.warn(
                    f"{path}: truncated record header after {record_index} "
                    "complete records; treating as end of stream",
                    stacklevel=2)
                return
            offset_s, length = _RECORD_HEADER.unpack(header)
            payload = fh.read(length)
            if len(payload) != length:
                if strict:
                    raise ValueError(f"{path}: truncated record payload")
                warnings.warn(
                    f"{path}: truncated record payload after {record_index} "
                    "complete records; treating as end of stream",
                    stacklevel=2)
                return
            record_index += 1
            yield offset_s, payload


class ScanLogReplaySource:
    """Drop-in stand-in for ``LiveScanReceiver`` that replays a ``.scanlog``.

    It exposes the small surface ``nvblox_live_map.main`` reads from a receiver
    (``start``/``stop``/``send_control`` + the ``bytes_received`` /
    ``bytes_by_type`` / ``frames_dropped`` diagnostics) but instead of owning a
    socket it decodes recorded payloads straight into the frame queue as fast as
    they can be consumed (direct injection -- original timing is ignored).

    When the recording is exhausted it sets ``replay_exhausted`` (and appends a
    synthetic finalize if the recording didn't end in one) so the driver loop can
    terminate cleanly whether or not the captured scan was stopped properly.
    """

    def __init__(
        self,
        path: str | Path,
        out_queue: queue.Queue,
        append_finalize: bool = True,
    ) -> None:
        self.path = Path(path)
        self.out_queue = out_queue
        self.append_finalize = append_finalize
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        # Receiver-compatible diagnostics the driver reads via getattr.
        self.bytes_received = 0
        self.bytes_by_type: dict[int, int] = {}
        self.frames_received = 0
        self.frames_dropped = 0
        self.connected = False
        self.last_error: str | None = None
        self.replay_exhausted = False

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run, name="scan-log-replay", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def send_control(self, _message_type: int) -> bool:
        # No phone to drive on replay; start/stop scan are no-ops.
        return False

    def _run(self) -> None:
        ended_in_finalize = False
        try:
            for _offset_s, payload in iter_records(self.path):
                if self._stop.is_set():
                    return
                try:
                    frame = decode_payload(payload)
                except LiveScanProtocolError as exc:
                    self.last_error = str(exc)
                    continue
                self.bytes_received += len(payload)
                self.bytes_by_type[frame.message_type] = (
                    self.bytes_by_type.get(frame.message_type, 0) + len(payload))
                if frame.is_imu:
                    # IMU belongs in the raw log for visual-inertial experiments,
                    # but the current nvblox mapper does not consume it.
                    continue
                # block=True so a slow GPU consumer applies natural backpressure
                # instead of us overrunning the queue and dropping geometry.
                self._put(frame)
                ended_in_finalize = frame.is_finalize
        except (OSError, ValueError) as exc:
            self.last_error = str(exc)
        finally:
            if self.append_finalize and not ended_in_finalize and not self._stop.is_set():
                self._put(_FINALIZE_FRAME)
            self.replay_exhausted = True

    def _put(self, frame: LiveMeshAnchorFrame) -> None:
        while not self._stop.is_set():
            try:
                self.out_queue.put(frame, timeout=0.25)
                self.frames_received += 1
                return
            except queue.Full:
                continue
