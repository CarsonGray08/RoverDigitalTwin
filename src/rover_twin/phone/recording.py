"""Scan recordings (``.scanlog``) and replay paced at the recorded timestamps.

A recording is the literal wire: every payload the phone sent, stamped with
the host time it arrived. Replaying it reproduces what the mapper saw live,
including the phone's frame timing and the network's jitter::

    magic  b"DTSCAN01"
    record <offset_s f64> <payload_len u32> <payload>     (repeated)

``offset_s`` is host monotonic seconds since the first record.
"""

from __future__ import annotations

import struct
import threading
import time
import warnings
from pathlib import Path
from typing import Callable, Iterator

from rover_twin.phone.protocol import Arrival

MAGIC = b"DTSCAN01"
_RECORD = struct.Struct("<dI")


class Recorder:
    """Appends arrivals to a recording; safe to call from the receiver thread."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._fh = open(self.path, "wb")
        self._fh.write(MAGIC)
        self._lock = threading.Lock()
        self._t0: float | None = None

    def write(self, arrival: Arrival) -> None:
        with self._lock:
            if self._t0 is None:
                self._t0 = arrival.t
            self._fh.write(_RECORD.pack(arrival.t - self._t0, len(arrival.payload)))
            self._fh.write(arrival.payload)

    def close(self) -> None:
        with self._lock:
            self._fh.close()


def read(path: str | Path) -> Iterator[tuple[float, bytes]]:
    """``(offset_s, payload)`` per record. A recorder killed mid-write leaves a
    truncated last record; everything before it is intact, so it ends the
    stream with a warning instead of an error."""
    with open(path, "rb") as fh:
        if fh.read(len(MAGIC)) != MAGIC:
            raise ValueError(f"{path} is not a scan recording")
        while header := fh.read(_RECORD.size):
            if len(header) == _RECORD.size:
                offset, length = _RECORD.unpack(header)
                payload = fh.read(length)
                if len(payload) == length:
                    yield offset, payload
                    continue
            warnings.warn(f"{path}: truncated final record ignored", stacklevel=2)
            return


def replay(
    path: str | Path,
    speed: float = 1.0,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> Iterator[Arrival]:
    """Yield a recording's payloads at their recorded arrival times.

    ``speed`` 1 is real time, 2 twice as fast; 0 replays as fast as the
    consumer takes payloads. Paced arrivals carry the time each payload was
    *due*, not the time the consumer got to it, so a mapper that falls behind
    shows the queueing delay in its latency metrics, as it would live.
    """
    t0 = clock()
    for offset, payload in read(path):
        if speed <= 0:
            yield Arrival(clock(), payload)
            continue
        due = t0 + offset / speed
        if (wait := due - clock()) > 0:
            sleep(wait)
        yield Arrival(due, payload)
