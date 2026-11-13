"""The operator's view for the dashboard: the twin's viewport, captured a few
times a second as a JPEG.

Isaac Sim only. The twin's loop calls :meth:`ViewportCapture.frame_drawn`
after each frame it draws; a capture asked for then arrives with a later
frame, as raw RGBA, and is encoded off the loop. The HTTP thread reads
:meth:`ViewportCapture.latest`.
"""

from __future__ import annotations

import ctypes
import io
import threading
import time

import numpy as np

MAX_WIDTH = 960          # px; the dashboard shows it at about this size
QUALITY = 80
LOST_S = 2.0             # a capture not back by then is asked for again

_capsule_pointer = ctypes.PYFUNCTYPE(ctypes.c_void_p, ctypes.py_object, ctypes.c_char_p)(
    ("PyCapsule_GetPointer", ctypes.pythonapi))


class ViewportCapture:
    def __init__(self, hz: float) -> None:
        from omni.kit.viewport.utility import get_active_viewport

        self._viewport = get_active_viewport()
        self._every_s = 1.0 / hz
        self._asked = self._due = 0.0
        self._pending = False
        self._raw: np.ndarray | None = None
        self._latest: tuple[bytes, float] | None = None
        self._wake = threading.Condition()
        self._closed = False
        self._thread = threading.Thread(target=self._encode, name="twin-view", daemon=True)
        self._thread.start()

    def frame_drawn(self) -> None:
        """From the loop: ask for the next capture when one is due."""
        from omni.kit.viewport.utility import capture_viewport_to_buffer

        now = time.monotonic()
        if self._viewport is None or now < self._due or (self._pending and now - self._asked < LOST_S):
            return
        self._asked, self._due, self._pending = now, now + self._every_s, True
        capture_viewport_to_buffer(self._viewport, self._on_capture)

    def latest(self) -> tuple[bytes, float] | None:
        """The last JPEG and its age (s), or None before the first."""
        latest = self._latest
        return None if latest is None else (latest[0], time.monotonic() - latest[1])

    def close(self) -> None:
        with self._wake:
            self._closed = True
            self._wake.notify()
        self._thread.join(timeout=1.0)

    def _on_capture(self, buffer, buffer_size: int, width: int, height: int, _format) -> None:
        self._pending = False
        if buffer_size < width * height * 4:
            return
        pixels = (ctypes.c_ubyte * buffer_size).from_address(_capsule_pointer(buffer, None))
        rgba = np.frombuffer(pixels, np.uint8, width * height * 4).reshape(height, width, 4).copy()
        with self._wake:
            self._raw = rgba
            self._wake.notify()

    def _encode(self) -> None:
        from PIL import Image

        while True:
            with self._wake:
                while self._raw is None and not self._closed:
                    self._wake.wait()
                if self._closed:
                    return
                rgba, self._raw = self._raw, None
            image = Image.fromarray(rgba[:, :, :3])
            if image.width > MAX_WIDTH:
                image = image.resize((MAX_WIDTH, round(image.height * MAX_WIDTH / image.width)), Image.BILINEAR)
            out = io.BytesIO()
            image.save(out, "JPEG", quality=QUALITY)
            self._latest = (out.getvalue(), time.monotonic())
