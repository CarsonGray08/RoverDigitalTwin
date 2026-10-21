"""The real rover: its HTTP controller (ezrassor_controller_server, port 5000)
forwards wheel commands over serial to the wheel Arduino (team2code.ino).

A wheel command is a step count at a speed level: the firmware runs all four
steppers until each has made that many steps, then stops by itself. Levels
1, 2 and 3 are 60, 100 and 140 motor RPM at 800 steps per revolution (1/4
microstepping); 0 is stop. The host sends more steps than the target and
stops the rover when its measured pose gets there, so wheel slip is
absorbed, and a host that dies mid-command still leaves the rover bounded
by the steps it was sent.

The controller reports each command's serial write on ``command_status``
("Success", "Timeout", "Exception", "Missing Arduino ...", with a counter
appended); it has no obstacle safety of its own.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

STEPS_PER_M = 23000.0                 # measured, at level 1 (~800 steps/s, 0.035 m/s)
STEPS_PER_DEG = 24000.0 / 90.0        # ~2x the real turn (a 180 deg trial turned 174 deg on 24107)
HEADROOM = 1.25                       # steps sent beyond the target; the pose stops the rover first
MAX_STEPS = 65535                     # uint16 in the serial packet
FAULTS = ("timeout", "exception", "missing", "invalid")


class RoverError(RuntimeError):
    pass


def steps(command: str, metres_or_degrees: float) -> int:
    per = STEPS_PER_M if command in ("FWD", "REV") else STEPS_PER_DEG
    return int(min(MAX_STEPS, round(HEADROOM * abs(metres_or_degrees) * per)))


class Rover:
    def __init__(self, ip: str, level: int = 1, port: int = 5000, name: str = "", timeout_s: float = 2.0) -> None:
        if level not in (1, 2, 3):
            raise ValueError("speed level is 1, 2 or 3")
        ip = ip.strip()
        self.level = level
        self._url = f"http://{ip}:{port}/"
        # The controller's namespace: ROVER_NAME on the rover if set, else its IP.
        self._status_url = f"{self._url}{name or 'ip_' + ip.replace('.', '_')}/command_status"
        self._timeout_s = timeout_s
        self._last_status: str | None = None

    def send(self, command: str, step_count: int = 0) -> None:
        self._last_status = self.status()
        action = {"command": command, "speed": 0 if command == "STOP" else self.level, "distance": step_count}
        request = urllib.request.Request(self._url, json.dumps({"wheel_action": action}).encode(),
                                         {"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=self._timeout_s) as response:
                reply = response.read()
        except (urllib.error.URLError, OSError) as exc:
            raise RoverError(f"rover controller at {self._url} unreachable: {exc}") from exc
        try:
            status = json.loads(reply).get("status") if reply else None
        except (ValueError, AttributeError):
            status = None
        if status not in (None, 200, "200"):
            raise RoverError(f"rover controller refused {command} {action}: {reply!r}")

    def stop(self) -> None:
        self.send("STOP")

    def fault(self) -> str | None:
        """The controller's status if it reported a failed command since the last send."""
        status = self.status()
        if status is None or status == self._last_status:
            return None
        return status if any(token in status.lower() for token in FAULTS) else None

    def status(self) -> str | None:
        try:
            with urllib.request.urlopen(self._status_url, timeout=min(self._timeout_s, 1.0)) as response:
                return str(json.loads(response.read()).get("status", ""))
        except (urllib.error.URLError, OSError, ValueError, AttributeError):
            return None
