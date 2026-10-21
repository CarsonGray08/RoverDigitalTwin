"""Stream a recording into a running mapper as the phone would, paced at the
recorded arrival times.

It connects to the mapper's phone port, so a recording goes through the same
receiver as the live phone (``mapper.main --replay`` reads it directly).

usage: python tools/replay.py SCANLOG [--host 127.0.0.1] [--port 9099] [--speed 1]
"""

from __future__ import annotations

import argparse
import socket
import threading
import time
from pathlib import Path

from rover_twin.phone import protocol, recording


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("scanlog", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9099)
    parser.add_argument("--speed", type=float, default=1.0, help="1 = real time, 0 = as fast as the socket takes it")
    args = parser.parse_args()

    stop = threading.Event()
    sock = socket.create_connection((args.host, args.port))
    sock.settimeout(0.25)
    threading.Thread(target=_discard, args=(sock, stop), daemon=True).start()
    sent = 0
    for arrival in recording.replay(args.scanlog, args.speed):
        sock.sendall(protocol.LENGTH.pack(len(arrival.payload)) + arrival.payload)
        sent += 1
    time.sleep(2.0)       # let the mapper drain before the connection closes
    stop.set()
    sock.close()
    print(f"sent {sent} messages from {args.scanlog}")


def _discard(sock: socket.socket, stop: threading.Event) -> None:
    """The mapper asks the phone to enable its IMU; a recording cannot obey."""
    try:
        while protocol.read_frame(sock, stop) is not None:
            pass
    except OSError:
        pass


if __name__ == "__main__":
    main()
