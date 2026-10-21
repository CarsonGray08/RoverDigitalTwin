import socket
import struct
import threading
import time

import numpy as np
import pytest

from rover_twin import frames
from rover_twin.frames import Intrinsics
from rover_twin.phone import protocol, recording
from rover_twin.phone.protocol import Arrival, Event, Imu, Msg, PhonePose, ProtocolError, RgbdFrame
from rover_twin.phone.receiver import PhoneLink


def rgbd(t: float = 12.5) -> RgbdFrame:
    rng = np.random.default_rng(0)
    return RgbdFrame(
        t_phone=t,
        T_A_C=frames.rot_z(0.4, (0.5, 1.0, -2.0)),
        depth=rng.uniform(0.2, 4.0, (6, 8)).astype(np.float32),
        confidence=rng.integers(0, 3, (6, 8), dtype=np.uint8),
        jpeg=b"\xff\xd8 not really a jpeg \xff\xd9",
        color_intrinsics=Intrinsics(64.0, 64.0, 32.0, 24.0, 64, 48),
    )


def payload(message) -> bytes:
    return protocol.encode(message)[protocol.LENGTH.size:]


def test_rgbd_round_trip():
    sent = rgbd()
    got = protocol.decode(payload(sent))
    assert got.t_phone == sent.t_phone and got.jpeg == sent.jpeg
    np.testing.assert_allclose(got.T_A_C, sent.T_A_C, atol=1e-6)   # f32 on the wire
    np.testing.assert_array_equal(got.depth, sent.depth)
    np.testing.assert_array_equal(got.confidence, sent.confidence)
    assert got.depth_intrinsics == Intrinsics(8.0, 8.0, 4.0, 3.0, 8, 6)


@pytest.mark.parametrize("message", [
    PhonePose(3.0, frames.rot_z(-1.0, (1.0, 2.0, 3.0)), protocol.TRACKING_NORMAL),
    Imu(4.0, (0.0, -9.75, 0.25), (0.5, 0.0, -0.125)),
    Event(Msg.REALIGN, 5.0),
])
def test_small_messages_round_trip(message):
    got = protocol.decode(payload(message))
    assert type(got) is type(message) and got.t_phone == message.t_phone
    if isinstance(message, PhonePose):
        np.testing.assert_allclose(got.T_A_C, message.T_A_C, atol=1e-6)
        assert got.tracking == message.tracking
    else:
        assert got == message


def test_malformed_payloads_are_refused_and_unknown_types_skipped():
    good = payload(rgbd())
    with pytest.raises(ProtocolError):
        protocol.decode(b"XXXX" + good[4:])
    with pytest.raises(ProtocolError):
        protocol.decode(good[:-1])
    mesh_anchor = good[:5] + bytes([1]) + good[6:]   # the app's mesh-anchor type, which the host skips
    assert protocol.decode(mesh_anchor) is None


def test_read_frame_survives_a_stall_mid_message():
    a, b = socket.socketpair()
    b.settimeout(0.05)
    wire = protocol.encode(rgbd())

    def send_in_two_parts():
        a.sendall(wire[:100])
        time.sleep(0.2)                 # several read timeouts
        a.sendall(wire[100:])
        a.close()

    threading.Thread(target=send_in_two_parts).start()
    stop = threading.Event()
    assert protocol.read_frame(b, stop) == wire[4:]
    assert protocol.read_frame(b, stop) is None     # EOF


def test_recording_round_trip_tolerates_a_truncated_tail(tmp_path):
    path = tmp_path / "scan.scanlog"
    rec = recording.Recorder(path)
    for t, p in [(10.0, b"a"), (10.25, b"bb"), (11.0, b"ccc")]:
        rec.write(Arrival(t, p))
    rec.close()
    with open(path, "ab") as fh:
        fh.write(struct.pack("<dI", 1.5, 99) + b"cut")
    with pytest.warns(UserWarning, match="truncated"):
        assert list(recording.read(path)) == [(0.0, b"a"), (0.25, b"bb"), (1.0, b"ccc")]


def test_replay_is_paced_at_recorded_arrival_times(tmp_path):
    path = tmp_path / "scan.scanlog"
    rec = recording.Recorder(path)
    for t in (0.0, 0.1, 0.2, 0.3):
        rec.write(Arrival(t, b"x"))
    rec.close()

    now = [100.0]
    sleeps = []

    def sleep(s):
        sleeps.append(round(s, 6))
        now[0] += s

    replay = recording.replay(path, speed=2.0, clock=lambda: now[0], sleep=sleep)
    assert next(replay).t == 100.0
    assert next(replay).t == pytest.approx(100.05)
    now[0] += 0.2        # the consumer stalls past the next two due times
    assert [a.t for a in replay] == pytest.approx([100.1, 100.15])
    assert sleeps == [0.05]


def test_phone_link_queues_arrivals_and_enables_imu():
    link = PhoneLink("127.0.0.1", 0)
    try:
        phone = socket.create_connection(("127.0.0.1", link.port))
        phone.sendall(protocol.encode(rgbd()) + protocol.encode(Event(Msg.FINALIZE, 13.0)))
        arrivals = iter(link)
        first, second = next(arrivals), next(arrivals)
        assert isinstance(protocol.decode(first.payload), RgbdFrame)
        assert protocol.decode(second.payload) == Event(Msg.FINALIZE, 13.0)
        assert first.t <= second.t <= time.monotonic()
        phone.settimeout(2.0)
        assert protocol.decode(protocol.read_frame(phone, threading.Event())) == Event(Msg.ENABLE_IMU)
        phone.close()
    finally:
        link.close()
