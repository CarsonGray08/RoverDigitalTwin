import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from digital_twin.executor import ConfirmingPublisher, DriveAborted
from digital_twin.geometry import advance_pose
from digital_twin.live_twin_session import (
    SESSION_NAME,
    new_session,
    resolve_session_dir,
    write_session,
)
from digital_twin.models import CommandName, CommandStep, Pose


class RecordingPublisher:
    def __init__(self):
        self.published: list[CommandStep] = []
        self.closed = False

    def publish(self, command: CommandStep) -> None:
        self.published.append(command)

    def close(self) -> None:
        self.closed = True


FWD = CommandStep(CommandName.FWD, 40, 1000)
STOP = CommandStep(CommandName.STOP, 0, 0)


class ConfirmingPublisherTest(unittest.TestCase):
    def test_stop_is_always_forwarded_without_prompting(self):
        inner = RecordingPublisher()
        prompted = []
        pub = ConfirmingPublisher(inner, prompt=lambda msg: prompted.append(msg) or "n")
        pub.publish(STOP)
        self.assertEqual(inner.published, [STOP])
        self.assertEqual(prompted, [])  # never asked

    def test_yes_forwards_the_command(self):
        inner = RecordingPublisher()
        pub = ConfirmingPublisher(inner, prompt=lambda _msg: "y")
        pub.publish(FWD)
        self.assertEqual(inner.published, [FWD])

    def test_decline_aborts_and_does_not_forward(self):
        inner = RecordingPublisher()
        pub = ConfirmingPublisher(inner, prompt=lambda _msg: "")
        with self.assertRaises(DriveAborted):
            pub.publish(FWD)
        self.assertEqual(inner.published, [])

    def test_auto_yes_skips_the_prompt(self):
        inner = RecordingPublisher()
        pub = ConfirmingPublisher(
            inner, auto_yes=True,
            prompt=lambda _msg: self.fail("should not prompt with auto_yes"))
        pub.publish(FWD)
        self.assertEqual(inner.published, [FWD])

    def test_close_delegates_to_inner(self):
        inner = RecordingPublisher()
        ConfirmingPublisher(inner).close()
        self.assertTrue(inner.closed)


class ResolveSessionDirTest(unittest.TestCase):
    def _make_session(self, path: Path) -> None:
        write_session(path, new_session("s"))

    def test_direct_session_dir_returns_itself(self):
        with TemporaryDirectory() as tmp:
            d = Path(tmp)
            self._make_session(d)
            self.assertEqual(resolve_session_dir(d), d)

    def test_latest_pointer_is_followed(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            session = root / "session-1"
            self._make_session(session)
            os.symlink(session, root / "latest")
            self.assertEqual(resolve_session_dir(root), session.resolve())

    def test_newest_session_dir_wins_without_pointer(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            old = root / "session-old"
            new = root / "session-new"
            self._make_session(old)
            self._make_session(new)
            os.utime(old / SESSION_NAME, (1, 1))
            os.utime(old, (1, 1))
            self.assertEqual(resolve_session_dir(root), new)

    def test_unknown_dir_returned_unchanged(self):
        with TemporaryDirectory() as tmp:
            d = Path(tmp) / "nope"
            self.assertEqual(resolve_session_dir(d), d)


class ForwardGoalTest(unittest.TestCase):
    def test_one_meter_ahead_follows_heading(self):
        # The goal a driver builds is exactly advance_pose along the current yaw.
        north = advance_pose(Pose(2.0, -1.0, 90.0), 1.0)
        self.assertAlmostEqual(north.x, 2.0, places=6)
        self.assertAlmostEqual(north.y, 0.0, places=6)
        self.assertAlmostEqual(north.yaw_deg, 90.0, places=6)


if __name__ == "__main__":
    unittest.main()
