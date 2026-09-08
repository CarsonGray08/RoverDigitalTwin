import base64
import tempfile
import unittest
from pathlib import Path

from digital_twin.observation_client import RoverObservation


class RoverObservationTests(unittest.TestCase):
    def test_saves_base64_vision_frame(self):
        image_bytes = b"\x89PNG\r\n\x1a\nfake"
        observation = RoverObservation(
            pose=None,
            payload={"vision": {"status": "ok", "rgb_base64": base64.b64encode(image_bytes).decode("ascii")}},
            vision_rgb_base64=base64.b64encode(image_bytes).decode("ascii"),
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = observation.save_vision_png(Path(tmp_dir) / "final.png")

            self.assertEqual(output_path.read_bytes(), image_bytes)

    def test_save_without_frame_raises_clear_error(self):
        observation = RoverObservation(pose=None, payload={}, vision_rgb_base64=None)

        with self.assertRaisesRegex(ValueError, "vision.rgb_base64"):
            observation.save_vision_png("unused.png")


if __name__ == "__main__":
    unittest.main()
