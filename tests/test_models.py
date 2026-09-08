import unittest

from digital_twin.models import CommandPlan, ValidationError


class CommandPlanSchemaTests(unittest.TestCase):
    def test_accepts_strict_command_plan(self):
        plan = CommandPlan.from_dict(
            {
                "goal": {"x": 3.2, "y": -1.1, "yaw_deg": 90},
                "commands": [
                    {"command": "FWD", "speed": 40, "distance": 800},
                    {"command": "STOP", "speed": 0, "distance": 0},
                ],
            }
        )

        self.assertEqual(plan.commands[0].to_ros_payload(), [1, 40, 800])

    def test_rejects_unknown_fields(self):
        with self.assertRaises(ValidationError):
            CommandPlan.from_dict(
                {
                    "goal": {"x": 0, "y": 0, "yaw_deg": 0, "z": 0},
                    "commands": [],
                }
            )

    def test_rejects_unknown_command(self):
        with self.assertRaises(ValidationError):
            CommandPlan.from_dict(
                {
                    "goal": {"x": 0, "y": 0, "yaw_deg": 0},
                    "commands": [{"command": "SPIN", "speed": 10, "distance": 1}],
                }
            )


if __name__ == "__main__":
    unittest.main()

