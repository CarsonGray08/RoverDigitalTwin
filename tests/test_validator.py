import unittest

from digital_twin.config import PlannerSettings
from digital_twin.models import CircleObstacle, CommandName, CommandStep, Pose, ValidationRequest, WorldModel
from digital_twin.validator import KinematicValidator


class KinematicValidatorTests(unittest.TestCase):
    def test_approves_direct_safe_plan(self):
        validator = KinematicValidator(PlannerSettings())
        result = validator.validate(
            ValidationRequest(
                start_pose=Pose(0, 0, 0),
                goal=Pose(1, 0, 0),
                commands=(CommandStep(CommandName.FWD, 40, 23000), CommandStep(CommandName.STOP, 0, 0)),
            )
        )

        self.assertTrue(result.approved)
        self.assertAlmostEqual(result.goal_error_m, 0.0)

    def test_turn_distance_is_degrees_in_simulation(self):
        validator = KinematicValidator(PlannerSettings())
        result = validator.validate(
            ValidationRequest(
                start_pose=Pose(0, 0, 0),
                goal=Pose(0, 0, 90),
                commands=(CommandStep(CommandName.LEFT, 30, 90), CommandStep(CommandName.STOP, 0, 0)),
            )
        )

        self.assertTrue(result.approved)
        self.assertAlmostEqual(result.final_pose.yaw_deg, 90.0)

    def test_rejects_unreasonable_turn_distance(self):
        validator = KinematicValidator(PlannerSettings())
        result = validator.validate(
            ValidationRequest(
                start_pose=Pose(0, 0, 0),
                goal=Pose(0, 0, 0),
                commands=(CommandStep(CommandName.LEFT, 40, 5000), CommandStep(CommandName.STOP, 0, 0)),
            )
        )

        self.assertFalse(result.approved)
        self.assertEqual(result.failure_type, "turn_distance_limit")
        self.assertEqual(result.failed_command_index, 0)

    def test_rejects_obstacle_clearance_violation(self):
        settings = PlannerSettings(
            world=WorldModel(obstacles=(CircleObstacle(name="table", x=0.5, y=0.0, radius_m=0.2),))
        )
        validator = KinematicValidator(settings)
        result = validator.validate(
            ValidationRequest(
                start_pose=Pose(0, 0, 0),
                goal=Pose(1, 0, 0),
                commands=(CommandStep(CommandName.FWD, 40, 23000),),
            )
        )

        self.assertFalse(result.approved)
        self.assertEqual(result.failure_type, "collision")
        self.assertEqual(result.failed_command_index, 0)


if __name__ == "__main__":
    unittest.main()
