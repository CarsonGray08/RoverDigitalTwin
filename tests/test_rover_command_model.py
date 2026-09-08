import unittest

from digital_twin.models import CommandName, SafetyLimits
from digital_twin.rover_command_model import (
    SCHEMA_MAX_DISTANCE,
    STEP_RATE_HZ,
    RoverCommandCalibration,
)


class StepRateTests(unittest.TestCase):
    def test_effective_step_rate_matches_firmware(self):
        # 100 rpm * 200 steps/rev / 60 = 333.3 nominal, /(6/7) = ~388.9 steps/s.
        self.assertAlmostEqual(STEP_RATE_HZ, 388.9, delta=0.5)


class LinearMappingTests(unittest.TestCase):
    def setUp(self):
        self.cal = RoverCommandCalibration()

    def test_default_matches_measured_safety_limits(self):
        # 23000 steps/m (SafetyLimits.distance_units_per_m).
        self.assertEqual(self.cal.distance_to_steps(1.0), 23000)
        self.assertEqual(self.cal.distance_to_steps(0.5), 11500)
        self.assertAlmostEqual(self.cal.steps_to_distance(23000), 1.0, places=4)

    def test_from_limits_binds_to_settings(self):
        cal = RoverCommandCalibration.from_limits(SafetyLimits())
        self.assertEqual(cal.steps_per_m, 23000.0)
        self.assertAlmostEqual(cal.steps_per_deg, 24000.0 / 90.0, places=4)
        # 90 deg turn -> 24000 steps (the measured turn anchor).
        self.assertEqual(cal.yaw_to_steps(90.0), 24000)

    def test_distance_round_trip(self):
        steps = self.cal.distance_to_steps(1.0)
        self.assertAlmostEqual(self.cal.steps_to_distance(steps), 1.0, places=3)

    def test_drive_command_direction_and_steps(self):
        fwd = self.cal.drive_command(0.5)
        rev = self.cal.drive_command(-0.5)
        self.assertEqual(fwd.command, CommandName.FWD)
        self.assertEqual(rev.command, CommandName.REV)
        self.assertEqual(fwd.distance, 11500)
        self.assertEqual(rev.distance, 11500)  # magnitude

    def test_distance_clamped_to_schema_max(self):
        self.assertEqual(self.cal.distance_to_steps(1000.0), SCHEMA_MAX_DISTANCE)


class TurnMappingTests(unittest.TestCase):
    def setUp(self):
        self.cal = RoverCommandCalibration()

    def test_turn_command_direction(self):
        left = self.cal.turn_command(90.0)
        right = self.cal.turn_command(-90.0)
        self.assertEqual(left.command, CommandName.LEFT)
        self.assertEqual(right.command, CommandName.RIGHT)
        self.assertEqual(left.distance, right.distance)  # magnitude
        self.assertGreater(left.distance, 0)

    def test_default_turn_is_calibrated_from_safety_limits(self):
        # The default turn constant comes from the measured SafetyLimits.
        self.assertTrue(self.cal.turn_calibrated)
        self.assertEqual(self.cal.yaw_to_steps(90.0), 24000)

    def test_turn_round_trip(self):
        steps = self.cal.yaw_to_steps(45.0)
        self.assertAlmostEqual(self.cal.steps_to_yaw(steps), 45.0, delta=0.5)


class DurationTests(unittest.TestCase):
    def test_duration_is_steps_over_rate_plus_latency(self):
        cal = RoverCommandCalibration(command_latency_s=0.2)
        # 800 steps takes ~1 s of stepping + 0.2 s latency at the measured rate.
        self.assertAlmostEqual(cal.duration_s(800), 1.0 + 0.2, delta=0.05)

    def test_ninety_degree_turn_waits_about_thirty_seconds(self):
        # Measured anchor: a 90 deg turn (24000 steps) completes in ~30 s.
        cal = RoverCommandCalibration()
        self.assertAlmostEqual(cal.duration_s(cal.yaw_to_steps(90.0)), 30.0, delta=1.0)

    def test_longer_motion_takes_longer(self):
        cal = RoverCommandCalibration()
        self.assertGreater(cal.duration_s(2000), cal.duration_s(1000))


class CalibrationFitTests(unittest.TestCase):
    def test_fit_linear_recovers_true_ratio(self):
        true_ratio = 2500.0  # steps per metre
        measurements = [(int(true_ratio * m), m) for m in (0.25, 0.5, 1.0, 1.5)]
        cal = RoverCommandCalibration().fit_linear(measurements)
        self.assertAlmostEqual(cal.steps_per_m, true_ratio, delta=1.0)

    def test_fit_turn_recovers_true_ratio_and_sets_flag(self):
        true_ratio = 14.0  # steps per degree
        measurements = [(int(true_ratio * d), d) for d in (30.0, 90.0, 180.0)]
        cal = RoverCommandCalibration().fit_turn(measurements)
        self.assertAlmostEqual(cal.steps_per_deg, true_ratio, delta=0.5)
        self.assertTrue(cal.turn_calibrated)

    def test_fit_handles_sign_conventions(self):
        # Negative degrees (right turns) still fit the same magnitude ratio.
        measurements = [(140, -10.0), (1260, -90.0)]
        cal = RoverCommandCalibration().fit_turn(measurements)
        self.assertAlmostEqual(cal.steps_per_deg, 14.0, delta=0.5)

    def test_fit_rejects_all_zero(self):
        with self.assertRaises(ValueError):
            RoverCommandCalibration().fit_linear([(0, 0.0)])


if __name__ == "__main__":
    unittest.main()
