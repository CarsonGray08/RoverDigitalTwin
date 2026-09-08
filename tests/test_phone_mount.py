import json
import math
import os
import tempfile
import unittest

import numpy as np

from digital_twin.models import Pose
from digital_twin.nvblox_feeder import (
    arkit_camera_pose_to_nvblox,
    turn_in_place_camera_pose_to_nvblox,
)
from digital_twin.phone_mount import (
    AlignmentStabilityGate,
    Mat4,
    PhoneMountCalibration,
    assert_rigid,
    compute_t_align,
    host_from_mat4,
    identity_mat4,
    mat4_multiply,
    quaternion_wxyz_to_mat4,
    rover_pose_from_camera,
    rover_pose_from_twin_camera_matrix,
)


def rot_y(deg: float) -> Mat4:
    """Rotation about the ARKit world up-axis (Y), column-vector convention."""
    c = math.cos(math.radians(deg))
    s = math.sin(math.radians(deg))
    return (
        (c, 0.0, s, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (-s, 0.0, c, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )


def make_T_A_R(px: float, pz: float, yaw_deg: float) -> Mat4:
    """Rover body pose in ARKit world: on the ground (Y=0), yawed about Y.

    After the M_axis swap (x, -z, y), twin XY = (px, -pz) and twin yaw = yaw_deg.
    """
    m = rot_y(yaw_deg)
    return (
        (m[0][0], m[0][1], m[0][2], px),
        (m[1][0], m[1][1], m[1][2], 0.0),
        (m[2][0], m[2][1], m[2][2], pz),
        (0.0, 0.0, 0.0, 1.0),
    )


def camera_host_transform(t_a_r: Mat4, calibration: PhoneMountCalibration):
    """The streamed camera transform a phone would report: T_A_C = T_A_R @ T_R_C."""
    t_a_c = mat4_multiply(t_a_r, calibration.T_R_C)
    return host_from_mat4(t_a_c)


# A realistic non-trivial mount: camera 0.2 m forward, 0.15 m up, 0.03 m right,
# pitched down ~30 deg (rotation about rover X). Quaternion for -30 deg about X.
_PITCH = math.radians(-30.0)
_MOUNT = PhoneMountCalibration.from_dict(
    {
        "rover_forward_axis": "x",
        "camera_in_rover": {
            "translation_m": [0.2, 0.03, 0.15],
            "quaternion_wxyz": [math.cos(_PITCH / 2), math.sin(_PITCH / 2), 0.0, 0.0],
        },
    }
)


class PhoneMountMathTests(unittest.TestCase):
    def test_identity_mount_recovers_axis_swapped_camera_pose(self):
        cal = PhoneMountCalibration.identity()
        host = camera_host_transform(make_T_A_R(1.5, -0.7, 0.0), cal)
        pose = rover_pose_from_camera(host, cal)
        self.assertAlmostEqual(pose.x, 1.5, places=5)
        self.assertAlmostEqual(pose.y, 0.7, places=5)  # twin y = -pz = -(-0.7)
        self.assertAlmostEqual(pose.yaw_deg, 0.0, places=5)

    def test_stationary_pose_is_stable_and_correct(self):
        host = camera_host_transform(make_T_A_R(0.4, 0.9, 25.0), _MOUNT)
        pose_a = rover_pose_from_camera(host, _MOUNT)
        pose_b = rover_pose_from_camera(host, _MOUNT)
        self.assertEqual((pose_a.x, pose_a.y, pose_a.yaw_deg), (pose_b.x, pose_b.y, pose_b.yaw_deg))
        # Rover centre, not the offset camera, is reported.
        self.assertAlmostEqual(pose_a.x, 0.4, places=5)
        self.assertAlmostEqual(pose_a.y, -0.9, places=5)
        self.assertAlmostEqual(pose_a.yaw_deg, 25.0, places=4)

    def test_twin_camera_matrix_recovers_rover_pose(self):
        host = camera_host_transform(make_T_A_R(0.4, 0.9, 25.0), _MOUNT)
        t_align = compute_t_align(host, _MOUNT, Pose(1.0, 2.0, 30.0))
        arkit_pose = rover_pose_from_camera(host, _MOUNT, t_align)
        camera_to_twin = arkit_camera_pose_to_nvblox(host, t_align)

        cuvslam_pose = rover_pose_from_twin_camera_matrix(camera_to_twin, _MOUNT)

        self.assertAlmostEqual(cuvslam_pose.x, arkit_pose.x, places=5)
        self.assertAlmostEqual(cuvslam_pose.y, arkit_pose.y, places=5)
        self.assertAlmostEqual(cuvslam_pose.yaw_deg, arkit_pose.yaw_deg, places=4)

    def test_rotating_rover_90_changes_yaw_by_90(self):
        host0 = camera_host_transform(make_T_A_R(0.0, 0.0, 0.0), _MOUNT)
        host90 = camera_host_transform(make_T_A_R(0.0, 0.0, 90.0), _MOUNT)
        self.assertAlmostEqual(rover_pose_from_camera(host0, _MOUNT).yaw_deg, 0.0, places=4)
        self.assertAlmostEqual(rover_pose_from_camera(host90, _MOUNT).yaw_deg, 90.0, places=4)

    def test_spin_in_place_keeps_rover_xy_fixed_despite_lever_arm(self):
        # Rover centre fixed; spin through a full turn. With the lever arm the
        # CAMERA orbits, but the recovered rover XY must stay put.
        center = (0.6, -0.25)
        cam_positions = []
        for yaw in range(0, 360, 30):
            t_a_r = make_T_A_R(center[0], center[1], float(yaw))
            host = camera_host_transform(t_a_r, _MOUNT)
            pose = rover_pose_from_camera(host, _MOUNT)
            self.assertAlmostEqual(pose.x, 0.6, places=4)
            self.assertAlmostEqual(pose.y, 0.25, places=4)  # -(-0.25)
            self.assertAlmostEqual(pose.yaw_deg, ((yaw + 180) % 360) - 180, places=4)
            cam_positions.append((host[12], host[13], host[14]))
        # Sanity: the camera really did move (otherwise the test is trivial).
        xs = [p[0] for p in cam_positions]
        self.assertGreater(max(xs) - min(xs), 0.2)

    def test_turn_in_place_mapper_pose_ignores_arkit_translation_drift(self):
        locked = Pose(1.0, 2.0, 0.0)
        host_a = camera_host_transform(make_T_A_R(0.0, 0.0, 0.0), _MOUNT)
        host_b = camera_host_transform(make_T_A_R(5.0, -4.0, 0.0), _MOUNT)

        pose_a = turn_in_place_camera_pose_to_nvblox(
            host_a, _MOUNT, None, locked, locked_rover_z=0.4)
        pose_b = turn_in_place_camera_pose_to_nvblox(
            host_b, _MOUNT, None, locked, locked_rover_z=0.4)

        self.assertTrue(np.allclose(pose_a, pose_b))
        self.assertTrue(np.allclose(pose_a[:3, 3], [1.2, 2.03, 0.55], atol=1e-6))

    def test_turn_in_place_mapper_pose_keeps_mount_orbit_from_yaw(self):
        locked = Pose(1.0, 2.0, 0.0)
        host = camera_host_transform(make_T_A_R(7.0, -3.0, 90.0), _MOUNT)

        pose = turn_in_place_camera_pose_to_nvblox(
            host, _MOUNT, None, locked, locked_rover_z=0.4)

        self.assertTrue(np.allclose(pose[:3, 3], [0.97, 2.2, 0.55], atol=1e-6))


class TwinAlignmentTests(unittest.TestCase):
    def test_start_pose_maps_to_origin(self):
        start_host = camera_host_transform(make_T_A_R(0.3, -0.2, 17.0), _MOUNT)
        t_align = compute_t_align(start_host, _MOUNT, Pose(0.0, 0.0, 0.0))
        pose = rover_pose_from_camera(start_host, _MOUNT, t_align)
        self.assertAlmostEqual(pose.x, 0.0, places=5)
        self.assertAlmostEqual(pose.y, 0.0, places=5)
        self.assertAlmostEqual(pose.yaw_deg, 0.0, places=4)

    def test_alignment_stability_gate_waits_for_stationary_window(self):
        gate = AlignmentStabilityGate(
            samples_required=3,
            max_translation_m=0.03,
            max_yaw_deg=2.0,
        )
        moving = [
            camera_host_transform(make_T_A_R(0.00, 0.0, 0.0), _MOUNT),
            camera_host_transform(make_T_A_R(0.10, 0.0, 0.0), _MOUNT),
            camera_host_transform(make_T_A_R(0.20, 0.0, 0.0), _MOUNT),
        ]
        for host in moving:
            ready, _ = gate.add(host, _MOUNT)
            self.assertFalse(ready)
        stable = camera_host_transform(make_T_A_R(0.20, 0.0, 0.0), _MOUNT)
        self.assertFalse(gate.add(stable, _MOUNT)[0])
        self.assertTrue(gate.add(stable, _MOUNT)[0])

    def test_start_pose_maps_to_configured_nonzero_pose(self):
        start_host = camera_host_transform(make_T_A_R(-1.1, 0.4, -40.0), _MOUNT)
        target = Pose(1.0, 2.0, 30.0)
        t_align = compute_t_align(start_host, _MOUNT, target)
        pose = rover_pose_from_camera(start_host, _MOUNT, t_align)
        self.assertAlmostEqual(pose.x, target.x, places=5)
        self.assertAlmostEqual(pose.y, target.y, places=5)
        self.assertAlmostEqual(pose.yaw_deg, target.yaw_deg, places=4)

    def test_rotation_direction_is_not_mirrored(self):
        # Pin the start at origin; rotate the rover +30 deg about ARKit-up. The
        # twin yaw must move +30 (same sign), not -30.
        start_host = camera_host_transform(make_T_A_R(0.3, -0.2, 10.0), _MOUNT)
        t_align = compute_t_align(start_host, _MOUNT, Pose(0.0, 0.0, 0.0))
        rotated_host = camera_host_transform(make_T_A_R(0.3, -0.2, 40.0), _MOUNT)
        pose = rover_pose_from_camera(rotated_host, _MOUNT, t_align)
        self.assertAlmostEqual(pose.x, 0.0, places=4)
        self.assertAlmostEqual(pose.y, 0.0, places=4)
        self.assertAlmostEqual(pose.yaw_deg, 30.0, places=4)

    def test_forward_motion_maps_to_correct_direction(self):
        # Start at origin facing +x (twin). Drive the rover forward 0.5 m; twin x
        # must increase by 0.5 (correct direction, lever arm removed).
        start_host = camera_host_transform(make_T_A_R(0.0, 0.0, 0.0), _MOUNT)
        t_align = compute_t_align(start_host, _MOUNT, Pose(0.0, 0.0, 0.0))
        moved_host = camera_host_transform(make_T_A_R(0.5, 0.0, 0.0), _MOUNT)
        pose = rover_pose_from_camera(moved_host, _MOUNT, t_align)
        self.assertAlmostEqual(pose.x, 0.5, places=5)
        self.assertAlmostEqual(pose.y, 0.0, places=5)
        self.assertAlmostEqual(pose.yaw_deg, 0.0, places=4)


class AssertRigidTests(unittest.TestCase):
    def test_accepts_proper_rotation(self):
        assert_rigid(identity_mat4())
        assert_rigid(rot_y(37.0))

    def test_rejects_mirrored_left_handed_frame(self):
        # Flip one axis -> determinant -1 (mirrored). Must be refused.
        mirrored = (
            (1.0, 0.0, 0.0, 0.0),
            (0.0, 1.0, 0.0, 0.0),
            (0.0, 0.0, -1.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        )
        with self.assertRaises(ValueError):
            assert_rigid(mirrored, name="T_R_C")

    def test_rejects_non_orthonormal(self):
        scaled = (
            (2.0, 0.0, 0.0, 0.0),
            (0.0, 1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        )
        with self.assertRaises(ValueError):
            assert_rigid(scaled)


class CalibrationLoadingTests(unittest.TestCase):
    def test_matrix_and_quaternion_paths_agree(self):
        quat_cal = PhoneMountCalibration.from_dict(
            {"camera_in_rover": {"translation_m": [0.1, 0.2, 0.3], "quaternion_wxyz": [1, 0, 0, 0]}}
        )
        matrix_cal = PhoneMountCalibration.from_dict(
            {
                "matrix_T_R_C": [
                    1, 0, 0, 0.1,
                    0, 1, 0, 0.2,
                    0, 0, 1, 0.3,
                    0, 0, 0, 1,
                ]
            }
        )
        self.assertEqual(quat_cal.lever_arm_m, (0.1, 0.2, 0.3))
        self.assertEqual(matrix_cal.lever_arm_m, (0.1, 0.2, 0.3))

    def test_rejects_unknown_forward_axis(self):
        with self.assertRaises(ValueError):
            PhoneMountCalibration.from_dict(
                {"rover_forward_axis": "diagonal", "camera_in_rover": {}}
            )

    def test_rejects_missing_mount_spec(self):
        with self.assertRaises(ValueError):
            PhoneMountCalibration.from_dict({"rover_forward_axis": "x"})

    def test_load_from_file(self):
        data = {
            "camera_in_rover": {"translation_m": [0.0, 0.0, 0.0], "quaternion_wxyz": [1, 0, 0, 0]}
        }
        fd, path = tempfile.mkstemp(suffix=".json")
        try:
            with os.fdopen(fd, "w") as handle:
                json.dump(data, handle)
            cal = PhoneMountCalibration.load(path)
            self.assertEqual(cal.lever_arm_m, (0.0, 0.0, 0.0))
        finally:
            os.unlink(path)

    def test_negative_forward_axis_flips_yaw_180(self):
        cal_x = PhoneMountCalibration.from_dict(
            {"rover_forward_axis": "x", "camera_in_rover": {"quaternion_wxyz": [1, 0, 0, 0]}}
        )
        cal_negx = PhoneMountCalibration.from_dict(
            {"rover_forward_axis": "-x", "camera_in_rover": {"quaternion_wxyz": [1, 0, 0, 0]}}
        )
        host = camera_host_transform(make_T_A_R(0.0, 0.0, 0.0), cal_x)
        self.assertAlmostEqual(rover_pose_from_camera(host, cal_x).yaw_deg, 0.0, places=5)
        self.assertAlmostEqual(abs(rover_pose_from_camera(host, cal_negx).yaw_deg), 180.0, places=5)


if __name__ == "__main__":
    unittest.main()
