import math
from pathlib import Path

import numpy as np
import pytest

from rover_twin import config, frames
from rover_twin.frames import M_AXIS, Intrinsics, Mount, Pose2D

LAB = config.load(Path(__file__).parents[1] / "config" / "rover_twin.toml")
MOUNT = Mount.from_config(LAB)


def camera_in_arkit(rover: Pose2D, mount: Mount = MOUNT) -> np.ndarray:
    """T_A_C a phone would report for a rover at ``rover`` (identity alignment).
    Yaw is the heading of the rover's forward axis, not of its body x axis."""
    body_yaw = math.radians(rover.yaw_deg) - math.atan2(mount.forward[1], mount.forward[0])
    T_W_R = frames.rot_z(body_yaw, (rover.x, rover.y, 0.0))
    return frames.rigid_inverse(M_AXIS) @ T_W_R @ mount.T_R_C


def test_wire_transform_is_simd_column_major():
    T = frames.rot_z(0.3, (1.0, 2.0, 3.0))
    wire = frames.to_wire(T)
    assert wire[12:15] == (1.0, 2.0, 3.0)
    np.testing.assert_allclose(frames.from_wire(wire), T)


def test_lab_mount_looks_forward_and_level():
    # CV forward (+z) is the rover's driving direction (-y); CV down (+y) is rover down.
    np.testing.assert_allclose(MOUNT.T_R_C[:3, 2], [0.0, -1.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(MOUNT.T_R_C[:3, 1], [0.0, 0.0, -1.0], atol=1e-9)


@pytest.mark.parametrize("rover", [Pose2D(0.0, 0.0, 0.0), Pose2D(1.5, -0.4, 135.0), Pose2D(-2.0, 3.0, -60.0)])
def test_rover_pose_inverts_the_mount_chain(rover):
    got = frames.rover_pose(M_AXIS @ camera_in_arkit(rover), MOUNT)
    assert (got.x, got.y) == pytest.approx((rover.x, rover.y), abs=1e-9)
    assert got.yaw_deg == pytest.approx(rover.yaw_deg, abs=1e-9)


def test_planar_alignment_anchors_the_rover_at_start_without_tilt():
    # ARKit's world has an arbitrary heading and origin; the camera is not level.
    tilt = np.eye(4)
    tilt[:3, :3] = frames.rotation_wxyz([math.cos(0.05), math.sin(0.05), 0.0, 0.0])
    T_A_C = frames.rigid_inverse(M_AXIS) @ frames.rot_z(1.1, (0.7, -2.0, 0.3)) @ M_AXIS @ camera_in_arkit(
        Pose2D(0.2, 0.1, 40.0)) @ tilt
    start = Pose2D(1.0, 2.0, 30.0)
    T_W_A = frames.planar_alignment(T_A_C, MOUNT, start, start_z=0.05)
    got = frames.rover_pose(T_W_A @ T_A_C, MOUNT)
    assert (got.x, got.y, got.yaw_deg) == pytest.approx((1.0, 2.0, 30.0), abs=1e-9)
    assert (T_W_A @ T_A_C @ MOUNT.T_C_R)[2, 3] == pytest.approx(0.05)
    np.testing.assert_allclose((T_W_A @ frames.rigid_inverse(M_AXIS))[:3, 2], [0.0, 0.0, 1.0], atol=1e-12)


def test_backproject_uses_z_depth():
    intr = Intrinsics(100.0, 100.0, 2.0, 1.0, 4, 3)
    depth = np.zeros((3, 4), np.float32)
    depth[1, 2] = 2.0          # principal point
    depth[2, 3] = 1.0
    points = frames.backproject(depth, intr)
    np.testing.assert_allclose(points, [[0.0, 0.0, 2.0], [0.01, 0.01, 1.0]])
