import time

import numpy as np
from support import map_scan

from rover_twin.frames import Pose2D
from rover_twin.link import messages
from rover_twin.link.client import LinkClient, Update
from rover_twin.link.messages import LivePose, Patch, Tiles, VisualMesh
from rover_twin.link.server import LinkServer
from rover_twin.mapping.tsdf import Mesh


def patch(key, value, n=4):
    height = np.full((n + 1, n + 1), value, np.float32)
    height[0, 0] = np.nan
    return Patch(key, height, np.where(np.isnan(height), 0, 1).astype(np.uint8))


def round_trip(message):
    return messages.decode(messages.encode(message)[4:])


def test_messages_round_trip():
    tiles = round_trip(Tiles(7, 12.5, True, 0.02, 4, (patch((-1, 2), 0.25), patch((0, 0), -0.5))))
    assert (tiles.rev, tiles.t_arrival, tiles.reset, tiles.cell_m, tiles.n) == (7, 12.5, True, 0.02, 4)
    assert [p.key for p in tiles.patches] == [(-1, 2), (0, 0)]
    np.testing.assert_array_equal(tiles.patches[0].height, patch((-1, 2), 0.25).height)   # NaN = unknown
    np.testing.assert_array_equal(tiles.patches[1].state, patch((0, 0), -0.5).state)
    pose = LivePose(7, 12.5, 3.25, Pose2D(1.0, -2.0, 45.0), 0.031)
    assert round_trip(pose) == pose
    mesh = Mesh(np.arange(9, dtype=np.float32).reshape(3, 3), np.array([[0, 1, 2]], np.int32),
                np.full((3, 3), 200, np.uint8))
    got = round_trip(VisualMesh(1.5, mesh)).mesh
    for a, b in ((got.vertices, mesh.vertices), (got.faces, mesh.faces), (got.colors, mesh.colors)):
        np.testing.assert_array_equal(a, b)


def test_update_keeps_every_unapplied_tile_and_only_the_latest_pose():
    update = Update()
    update.merge(Tiles(1, 1.0, False, 0.02, 4, (patch((0, 0), 0.1), patch((1, 0), 0.1))))
    update.merge(LivePose(1, 1.0, 1.0, Pose2D(0.0, 0.0, 0.0)))
    update.merge(Tiles(2, 2.0, True, 0.02, 4, (patch((0, 0), 0.2),)))      # a re-fuse: the old terrain goes
    update.merge(Tiles(3, 3.0, False, 0.02, 4, (patch((2, 0), 0.3),)))
    update.merge(LivePose(3, 3.0, 3.0, Pose2D(1.0, 0.0, 0.0)))
    assert update.reset and set(update.patches) == {(0, 0), (2, 0)}
    assert update.patches[(0, 0)].height[1, 1] == np.float32(0.2)
    assert update.pose.rev == 3 and update.tile_arrivals == [1.0, 2.0, 3.0]


def test_a_twin_connecting_mid_run_gets_the_map_the_mapper_holds(scan, tmp_path):
    server = LinkServer("127.0.0.1", 0)
    client = LinkClient("127.0.0.1", server.port)
    try:
        deadline = time.monotonic() + 5.0
        while not server.connected and time.monotonic() < deadline:
            time.sleep(0.01)
        mapper = map_scan(scan("drive"), tmp_path, link=server)
        terrain, pose = {}, None
        while (pose is None or pose.t_phone != mapper.track[-1].t_phone) and time.monotonic() < deadline:
            update = client.take()
            if update is None:
                time.sleep(0.01)
                continue
            if update.reset:
                terrain.clear()
            terrain.update(update.patches)
            pose = update.pose or pose
        assert set(terrain) == set(mapper.grid.tiles)
        for key, p in terrain.items():
            np.testing.assert_array_equal(p.height, mapper.grid.patch(key)[0])
        assert pose.rev == mapper.rev
    finally:
        client.close()
        server.close()
