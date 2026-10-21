import numpy as np
import pytest
from support import TEST_CONFIG, saved_map

from rover_twin.client import TwinClient
from rover_twin.twin import offline


def test_a_client_plans_on_the_offline_twins_map_and_rehearses_on_it(tmp_path):
    path = saved_map(tmp_path / "map.npz", (-1.0, 3.0, -1.5, 1.5, 0.0), (1.5, 1.8, 0.6, 0.9, 0.25))
    _, server = offline.start(path, TEST_CONFIG, "127.0.0.1", 0)
    try:
        twin = TwinClient(f"http://127.0.0.1:{server.server_address[1]}")
        grid = twin.map()
        assert grid.obstacle[grid.cell(1.65, 0.75)] and not grid.obstacle[grid.cell(1.0, 0.0)]
        assert np.isnan(grid.height_m[grid.cell(0.0, 2.0)]) and grid.state[grid.cell(0.0, 2.0)] == 0   # never seen

        clear = twin.rehearse((0.0, 0.0, 0.0), [twin.forward(1.0), twin.left(90), twin.forward(0.4)])
        assert clear["approved"], clear["reason"]
        assert clear["final_pose"]["y"] == pytest.approx(0.4, abs=0.03)
        # 1.4 m at 0.1875 m/s and 90 deg at the Newton rover's 19.5 deg/s, plus stopping
        assert clear["duration_s"] == pytest.approx(1.4 / 0.1875 + 90 / 19.5, rel=0.1)
        assert sum(s["duration_s"] for s in clear["step_results"]) == pytest.approx(clear["duration_s"], abs=0.05)

        boxed = twin.rehearse((1.0, 0.75, 0.0), [twin.forward(1.0)])
        assert not boxed["approved"] and boxed["failure_type"] == "collision"
        # A refusal no single command caused still carries the index, as null.
        for refused in (twin.rehearse((0.0, 0.0, 0.0), [twin.left(10)] * 13),
                        twin.rehearse((0.0, 0.0, 0.0), [twin.forward(0.5)], goal=(2.0, 0.0, 0.0))):
            assert not refused["approved"] and refused["failed_command_index"] is None
    finally:
        server.shutdown()
