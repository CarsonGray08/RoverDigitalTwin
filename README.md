# RASSOR Digital Twin

An iPhone on the rover streams LiDAR depth and pose to a mapper. Isaac Sim
runs the rover on that terrain and rehearses plans before the real rover
executes them. Student software uses the twin through HTTP or the Python client.

## Student development

Keep student software in its own repository. Use [docs/twin-api.md](docs/twin-api.md)
as the interface reference for you and your coding assistant.

On the configured lab workstation, use the running twin at `http://127.0.0.1:8765`.
For development on a saved map without Isaac Sim:

```bash
git clone https://github.com/CarsonGray08/RoverDigitalTwin.git
cd RoverDigitalTwin
python3 -m venv .venv
.venv/bin/pip install -e .
.venv/bin/python -m rover_twin.twin.offline assets/lab_map.npz
```

In another terminal, using the same environment:

```python
from rover_twin.client import TwinClient

twin = TwinClient("http://127.0.0.1:8765")
grid = twin.map()
result = twin.rehearse((0.0, 0.0, 0.0), [twin.forward(1.0), twin.left(90)])
```

## Lab use

Start `./scripts/dashboard.sh`, open the printed URL, and press **Start mapper
and twin**. Connect LiveScanner on the phone to the workstation, then scan.
Motion is rehearsed only until armed. See [operations](docs/operations.md) for
driving and recordings, and [setup](docs/setup.md) for a new workstation or rover.

Code is in `src/rover_twin/`; operator UI in `dashboard/`; runtime settings in
`config/`; rover assets and the saved lab map in `assets/`; phone app in
`ios/LiveScanner/`; firmware and ROS bridge in `robot/`.

Tests (no GPU or Isaac Sim required):

```bash
.venv/bin/pip install -e '.[dev]'
env -u PYTHONPATH .venv/bin/python -m pytest -q
```
