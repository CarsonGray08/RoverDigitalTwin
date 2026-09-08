# Digital Twin Capstone

This project connects the lab rover and mounted iPhone to a live digital twin
using nvblox, Isaac Sim, and the rover's HTTP-to-ROS command bridge.

## Supported workflow

1. Start the dashboard with `./scripts/dashboard.sh`.
2. Start the live twin from the dashboard.
3. Connect the LiveScanner iPhone app and run the 360-degree baseline scan.
4. Review map and tracking readiness.
5. Run a dry rehearsal, then explicitly arm a validated drive.

The command-line equivalents are:

```bash
./scripts/live_twin.sh
python -m digital_twin.scan --rover-ip 192.168.0.157 --arm --telemetry-stop
python -m digital_twin.drive --rover-ip 192.168.0.157 --arm --telemetry-stop --distance-m 1.0
```

Real motion is never the default. The scan and drive commands require `--arm`;
without it they use the dry-run publisher. The scan command controls the phone
scan unless `--no-auto-scan` is used. Live-pose stopping is enabled with
`--telemetry-stop`. The
dashboard supplies that flag automatically whenever motion is armed.

Rover commands are sent to the rover's HTTP controller on port 5000. They
default to `continual`: the host sends Mode-A `linear_x`/`angular_z` JSON and
live odometry sends an explicit HTTP STOP. The controller translates that into
the rover's internal ROS topic. Select `step` only when bounded Mode-B
`command`/`speed`/`distance` messages are needed.

## Setup

Create a Python environment and install the host package:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]'
pytest
```

Edit `config/system.env` for the workstation's Isaac path and rover IP. Phone
mount calibration lives in `config/phone_mount.json`.

The mapper and Isaac validator run in separate Python
environments. `scripts/live_twin.sh` starts both and passes this project's
`src/` directory to Isaac's Python process.

## Project layout

- `src/digital_twin/`: mapping, simulation, planning, and rover runtime code.
- `dashboard/`: operator web interface and server assets.
- `ios/LiveScanner/`: source for the mounted iPhone RGB-D streamer.
- `robot/`: ROS 2 serial bridge and rover firmware.
- `assets/rover.usd`: Isaac rover scene.
- `tests/`: focused unit and integration tests for the supported workflow.
- `ARCHIVE.md`: scope of this repository and its research archive.

More detail is in [setup](docs/setup.md), [architecture](docs/architecture.md),
and [field operations](docs/operations.md).
