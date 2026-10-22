# Lab operations

Run commands from the repository root on the [configured workstation](setup.md).

## Start and drive

1. Run `./scripts/dashboard.sh` and open the printed URL.
2. Press **Start mapper and twin**. Connect LiveScanner to the workstation IP
   and mount the phone on the rover.
3. Press **360° scan**: it starts the phone scanning, waits for the map to
   anchor (the rover must hold still for a second; the map anchors only once
   the phone is scanning, so it can be handled before that), then turns. The
   scan's turns are not rehearsed on the twin; the live pose still stops each.
   Starting the scan on the phone itself works too, once the phone is mounted.
   Then use **Rehearse & drive**. Motion is rehearsed only until the arm
   switch is on and a rover IP is set. **STOP** stops the rover.

`./scripts/rover_twin.sh` starts mapper and twin without the dashboard; Ctrl-C
stops both. The arm switch resets on page load.

To run a plan (`{"commands": [...], "goal": {...}}`; goal optional):

```bash
.venv/bin/python -m rover_twin.ops.drive plan.json                                # rehearse from live pose
.venv/bin/python -m rover_twin.ops.drive plan.json --arm --rover-ip 192.168.0.157   # drive
```

See [the API reference](twin-api.md) for command units, limits and refusal results.
The driver rehearses the remaining plan before each command and stops on a
refusal, stale/lost pose, controller fault, completion or Ctrl-C.

The drive tool has only been tested against a simulated rover. For the first
hardware trial, have someone at the hardware stop, confirm the controller's
`/<rover-name>/command_status` endpoint responds, then rehearse and fly a
0.25 m drive (`FWD`, speed 30, distance 5750) and a 90° turn. Keep the logs.
Real speed levels 2/3 and the rover's climbable step height still need measuring.

## Record and replay

Live sessions save `phone.scanlog`, `elevation.npz`, `metrics.json` and other
mapper outputs under `runs/<time>/`; the twin adds `twin_metrics.json`.
Drive timing logs go to `runs/drive_log.jsonl` from the dashboard, or
`drive_log.jsonl` from the CLI (`--log` overrides it).

```bash
./scripts/rover_twin.sh --replay runs/spin1.scanlog --speed 1
.venv/bin/python -m rover_twin.mapper.main --replay runs/spin1.scanlog --speed 0
.venv/bin/python tools/evaluate.py runs/spin1.truth.toml run=runs/<time>
```

`--speed 1` preserves recording timing; `--speed 0` replays quickly. Scoring
needs measured ground truth, for example:

```toml
kind = "spin"  # spin, drive, loop or scene
[end]
x_m = 0.0
y_m = 0.0
yaw_deg = 0.0
[[boxes]]
name = "crate"
center_m = [1.4, -1.2]
size_m = [0.40, 0.30, 0.25]
```

Lab reference recordings (`drift_run` and `loop`) stay on the workstation,
outside git. For Python commands, use `env -u PYTHONPATH` if ROS is sourced.
