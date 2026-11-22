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

Under the map, **Clear map** forgets everything mapped so far (something left
behind as an obstacle, or a long session whose re-fuses have grown slow) and
keeps the rover where it is; the map grows again as the phone looks around,
so turn the rover or run a scan before driving. **Reset phone tracking**
restarts ARKit on the phone, for a pose that runs off or freezes while the
rover stands still: the map is cleared and anchors again once the rover holds
still, with the rover at the start pose. It needs the LiveScanner built from
this repository (older builds ignore it). Neither runs while the rover moves.

**Live views** shows the phone's camera (once it scans; **Rotate** turns the
image to match how the phone is mounted) and the twin's 3D view, each with
its age. The twin's view, on screen and in the dashboard, looks through the
rover's chase camera at its live pose, so it follows the rover; a rehearsal
drives off from there. `twin.camera` names another camera on the rover, or
`""` for Isaac's own. The twin captures its view at `twin.view_hz` (5 Hz by
default), which costs it a little GPU time; set it to 0 under `[twin]` in
`config/rover_twin.toml` to turn it off.

`./scripts/rover_twin.sh` starts mapper and twin without the dashboard; Ctrl-C
stops both. The arm switch resets on page load.

To run a plan (`{"commands": [...], "goal": {...}}`; goal optional):

```bash
.venv/bin/python -m rover_twin.ops.drive plan.json                                # rehearse from live pose
.venv/bin/python -m rover_twin.ops.drive plan.json --arm --rover-ip 192.168.0.157   # drive
```

To drive to a point instead, let the twin plan a safe route there:

```bash
.venv/bin/python -m rover_twin.ops.drive --to 1.0 -0.8                              # plan and rehearse from live pose
.venv/bin/python -m rover_twin.ops.drive --to 1.0 -0.8 90 --arm --rover-ip 192.168.0.157
```

See [the API reference](twin-api.md) for command units, limits and refusal results.

## Run a mission (team1 branch)

A mission file lists excavation jobs in the twin frame and the
[`/plan_mission`](twin-api.md) options; `docs/examples/lab_mission.json` is
five jobs on the saved lab map with a 20 min limit. Team 1's scheduler picks
the order, the twin routes and rehearses every leg.

```bash
.venv/bin/python -m rover_twin.ops.mission docs/examples/lab_mission.json --start 0 0 0 --out plan.json   # offline twin
.venv/bin/python -m rover_twin.ops.mission mission.json                                                   # live pose
.venv/bin/python -m rover_twin.ops.mission mission.json --arm --rover-ip 192.168.0.157
```

Unarmed, it prints the plan (stops, legs, jobs left out and why) and moves
nothing. Armed, it shows the plan and waits for `y`, then for each step:

| Step | Rover | Operator |
|---|---|---|
| Drive | Goes to the next stop on routes the twin plans and rehearses, replanning the route from the live pose | Watch; STOP or Ctrl-C stops it |
| Dig | Waits at the job | Run the drums; Enter when done, or `skip` |
| Unload, charge | Waits at base | Do it; Enter when done |

It keeps the mission's own account (time from the clock, cargo from your
answers, battery estimated from the scheduler's rover model: there is no
battery telemetry) and plans the rest again from there and the live pose
when the twin refuses a route (at once), a drive fails for another reason
(you choose to replan or end), you skip a job, or it falls more than 2 min
and 20% behind its plan. Base stays where the mission started unless the
file gives `base`. Steps and plans go to `mission_log.jsonl`, commands to
`drive_log.jsonl`. The first armed run needs the same supervision as the
hardware trial above.
The driver rehearses the remaining plan before each command and stops on a
refusal, stale/lost pose, controller fault, completion or Ctrl-C.

The drive tool has only been tested against a simulated rover. For the first
hardware trial, have someone at the hardware stop, confirm the controller's
`/<rover-name>/command_status` endpoint responds, then rehearse and fly a
0.25 m drive (`FWD`, speed 30, distance 5750) and a 90° turn. Keep the logs.
Real speed levels 2/3 and the rover's climbable step height still need measuring.

## Remote operation

`./scripts/remote_dashboard.sh` starts the dashboard and a Cloudflare tunnel
that publishes it, and prints the public address; Ctrl-C stops both. Mapper,
twin, phone and rover stay on the workstation's local network: only the
dashboard crosses the internet. In the field, take the workstation with the
rover and put all three on the rover's router. It needs, in `.env` (not in git):

```bash
APP_USERNAME=...            # the dashboard's login; also asked on the local network
APP_PASSWORD=...
TUNNEL_CONFIG=/path/to/cloudflared.yml   # its ingress routes the hostname to http://127.0.0.1:8090
CLOUDFLARED=/path/to/cloudflared         # if it is not on PATH
```

Through the tunnel, plain HTTP is redirected to HTTPS before the login. A new
hostname needs a DNS route once:
`cloudflared tunnel route dns <tunnel> <hostname>`. STOP crosses the internet,
but the live pose stops a drive on the workstation itself. Keep someone at
the rover for remote runs.

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
