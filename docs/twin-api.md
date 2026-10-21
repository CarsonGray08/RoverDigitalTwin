# Twin API

Reference for student software and coding assistants. Base URL on the lab
workstation: `http://127.0.0.1:8765`. All endpoints use JSON.

The offline twin serves the same API on a saved map, using kinematic motion:

```bash
.venv/bin/python -m rover_twin.twin.offline assets/lab_map.npz
```

Use it for development and search. Rehearse plans intended for the rover on
the lab twin, which models physics, contact and stalls. Neither approval
replaces a supervised hardware trial. Battery, energy, thermal state and time
spent doing work at stops are outside this API.

## Coordinates and commands

- Metres in the twin frame: origin and heading are the rover at map anchoring;
  `+x` forward, `+y` left, `+z` up. `yaw_deg` is counter-clockwise from `+x`,
  wrapped to -180..180.
- A pose is `{"x": 0.0, "y": 0.0, "yaw_deg": 0.0}`.
- A command is `{"command": "FWD", "speed": 30, "distance": 23000}`.
  `FWD`/`REV` distance is integer rover units (23000/m by default);
  `LEFT`/`RIGHT` distance is integer degrees. Moving speeds are integers 1–60.
  `STOP` requires speed 0 and distance 0.
- Default limits: 12 commands/plan, 30000 distance units/drive, 360°/turn,
  lab bounds -5..5 m in x/y. Read `/health` for configurable limits.

## Endpoints

### `GET /health`

```json
{"status": "ok", "rover": {
  "distance_units_per_m": 23000.0, "speed_range": [1, 60],
  "max_distance_units": 30000, "max_turn_deg": 360, "max_commands": 12,
  "footprint_half_m": [0.26, 0.30], "max_step_m": 0.12,
  "goal_tolerance": [0.15, 8.0]}}
```

Footprint half sizes are along/across the heading; goal tolerance is metres/degrees.

### `GET /map`

Returns `cell_m`, `origin_m: [x0, y0]`, `shape: [nx, ny]`, and matching 2D arrays
`height_m`, `state`, `obstacle`. Cell `[i][j]` spans x from `x0 + i*cell_m` to
`x0 + (i+1)*cell_m`, and y likewise with j.

`height_m` is null where unknown. State: 0 unknown, 1 observed, 2 inferred
under the rover. `obstacle` is 0/1: observed height at least `max_step_m` above
or below the starting floor (z=0). The grid keeps the highest surface, so
tables/shelves count as solid. Heights are absolute; uneven-ground planning
still needs a local-ground model. Fetch a new map when the live pose's `rev` changes.

### `POST /validate_plan`

```json
{"start_pose": {"x": 0.0, "y": 0.0, "yaw_deg": 0.0},
 "goal": {"x": 1.0, "y": 0.0, "yaw_deg": 0.0},
 "commands": [{"command": "FWD", "speed": 30, "distance": 23000}]}
```

Checks limits and start pose, sweeps the footprint along each command
(at least 80% known terrain, no obstacle cells), simulates motion, and checks
the final goal. HTTP 200 includes approved and refused plans:

| Field | Meaning |
|---|---|
| `approved`, `reason` | Boolean outcome and explanation |
| `final_pose` | End pose |
| `duration_s` | Total simulated time, including stopping after commands |
| `min_clearance_m` | Approximate centre-to-obstacle distance minus half the footprint diagonal; 999 if no obstacles |
| `trajectory_dt_s`, `trajectory` | Sample interval and list of poses |
| `step_results` | Completed steps: `command_index`, `status`, `duration_s`, `end_pose`, `command` |
| `expected_step_end_poses` | End pose of each completed command |
| `goal_error_m`, `goal_error_yaw_deg` | Present when the plan reaches the final goal check |
| `failure_type`, `failed_command_index` | Always both on refusal; index is null if no single command is at fault |

Failure types: `too_many_commands`, `speed_limit`, `distance_limit`,
`turn_distance_limit`, `invalid_stop`, `unknown_not_ready`, `collision`,
`out_of_bounds`, `turn_not_completed`, `goal_not_reached`.
Malformed input returns HTTP 400 with `error: "schema_error"`; internal
failure returns 500 with `error: "twin_error"`.

Simulated speed at 30 is about 0.19 m/s and 19.5°/s. The real rover is
slower (~0.035 m/s at level 1). Fit real duration models from drive logs,
not `duration_s`.

### `GET /observation`

Returns `live_pose` (null offline/before tracking), `rover_pose`,
`terrain_tiles` and `mapper_connected`. A live pose has `x`, `y`, `yaw_deg`,
`rev` and `age_s`. Treat age over ~1 s as unavailable. Limited ARKit tracking
stops pose updates, so its age grows.

`/map` and `/validate_plan` queue in the simulation loop; the map stays fixed
during a rehearsal. `/health` and `/observation` answer without waiting for it.

## Python and real execution

```python
from rover_twin.client import TwinClient

twin = TwinClient()
grid = twin.map()  # numpy arrays; grid.cell(x, y) -> (i, j)
plan = [twin.forward(1.0), twin.left(90), twin.forward(0.5)]
result = twin.rehearse((0.0, 0.0, 0.0), plan)
if not result["approved"]:
    print(result["failure_type"], result["failed_command_index"], result["reason"])
```

Client helpers: `forward`/`reverse` take metres, `left`/`right` degrees, with
optional `speed` (default 30). `rehearse(start, commands, goal=None)` takes
pose tuples `(x, y, yaw_deg)`; its default goal is the nominal plan endpoint.
`observation()` reads live state.

To fly a plan, save `{"commands": [...], "goal": {...}}` (goal optional)
and use `python -m rover_twin.ops.drive plan.json`. It rehearses only unless
`--arm --rover-ip IP` is supplied. Offline rehearsal needs `--start X Y YAW_DEG`.
An armed drive uses fresh live poses and rehearses the remaining plan before
each command; refusal stops execution so your software can replan.
`--level 1|2|3` sets real wheel speed (default 1), independently of simulated speed.
See [operations](operations.md) for the hardware trial and logs.
