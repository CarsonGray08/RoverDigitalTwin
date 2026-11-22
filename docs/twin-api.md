# Twin API

Reference for student software and coding assistants. Base URL on the lab
workstation: `http://127.0.0.1:8765`. All endpoints use JSON, except the
`/view.jpg` image.

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

### `POST /plan_route`

Plans a safe route to a goal and rehearses it. Send coordinates; the twin
finds the way and writes the commands.

```json
{"start_pose": {"x": 0.0, "y": 0.0, "yaw_deg": 0.0},
 "goal": {"x": 2.4, "y": 0.0, "yaw_deg": 90.0},
 "speed": 30, "rehearse": true}
```

`goal.yaw_deg` is optional (omitted: arrive facing along the last straight);
`speed` (simulated, default `route.speed`) and `rehearse` (default true) too.

A route keeps the rover's centre where it could turn on the spot without
its footprint coming within `route.clearance_m` (0.10 m) of an obstacle, on
known ground at every heading, inside the lab bounds. That is the footprint's
half diagonal (0.40 m) plus the clearance on every side, so gaps narrower
than about 1 m are treated as closed. The shortest such path is cut to
straight legs; each corner is a turn, each straight one or more `FWD` of at
most `max_distance_units`, grouped into plans of at most `max_commands`. The
twin then rehearses the plans in order, each from where the last ended.

| Field | Meaning |
|---|---|
| `approved`, `reason` | Route found and, when rehearsed, every plan approved |
| `rehearsed` | Whether the plans were rehearsed |
| `waypoints` | Start, corners, goal: `{"x", "y"}` |
| `length_m`, `turn_deg` | Route length and total turning, including to the goal heading |
| `plans` | `{"commands": [...], "goal": {...}}` per plan, ready for `/validate_plan` or `ops.drive` |
| `rehearsals` | Per plan rehearsed: `approved`, `reason`, `duration_s`, `final_pose`, `min_clearance_m`, and on refusal `failure_type`, `failed_command_index` |
| `failure_type`, `failed_plan_index` | On refusal; index is null when no route was found |

Failure types, besides those of `/validate_plan` from a rehearsal:
`start_blocked`, `goal_blocked` (the reason says why: an obstacle within
reach, ground not observed, outside the lab bounds or the mapped area),
`no_route`, `unknown_not_ready` (nothing mapped) and `speed_limit`.

### `POST /route_costs`

Route length and corner turning between every ordered pair of up to 32
points, on the same routes `/plan_route` plans, without commands or
rehearsal: what a scheduler needs to cost its legs.

```json
{"points": [{"x": 0.0, "y": 0.0}, {"x": 2.4, "y": 0.0}, {"x": 0.0, "y": 1.0}]}
```

Returns `length_m` and `turn_deg`, matrices indexed `[from][to]` (null where
no route exists), and `blocked`: per point, null or why the rover cannot
stand there. `turn_deg` counts only the corners: turning to leave along the
first leg or to a heading on arrival depends on the legs either side.

### `POST /plan_mission` (team1 branch)

Schedules excavation jobs with Team 1's scheduler on the twin's routes, then
plans and rehearses every leg. The scheduler is vendored in
`src/rover_twin/mission/scheduler/` (see its `VENDORED.md`).

```json
{"start_pose": {"x": 0.0, "y": 0.0, "yaw_deg": 0.0},
 "jobs": [{"id": "A", "x": 2.4, "y": 0.0, "material_kg": 4, "task_min": 4},
          {"id": "B", "x": 0.0, "y": 1.2, "material_kg": 3, "task_min": 3, "yaw_deg": 90}],
 "sol_limit_min": 25, "algorithm": "simulated_annealing", "seed": 0}
```

| Field | Default | Meaning |
|---|---|---|
| `jobs` | required | 1 to 30 jobs: `id`, `x`, `y`, `material_kg`, `task_min`; optional `excavation_power_w`, and `yaw_deg` to arrive facing |
| `base` | the start pose | `{"x", "y"}` where cargo is unloaded and the battery charged |
| `rover` | calibration v2 | Overrides of the scheduler's rover parameters (`speed_m_per_min`, `capacity_kg`, `battery_wh`, ...) |
| `sol_limit_min` | 120 (calibration policy) | Mission time limit; `null` for none |
| `state` | a fresh start | Replanning mid-way: `battery_wh`, `elapsed_min`, `load_kg`, `delivered_kg` |
| `turn_s_per_90deg` | 40 (assumed) | Turn rate for leg times |
| `algorithm`, `seed` | `simulated_annealing`, 0 | Or `genetic_algorithm`, `baseline` |
| `rehearse` | true | Rehearse each leg, or only plan it |

The twin's route costs become each leg's time: route length at the rover's
speed, plus the turning at its corners and an average 90° to set off along
it, at `turn_s_per_90deg`. The scheduler orders the jobs on those times; the
twin then plans each leg from where the last ended. If it refuses a leg,
that leg is taken out and the jobs are scheduled again (up to 3 times).

| Field | Meaning |
|---|---|
| `approved`, `reason` | Every leg of the chosen schedule planned (and rehearsed) |
| `order`, `route` | Jobs in the order done; every stop, `BASE` included (`START` first when replanning away from base) |
| `legs` | Per leg: `from`, `to`, `time_min` (scheduled), and the `/plan_route` answer (`plans`, `rehearsals`, ...) |
| `jobs` | Per job: `status` (`delivered`, `skipped_infeasible`), `reasons`, and `unreachable` with the twin's reason when no route reaches it |
| `objective_kg`, `total_time_min`, `total_distance_m`, `battery_remaining_wh`, `events` | The scheduler's result |
| `refused_legs` | Legs the twin refused, scheduled around |
| `assumptions` | The rover values, speed and turn rate used |

Refusals: `base_blocked`, `start_blocked`, `no_route_home`, `leg_refused`
(after the retries). A rover or job the scheduler rejects returns 400 naming
the field. The scheduler runs outside the simulation loop; only route costs
and each leg's rehearsal queue in it.

To fly a mission, `python -m rover_twin.ops.mission mission.json` (a body
without `start_pose`) plans it; with `--arm --rover-ip IP` it flies it step
by step with the operator, replanning from the live state on the way. See
[operations](operations.md#run-a-mission-team1-branch).

### `GET /observation`

Returns `live_pose` (null offline/before tracking), `rover_pose`,
`terrain_tiles` and `mapper_connected`. A live pose has `x`, `y`, `yaw_deg`,
`rev` and `age_s`. Treat age over ~1 s as unavailable. Limited ARKit tracking
stops pose updates, so its age grows.

### `GET /view.jpg`

The lab twin's 3D view as a JPEG (the dashboard shows it), captured at
`twin.view_hz` (5 Hz by default; 0 turns it off), with its age in seconds in
the `X-Age-S` header. Returns 404 JSON when the view is off, not yet captured,
or on the offline twin.

`/map`, `/validate_plan`, `/plan_route` and `/route_costs` queue in the
simulation loop; the map stays fixed during a rehearsal and a route's rehearsals. `/health`, `/observation` and `/view.jpg` answer without waiting for it.

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
`plan_route(start, goal)` takes a goal `(x, y)` or `(x, y, yaw_deg)`;
`route_costs(points)` a list of `(x, y)`. `observation()` reads live state.

```python
route = twin.plan_route((0.0, 0.0, 0.0), (2.4, 0.0))
if route["approved"]:
    print(route["length_m"], [p["goal"] for p in route["plans"]])
legs = twin.route_costs([(0.0, 0.0), (2.4, 0.0), (0.0, 1.0)])["length_m"]
```

To drive to a goal, `python -m rover_twin.ops.drive --to X Y [YAW_DEG]`
plans and rehearses the route (offline: add `--start X Y YAW_DEG`). Armed
(`--arm --rover-ip IP`), it flies the route's first plan, then plans again
from the live pose, until the rover is within the goal tolerance; a route
the twin refuses stops it before anything moves.

To fly a plan, save `{"commands": [...], "goal": {...}}` (goal optional)
and use `python -m rover_twin.ops.drive plan.json`. It rehearses only unless
`--arm --rover-ip IP` is supplied. Offline rehearsal needs `--start X Y YAW_DEG`.
An armed drive uses fresh live poses and rehearses the remaining plan before
each command; refusal stops execution so your software can replan. Without a
goal, each rehearsal's goal is where the remaining commands end from the live
pose then, so drift during one command does not refuse the next: give a goal
when where the rover ends up matters. A plan with `"rehearse": false` is
flown without rehearsing (the dashboard's 360° scan), still stopping each
command on the live pose.
`--level 1|2|3` sets real wheel speed (default 1), independently of simulated speed.
See [operations](operations.md) for the hardware trial and logs.
