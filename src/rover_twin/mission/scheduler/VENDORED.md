# Vendored: Team 1's rover scheduler

A copy of Team 1's scheduler (`Rover_Scheduler`, Phase 3B, calibration v2,
received 21 Nov 2026), used by `POST /plan_mission` (`../planner.py`). The
students' repository is the source of truth: changes to the algorithms belong
there first, then are copied here.

## What was copied

| File | Their file | Changed here |
|---|---|---|
| `scheduler_core.py` | same name | imports; twin hooks (below) |
| `simulator.py` | same name | imports; twin hooks |
| `scheduler.py` | same name | imports; one line (nearest job by `sim.distance_to`) |
| `sa.py`, `ga.py` | same names | imports only |
| `calibration.py` | same name | imports only |
| `rover_calibration_preliminary_v2.json` | `config/` | none |

Imports were made package-relative (`from .simulator import ...`). Their
command-line `main()` functions are not used here and still import
`reporting` and `calibration` by top-level name, so they do not run from this
copy. Benchmarks, experiments, plots, scenarios, the workbook importer and
their tests stay in their repository.

## The twin hooks

`twin_hooks.patch` holds them as a patch against their files, ready to apply
in their repository (`patch -p1 < twin_hooks.patch` from its root). Every
added line is marked `[twin]`. Both hooks are optional keys in the mission
dict, so SA, GA, the exhaustive solver and the baseline use them without
change:

- `travel`: `{from_id: {to_id: {"distance_m", "time_min"} | null}}` replaces
  straight-line distance and constant speed with the twin's routes. A null or
  missing leg has no route; a job without one is skipped with the reason
  "no route on the twin's map". Travel energy is `travel_power_w` times the
  leg's minutes.
- `start`: a mission replanned mid-way, at `START` (`x_m`, `y_m`) or `BASE`,
  with `battery_wh`, `elapsed_min`, `load_kg` and `delivered_kg`. Results'
  `route` then begins at the start location.

Without either key, nothing changes: all 169 of their tests passed with the
hooks applied (run on their copy, 22 Nov 2026), and `tests/test_mission.py`
pins their published results (baseline 24 kg in 58.61 min, 47 kg in
168.22 min with no sol limit, SA and GA 35 kg on their standard mission).
