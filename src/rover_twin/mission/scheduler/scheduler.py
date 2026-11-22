"""Baseline scheduler: nearest feasible job (simulation only).

Python 3.10+, standard library only. Since Phase 2A this file holds only the
*policy* - which job to do next. All physics and rules (energy, time, cargo,
reserve, sol limit, returns, unloading, charging) live in ``simulator.py`` and
are shared with the sequence evaluator that GA / SA will use later.

Policy, applied until no job remains or none is serviceable:

1. Among unfinished jobs that fit right now (cargo, battery incl. direct
   return with reserve, and sol limit incl. return + unload), take the nearest
   by straight-line distance from the current position; ties go to the
   alphabetically first job ID.
2. If none fits and the rover is away from base: return and unload, then
   retry on the current charge.
3. If none fits and the rover is at base: recharge fully only if at least one
   job would fit after charging (with empty cargo and within the sol); the
   same check then guarantees a job is taken, so charging cannot loop.
4. Otherwise every remaining job is unserviceable: finish at base, unload,
   and stop. No recharge after the final delivery.
"""
from __future__ import annotations

import time

from .scheduler_core import BASE_ID, TIE_DECIMALS, MissionError, Task, distance, parse_mission  # noqa: F401 (re-exported)
from .simulator import MissionSimulator, MissionState

ALGORITHM_NAME = "nearest_feasible_baseline"


def nearest_feasible_sequence(sim: MissionSimulator) -> tuple[MissionState, list[dict]]:
    """Run the greedy policy on the simulator; return final state and skipped jobs."""
    s = sim.initial_state()
    remaining: list[Task] = list(sim.tasks)

    def nearest(candidates: list[Task]) -> Task:
        # Round the distance so mathematically equal distances that differ only
        # in the last floating-point digit still tie and fall back to the ID.
        return min(candidates, key=lambda t: (round(sim.distance_to(s, t), TIE_DECIMALS), t.id))

    while remaining:
        feasible_now = [t for t in remaining if sim.feasible_now(t, s)]
        if feasible_now:
            task = nearest(feasible_now)
            sim.do_job(s, task)
            remaining.remove(task)
        elif s.location != BASE_ID:
            sim.return_and_unload(s)
        elif any(not sim.why_not_feasible(t, s, from_base_full=True) for t in remaining):
            sim.charge(s)
            if not any(sim.feasible_now(t, s) for t in remaining):
                raise RuntimeError("internal error: charged but no job became feasible")
        else:
            break

    skipped = [{"id": t.id, "reasons": sim.why_not_feasible(t, s, from_base_full=True)}
               for t in remaining]
    sim.finish(s)
    return s, skipped


def schedule(mission: dict) -> dict:
    """Nearest-feasible-job baseline. Dictionary in, result dictionary out."""
    t_start = time.perf_counter()
    sim = MissionSimulator(mission)
    state, skipped = nearest_feasible_sequence(sim)
    return sim.result(state, skipped=skipped, algorithm=ALGORITHM_NAME,
                      sequence=list(state.completed),
                      runtime_s=time.perf_counter() - t_start)
