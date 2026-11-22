"""Simulated annealing (SA) over job orders, scored by the shared MissionSimulator.

    python sa.py                      # standard mission, default settings
    python sa.py --seed 3 --iters 20000 --t0 3 --alpha 0.999

SA proposes job orders; the simulator decides what they mean physically. This
module never touches energy, time, cargo or sol rules - see simulator.py.

Representation
    A full permutation of all job IDs. The simulator decodes it (automatic
    returns / unloads / charges) and labels jobs it cannot fit as
    ``skipped_infeasible``. With ``allow_omission=True`` the solution is an
    ordered *active* list plus an inactive pool, and dropped jobs are labelled
    ``excluded_from_sequence`` by the simulator.

Score (maximised)
    objective_kg + TIME_TIEBREAK * sol_time_remaining_min. The tie-break is far
    below 1 kg, so delivered regolith always dominates; among equal-kg
    schedules the faster one wins.

Moves  (uniform random choice each iteration)
    swap two positions | relocate one job | reverse a subsequence (2-opt)
    | drop / insert (only when allow_omission)

Acceptance (maximisation)
    accept if delta >= 0 else with probability exp(delta / T)

Cooling / stopping
    geometric T <- T * alpha each iteration; stop at max_iterations or T < t_min.

Guarantees
    * starts from the baseline order, tracks the best-so-far separately, and
      returns the baseline if nothing better is found -> never worse in kg;
    * fully reproducible for a given seed;
    * runtime_s is optimisation wall-clock, independent of simulated mission time.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from .scheduler import nearest_feasible_sequence, schedule
from .simulator import MissionSimulator

ALGORITHM_NAME = "simulated_annealing"
TIME_TIEBREAK = 1e-6     # kg-equivalent per unused sol minute; << 1 kg by design


@dataclass(frozen=True)
class SAConfig:
    initial_temperature: float = 2.0   # kg; p(accept 2 kg loss) ~ e^-1 at start
    cooling_rate: float = 0.9989       # geometric alpha; T0 -> t_min in ~5000 steps
    min_temperature: float = 0.01      # kg; stop below this
    max_iterations: int = 5000
    seed: int = 0
    allow_omission: bool = False       # enable drop/insert moves (inactive pool)
    time_limit_s: float | None = None  # optional wall-clock stop (experiments only)

    def validate(self) -> None:
        if not (0 < self.cooling_rate < 1):
            raise ValueError("cooling_rate must be in (0, 1)")
        if self.initial_temperature <= 0 or self.min_temperature <= 0:
            raise ValueError("temperatures must be positive")
        if self.max_iterations < 0:
            raise ValueError("max_iterations must be >= 0")


def score_of(result: dict) -> float:
    """Primary objective with a negligible time tie-break (see module doc)."""
    remaining = result["sol_time_remaining_min"] or 0.0
    return result["objective_kg"] + TIME_TIEBREAK * remaining


class SimulatedAnnealing:
    def __init__(self, mission: dict, config: SAConfig | None = None):
        self.config = config or SAConfig()
        self.config.validate()
        self.sim = MissionSimulator(mission)
        self.all_ids = [t.id for t in self.sim.tasks]
        self.rng = random.Random(self.config.seed)
        self.evaluations = 0

    # ------------------------------------------------------------ helpers
    def _evaluate(self, order: list[str]) -> dict:
        self.evaluations += 1
        return self.sim.evaluate(order)

    def baseline_order(self) -> list[str]:
        """Baseline completed jobs followed by its skipped ones: a full permutation
        that the simulator decodes to exactly the baseline schedule."""
        state, skipped = nearest_feasible_sequence(self.sim)
        return list(state.completed) + [d["id"] for d in skipped]

    def neighbour(self, active: list[str], pool: list[str]) -> tuple[list[str], list[str]]:
        """Return a new (active, pool) pair; inputs are not modified."""
        a, p = list(active), list(pool)
        n = len(a)
        moves = ["swap", "relocate", "reverse"] if n >= 2 else []
        if self.config.allow_omission and (n >= 1 or p):
            moves.append("drop_insert")
        if not moves:
            return a, p
        move = self.rng.choice(moves)
        if move == "swap":
            i, j = self.rng.sample(range(n), 2)
            a[i], a[j] = a[j], a[i]
        elif move == "relocate":
            i = self.rng.randrange(n)
            job = a.pop(i)
            a.insert(self.rng.randrange(n), job)
        elif move == "reverse":
            i, j = sorted(self.rng.sample(range(n), 2))
            a[i:j + 1] = reversed(a[i:j + 1])
        else:  # drop_insert
            if p and (not a or self.rng.random() < 0.5):
                job = p.pop(self.rng.randrange(len(p)))
                a.insert(self.rng.randrange(n + 1), job)
            else:
                p.append(a.pop(self.rng.randrange(n)))
        return a, p

    # --------------------------------------------------------------- run
    def run(self) -> dict:
        cfg = self.config
        t_start = time.perf_counter()

        current = self.baseline_order()
        pool: list[str] = []
        current_res = self._evaluate(current)
        current_score = score_of(current_res)
        baseline_res = current_res
        best, best_res, best_score = list(current), current_res, current_score
        eval_trace: list[tuple[int, float]] = [(self.evaluations, best_res["objective_kg"])]

        T = cfg.initial_temperature
        iterations = accepted = improvements = 0
        trace: list[tuple[int, float, float]] = []   # (iteration, T, best_kg)
        while iterations < cfg.max_iterations and T >= cfg.min_temperature:
            if cfg.time_limit_s is not None and time.perf_counter() - t_start >= cfg.time_limit_s:
                break
            cand, cand_pool = self.neighbour(current, pool)
            cand_res = self._evaluate(cand)
            cand_score = score_of(cand_res)
            delta = cand_score - current_score
            if delta >= 0 or self.rng.random() < math.exp(delta / T):
                current, pool, current_res, current_score = cand, cand_pool, cand_res, cand_score
                accepted += 1
                if current_score > best_score:
                    best, best_res, best_score = list(current), current_res, current_score
                    improvements += 1
                    eval_trace.append((self.evaluations, best_res["objective_kg"]))
            iterations += 1
            T *= cfg.cooling_rate
            if iterations % 100 == 0:
                trace.append((iterations, T, best_res["objective_kg"]))

        # Fallback guarantee: never worse than the baseline on the primary objective.
        if best_res["objective_kg"] < baseline_res["objective_kg"]:
            best, best_res = self.baseline_order(), baseline_res

        result = dict(best_res)
        result["algorithm"] = ALGORITHM_NAME
        result["runtime_s"] = time.perf_counter() - t_start          # optimisation wall-clock
        result["baseline_objective_kg"] = baseline_res["objective_kg"]
        result["improvement_kg"] = best_res["objective_kg"] - baseline_res["objective_kg"]
        result["sa_config"] = asdict(cfg)
        result["sa_stats"] = {
            "iterations": iterations, "evaluations": self.evaluations,
            "accepted": accepted, "improvements": improvements,
            "final_temperature": T,
            "stopped_by": ("time_limit" if cfg.time_limit_s is not None
                           and time.perf_counter() - t_start >= cfg.time_limit_s
                           and iterations < cfg.max_iterations and T >= cfg.min_temperature
                           else "temperature" if T < cfg.min_temperature else "iterations"),
            "cache_hits": 0, "unique_evaluations": self.evaluations,
            "best_score": best_score, "trace": trace,
            "eval_trace": eval_trace + [(self.evaluations, best_res["objective_kg"])],
        }
        return result


def optimise(mission: dict, config: SAConfig | None = None) -> dict:
    """benchmark.ALGORITHMS-compatible entry point: f(mission) -> result dict."""
    return SimulatedAnnealing(mission, config).run()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mission", nargs="?", type=Path, default=Path(__file__).with_name("mission.json"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--iters", type=int, default=SAConfig.max_iterations)
    parser.add_argument("--t0", type=float, default=SAConfig.initial_temperature)
    parser.add_argument("--alpha", type=float, default=SAConfig.cooling_rate)
    parser.add_argument("--tmin", type=float, default=SAConfig.min_temperature)
    parser.add_argument("--allow-omission", action="store_true")
    parser.add_argument("-o", "--output", type=Path, default=Path("sa_result.json"))
    args = parser.parse_args(argv)
    from calibration import load_mission
    mission = load_mission(args.mission)      # merges calibration_file if present
    cfg = SAConfig(args.t0, args.alpha, args.tmin, args.iters, args.seed, args.allow_omission)
    res = optimise(mission, cfg)
    base = schedule(mission)
    from reporting import summary_text, write_csv, write_json
    print(summary_text(res))
    print(f"Baseline objective:  {base['objective_kg']:.2f} kg   SA improvement: {res['improvement_kg']:+.2f} kg")
    st = res["sa_stats"]
    print(f"SA runtime:          {res['runtime_s']*1000:.1f} ms  ({st['iterations']} iterations, "
          f"{st['accepted']} accepted, {st['improvements']} improvements, stopped by {st['stopped_by']})")
    write_json(res, args.output)
    write_csv(res, args.output.with_suffix(".csv"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
