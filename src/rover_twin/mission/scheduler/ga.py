"""Genetic algorithm (GA) over job orders, scored by the shared MissionSimulator.

    python ga.py                        # standard mission, default settings
    python ga.py --seed 3 --pop 100 --gens 50 --mutation 0.3

Same contract as sa.py: the GA proposes full permutations of job IDs and the
simulator decides what they mean. Fitness is ``sa.score_of`` (delivered kg with
a negligible time tie-break) so SA and GA optimise the identical function.

Design
    population      50, tournament selection (size 3)
    initial pop     baseline order + light mutations of it + random permutations
    crossover       Order Crossover (OX), permutation-safe, p = crossover_rate
    mutation        one of swap / relocate / reverse (the SA moves), p = mutation_rate
    elitism         top ``elitism`` individuals copied unchanged (not re-evaluated)
    budget          max_generations, and a hard cap max_evaluations on simulator
                    calls so GA never spends more evaluations than SA (~5 000)
    memo cache      identical orders are not re-simulated; cache hits do not
                    count as evaluations (reported separately)

Guarantees: best-so-far tracked over every evaluation, baseline fallback (never
worse in kg), reproducible per seed, runtime_s = optimiser wall-clock only.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from .sa import SAConfig, SimulatedAnnealing, score_of
from .scheduler import schedule
from .simulator import MissionSimulator

ALGORITHM_NAME = "genetic_algorithm"


@dataclass(frozen=True)
class GAConfig:
    population_size: int = 50
    max_generations: int = 100
    crossover_rate: float = 0.9
    mutation_rate: float = 0.2
    elitism: int = 2
    tournament_size: int = 3
    max_evaluations: int = 5000     # hard cap on simulator calls (SA parity)
    seed: int = 0
    seeded_fraction: float = 1 / 3  # share of initial pop that are mutated baselines
    time_limit_s: float | None = None  # optional wall-clock stop (experiments only)
    max_proposals: int | None = None   # optional cap on unique + cache-hit proposals (safety)

    def validate(self) -> None:
        if self.population_size < 2:
            raise ValueError("population_size must be >= 2")
        if not 0 <= self.crossover_rate <= 1 or not 0 <= self.mutation_rate <= 1:
            raise ValueError("rates must be in [0, 1]")
        if not 0 <= self.elitism < self.population_size:
            raise ValueError("elitism must be in [0, population_size)")
        if self.tournament_size < 1 or self.max_generations < 0 or self.max_evaluations < 1:
            raise ValueError("tournament_size >= 1, max_generations >= 0, max_evaluations >= 1")


def order_crossover(p1: list[str], p2: list[str], rng: random.Random) -> list[str]:
    """OX: keep a random slice of p1 in place, fill the rest in p2's order."""
    n = len(p1)
    if n < 2:
        return list(p1)
    i, j = sorted(rng.sample(range(n), 2))
    child: list[str | None] = [None] * n
    child[i:j + 1] = p1[i:j + 1]
    held = set(p1[i:j + 1])
    fill = [g for g in p2 if g not in held]
    k = 0
    for pos in list(range(j + 1, n)) + list(range(0, i)):
        child[pos] = fill[k]
        k += 1
    return child  # type: ignore[return-value]


class GeneticAlgorithm:
    def __init__(self, mission: dict, config: GAConfig | None = None):
        self.config = config or GAConfig()
        self.config.validate()
        self.sim = MissionSimulator(mission)
        self.rng = random.Random(self.config.seed)
        # Reuse SA's move operators and baseline-order construction verbatim.
        self._moves = SimulatedAnnealing(mission, SAConfig(seed=self.config.seed, max_iterations=0))
        self._moves.rng = self.rng
        self.evaluations = 0
        self.cache_hits = 0
        self._cache: dict[tuple[str, ...], dict] = {}

    # ---------------------------------------------------------- evaluation
    def evaluate(self, order: list[str]) -> dict | None:
        """Simulator result for ``order``; None once the evaluation budget is spent."""
        key = tuple(order)
        if key in self._cache:
            self.cache_hits += 1
            return self._cache[key]
        if self.evaluations >= self.config.max_evaluations:
            return None
        self.evaluations += 1
        res = self.sim.evaluate(order)
        self._cache[key] = res
        return res

    def mutate(self, order: list[str]) -> list[str]:
        new, _ = self._moves.neighbour(order, [])
        return new

    def tournament(self, pop: list[tuple[list[str], float]]) -> list[str]:
        k = min(self.config.tournament_size, len(pop))
        return max(self.rng.sample(pop, k), key=lambda x: x[1])[0]

    # ------------------------------------------------------------- run
    def run(self) -> dict:
        cfg = self.config
        t_start = time.perf_counter()
        base_order = self._moves.baseline_order()
        base_res = self.evaluate(base_order)
        assert base_res is not None
        best_order, best_res, best_score = list(base_order), base_res, score_of(base_res)
        eval_trace: list[tuple[int, float]] = [(self.evaluations, base_res["objective_kg"])]

        def consider(order, res):
            nonlocal best_order, best_res, best_score
            s = score_of(res)
            if s > best_score:
                best_order, best_res, best_score = list(order), res, s
                eval_trace.append((self.evaluations, res["objective_kg"]))
            return s

        def out_of_time():
            return cfg.time_limit_s is not None and time.perf_counter() - t_start >= cfg.time_limit_s

        def over_proposals():
            return (cfg.max_proposals is not None
                    and self.evaluations + self.cache_hits >= cfg.max_proposals)

        # initial population: baseline, mutated baselines, random permutations
        orders = [list(base_order)]
        n_seeded = int(round(cfg.seeded_fraction * (cfg.population_size - 1)))
        for _ in range(n_seeded):
            o = list(base_order)
            for _ in range(self.rng.randint(1, 2)):
                o = self.mutate(o)
            orders.append(o)
        while len(orders) < cfg.population_size:
            o = list(base_order)
            self.rng.shuffle(o)
            orders.append(o)

        population: list[tuple[list[str], float]] = []
        for o in orders:
            res = self.evaluate(o)
            if res is None:
                break
            population.append((o, consider(o, res)))

        generations = 0
        trace: list[tuple[int, float]] = [(0, best_res["objective_kg"])]
        budget_exhausted = len(population) < cfg.population_size
        timed_out = capped = False
        while (generations < cfg.max_generations and not budget_exhausted
               and not timed_out and not capped):
            population.sort(key=lambda x: x[1], reverse=True)
            next_pop = population[:cfg.elitism]            # elites: no re-evaluation
            while len(next_pop) < cfg.population_size:
                if out_of_time():
                    timed_out = True
                    break
                if over_proposals():
                    capped = True
                    break
                p1 = self.tournament(population)
                if self.rng.random() < cfg.crossover_rate:
                    p2 = self.tournament(population)
                    child = order_crossover(p1, p2, self.rng)
                else:
                    child = list(p1)
                if self.rng.random() < cfg.mutation_rate:
                    child = self.mutate(child)
                res = self.evaluate(child)
                if res is None:                              # budget spent mid-generation
                    budget_exhausted = True
                    break
                next_pop.append((child, consider(child, res)))
            if len(next_pop) < cfg.population_size:        # partial generation: keep best of both
                population = sorted(population + next_pop[cfg.elitism:], key=lambda x: x[1],
                                    reverse=True)[:cfg.population_size]
            else:
                population = next_pop
            generations += 1
            trace.append((generations, best_res["objective_kg"]))

        if best_res["objective_kg"] < base_res["objective_kg"]:   # fallback (defensive)
            best_order, best_res = base_order, base_res

        result = dict(best_res)
        result["algorithm"] = ALGORITHM_NAME
        result["runtime_s"] = time.perf_counter() - t_start
        result["baseline_objective_kg"] = base_res["objective_kg"]
        result["improvement_kg"] = best_res["objective_kg"] - base_res["objective_kg"]
        result["ga_config"] = asdict(cfg)
        result["ga_stats"] = {
            "generations": generations, "evaluations": self.evaluations,
            "cache_hits": self.cache_hits, "population_size": cfg.population_size,
            "stopped_by": ("time_limit" if timed_out else "proposal_cap" if capped else
                           "evaluation_budget" if budget_exhausted else "generations"),
            "unique_evaluations": self.evaluations,
            "best_score": best_score, "trace": trace,
            "eval_trace": eval_trace + [(self.evaluations, best_res["objective_kg"])],
        }
        return result


def optimise(mission: dict, config: GAConfig | None = None) -> dict:
    """benchmark.ALGORITHMS-compatible entry point: f(mission) -> result dict."""
    return GeneticAlgorithm(mission, config).run()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mission", nargs="?", type=Path, default=Path(__file__).with_name("mission.json"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pop", type=int, default=GAConfig.population_size)
    parser.add_argument("--gens", type=int, default=GAConfig.max_generations)
    parser.add_argument("--crossover", type=float, default=GAConfig.crossover_rate)
    parser.add_argument("--mutation", type=float, default=GAConfig.mutation_rate)
    parser.add_argument("--elitism", type=int, default=GAConfig.elitism)
    parser.add_argument("--max-evals", type=int, default=GAConfig.max_evaluations)
    parser.add_argument("-o", "--output", type=Path, default=Path("ga_result.json"))
    args = parser.parse_args(argv)
    from calibration import load_mission
    mission = load_mission(args.mission)      # merges calibration_file if present
    cfg = GAConfig(args.pop, args.gens, args.crossover, args.mutation, args.elitism,
                   max_evaluations=args.max_evals, seed=args.seed)
    res = optimise(mission, cfg)
    base = schedule(mission)
    from reporting import summary_text, write_csv, write_json
    print(summary_text(res))
    print(f"Baseline objective:  {base['objective_kg']:.2f} kg   GA improvement: {res['improvement_kg']:+.2f} kg")
    st = res["ga_stats"]
    print(f"GA runtime:          {res['runtime_s']*1000:.1f} ms  ({st['generations']} generations, "
          f"{st['evaluations']} evaluations, {st['cache_hits']} cache hits, stopped by {st['stopped_by']})")
    write_json(res, args.output)
    write_csv(res, args.output.with_suffix(".csv"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
