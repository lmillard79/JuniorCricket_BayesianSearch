"""Batting-order strategies for U11, judged against an unknown opposition.

The question: with wide differences between our batters, does the order
matter, and does it pay to pair a strong batter with a weaker one rather
than bat strongest to weakest?

The method, in brief:

* A *world* is one draw of everything we cannot know before the match: how
  good each of our players really is (a draw from the fitted posterior, plus
  optional drift since the U10 data), who the opposition are (nine players
  drawn from the grade), their bowling tactics (who bowls how many overs and
  in what rotation), and the ground.
* Every strategy is played through the *same* worlds, so a difference
  between two strategies is not muddied by one of them meeting a tougher
  attack (a paired comparison). Ball-level random numbers are also shared
  per world and repetition.
* A strategy is a rule that turns our estimated skills into a batting order.
  It only knows the posterior means, while the worlds contain the truth, so
  estimation error is part of the test.
* Orders found by search are judged on fresh worlds, never the worlds they
  were tuned on.

Skills come from U10 games; U11 differs (25 overs, dismissed batters leave,
retire at 25 to 35 balls, bigger boundaries). ``Scenario`` carries the
assumptions about that transfer so their effect can be tested.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from junior_cricket import replay
from junior_cricket.ball_model import (
    COMPONENTS,
    JointOutcomes,
    _expected_boundary_runs,
    _expected_scoring_runs,
    _invlogit,
)
from junior_cricket.replay import Runner, stub
from junior_cricket.rules_u11 import U11
from junior_cricket.simulator import simulate_innings

OVERS = U11.overs_per_innings
SIDE = 9                                         # players a team fields


@dataclass(frozen=True)
class Scenario:
    """Assumptions about the unknown U11 world.

    Attributes:
        retire_at: Balls at which our batters retire (25 to 35); None
            leaves only the mandatory cap of 35.
        boundary_shift: Shift on the log-odds of a boundary per ball. U11
            boundaries are 45 m against 30 to 35 m in U10, so a negative
            shift models the bigger ground (-0.7 roughly halves the odds).
        drift_sd: Spread (logit scale) of how far each of our players'
            skills has moved since the U10 data.
        tactics: ``random`` gives the opposition's overs and rotation out
            at random; ``smart`` gives the biggest shares and the opening
            overs to their best bowlers.
    """

    retire_at: Optional[int] = None
    boundary_shift: float = 0.0
    drift_sd: float = 0.0
    tactics: str = "random"

    def label(self) -> str:
        retire = self.retire_at or U11.retirement_mandatory_balls
        return (f"retire {retire}, boundary {self.boundary_shift:+.1f}, "
                f"drift {self.drift_sd:.1f}, {self.tactics} attack")


# --------------------------------------------------------------------------
# Skill and strategies


def runs_per_ball(
    rule: JointOutcomes, batter: Optional[str] = None, bowler: Optional[str] = None,
) -> Tuple[float, float]:
    """(dismissal chance, expected runs) for one ball on an average ground.

    A missing batter or bowler stands for an average one (zero effects).

    Args:
        rule: The joint-model rule.
        batter: Batter's name, or None for an average batter.
        bowler: Bowler's name, or None for an average bowler.

    Returns:
        The chance the ball is a dismissal, and the expected runs off the
        ball counting a dismissal as no runs.
    """
    zero = {c: 0.0 for c in COMPONENTS}
    bat = rule.bat[batter] if batter else zero
    bowl = rule.bowl[bowler] if bowler else zero
    pd_, pb, ps = (float(_invlogit(rule.a[c] + bat[c] + bowl[c])) for c in COMPONENTS)
    runs = pb * _expected_boundary_runs() + (1 - pb) * ps * _expected_scoring_runs()
    return pd_, (1 - pd_) * runs


def stint_value(rule: JointOutcomes, name: str, balls: int = 30) -> float:
    """Expected runs from a stint of up to ``balls`` balls, a dismissal ending it.

    Against an average bowler on an average ground. A batter who scores
    quickly but gets out often is worth less than the same rate sustained.
    """
    pd_, per_ball = runs_per_ball(rule, batter=name)
    if pd_ < 1e-9:
        return balls * per_ball
    return per_ball * (1 - (1 - pd_) ** balls) / pd_


def named_orders(players: Sequence[str], value: Dict[str, float]) -> Dict[str, Tuple[str, ...]]:
    """The batting-order strategies under test, built from each batter's value.

    With the squad ranked r1 (best) to rN (worst), the pair at the crease
    at the start is the first two in the order, the next pair the next two,
    and so on (a dismissal or retirement changes who is paired). Batters
    change ends only at the end of an over, so a pair shares the strike
    over by over.

    * ``strongest to weakest``: r1, r2, r3 ... the conventional order.
    * ``weakest to strongest``: the reverse.
    * ``strong-weak alternating``: r1, rN, r2, rN-1 ... every pair holds
      one strong and one weak batter.
    * ``balanced pairs (top six)``: r1 with r6, r2 with r5, r3 with r4,
      then r7 onwards; the strong-weak idea applied only among the six
      batters who normally get to bat.
    * ``strong-middle alternating``: each of the top third paired with one
      of the middle third, then the rest.

    Returns:
        Strategy name -> order.
    """
    r = sorted(players, key=lambda p: value[p], reverse=True)
    n = len(r)
    top, mid, rest = r[: n // 3], r[n // 3: 2 * n // 3], r[2 * n // 3:]
    strong_middle = [p for pair in zip(top, mid) for p in pair] + rest
    return {
        "strongest to weakest": tuple(r),
        "weakest to strongest": tuple(reversed(r)),
        "strong-weak alternating": tuple(_zigzag(r)),
        "balanced pairs (top six)": tuple(_zigzag(r[:6]) + r[6:]),
        "strong-middle alternating": tuple(strong_middle),
    }


def _zigzag(ranked: Sequence[str]) -> List[str]:
    """Best, worst, second best, second worst ... (each strong batter partnered by a weak one)."""
    out: List[str] = []
    lo, hi = 0, len(ranked) - 1
    while lo <= hi:
        out.append(ranked[lo])
        if lo != hi:
            out.append(ranked[hi])
        lo, hi = lo + 1, hi - 1
    return out


# --------------------------------------------------------------------------
# Worlds and simulation


@dataclass(frozen=True)
class StrategyTask:
    """A batch of worlds through which one batting order is played.

    Attributes:
        order: Our batting order.
        ranking: Our players best to worst; balls and runs are reported
            by this ranking so different orders can be compared.
        scenario: Assumptions about the U11 world.
        worlds: [start, stop) world numbers; a world is fully determined by
            its number and ``seed``.
        reps: Innings played in each world.
        seed: Base entropy.
    """

    order: Tuple[str, ...]
    ranking: Tuple[str, ...]
    scenario: Scenario
    worlds: Tuple[int, int]
    reps: int
    seed: int


def _bowler_economy(rule: JointOutcomes, name: str) -> float:
    """Runs per ball an average batter scores off this bowler (lower is better)."""
    return runs_per_ball(rule, bowler=name)[1]


def make_world(pool, scenario: Scenario, ours: Sequence[str], seed: int, number: int):
    """One draw of everything unknown before the match.

    Returns:
        (rule, opposition names, their overs per bowler, their rotation,
        the attack's rating: runs per ball an average batter scores, lower
        being a stronger attack).
    """
    rng = np.random.default_rng([seed, number])
    base = pool[int(rng.integers(len(pool)))]
    a = dict(base.a)
    a["b"] += scenario.boundary_shift
    bat, bowl = dict(base.bat), dict(base.bowl)
    if scenario.drift_sd:
        for name in ours:
            bat[name] = {c: v + rng.normal(0, scenario.drift_sd) for c, v in bat[name].items()}
            bowl[name] = {c: v + rng.normal(0, scenario.drift_sd) for c, v in bowl[name].items()}
    rule = JointOutcomes(a, bat, bowl, base.bat_sd, base.bowl_sd, base.game_sd)
    rule.begin_game(rng)
    opposition = [f"opp_{i}" for i in range(SIDE)]
    for name in opposition:
        rule.register(name, rng)
    economy = {n: _bowler_economy(rule, n) for n in opposition}
    shares = sorted(U11.bowling_allocations[SIDE], reverse=True)
    if scenario.tactics == "smart":
        ordered = sorted(opposition, key=economy.get)
        rotation = list(ordered)
    else:
        ordered = [opposition[i] for i in rng.permutation(SIDE)]
        rotation = [opposition[i] for i in rng.permutation(SIDE)]
    allocation = dict(zip(ordered, shares))
    rating = sum(allocation[n] * economy[n] for n in opposition) / OVERS
    return rule, opposition, allocation, rotation, rating


def run_strategy_task(task: StrategyTask) -> Dict[str, np.ndarray]:
    """Play the order through each world; return per-world, per-repetition results.

    Keys: ``total``, ``wickets`` (worlds, reps); ``balls`` and ``runs`` by
    rank (worlds, reps, players); ``over_runs`` (worlds, reps, overs);
    ``rating`` (worlds,).
    """
    start, stop = task.worlds
    n, size = stop - start, len(task.order)
    out = {"total": np.zeros((n, task.reps), dtype=np.int16),
           "wickets": np.zeros((n, task.reps), dtype=np.int16),
           "balls": np.zeros((n, task.reps, size), dtype=np.int16),
           "runs": np.zeros((n, task.reps, size), dtype=np.int16),
           "over_runs": np.zeros((n, task.reps, OVERS), dtype=np.int16),
           "rating": np.zeros(n)}
    batting = [stub(name) for name in task.order]
    for i in range(n):
        rule, opposition, allocation, rotation, rating = make_world(
            replay._POOL, task.scenario, task.ranking, task.seed, start + i)
        out["rating"][i] = rating
        fielders = {name: stub(name) for name in opposition}
        for r in range(task.reps):
            rng = np.random.default_rng([task.seed, start + i, r, 1])
            result = simulate_innings(
                batting, fielders, rotation, allocation, rng, conditions=U11,
                retire_at_balls=task.scenario.retire_at, outcomes=rule)
            cards = {c.name: c for c in result.batter_cards}
            out["total"][i, r], out["wickets"][i, r] = result.runs, result.wickets
            out["balls"][i, r] = [cards[p].balls for p in task.ranking]
            out["runs"][i, r] = [cards[p].runs for p in task.ranking]
            out["over_runs"][i, r, :len(result.over_runs)] = result.over_runs
    return out


def evaluate(
    runner: Runner, orders: Dict[str, Sequence[str]], ranking: Sequence[str],
    scenario: Scenario, worlds: Tuple[int, int], reps: int, seed: int, chunks: int = 14,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Play every order through the same worlds.

    Args:
        runner: Where to run the simulations.
        orders: Strategy name -> batting order.
        ranking: Our players best to worst.
        scenario: Assumptions about the U11 world.
        worlds: [start, stop) world numbers.
        reps: Innings per world.
        seed: Base entropy.
        chunks: Pieces each order's worlds are split into for parallel work.

    Returns:
        Strategy name -> the arrays of ``run_strategy_task`` joined across chunks.
    """
    start, stop = worlds
    edges = np.linspace(start, stop, min(chunks, stop - start) + 1).astype(int)
    tasks, index = [], []
    for name, order in orders.items():
        for lo, hi in zip(edges[:-1], edges[1:]):
            tasks.append(StrategyTask(tuple(order), tuple(ranking), scenario, (int(lo), int(hi)),
                                      reps, seed))
            index.append(name)
    results = runner.run(tasks, worker=run_strategy_task)
    joined: Dict[str, Dict[str, list]] = {}
    for name, part in zip(index, results):
        bucket = joined.setdefault(name, {k: [] for k in part})
        for k, v in part.items():
            bucket[k].append(v)
    return {name: {k: np.concatenate(v) for k, v in parts.items()} for name, parts in joined.items()}


# --------------------------------------------------------------------------
# Reading the results


def world_means(result: Dict[str, np.ndarray]) -> np.ndarray:
    """Mean total in each world (averaging the repetitions)."""
    return result["total"].mean(axis=1)


def paired(a: Dict[str, np.ndarray], b: Dict[str, np.ndarray]) -> Tuple[float, float, float]:
    """How much better ``a`` is than ``b`` over the same worlds.

    Returns:
        (mean difference in runs, its standard error, share of worlds where
        ``a`` scored more).
    """
    diff = world_means(a) - world_means(b)
    return float(diff.mean()), float(diff.std(ddof=1) / math.sqrt(len(diff))), float((diff > 0).mean())


def by_attack(result: Dict[str, np.ndarray], parts: int = 3) -> List[np.ndarray]:
    """World indices split into equal groups from the strongest attack to the weakest."""
    return np.array_split(np.argsort(result["rating"]), parts)


def random_orders(players: Sequence[str], n: int, seed: int) -> Dict[str, Tuple[str, ...]]:
    """``n`` distinct random batting orders, labelled ``random 1`` ...."""
    rng = np.random.default_rng([seed, 99])
    seen, out = set(), {}
    while len(out) < n:
        order = tuple(players[i] for i in rng.permutation(len(players)))
        if order not in seen:
            seen.add(order)
            out[f"random {len(out) + 1}"] = order
    return out


def hill_climb(
    runner: Runner, start: Sequence[str], ranking: Sequence[str], scenario: Scenario,
    worlds: Tuple[int, int], reps: int, seed: int, rounds: int = 6, z: float = 2.0,
) -> Tuple[Tuple[str, ...], List[Tuple[Tuple[str, ...], float, float]]]:
    """Search for a better batting order by trying every swap of two batters.

    Each round tries every pair swap on the same worlds and takes the best if
    it beats the current order by more than ``z`` standard errors, so noise is
    not chased. The result must be re-judged on fresh worlds.

    Returns:
        (the order found, history of (order, mean total, gain over start)).
    """
    current = tuple(start)
    base = evaluate(runner, {"current": current}, ranking, scenario, worlds, reps, seed)["current"]
    history = [(current, float(world_means(base).mean()), 0.0)]
    total_gain = 0.0
    for _ in range(rounds):
        swaps = {}
        for i in range(len(current)):
            for j in range(i + 1, len(current)):
                trial = list(current)
                trial[i], trial[j] = trial[j], trial[i]
                swaps[f"{i}-{j}"] = tuple(trial)
        results = evaluate(runner, swaps, ranking, scenario, worlds, reps, seed)
        scored = {k: paired(r, base) for k, r in results.items()}
        best = max(scored, key=lambda k: scored[k][0])
        gain, se, _ = scored[best]
        if gain <= z * se:
            break
        current, base = swaps[best], results[best]
        total_gain += gain
        history.append((current, float(world_means(base).mean()), total_gain))
    return current, history
