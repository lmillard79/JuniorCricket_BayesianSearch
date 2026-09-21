"""Replay real U10 games under the joint ball model, and reshuffle decisions.

A finished game is one draw from a wide spread of things that could have
happened. This module rebuilds a real game (who batted in what order, who
bowled each over) and plays it again many times with the joint
batter-by-bowler model. That supports three questions:

1. As played: where did the recorded totals sit in the simulated spread,
   above or below the median?
2. Batting order: the same game with our batters in a different order.
3. Bowling: the same game with our overs shared out differently.

Under Rule 16.10(vi) a team's total is its own runs plus 4 for every
dismissal its bowlers take, so the margin splits into two independent parts:

    margin = (our runs - 4 x our dismissals)      [our batting: order]
           + (4 x their dismissals - their runs)  [our bowling: allocation]

Batting order can only move the first part and the bowling split only the
second, so each decision is judged on its own part.

This is in-sample. The model has already seen these games, so a replay
shows what the model thinks a typical version of the day looks like, not a
forecast.
"""

from __future__ import annotations

import json
import math
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from junior_cricket.ball_model import JointOutcomes
from junior_cricket.playhq_parse import Appearance, parse_scorecard
from junior_cricket.playhq_scorebook import PlayerAliases, completed_games
from junior_cricket.rules_u10 import U10
from junior_cricket.simulator import (
    InningsResult,
    PlayerSkills,
    generate_over_sequence,
    simulate_innings_u10,
)

OVERS = U10.overs_per_innings
PENALTY = U10.dismissal_penalty_runs


# --------------------------------------------------------------------------
# Real games


@dataclass
class RealInnings:
    """One recorded innings, over by over.

    Attributes:
        runs: Runs scored in each over.
        wickets: Dismissals in each over, run outs included.
        bowlers: Who bowled each over (an alias).
    """

    runs: List[int]
    wickets: List[int]
    bowlers: List[str]

    @property
    def total_runs(self) -> int:
        return int(sum(self.runs))

    @property
    def total_wickets(self) -> int:
        return int(sum(self.wickets))


@dataclass
class RealGame:
    """A completed U10 game, as recorded.

    Attributes:
        game_id: PlayHQ game ID.
        date: ISO date.
        opponent: Opposition team name.
        ours: Our batting order (aliases).
        theirs: Their batting order (aliases).
        our_innings: Our batting innings (bowlers are theirs).
        their_innings: Their batting innings (bowlers are ours).
        our_total: Final team total including penalty runs.
        their_total: The opposition's final total.
    """

    game_id: str
    date: str
    opponent: str
    ours: Tuple[str, ...]
    theirs: Tuple[str, ...]
    our_innings: RealInnings
    their_innings: RealInnings
    our_total: int
    their_total: int

    @property
    def our_overs(self) -> Tuple[str, ...]:
        """Which of our players bowled each over of their innings."""
        return tuple(self.their_innings.bowlers)

    @property
    def their_overs(self) -> Tuple[str, ...]:
        """Which of their players bowled each over of our innings."""
        return tuple(self.our_innings.bowlers)


def key_to_alias(path: Path) -> Dict[str, str]:
    """Map PlayHQ profile keys to pseudonyms."""
    frame = pd.read_csv(path)
    return dict(zip(frame["key"], frame["alias"]))


def batting_order(
    lines: Iterable[Appearance], aliases: Dict[str, str]
) -> Optional[Tuple[str, ...]]:
    """Aliases of the players who batted, in batting order, or None if one is unknown."""
    batters = sorted((a for a in lines if a.batting and a.batting.balls_faced > 0),
                     key=lambda a: a.batting.order)
    names = [aliases.get(PlayerAliases.key_for(a)) for a in batters]
    return None if any(n is None for n in names) else tuple(names)


def real_innings(balls: pd.DataFrame) -> Optional[RealInnings]:
    """Over-by-over record of one innings from its ball rows.

    Args:
        balls: The innings' deliveries in the order bowled. The scorer's
            ``over`` number is used when present so an over with a missing
            ball stays aligned; otherwise overs are counted six balls at a time.

    Returns:
        The record, or None unless it covers a full ``OVERS`` overs.
    """
    frame = balls.copy()
    if "over" not in frame:
        frame["over"] = np.arange(len(frame)) // U10.fair_balls_per_over + 1
    grouped = frame.groupby("over", sort=True)
    if list(grouped.groups) != list(range(1, OVERS + 1)):
        return None
    return RealInnings(
        runs=[int(x) for x in grouped["runs"].sum()],
        wickets=[int(x) for x in grouped["dismissed"].sum()],
        bowlers=[str(x) for x in grouped["bowler"].agg(lambda s: s.value_counts().idxmax())],
    )


def load_real_games(
    raw_dir: Path, balls_path: Path, team_ids: Sequence[str]
) -> Tuple[List[RealGame], List[str]]:
    """Rebuild every usable completed game from the cached PlayHQ data.

    A game is used when its scorecard, totals and ball-by-ball are all
    present, both innings ran the full length, and every batter has an alias.

    Args:
        raw_dir: ``data/raw/playhq``.
        balls_path: ``playhq_balls.csv`` from ``fetch_scorecards.py``.
        team_ids: PlayHQ team IDs whose cached fixtures to read.

    Returns:
        (games in date order, one note per game that was left out).
    """
    ours_map = key_to_alias(raw_dir / "player_map.csv")
    theirs_map = key_to_alias(raw_dir / "opposition_map.csv")
    our_aliases = set(ours_map.values())
    balls = pd.read_csv(balls_path)
    games: List[RealGame] = []
    notes: List[str] = []
    for team_id in team_ids:
        fixture = json.loads((raw_dir / "fixtures" / f"team_{team_id}.json").read_text(
            encoding="utf-8"))
        for info in completed_games(fixture):
            card_path = raw_dir / "games" / info.game_id / "scorecard.json"
            rows = balls[balls["game_id"] == info.game_id]
            if not card_path.exists() or info.our_total is None or info.their_total is None:
                notes.append(f"{info.game_id}: no scorecard or totals")
                continue
            if rows.empty:
                notes.append(f"{info.game_id}: no ball-by-ball events")
                continue
            card = parse_scorecard(json.loads(card_path.read_text(encoding="utf-8")))
            other = "AWAY" if info.our_side == "HOME" else "HOME"
            ours = batting_order(card.side(info.our_side), ours_map)
            theirs = batting_order(card.side(other), theirs_map)
            mine = rows["batter"].isin(our_aliases)
            our_inn = real_innings(rows[mine])
            their_inn = real_innings(rows[~mine])
            if ours is None or theirs is None or our_inn is None or their_inn is None:
                notes.append(f"{info.game_id}: shortened game or unlinked player")
                continue
            if len(ours) not in U10.batting_ball_allotments or len(
                    theirs) not in U10.batting_ball_allotments:
                notes.append(f"{info.game_id}: team size outside the allotment table")
                continue
            games.append(RealGame(
                game_id=info.game_id, date=info.date, opponent=info.opponent,
                ours=ours, theirs=theirs, our_innings=our_inn, their_innings=their_inn,
                our_total=info.our_total, their_total=info.their_total))
    return sorted(games, key=lambda g: g.date), notes


# --------------------------------------------------------------------------
# Simulation


@lru_cache(maxsize=None)
def stub(name: str) -> PlayerSkills:
    """A skills placeholder; the joint model reads its effects by name."""
    return PlayerSkills(name=name, p_out=0.05, p_bound=0.07, srr=0.5,
                        p_wicket=0.04, econ=1.0, p_extra=0.0)


def play_innings(
    outcomes: JointOutcomes,
    batting: Sequence[str],
    overs: Sequence[str],
    rng: np.random.Generator,
) -> InningsResult:
    """One innings: ``batting`` in order against ``overs`` (the bowler of each over)."""
    fielders = {name: stub(name) for name in dict.fromkeys(overs)}
    return simulate_innings_u10(
        [stub(n) for n in batting], fielders, list(fielders), dict(Counter(overs)), rng,
        outcomes=outcomes, enforce_bowling_table=False, over_sequence=list(overs))


@dataclass(frozen=True)
class Task:
    """A batch of simulations of one set of decisions.

    Attributes:
        kind: ``match`` (both innings), ``bat`` (our innings only) or
            ``bowl`` (their innings only).
        game_id: The game, for its fitted conditions effect.
        our_bat: Our batting order.
        their_bat: Their batting order.
        our_overs: Our bowler in each over of their innings.
        their_overs: Their bowler in each over of our innings.
        n: Simulations to run.
        seed: Entropy for this batch's random generator.
        day: True to use the game's fitted ground and conditions effect;
            False to draw a typical one for each simulation.
    """

    kind: str
    game_id: str
    our_bat: Tuple[str, ...]
    their_bat: Tuple[str, ...]
    our_overs: Tuple[str, ...]
    their_overs: Tuple[str, ...]
    n: int
    seed: Tuple[int, ...]
    day: bool = False


_POOL: List[JointOutcomes] = []


def _init_worker(pool: List[JointOutcomes]) -> None:
    global _POOL
    _POOL = pool


def run_task(task: Task) -> Dict[str, np.ndarray]:
    """Run a batch; return per-simulation, per-over runs and dismissals.

    Keys are ``our_runs`` and ``our_wkts`` (our batting innings) and
    ``their_runs`` and ``their_wkts`` (theirs), each shaped
    (simulations, overs), for the innings the task's kind asks for.
    """
    rng = np.random.default_rng(list(task.seed))
    want_ours = task.kind in ("match", "bat")
    want_theirs = task.kind in ("match", "bowl")
    out = {}
    for side, wanted in (("our", want_ours), ("their", want_theirs)):
        if wanted:
            out[f"{side}_runs"] = np.zeros((task.n, OVERS), dtype=np.int16)
            out[f"{side}_wkts"] = np.zeros((task.n, OVERS), dtype=np.int16)
    for k in range(task.n):
        outcomes = _POOL[int(rng.integers(len(_POOL)))]
        if task.day:
            outcomes.use_day(task.game_id)
        else:
            outcomes.begin_game(rng)
        if want_ours:
            inn = play_innings(outcomes, task.our_bat, task.their_overs, rng)
            out["our_runs"][k], out["our_wkts"][k] = inn.over_runs, inn.over_wickets
        if want_theirs:
            inn = play_innings(outcomes, task.their_bat, task.our_overs, rng)
            out["their_runs"][k], out["their_wkts"][k] = inn.over_runs, inn.over_wickets
    return out


class Runner:
    """Runs simulation tasks inline or across worker processes.

    Args:
        pool: Joint-model rules, one per posterior draw; each simulation
            picks one at random so uncertainty about every player's skill
            carries into the results.
        workers: Worker processes; 1 runs in this process.
    """

    def __init__(self, pool: List[JointOutcomes], workers: int = 1) -> None:
        self.pool = pool
        self.workers = workers
        self._executor: Optional[ProcessPoolExecutor] = None

    def __enter__(self) -> "Runner":
        if self.workers > 1:
            self._executor = ProcessPoolExecutor(
                max_workers=self.workers, initializer=_init_worker, initargs=(self.pool,))
        else:
            _init_worker(self.pool)
        return self

    def __exit__(self, *exc) -> None:
        if self._executor is not None:
            self._executor.shutdown()

    def run(self, tasks: Sequence, worker=None) -> List[Dict[str, np.ndarray]]:
        """Results in the order of ``tasks``.

        Args:
            tasks: What to run.
            worker: A top-level function taking one task; ``run_task`` by
                default. It reads the pool through this module's ``_POOL``.
        """
        worker = worker or run_task
        if self._executor is None:
            return [worker(t) for t in tasks]
        return list(self._executor.map(worker, tasks, chunksize=1))


def default_workers() -> int:
    """Leave a couple of cores free."""
    return max(1, (os.cpu_count() or 2) - 2)


def match_task(
    game: RealGame, n: int, seed: Sequence[int], kind: str = "match", day: bool = False,
    our_bat: Optional[Sequence[str]] = None, our_overs: Optional[Sequence[str]] = None,
) -> Task:
    """A task for ``game`` as played, with our batting order and/or bowling overs replaced."""
    return Task(
        kind=kind, game_id=game.game_id,
        our_bat=tuple(our_bat) if our_bat is not None else game.ours,
        their_bat=game.theirs,
        our_overs=tuple(our_overs) if our_overs is not None else game.our_overs,
        their_overs=game.their_overs, n=n, seed=tuple(seed), day=day)


# --------------------------------------------------------------------------
# Reading the results


def totals(sims: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Final team totals and margin per simulation (both innings needed)."""
    our_runs, their_runs = sims["our_runs"].sum(axis=1), sims["their_runs"].sum(axis=1)
    our_wkts, their_wkts = sims["our_wkts"].sum(axis=1), sims["their_wkts"].sum(axis=1)
    our_total = our_runs + PENALTY * their_wkts
    their_total = their_runs + PENALTY * our_wkts
    return {"our_total": our_total, "their_total": their_total,
            "margin": our_total - their_total,
            "our_runs": our_runs, "their_runs": their_runs,
            "our_wkts": our_wkts, "their_wkts": their_wkts}


def percentile_rank(values: np.ndarray, actual: float) -> float:
    """Share of simulations below ``actual``, counting ties as half (0 to 1)."""
    return float((np.sum(values < actual) + 0.5 * np.sum(values == actual)) / len(values))


def ordinal(share: float) -> str:
    """A share from 0 to 1 as a percentile with its suffix: 0.92 gives ``92nd``."""
    n = min(max(int(round(share * 100)), 0), 100)
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def net_batting(sims: Dict[str, np.ndarray]) -> np.ndarray:
    """Our batting's part of the margin: runs minus the penalty for our dismissals."""
    return sims["our_runs"].sum(axis=1) - PENALTY * sims["our_wkts"].sum(axis=1)


def net_bowling(sims: Dict[str, np.ndarray]) -> np.ndarray:
    """Our bowling's part of the margin: the penalty for their dismissals minus their runs."""
    return PENALTY * sims["their_wkts"].sum(axis=1) - sims["their_runs"].sum(axis=1)


def over_summary(
    runs: np.ndarray, wickets: np.ndarray, actual_runs: Sequence[int],
    actual_wickets: Sequence[int],
) -> pd.DataFrame:
    """Over-by-over simulated spread beside the recorded innings.

    Args:
        runs: Simulated runs per over, (simulations, overs).
        wickets: Simulated dismissals per over.
        actual_runs: Recorded runs per over.
        actual_wickets: Recorded dismissals per over.

    Returns:
        One row per over: per-over median and quartiles, the cumulative
        median and 50% and 90% bands, mean dismissals, and the record.
    """
    cum = np.cumsum(runs, axis=1)
    q = lambda a, p: np.percentile(a, p, axis=0)                     # noqa: E731
    frame = pd.DataFrame({
        "over": np.arange(1, runs.shape[1] + 1),
        "sim_median": q(runs, 50), "sim_q25": q(runs, 25), "sim_q75": q(runs, 75),
        "sim_mean": runs.mean(axis=0),
        "cum_median": q(cum, 50), "cum_q05": q(cum, 5), "cum_q25": q(cum, 25),
        "cum_q75": q(cum, 75), "cum_q95": q(cum, 95),
        "sim_wickets": wickets.mean(axis=0),
        "actual_runs": list(actual_runs),
        "actual_cum": np.cumsum(list(actual_runs)),
        "actual_wickets": list(actual_wickets),
    })
    return frame


# --------------------------------------------------------------------------
# Decisions to compare


def random_orders(
    order: Sequence[str], n: int, rng: np.random.Generator
) -> List[Tuple[str, ...]]:
    """Up to ``n`` distinct random reorderings of a batting order, never the original."""
    seen = {tuple(order)}
    out: List[Tuple[str, ...]] = []
    n = min(n, math.factorial(len(order)) - 1)
    while len(out) < n:
        candidate = tuple(order[i] for i in rng.permutation(len(order)))
        if candidate not in seen:
            seen.add(candidate)
            out.append(candidate)
    return out


def ranked_orders(
    order: Sequence[str], value: Dict[str, float]
) -> Dict[str, Tuple[str, ...]]:
    """Rule-of-thumb batting orders built from each batter's model value.

    Args:
        order: The order as played (its members are the batters).
        value: Batter -> value (higher is better).

    Returns:
        ``best first``, ``worst first``, ``alternating`` (best, worst,
        next best, ...) and ``reversed`` (the played order backwards).
    """
    ranked = sorted(order, key=lambda p: value[p], reverse=True)
    alternating: List[str] = []
    lo, hi = 0, len(ranked) - 1
    while lo <= hi:
        alternating.append(ranked[lo])
        if lo != hi:
            alternating.append(ranked[hi])
        lo, hi = lo + 1, hi - 1
    return {"best first": tuple(ranked), "worst first": tuple(reversed(ranked)),
            "alternating": tuple(alternating), "reversed": tuple(reversed(order))}


def rotation_of(overs: Sequence[str]) -> List[str]:
    """Bowlers in order of first appearance."""
    return list(dict.fromkeys(overs))


def keepers_of(allocation: Dict[str, int]) -> Tuple[str, ...]:
    """The wicketkeepers, taken to be the two players given the fewest overs.

    The U10 tables give the two keepers the smallest share. Returns an empty
    tuple when the split has no such pair (a team of five all bowl equally).
    """
    fewest = min(allocation.values())
    pair = [p for p, t in allocation.items() if t == fewest]
    return tuple(pair) if len(pair) == 2 and fewest < max(allocation.values()) else ()


def sequence_for(allocation: Dict[str, int], rotation: Sequence[str]) -> Tuple[str, ...]:
    """The over-by-over round robin for an allocation of overs."""
    return tuple(generate_over_sequence(
        [p for p in rotation if allocation.get(p, 0) > 0], allocation, OVERS))


def reassigned(
    allocation: Dict[str, int], order: Sequence[str], fixed: Iterable[str] = (),
) -> Dict[str, int]:
    """Hand the same overs to the same bowlers in the order given.

    Args:
        allocation: Bowler -> overs as played.
        order: The free bowlers, best to worst: the largest shares go first.
        fixed: Bowlers who keep their share (the keepers).
    """
    fixed = set(fixed)
    shares = sorted((t for p, t in allocation.items() if p not in fixed), reverse=True)
    out = {p: allocation[p] for p in fixed}
    out.update(dict(zip(order, shares)))
    return out


def allocation_candidates(
    allocation: Dict[str, int], value: Dict[str, float], rng: np.random.Generator,
    n_random: int, keepers_fixed: bool,
) -> Dict[str, Dict[str, int]]:
    """Alternative ways to share out the same overs among the same bowlers.

    Args:
        allocation: Bowler -> overs as played.
        value: Bowler -> value (higher is better).
        rng: Random generator for the random reassignments.
        n_random: How many random reassignments to include.
        keepers_fixed: Keep the two keepers on their one over each.

    Returns:
        Label -> allocation: ``best first``, ``worst first`` and random ones.
    """
    fixed = keepers_of(allocation) if keepers_fixed else ()
    free = [p for p in allocation if p not in fixed]
    best = sorted(free, key=lambda p: value[p], reverse=True)
    out = {"best first": reassigned(allocation, best, fixed),
           "worst first": reassigned(allocation, best[::-1], fixed)}
    seen = {tuple(sorted(allocation.items())), *(tuple(sorted(a.items())) for a in out.values())}
    tries = 0
    while len(out) < 2 + n_random and tries < 50 * max(n_random, 1):
        tries += 1
        cand = reassigned(allocation, [free[i] for i in rng.permutation(len(free))], fixed)
        key = tuple(sorted(cand.items()))
        if key not in seen:
            seen.add(key)
            out[f"random {len(out) - 1}"] = cand
    return out


# --------------------------------------------------------------------------
# Screening and confirming decisions


@dataclass
class DecisionStudy:
    """The outcome of comparing alternatives for one decision in one game.

    Attributes:
        screen: Label -> (mean, standard error) over the screening runs,
            one row per alternative.
        confirm: Label -> (mean, standard error) from fresh runs of the
            alternatives worth a second look.
        baseline: The label everything is compared with.
        true_sd: Spread of the real effect of a random alternative, with
            the simulation noise taken out (runs).
    """

    screen: Dict[str, Tuple[float, float]]
    confirm: Dict[str, Tuple[float, float]]
    baseline: str
    true_sd: float
    picks: Dict[str, str] = field(default_factory=dict)

    def gain(self, label: str, against: Optional[str] = None) -> Tuple[float, float]:
        """Confirmed gain over the baseline (or ``against``) and its standard error (runs)."""
        m, se = self.confirm[label]
        m0, se0 = self.confirm[against or self.baseline]
        return m - m0, math.hypot(se, se0)

    def best_label(self) -> str:
        """The alternative that looked best in screening, ties broken by label order.

        Chosen on the screening runs and judged on the confirming runs, so
        the reported gain is not flattered by having picked the maximum.
        """
        labels = [x for x in self.confirm if x != self.baseline and not x.startswith("as played")]
        return max(labels, key=lambda x: self.screen[x][0])


def _mean_se(values: np.ndarray) -> Tuple[float, float]:
    return float(values.mean()), float(values.std(ddof=1) / math.sqrt(len(values)))


def study_decision(
    runner: Runner,
    game: RealGame,
    kind: str,
    candidates: Dict[str, Tuple[str, ...]],
    baseline: str,
    seed: Sequence[int],
    screen_n: int = 1500,
    confirm_n: int = 10000,
    day: bool = False,
) -> DecisionStudy:
    """Compare alternatives for one decision, guarding against picking noise.

    Every alternative is simulated ``screen_n`` times. The baseline, the
    named rules of thumb, and the best and worst random alternative *by the
    screening runs* are then simulated afresh ``confirm_n`` times. Only the
    fresh runs are reported as results: choosing the best of many noisy
    estimates flatters it, and a re-run removes that flattery.

    Args:
        runner: Where to run the simulations.
        game: The game being replayed.
        kind: ``bat`` (candidates are batting orders, judged by our runs
            minus the penalty for our dismissals) or ``bowl`` (candidates
            are over-by-over bowler sequences, judged by the penalty for
            their dismissals minus their runs).
        candidates: Label -> the decision (a batting order, or the bowler
            of each over). Labels starting with ``random`` are the random set.
        baseline: Label in ``candidates`` to compare against.
        seed: Base entropy.
        screen_n: Screening simulations per alternative.
        confirm_n: Fresh simulations per finalist.
        day: Use the game's fitted conditions instead of typical ones.

    Returns:
        The screen and confirm tables and the spread of true effects.
    """
    metric = net_batting if kind == "bat" else net_bowling

    def tasks_for(labels: Sequence[str], n: int, stage: int) -> List[Task]:
        out = []
        for i, label in enumerate(labels):
            arg = ({"our_bat": candidates[label]} if kind == "bat"
                   else {"our_overs": candidates[label]})
            out.append(match_task(game, n, (*seed, stage, i), kind=kind, day=day, **arg))
        return out

    labels = list(candidates)
    screened = {label: _mean_se(metric(r))
                for label, r in zip(labels, runner.run(tasks_for(labels, screen_n, 0)))}
    randoms = [x for x in labels if x.startswith("random")]
    picks = {}
    if randoms:
        by_mean = sorted(randoms, key=lambda x: screened[x][0])
        picks = {"best random (by screening)": by_mean[-1],
                 "worst random (by screening)": by_mean[0]}
    finalists = [x for x in labels if not x.startswith("random")]
    finalists += list(dict.fromkeys(picks.values()))
    confirmed = {label: _mean_se(metric(r))
                 for label, r in zip(finalists, runner.run(tasks_for(finalists, confirm_n, 1)))}
    means = np.array([screened[x][0] for x in randoms]) if randoms else np.array([])
    noise = np.array([screened[x][1] ** 2 for x in randoms]) if randoms else np.array([])
    true_sd = math.sqrt(max(0.0, means.var(ddof=1) - noise.mean())) if len(means) > 2 else float("nan")
    return DecisionStudy(screen=screened, confirm=confirmed, baseline=baseline,
                         true_sd=true_sd, picks=picks)
