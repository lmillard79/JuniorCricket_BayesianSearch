"""Tests for the game replays and decision studies."""

import numpy as np
import pandas as pd
import pytest

from junior_cricket import replay as R
from junior_cricket.ball_model import JointOutcomes
from junior_cricket.playhq_parse import Appearance, BattingLine
from junior_cricket.playhq_scorebook import PlayerAliases

OURS = [f"P{i}" for i in range(9)]
THEIRS = [f"O{i}" for i in range(9)]


def rule(boundary_day: float = 0.0, fielding_mu: float = 0.0) -> JointOutcomes:
    zeros = {"d": 0.0, "b": 0.0, "s": 0.0}
    names = OURS + THEIRS
    return JointOutcomes(
        intercepts={"d": float(np.log(0.05 / 0.95)), "b": float(np.log(0.08 / 0.92)), "s": 0.0},
        bat={n: dict(zeros) for n in names}, bowl={n: dict(zeros) for n in names},
        bat_sd={"d": 0.3, "b": 0.8, "s": 0.4}, bowl_sd={"d": 0.3, "b": 0.2, "s": 0.4},
        game_sd={"b": 0.25, "s": 0.3}, fielding_mu=fielding_mu, fielding_sd=0.0,
        days={"g1": {"b": boundary_day, "s": 0.0}},
    )


ALLOCATION = {"P0": 3, "P1": 3, "P2": 3, "P3": 3, "P4": 2, "P5": 2, "P6": 2, "P7": 1, "P8": 1}
ROTATION = list(ALLOCATION)
PLAYED = R.sequence_for(ALLOCATION, ROTATION)


def game() -> R.RealGame:
    innings = R.RealInnings(runs=[5] * 20, wickets=[0] * 20, bowlers=list(PLAYED))
    theirs = R.RealInnings(runs=[6] * 20, wickets=[1] * 20, bowlers=list(PLAYED))
    return R.RealGame("g1", "2025-10-11", "Opponents", tuple(OURS), tuple(THEIRS),
                      our_innings=innings, their_innings=theirs, our_total=180, their_total=100)


def test_percentile_rank_counts_ties_as_half() -> None:
    values = np.array([1, 2, 3, 3, 4])
    assert R.percentile_rank(values, 3) == pytest.approx(0.6)
    assert R.percentile_rank(values, 0) == 0.0 and R.percentile_rank(values, 9) == 1.0


def test_percentiles_read_naturally() -> None:
    got = {p: R.ordinal(p / 100) for p in (0, 1, 2, 3, 4, 11, 12, 13, 21, 22, 23, 50, 61, 92, 100)}
    assert got == {0: "0th", 1: "1st", 2: "2nd", 3: "3rd", 4: "4th", 11: "11th", 12: "12th",
                   13: "13th", 21: "21st", 22: "22nd", 23: "23rd", 50: "50th", 61: "61st",
                   92: "92nd", 100: "100th"}


def balls(overs: int = 20, per_over: int = 6) -> pd.DataFrame:
    n = overs * per_over
    return pd.DataFrame({
        "runs": [1] * n, "dismissed": [1 if i % 6 == 0 else 0 for i in range(n)],
        "bowler": [f"O{(i // 6) % 3}" for i in range(n)],
        "over": [i // per_over + 1 for i in range(n)],
    })


def test_real_innings_reads_runs_wickets_and_bowlers_by_over() -> None:
    inn = R.real_innings(balls())
    assert inn.runs == [6] * 20 and inn.wickets == [1] * 20
    assert inn.bowlers[:4] == ["O0", "O1", "O2", "O0"]
    assert inn.total_runs == 120 and inn.total_wickets == 20


def test_a_missing_ball_leaves_a_short_over_but_keeps_the_overs_aligned() -> None:
    frame = balls().drop(index=20)                      # one ball missing from over 4
    inn = R.real_innings(frame)
    assert inn.runs[3] == 5 and inn.runs[4] == 6 and len(inn.runs) == 20


def test_real_innings_counts_overs_by_position_without_the_scorers_numbers() -> None:
    inn = R.real_innings(balls().drop(columns="over"))
    assert inn.runs == [6] * 20


def test_an_innings_that_is_not_full_length_is_not_replayed() -> None:
    assert R.real_innings(balls(overs=18)) is None


def test_batting_order_follows_the_scorecard_and_needs_every_alias() -> None:
    def player(name, order, faced=10):
        return Appearance(name, None, name, order, BattingLine(faced, 5, 0, 0, False, order), None)
    lines = [player("B Two", 2), player("A One", 1), player("Did Not Bat", 3, faced=0)]
    aliases = {PlayerAliases.key_for(lines[1]): "P1", PlayerAliases.key_for(lines[0]): "P2"}
    assert R.batting_order(lines, aliases) == ("P1", "P2")
    assert R.batting_order(lines, {PlayerAliases.key_for(lines[1]): "P1"}) is None


def test_random_orders_are_distinct_reorderings_of_the_same_batters() -> None:
    rng = np.random.default_rng(0)
    orders = R.random_orders(OURS, 30, rng)
    assert len(orders) == len(set(orders)) == 30 and tuple(OURS) not in orders
    assert all(sorted(o) == sorted(OURS) for o in orders)
    assert len(R.random_orders(["a", "b", "c"], 50, rng)) == 5      # 3! - 1 available


def test_ranked_orders_follow_the_value_given() -> None:
    value = {"a": 5, "b": 4, "c": 3, "d": 2, "e": 1}
    orders = R.ranked_orders(["c", "a", "e", "b", "d"], value)
    assert orders["best first"] == ("a", "b", "c", "d", "e")
    assert orders["worst first"] == ("e", "d", "c", "b", "a")
    assert orders["alternating"] == ("a", "e", "b", "d", "c")
    assert orders["reversed"] == ("d", "b", "e", "a", "c")


def test_keepers_are_the_two_bowlers_with_the_fewest_overs() -> None:
    assert set(R.keepers_of(ALLOCATION)) == {"P7", "P8"}
    assert R.keepers_of({"a": 4, "b": 4, "c": 4, "d": 4, "e": 4}) == ()


def test_sequences_give_every_bowler_their_overs() -> None:
    assert len(PLAYED) == 20
    assert {p: PLAYED.count(p) for p in ALLOCATION} == ALLOCATION


def test_reassigning_overs_keeps_the_same_shares_and_the_keepers() -> None:
    order = ["P6", "P5", "P4", "P3", "P2", "P1", "P0"]
    new = R.reassigned(ALLOCATION, order, fixed=("P7", "P8"))
    assert sorted(new.values()) == sorted(ALLOCATION.values())
    assert new["P6"] == 3 and new["P0"] == 2 and new["P7"] == new["P8"] == 1


def test_allocation_candidates_can_hold_the_keepers_still_or_free_them() -> None:
    value = {p: float(i) for i, p in enumerate(reversed(ROTATION))}        # P0 best ... P8 worst
    rng = np.random.default_rng(0)
    held = R.allocation_candidates(ALLOCATION, value, rng, 6, keepers_fixed=True)
    assert held["best first"]["P0"] == 3 and held["worst first"]["P6"] == 3
    assert all(a["P7"] == a["P8"] == 1 for a in held.values())
    free = R.allocation_candidates(ALLOCATION, value, rng, 6, keepers_fixed=False)
    assert free["worst first"]["P8"] == 3                    # the keepers are free to move
    assert sum(1 for k in held if k.startswith("random")) == 6


def test_a_task_returns_over_by_over_arrays_that_reconcile_with_the_totals() -> None:
    R._init_worker([rule()])
    task = R.match_task(game(), 25, (1, 2, 3))
    sims = R.run_task(task)
    assert sims["our_runs"].shape == (25, 20) and sims["their_wkts"].shape == (25, 20)
    t = R.totals(sims)
    assert (t["our_total"] == sims["our_runs"].sum(axis=1) + 4 * sims["their_wkts"].sum(axis=1)).all()
    assert (t["margin"] == t["our_total"] - t["their_total"]).all()
    again = R.run_task(R.match_task(game(), 25, (1, 2, 3)))
    assert (again["our_runs"] == sims["our_runs"]).all()               # same seed, same replays


def test_kinds_run_only_the_innings_they_need() -> None:
    R._init_worker([rule()])
    assert set(R.run_task(R.match_task(game(), 5, (0,), kind="bat"))) == {"our_runs", "our_wkts"}
    assert set(R.run_task(R.match_task(game(), 5, (0,), kind="bowl"))) == {"their_runs", "their_wkts"}


def test_the_days_fitted_conditions_replace_a_typical_draw() -> None:
    R._init_worker([rule(boundary_day=6.0)])                            # a boundary every ball
    typical = R.run_task(R.match_task(game(), 30, (5,), kind="bat"))["our_runs"].sum(axis=1)
    day = R.run_task(R.match_task(game(), 30, (5,), kind="bat", day=True))["our_runs"].sum(axis=1)
    assert day.mean() > typical.mean() + 50


def test_the_fielding_edge_only_lifts_wickets_in_our_bowling_innings() -> None:
    """A strong fielding edge boosts wickets when we bowl (their innings), not when we bat."""
    R._init_worker([rule(fielding_mu=3.0)])
    sims = R.run_task(R.match_task(game(), 30, (7,)))
    their, ours = sims["their_wkts"].sum(axis=1), sims["our_wkts"].sum(axis=1)
    assert their.mean() > ours.mean() + 20                             # we bowl their innings
    R._init_worker([rule()])                                          # no edge: roughly level
    level = R.run_task(R.match_task(game(), 30, (7,)))
    their0, ours0 = level["their_wkts"].sum(axis=1), level["our_wkts"].sum(axis=1)
    assert abs(their0.mean() - ours0.mean()) < 3


def test_a_decision_study_screens_then_confirms_on_fresh_runs() -> None:
    candidates = {"as played": tuple(OURS)}
    candidates.update(R.ranked_orders(OURS, {p: float(i) for i, p in enumerate(reversed(OURS))}))
    for j, order in enumerate(R.random_orders(OURS, 5, np.random.default_rng(0)), 1):
        candidates[f"random {j}"] = order
    with R.Runner([rule(), rule()], workers=1) as runner:
        study = R.study_decision(runner, game(), "bat", candidates, "as played", (9,),
                                 screen_n=30, confirm_n=60)
    assert set(study.screen) == set(candidates)
    assert {"as played", "best first", "worst first", "alternating", "reversed"} <= set(study.confirm)
    assert set(study.picks) == {"best random (by screening)", "worst random (by screening)"}
    assert study.gain("as played")[0] == 0.0
    assert study.best_label() not in ("as played",) and not study.best_label().startswith("as played")
    assert np.isfinite(study.true_sd) or np.isnan(study.true_sd)


def test_bowling_studies_judge_the_penalty_for_their_dismissals_minus_their_runs() -> None:
    sims = {"their_runs": np.array([[10, 10]]), "their_wkts": np.array([[1, 2]])}
    assert R.net_bowling(sims)[0] == 4 * 3 - 20
    sims = {"our_runs": np.array([[10, 10]]), "our_wkts": np.array([[1, 2]])}
    assert R.net_batting(sims)[0] == 20 - 4 * 3


def test_over_summary_gives_the_spread_beside_the_record() -> None:
    runs = np.tile(np.arange(20), (50, 1))
    table = R.over_summary(runs, np.zeros((50, 20)), [1] * 20, [0] * 20)
    assert list(table["over"]) == list(range(1, 21))
    assert table["sim_median"].iloc[4] == 4 and table["cum_median"].iloc[-1] == sum(range(20))
    assert table["actual_cum"].iloc[-1] == 20
