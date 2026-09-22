"""Tests for the U11 batting-order strategy comparison."""

import numpy as np
import pytest

from junior_cricket import replay as R
from junior_cricket import strategies as S
from junior_cricket.ball_model import JointOutcomes

SQUAD = [f"P{i}" for i in range(9)]


def rule(strength: float = 0.0) -> JointOutcomes:
    """A rule where batter Pi is stronger for lower i (more boundaries, fewer dismissals)."""
    bat = {f"P{i}": {"d": 0.15 * (i - 4), "b": -strength * (i - 4), "s": -0.1 * (i - 4)}
           for i in range(9)}
    zeros = {"d": 0.0, "b": 0.0, "s": 0.0}
    bowl = {f"P{i}": dict(zeros) for i in range(9)}
    return JointOutcomes(
        intercepts={"d": float(np.log(0.045 / 0.955)), "b": float(np.log(0.08 / 0.92)), "s": 0.0},
        bat=bat, bowl=bowl, bat_sd={"d": 0.3, "b": 0.8, "s": 0.4},
        bowl_sd={"d": 0.3, "b": 0.2, "s": 0.4}, game_sd={"b": 0.25, "s": 0.3})


VALUE = {f"P{i}": float(9 - i) for i in range(9)}                # P0 best ... P8 worst


def test_the_named_strategies_are_permutations_of_the_squad() -> None:
    orders = S.named_orders(SQUAD, VALUE)
    assert all(sorted(o) == sorted(SQUAD) for o in orders.values())
    assert orders["strongest to weakest"] == tuple(SQUAD)
    assert orders["weakest to strongest"] == tuple(reversed(SQUAD))
    assert orders["strong-weak alternating"][:4] == ("P0", "P8", "P1", "P7")
    assert orders["balanced pairs (top six)"] == ("P0", "P5", "P1", "P4", "P2", "P3", "P6", "P7", "P8")
    assert orders["strong-middle alternating"] == ("P0", "P3", "P1", "P4", "P2", "P5", "P6", "P7", "P8")


def test_ranking_uses_the_value_not_the_input_order() -> None:
    shuffled = ["P4", "P8", "P0", "P2", "P6", "P1", "P7", "P3", "P5"]
    assert S.named_orders(shuffled, VALUE)["strongest to weakest"] == tuple(SQUAD)


def test_a_stint_is_worth_more_for_a_batter_who_scores_faster_and_survives() -> None:
    r = rule(strength=0.3)
    assert S.stint_value(r, "P0") > S.stint_value(r, "P4") > S.stint_value(r, "P8")
    never_out = rule()
    never_out.a["d"] = -30.0
    d, per_ball = S.runs_per_ball(never_out, batter="P4")
    assert d < 1e-9 and S.stint_value(never_out, "P4", balls=30) == pytest.approx(30 * per_ball)


def test_an_average_bowler_and_batter_are_the_zero_effects() -> None:
    r = rule()
    d, runs = S.runs_per_ball(r)
    assert d == pytest.approx(0.045) and runs > 0


def test_a_world_is_fixed_by_its_number_and_the_seed() -> None:
    pool = [rule(), rule(0.2)]
    scenario = S.Scenario(drift_sd=0.3)
    a = S.make_world(pool, scenario, SQUAD, seed=5, number=7)
    b = S.make_world(pool, scenario, SQUAD, seed=5, number=7)
    assert a[1:] == b[1:] and a[0].bat["P3"] == b[0].bat["P3"]
    c = S.make_world(pool, scenario, SQUAD, seed=5, number=8)
    assert c[4] != a[4]                                       # a different attack
    assert sorted(a[2].values(), reverse=True) == [4, 4, 3, 3, 3, 3, 3, 1, 1]


def test_smart_tactics_give_the_biggest_shares_to_the_best_bowlers() -> None:
    pool = [rule()]
    _, opposition, allocation, rotation, _ = S.make_world(
        pool, S.Scenario(tactics="smart"), SQUAD, seed=1, number=0)
    rule_, *_ = S.make_world(pool, S.Scenario(tactics="smart"), SQUAD, seed=1, number=0)
    economy = {n: S.runs_per_ball(rule_, bowler=n)[1] for n in opposition}
    best_two = sorted(opposition, key=economy.get)[:2]
    assert {allocation[n] for n in best_two} == {4} and rotation[:2] == best_two


def test_drift_and_bigger_grounds_change_the_world() -> None:
    pool = [rule()]
    plain = S.make_world(pool, S.Scenario(), SQUAD, 3, 0)[0]
    drifted = S.make_world(pool, S.Scenario(drift_sd=0.5), SQUAD, 3, 0)[0]
    bigger = S.make_world(pool, S.Scenario(boundary_shift=-1.0), SQUAD, 3, 0)[0]
    assert plain.bat["P2"] != drifted.bat["P2"]
    assert bigger.a["b"] == pytest.approx(plain.a["b"] - 1.0)
    assert plain.bat["P2"] == pool[0].bat["P2"]               # the shared pool is not disturbed


def test_every_strategy_meets_the_same_worlds() -> None:
    orders = S.named_orders(SQUAD, VALUE)
    with R.Runner([rule(0.3), rule(0.1)], workers=1) as runner:
        results = S.evaluate(runner, orders, SQUAD, S.Scenario(), (0, 40), 2, seed=3, chunks=4)
    ratings = [r["rating"] for r in results.values()]
    assert all((x == ratings[0]).all() for x in ratings)
    first = results["strongest to weakest"]
    assert first["total"].shape == (40, 2) and first["balls"].shape == (40, 2, 9)
    assert first["over_runs"].sum(axis=2).tolist() == first["total"].tolist()
    assert (first["balls"].sum(axis=2) <= 9 * 35).all()


def test_the_best_batters_should_bat_first_when_skill_differs_a_lot() -> None:
    orders = {"strongest to weakest": tuple(SQUAD), "weakest to strongest": tuple(reversed(SQUAD))}
    with R.Runner([rule(0.4)], workers=1) as runner:
        results = S.evaluate(runner, orders, SQUAD, S.Scenario(), (0, 300), 2, seed=1, chunks=3)
    diff, se, better = S.paired(results["strongest to weakest"], results["weakest to strongest"])
    assert diff > 5 * se and better > 0.5


def test_paired_differences_use_the_worlds_as_the_unit() -> None:
    a = {"total": np.array([[10, 12], [20, 22]])}
    b = {"total": np.array([[8, 8], [21, 21]])}
    diff, se, better = S.paired(a, b)
    assert diff == pytest.approx(((11 - 8) + (21 - 21)) / 2) and better == 0.5
    assert se == pytest.approx(np.std([3, 0], ddof=1) / np.sqrt(2))


def test_attack_groups_run_from_strongest_to_weakest() -> None:
    groups = S.by_attack({"rating": np.array([0.9, 0.5, 0.7, 0.6, 0.8, 0.4])}, parts=3)
    assert [g.tolist() for g in groups] == [[5, 1], [3, 2], [4, 0]]


def test_random_orders_are_distinct_and_reproducible() -> None:
    a, b = S.random_orders(SQUAD, 6, seed=2), S.random_orders(SQUAD, 6, seed=2)
    assert a == b and len(set(a.values())) == 6
    assert all(sorted(o) == sorted(SQUAD) for o in a.values())


def test_a_search_does_not_chase_noise() -> None:
    """Starting from the best order with a strong skill gradient, nothing should beat it."""
    with R.Runner([rule(0.4)], workers=1) as runner:
        found, history = S.hill_climb(runner, tuple(SQUAD), SQUAD, S.Scenario(), (0, 150), 2,
                                      seed=4, rounds=2, z=3.0)
    assert found[:3] == tuple(SQUAD[:3]) and history[0][2] == 0.0


def test_a_search_finds_an_obviously_better_order() -> None:
    start = tuple(reversed(SQUAD))
    with R.Runner([rule(0.4)], workers=1) as runner:
        found, history = S.hill_climb(runner, start, SQUAD, S.Scenario(), (0, 200), 2,
                                      seed=4, rounds=3, z=2.0)
    assert found != start and history[-1][1] > history[0][1]


def test_evaluate_one_matches_evaluate_when_nobody_is_bankable() -> None:
    """With no bankable batters, evaluate_one is the same rule evaluate uses."""
    with R.Runner([rule(0.3), rule(0.1)], workers=1) as runner:
        via_evaluate = S.evaluate(runner, {"strongest to weakest": tuple(SQUAD)}, SQUAD,
                                  S.Scenario(), (0, 30), 2, seed=6, chunks=3)["strongest to weakest"]
        via_one = S.evaluate_one(runner, tuple(SQUAD), SQUAD, S.Scenario(), (0, 30), 2, seed=6, chunks=3)
    assert via_evaluate["total"].tolist() == via_one["total"].tolist()


def test_evaluate_one_bank_and_recall_changes_who_faces_the_balls() -> None:
    """Bankable top batters recalled on a cheap wicket face more balls than plain retirement."""
    scenario = S.Scenario(retire_at=25)
    with R.Runner([rule(0.4)], workers=1) as runner:
        plain = S.evaluate_one(runner, tuple(SQUAD), SQUAD, scenario, (0, 200), 2, seed=8, chunks=4)
        banked = S.evaluate_one(runner, tuple(SQUAD), SQUAD, scenario, (0, 200), 2, seed=8, chunks=4,
                                bankable=SQUAD[:4], recall_within_balls=6, recall_below_runs=5)
    # The top four are on strike more often when they can be recalled early.
    assert banked["balls"][:, :, :4].mean() > plain["balls"][:, :, :4].mean()
