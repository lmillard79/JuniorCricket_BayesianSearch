"""Tests for the joint batter-by-bowler ball model."""

from types import SimpleNamespace

import numpy as np
import pytest

from junior_cricket.ball_model import (
    BallData,
    JointOutcomes,
    effect_table,
    fit_ball_model,
    held_out_lpd,
)


def tiny_data() -> BallData:
    return BallData(
        players=["A", "B", "C"], games=["g1", "g2"],
        bat=np.array([0, 0, 1, 1, 2, 2]), bowl=np.array([1, 1, 2, 2, 0, 0]),
        game=np.array([0, 0, 0, 1, 1, 1]),
        runs=np.array([0, 4, 1, 0, 6, 2]),
        dismissed=np.array([0, 0, 0, 1, 0, 0]),
        boundary=np.array([0, 1, 0, 0, 1, 0]),
    )


def test_components_select_the_balls_that_inform_them() -> None:
    data = tiny_data()
    mask, y = data.component("d")
    assert mask.all() and list(y) == [0, 0, 0, 1, 0, 0]
    mask, y = data.component("b")                      # not out: 5 balls
    assert int(mask.sum()) == 5 and list(y) == [0, 1, 0, 1, 0]
    mask, y = data.component("s")                      # not out, not a boundary
    assert list(np.nonzero(mask)[0]) == [0, 2, 5] and list(y) == [0, 1, 1]


def test_take_keeps_the_player_and_game_coding() -> None:
    subset = tiny_data().take(np.array([True, False, True, False, False, True]))
    assert len(subset.bat) == 3 and subset.players == ["A", "B", "C"]


def outcomes(a=(0.0, 0.0, 0.0), bat=None, bowl=None) -> JointOutcomes:
    zeros = {"d": 0.0, "b": 0.0, "s": 0.0}
    return JointOutcomes(
        intercepts=dict(zip("dbs", a)),
        bat=bat or {"x": zeros}, bowl=bowl or {"y": zeros},
        bat_sd={"d": 0.3, "b": 0.8, "s": 0.4}, bowl_sd={"d": 0.3, "b": 0.2, "s": 0.4},
        game_sd={"b": 0.25, "s": 0.3},
    )


X, Y = SimpleNamespace(name="x"), SimpleNamespace(name="y")


def test_dismissal_probability_adds_batter_and_bowler_on_the_logit_scale() -> None:
    base = np.log(0.1 / 0.9)
    rule = outcomes(a=(base, 0, 0),
                    bat={"x": {"d": 1.0, "b": 0, "s": 0}},
                    bowl={"y": {"d": -0.5, "b": 0, "s": 0}})
    expected = 1 / (1 + np.exp(-(base + 1.0 - 0.5)))
    assert rule.dismissal_probability(X, Y) == pytest.approx(expected)
    assert outcomes(a=(base, 0, 0)).dismissal_probability(X, Y) == pytest.approx(0.1)


def test_runs_follow_the_boundary_then_scoring_logic() -> None:
    rng = np.random.default_rng(0)
    always_boundary = outcomes(a=(0, 50.0, 0))
    seen = {always_boundary.runs(X, Y, rng) for _ in range(300)}
    assert {r for r, _ in seen} <= {4, 5, 6} and all(b for _, b in seen)
    always_score = outcomes(a=(0, -50.0, 50.0))
    runs = [always_score.runs(X, Y, rng) for _ in range(300)]
    assert all(r >= 1 and not b for r, b in runs) and {r for r, _ in runs} <= {1, 2, 3}
    never = outcomes(a=(0, -50.0, -50.0))
    assert all(never.runs(X, Y, rng) == (0, False) for _ in range(50))


def test_new_players_and_games_are_drawn_from_the_fitted_spread() -> None:
    rule = outcomes()
    rng = np.random.default_rng(1)
    for i in range(400):
        rule.register(f"n{i}", rng)
    boundary_effects = np.array([rule.bat[f"n{i}"]["b"] for i in range(400)])
    assert abs(boundary_effects.std() - 0.8) < 0.1          # the batter spread
    rule.begin_game(rng)
    assert rule.game != {"b": 0.0, "s": 0.0}
    with pytest.raises(KeyError):
        rule.dismissal_probability(SimpleNamespace(name="unknown"), Y)


@pytest.fixture(scope="module")
def idata():
    """A small real fit through the JAX sampler (needs numpyro)."""
    pytest.importorskip("numpyro")
    rng = np.random.default_rng(4)
    n = 400
    data = BallData(
        players=[f"P{i}" for i in range(6)], games=["g1", "g2", "g3"],
        bat=rng.integers(0, 6, n), bowl=rng.integers(0, 6, n), game=rng.integers(0, 3, n),
        runs=rng.integers(0, 5, n), dismissed=(rng.random(n) < 0.06).astype(int),
        boundary=(rng.random(n) < 0.1).astype(int),
    )
    return fit_ball_model(data, draws=60, tune=100, chains=2, seed=5), data


def test_fit_yields_outcomes_and_effect_tables(idata) -> None:
    fitted, data = idata
    rule = JointOutcomes.from_posterior(fitted)
    assert set(rule.bat) == set(data.players)
    p = rule.dismissal_probability(SimpleNamespace(name="P0"), SimpleNamespace(name="P1"))
    assert 0 < p < 1
    table = effect_table(fitted)
    assert len(table) == 2 * 3 * 6                      # roles x components x players
    assert set(table["role"]) == {"bat", "bowl"}
    assert (table["lo"] <= table["effect"]).all() and (table["effect"] <= table["hi"]).all()


def test_expected_run_constants_match_the_measured_shares() -> None:
    from junior_cricket.ball_model import _expected_boundary_runs, _expected_scoring_runs
    assert _expected_boundary_runs() == pytest.approx(4.215)
    assert 1.10 < _expected_scoring_runs() < 1.16


def test_player_profiles_give_ordered_intervals_and_sane_figures(idata) -> None:
    from junior_cricket.ball_model import player_profiles
    fitted, data = idata
    table = player_profiles(fitted, ["P0", "P3"])
    assert list(table["player"]) == ["P0", "P3"]
    assert (table["net_lo"] <= table["net"]).all() and (table["net"] <= table["net_hi"]).all()
    assert (table["wickets_lo"] <= table["wickets"]).all() and (table["wickets"] <= table["wickets_hi"]).all()
    assert ((table["boundaries"] >= 0) & (table["boundaries"] < 13)).all()
    assert ((table["dismissals"] >= 0) & (table["dismissals"] < 13)).all()
    assert (table["runs_conceded"] > 0).all()


def test_held_out_lpd_is_finite_for_every_variant(idata) -> None:
    fitted, data = idata
    for component in ("d", "b", "s"):
        for variant in ("null", "batter", "both"):
            value = held_out_lpd(fitted, data.take(np.arange(len(data.bat)) < 100),
                                 component, variant)
            assert np.isfinite(value) and value < 0
