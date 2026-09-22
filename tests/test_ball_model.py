"""Tests for the joint batter-by-bowler ball model."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from junior_cricket.ball_model import (
    BallData,
    JointOutcomes,
    build_ball_model,
    effect_table,
    fielding_credit_table,
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
        ours_fielding=np.array([1, 1, 0, 0, 1, 1]),
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


def test_fielding_edge_only_applies_when_toggled_on() -> None:
    """set_fielding_ours gates the dismissal-only fielding term, our side only."""
    base = np.log(0.1 / 0.9)
    rule = JointOutcomes(
        intercepts=dict(zip("dbs", (base, 0.0, 0.0))),
        bat={"x": {"d": 0.0, "b": 0.0, "s": 0.0}}, bowl={"y": {"d": 0.0, "b": 0.0, "s": 0.0}},
        bat_sd={"d": 0.3, "b": 0.8, "s": 0.4}, bowl_sd={"d": 0.3, "b": 0.2, "s": 0.4},
        game_sd={"b": 0.25, "s": 0.3}, fielding_mu=1.5, fielding_sd=0.0,
    )
    rule.begin_game(np.random.default_rng(0))
    assert rule.dismissal_probability(X, Y) == pytest.approx(0.1)           # off by default
    rule.set_fielding_ours(True)
    assert rule.dismissal_probability(X, Y) == pytest.approx(
        1 / (1 + np.exp(-(base + 1.5))))
    rule.set_fielding_ours(False)
    assert rule.dismissal_probability(X, Y) == pytest.approx(0.1)           # toggled back off
    rule.set_fielding_ours(True)
    unaffected = outcomes(a=(base, 0, 0))
    rng_a, rng_b = np.random.default_rng(2), np.random.default_rng(2)
    assert rule.runs(X, Y, rng_a) == unaffected.runs(X, Y, rng_b)           # never touches b/s


def test_fielding_edge_defaults_to_zero_for_an_older_posterior() -> None:
    """A JointOutcomes built without fielding_mu/fielding_sd behaves exactly as before."""
    rule = outcomes()
    before = rule.dismissal_probability(X, Y)
    rule.begin_game(np.random.default_rng(0))
    assert rule.field_today == 0.0
    rule.set_fielding_ours(True)
    assert rule.dismissal_probability(X, Y) == pytest.approx(before)


def test_fielding_effects_flag_adds_or_drops_the_model_terms() -> None:
    data = tiny_data()
    with_field = build_ball_model(data, fielding_effects=True)
    without = build_ball_model(data, fielding_effects=False)
    assert "mu_f_d" in {v.name for v in with_field.free_RVs}
    assert "mu_f_d" not in {v.name for v in without.free_RVs}


def test_fielding_credit_table_counts_catches_and_run_outs() -> None:
    frame = pd.DataFrame({
        "fielder": ["P02", "", "P02", "P05", ""],
        "run_out": [0, 0, 0, 1, 0],
    })
    table = fielding_credit_table(frame).set_index("player")
    assert table.loc["P02", ["catches", "run_outs", "total"]].tolist() == [2, 0, 2]
    assert table.loc["P05", ["catches", "run_outs", "total"]].tolist() == [0, 1, 1]
    assert list(table.index) == ["P02", "P05"]                              # most credited first


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
        ours_fielding=rng.integers(0, 2, n),
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


def test_fit_learns_a_fielding_edge_and_from_posterior_reads_it(idata) -> None:
    fitted, data = idata
    assert {"mu_f_d", "sf_d", "field_d"} <= set(fitted.posterior.data_vars)
    rule = JointOutcomes.from_posterior(fitted)
    assert set(rule.days) == set(data.games)
    assert "field_d" in rule.days["g1"]
    assert isinstance(rule.fielding_mu, float) and isinstance(rule.fielding_sd, float)


def test_a_posterior_draw_gives_one_plausible_set_of_effects(idata) -> None:
    fitted, _ = idata
    mean_rule = JointOutcomes.from_posterior(fitted)
    one = JointOutcomes.from_posterior(fitted, draw=3)
    other = JointOutcomes.from_posterior(fitted, draw=7)
    assert mean_rule.draw is None and one.draw == 3
    assert set(one.bat) == set(mean_rule.bat)
    assert one.bat["P0"]["b"] != other.bat["P0"]["b"]
    every = JointOutcomes.pool(fitted, 10_000, np.random.default_rng(0))   # 2 chains x 60 draws
    assert len(every) == 120
    assert np.mean([r.bat["P0"]["d"] for r in every]) == pytest.approx(mean_rule.bat["P0"]["d"])


def test_a_pool_is_made_of_distinct_draws(idata) -> None:
    fitted, _ = idata
    pool = JointOutcomes.pool(fitted, 10, np.random.default_rng(1))
    assert len({r.draw for r in pool}) == 10


def test_use_day_fixes_the_games_fitted_conditions(idata) -> None:
    fitted, data = idata
    rule = JointOutcomes.from_posterior(fitted)
    assert set(rule.days) == set(data.games)
    rule.use_day("g2")
    assert rule.game == {"b": rule.days["g2"]["b"], "s": rule.days["g2"]["s"]}
    assert rule.field_today == rule.days["g2"]["field_d"]
    rule.begin_game(np.random.default_rng(0))            # a typical day replaces it
    assert rule.game != {"b": rule.days["g2"]["b"], "s": rule.days["g2"]["s"]}


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
