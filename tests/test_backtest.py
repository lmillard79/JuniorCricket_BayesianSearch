"""Tests for the out-of-sample backtest scoring and split logic."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from junior_cricket.backtest import (
    FAMILIES,
    FamilySpec,
    baseline_rates,
    binomial_lpd,
    game_splits,
    logmeanexp,
    poisson_lpd,
    run_split,
    score_family,
    summarise,
)

BINOMIAL = FamilySpec("out", "binomial", "logit", "balls_faced", "outs")
POISSON = FamilySpec("srr", "poisson", "log", "non_boundary_balls", "non_boundary_runs")


def test_logmeanexp_matches_the_direct_calculation() -> None:
    values = np.log(np.array([[0.2, 0.5], [0.4, 0.1]]))
    assert np.allclose(np.exp(logmeanexp(values, axis=0)), [0.3, 0.3])


def test_binomial_lpd_plug_in_and_draws() -> None:
    y, n = np.array([2, 5]), np.array([20, 20])
    plug = binomial_lpd(y, n, np.array([0.1, 0.25]))
    assert np.allclose(plug, stats.binom.logpmf(y, n, [0.1, 0.25]))
    draws = np.array([[0.1, 0.2], [0.3, 0.3]])
    expected = np.log(stats.binom.pmf(y, n, draws).mean(axis=0))
    assert np.allclose(binomial_lpd(y, n, draws), expected)


def test_poisson_lpd_uses_rate_times_exposure() -> None:
    y, n = np.array([6]), np.array([10.0])
    assert np.allclose(poisson_lpd(y, n, np.array([0.5])),
                       stats.poisson.logpmf(6, 5.0))


def test_baselines_pool_and_smooth() -> None:
    """Pooled is total/total; raw never predicts an impossible zero."""
    trials = np.array([100.0, 50.0, 0.0])
    counts = np.array([4.0, 0.0, 0.0])
    rates = baseline_rates(BINOMIAL, trials, counts)
    assert np.allclose(rates["pooled"], 4 / 150)
    assert rates["raw"][1] > 0                       # smoothed away from zero
    assert rates["raw"][2] == pytest.approx(4 / 150)  # no exposure -> pooled
    poisson = baseline_rates(POISSON, np.array([10.0, 0.0]), np.array([6.0, 0.0]))
    assert poisson["raw"][0] == pytest.approx(0.6)
    assert poisson["raw"][1] == pytest.approx(0.6)   # falls back to pooled


def test_model_that_knows_the_players_beats_the_pooled_baseline() -> None:
    """When players truly differ, informed draws beat one pooled rate."""
    rng = np.random.default_rng(0)
    true = np.array([0.02, 0.10, 0.05, 0.08])
    n = np.full(4, 400.0)
    y = np.rint(true * n)
    model = np.clip(true + rng.normal(0, 0.003, size=(200, 4)), 0.005, 0.5)
    baselines = baseline_rates(BINOMIAL, n, y)        # pooled = mean of the truth
    rows = score_family(BINOMIAL, model, baselines, n, y, rng)
    by_method = {r["method"]: r for r in rows}
    assert by_method["model"]["lpd"] > by_method["pooled"]["lpd"]
    assert 0.0 <= by_method["model"]["coverage90"] <= 1.0
    assert np.isnan(by_method["pooled"]["coverage90"])
    assert by_method["model"]["players"] == 4


def test_players_without_test_exposure_are_not_scored() -> None:
    rng = np.random.default_rng(1)
    model = np.full((50, 3), 0.05)
    rows = score_family(BINOMIAL, model, baseline_rates(BINOMIAL, np.ones(3), np.ones(3)),
                        np.array([100.0, 0.0, 100.0]), np.array([5.0, 0.0, 4.0]), rng)
    assert {r["players"] for r in rows} == {2}


def test_game_splits_grow_the_training_window() -> None:
    dates = [f"2025-10-{d:02d}" for d in range(1, 16)]      # 15 games
    splits = game_splits(dates, min_train=6, test_games=3)
    assert len(splits) == 3
    assert splits[0] == (("2025-10-01", "2025-10-06"), ("2025-10-07", "2025-10-09"))
    assert splits[-1][1] == ("2025-10-13", "2025-10-15")


def test_summarise_totals_scores_across_splits() -> None:
    rows = []
    for split in range(2):
        for method, lpd in (("model", -10.0), ("pooled", -12.0), ("raw", -15.0)):
            rows.append({"family": "out", "method": method, "lpd": lpd, "rmse": 1.0,
                         "corr": 0.5, "coverage90": 0.9 if method == "model" else np.nan,
                         "players": 5, "split": split})
    summary = summarise(pd.DataFrame(rows)).set_index("method")
    assert summary.loc["model", "lpd"] == -20.0
    assert summary.loc["model", "model_minus_pooled"] == 4.0
    assert summary.loc["model", "model_minus_raw"] == 10.0


def _synthetic_games(n_players: int = 4, n_games: int = 9):
    rng = np.random.default_rng(3)
    dates = pd.date_range("2025-10-11", periods=n_games, freq="7D")
    bat, bowl = [], []
    for date in dates:
        for i in range(n_players):
            balls = int(rng.integers(12, 16))
            bat.append({"date": date, "opponent": "X", "player_name": f"P{i}",
                        "balls_faced": balls, "runs_scored": int(rng.integers(3, 15)),
                        "fours": int(rng.integers(0, 3)), "sixes": 0,
                        "dismissed": int(rng.integers(0, 2)), "retired": 1,
                        "source": "playhq"})
            bowl.append({"date": date, "opponent": "X", "player_name": f"P{i}",
                         "balls_bowled": 12, "runs_conceded": int(rng.integers(4, 14)),
                         "wickets": int(rng.integers(0, 2)), "wides": 0,
                         "no_balls": 0, "source": "playhq"})
    return pd.DataFrame(bat), pd.DataFrame(bowl), [f"P{i}" for i in range(n_players)]


def test_run_split_end_to_end_on_synthetic_games() -> None:
    """Fitting, prediction and scoring run through the real code path."""
    pytest.importorskip("numpyro")
    batting, bowling, players = _synthetic_games()
    scores = run_split(batting, bowling, players,
                       ("2025-10-11", "2025-11-08"), ("2025-11-15", "2025-11-29"),
                       seed=2, draws=60)
    assert set(scores["family"]) == {spec.name for spec in FAMILIES}
    assert set(scores["method"]) == {"model", "pooled", "raw"}
    assert np.isfinite(scores["lpd"]).all()
