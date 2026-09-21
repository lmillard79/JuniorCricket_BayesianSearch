"""Out-of-sample backtest of the skill model against simple baselines.

The model is fitted on earlier games and asked to predict the counts a
later block of games actually produced, for five rate families
(dismissal hazard, boundary rate, bowler wicket rate, non-boundary
scoring rate, bowler economy). Each family is scored by the log
predictive density of the observed test counts, and compared with two
plug-in baselines that use only the training games:

* ``pooled``: every player at the pooled training rate (no player
  differences at all);
* ``raw``: each player's own training rate (no pooling).

If partial pooling is doing useful work the model should beat both.
The scoring functions are pure numpy/scipy so they can be tested
without sampling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pymc as pm
from scipy import stats
from scipy.special import logsumexp

from junior_cricket.data_loader import (
    aggregate_batting,
    aggregate_bowling,
    define_periods,
)
from junior_cricket.model import ModelData, build_marcel_model_v2, prepare_model_data

METHODS = ("model", "pooled", "raw")


@dataclass(frozen=True)
class FamilySpec:
    """One rate family in the backtest.

    Attributes:
        name: Family name as used in the model variables.
        kind: ``binomial`` or ``poisson`` likelihood.
        working: Scale of the latent state, ``logit`` or ``log``.
        trials: ``ModelData`` field holding the exposure.
        counts: ``ModelData`` field holding the observed counts.
    """

    name: str
    kind: str
    working: str
    trials: str
    counts: str


FAMILIES: Tuple[FamilySpec, ...] = (
    FamilySpec("out", "binomial", "logit", "balls_faced", "outs"),
    FamilySpec("bound", "binomial", "logit", "balls_faced", "boundaries"),
    FamilySpec("wicket", "binomial", "logit", "fair_balls", "wickets"),
    FamilySpec("srr", "poisson", "log", "non_boundary_balls", "non_boundary_runs"),
    FamilySpec("econ", "poisson", "log", "fair_balls", "runs_off_bat"),
)


def logmeanexp(values: np.ndarray, axis: int = 0) -> np.ndarray:
    """Numerically stable log of the mean of exponentials."""
    return logsumexp(values, axis=axis) - np.log(values.shape[axis])


def binomial_lpd(y: np.ndarray, n: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Per-player log predictive density under Binomial(n, p).

    Args:
        y: Observed counts, shape (players,).
        n: Trials, shape (players,).
        p: Probability draws, shape (draws, players), or (players,) for
            a plug-in point estimate.

    Returns:
        Log predictive density per player, averaged over draws.
    """
    p = np.clip(np.atleast_2d(p), 1e-9, 1 - 1e-9)
    return logmeanexp(stats.binom.logpmf(y, n.astype(int), p), axis=0)


def poisson_lpd(y: np.ndarray, n: np.ndarray, rate: np.ndarray) -> np.ndarray:
    """Per-player log predictive density under Poisson(rate * n).

    Args:
        y: Observed counts, shape (players,).
        n: Exposure, shape (players,).
        rate: Rate draws, shape (draws, players), or (players,).

    Returns:
        Log predictive density per player, averaged over draws.
    """
    rate = np.clip(np.atleast_2d(rate), 1e-9, None)
    return logmeanexp(stats.poisson.logpmf(y, rate * n), axis=0)


def baseline_rates(
    spec: FamilySpec, trials: np.ndarray, counts: np.ndarray
) -> Dict[str, np.ndarray]:
    """Plug-in rates for the ``pooled`` and ``raw`` baselines.

    Args:
        spec: Family being predicted.
        trials: Training exposure per player.
        counts: Training counts per player.

    Returns:
        Rates per player for each baseline. ``raw`` uses add-half
        smoothing for binomial families so a zero-count history does
        not predict an impossible event; players with no training
        exposure fall back to the pooled rate.
    """
    total = trials.sum()
    pooled = float(counts.sum() / total) if total > 0 else 0.05
    if spec.kind == "binomial":
        raw = (counts + 0.5) / (trials + 1.0)
    else:
        raw = np.divide(counts, trials, out=np.full_like(trials, pooled, dtype=float),
                        where=trials > 0)
    raw = np.where(trials > 0, raw, pooled)
    return {"pooled": np.full_like(trials, pooled, dtype=float), "raw": raw}


def predictive_rates(
    idata, spec: FamilySpec, rng: np.random.Generator
) -> np.ndarray:
    """Draw next-period rates from a fitted v2 posterior.

    The next period's state is the last period's state plus one more
    drift step, so both parameter uncertainty and period-to-period
    change are carried into the prediction.

    Args:
        idata: Posterior from ``build_marcel_model_v2``.
        spec: Family to predict.
        rng: Random generator for the drift step.

    Returns:
        Array shaped (draws, players) of natural-scale rates.
    """
    posterior = idata.posterior
    theta = np.asarray(posterior[f"theta_{spec.name}"].isel(period=-1))
    tau = np.asarray(posterior[f"tau_{spec.name}"])[..., None]
    state = theta + tau * rng.standard_normal(theta.shape)
    rate = 1.0 / (1.0 + np.exp(-state)) if spec.working == "logit" else np.exp(state)
    return rate.reshape(-1, rate.shape[-1])


def score_family(
    spec: FamilySpec,
    model_rates: np.ndarray,
    baselines: Dict[str, np.ndarray],
    test_trials: np.ndarray,
    test_counts: np.ndarray,
    rng: np.random.Generator,
) -> List[Dict[str, float]]:
    """Score every method on one family's test counts.

    Args:
        spec: Family being scored.
        model_rates: Model rate draws, shape (draws, players).
        baselines: Plug-in rates from ``baseline_rates``.
        test_trials: Test exposure per player.
        test_counts: Observed test counts per player.
        rng: Generator for the predictive-interval simulation.

    Returns:
        One row per method with the summed log predictive density,
        the RMSE of expected counts, the correlation of predicted and
        observed rates across players, and (model only) the share of
        players whose observed count fell inside the 90 percent
        predictive interval. Only players with test exposure count.
    """
    keep = test_trials > 0
    y, n = test_counts[keep], test_trials[keep]
    lpd = binomial_lpd if spec.kind == "binomial" else poisson_lpd
    draws = {"model": model_rates[:, keep]}
    for name, rate in baselines.items():
        draws[name] = rate[keep]
    rows = []
    for method in METHODS:
        rate = draws[method]
        mean_rate = rate.mean(axis=0) if rate.ndim == 2 else rate
        expected = mean_rate * n
        observed_rate = y / n
        corr = (
            float(np.corrcoef(mean_rate, observed_rate)[0, 1])
            if len(y) > 2 and np.ptp(mean_rate) > 1e-12 and np.ptp(observed_rate) > 1e-12
            else float("nan")
        )
        row = {
            "family": spec.name,
            "method": method,
            "lpd": float(lpd(y, n, rate).sum()),
            "rmse": float(np.sqrt(np.mean((expected - y) ** 2))),
            "corr": corr,
            "players": int(keep.sum()),
            "coverage90": float("nan"),
        }
        if method == "model":
            if spec.kind == "binomial":
                sims = rng.binomial(n.astype(int), rate)
            else:
                sims = rng.poisson(rate * n)
            lo, hi = np.percentile(sims, [5, 95], axis=0)
            row["coverage90"] = float(np.mean((y >= lo) & (y <= hi)))
        rows.append(row)
    return rows


def window_data(
    batting: pd.DataFrame,
    bowling: pd.DataFrame,
    start: str,
    end: str,
    players: Sequence[str],
) -> ModelData:
    """Aggregate all games in [start, end] into one period of arrays.

    Args:
        batting: Validated batting records.
        bowling: Validated bowling records.
        start: First date included (ISO).
        end: Last date included (ISO).
        players: Player order shared by every window.

    Returns:
        A one-period ``ModelData`` over the given player order.
    """
    periods = define_periods([("window", start, end)])
    return prepare_model_data(
        aggregate_batting(batting, periods),
        aggregate_bowling(bowling, periods),
        period_labels=["window"],
        player_order=list(players),
    )


def fit_v2(
    data: ModelData, draws: int = 1000, tune: int = 1000, chains: int = 4,
    seed: int = 0,
):
    """Fit the v2 model with the JAX sampler (no C compiler needed)."""
    model = build_marcel_model_v2(data)
    with model:
        return pm.sample(
            draws=draws, tune=tune, chains=chains, target_accept=0.9,
            random_seed=seed, progressbar=False, nuts_sampler="numpyro",
        )


def run_split(
    batting: pd.DataFrame,
    bowling: pd.DataFrame,
    players: Sequence[str],
    train: Tuple[str, str],
    test: Tuple[str, str],
    seed: int = 0,
    draws: int = 1000,
) -> pd.DataFrame:
    """Fit on one date window and score the next.

    Args:
        batting: Validated batting records.
        bowling: Validated bowling records.
        players: Common player order.
        train: (start, end) of the training games.
        test: (start, end) of the held-out games.
        seed: Random seed.
        draws: Posterior draws per chain.

    Returns:
        Scores for every family and method on this split.
    """
    train_data = window_data(batting, bowling, *train, players)
    test_data = window_data(batting, bowling, *test, players)
    idata = fit_v2(train_data, draws=draws, seed=seed)
    rng = np.random.default_rng(seed)
    rows: List[Dict[str, float]] = []
    for spec in FAMILIES:
        trials = getattr(train_data, spec.trials)[:, 0]
        counts = getattr(train_data, spec.counts)[:, 0]
        rows.extend(score_family(
            spec,
            predictive_rates(idata, spec, rng),
            baseline_rates(spec, trials, counts),
            getattr(test_data, spec.trials)[:, 0],
            getattr(test_data, spec.counts)[:, 0],
            rng,
        ))
    frame = pd.DataFrame(rows)
    frame["train"] = f"{train[0]}..{train[1]}"
    frame["test"] = f"{test[0]}..{test[1]}"
    return frame


def summarise(scores: pd.DataFrame) -> pd.DataFrame:
    """Total the scores across splits, per family and method.

    Args:
        scores: Concatenated ``run_split`` output.

    Returns:
        One row per family and method: summed log predictive density,
        mean RMSE, mean correlation, mean coverage, plus the model's
        advantage over each baseline (positive means the model won).
    """
    grouped = scores.groupby(["family", "method"], sort=False).agg(
        lpd=("lpd", "sum"), rmse=("rmse", "mean"), corr=("corr", "mean"),
        coverage90=("coverage90", "mean"),
    ).reset_index()
    lpd = grouped.pivot(index="family", columns="method", values="lpd")
    grouped["model_minus_pooled"] = grouped["family"].map(lpd["model"] - lpd["pooled"])
    grouped["model_minus_raw"] = grouped["family"].map(lpd["model"] - lpd["raw"])
    grouped.loc[grouped["method"] != "model", ["model_minus_pooled", "model_minus_raw"]] = np.nan
    return grouped


def game_splits(
    dates: Sequence[str], min_train: int, test_games: int
) -> List[Tuple[Tuple[str, str], Tuple[str, str]]]:
    """Rolling-origin splits by game date.

    Args:
        dates: Sorted unique game dates (ISO strings).
        min_train: Games in the first training window.
        test_games: Games in each test block; the training window
            then grows to include the previous test block.

    Returns:
        (train window, test window) date pairs.
    """
    splits = []
    k = min_train
    while k + test_games <= len(dates):
        splits.append(
            ((dates[0], dates[k - 1]), (dates[k], dates[k + test_games - 1]))
        )
        k += test_games
    return splits
