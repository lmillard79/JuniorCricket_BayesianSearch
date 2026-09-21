"""Adapted Bayesian MARCEL model for junior cricket skills.

Re-casts the MARCEL baseball projection framework (PyMC Labs,
"Bayesian MARCEL") for BNJCA junior cricket:

* Hierarchical Beta-Binomial rates for dismissal hazard, boundary
  rate, bowler wicket rate and bowler extras rate - partial
  pooling gives sample-size-aware shrinkage toward the grade
  population, which also serves as the prior for unknown
  opposition players.
* Gamma-Poisson rates for non-boundary scoring rate and bowler
  economy.
* Dirichlet-learned recency weights over three history periods
  (replacing MARCEL's hard-coded 5/4/3 season weighting).
* Optional monotone relative-age effect on the logit scale for
  the binary rates (replacing MARCEL's peak-age curve).

All rates are per ball faced (batting) or per delivery (bowling).

Two model builders live here. ``build_marcel_model`` is the original
specification. Fitted to real U10 data it did not sample (every draw
diverged), and its recency weights only ever appear in deterministic
projections, so no likelihood ever informs them. ``build_marcel_model_v2``
replaces it: a non-centred local-level model in which each player's
rate on the logit (or log) scale carries over from period to period
with a learned drift, so recency weighting emerges from the data
instead of being fixed. It exposes the same variable names the
optimiser reads.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt

# Weakly informative population priors based on typical BNJCA
# junior cricket rates per ball. The data dominate these quickly.
PRIOR_MU_OUT = 0.05
PRIOR_MU_BOUND = 0.04
PRIOR_MU_SRR = 0.30
PRIOR_MU_WICKET = 0.05
PRIOR_MU_ECON = 0.55
PRIOR_MU_EXTRA = 0.12

# Dirichlet concentration over the three history periods,
# oldest -> newest (mirrors the Bayesian MARCEL blog's scheme of
# letting the data shift the weights toward recency).
DIRICHLET_CONCENTRATION = np.array([3.0, 4.0, 5.0])


@dataclass
class ModelData:
    """Observed arrays shaped (player, period) for the model.

    Attributes:
        player_names: Player labels, row order of all arrays.
        period_labels: Period labels oldest-first, column order.
        balls_faced: Legal balls faced while batting.
        outs: Dismissals while batting.
        boundaries: Fours plus sixes while batting.
        non_boundary_runs: Runs scored off non-boundary balls.
        non_boundary_balls: Balls faced that were not boundaries.
        deliveries: Total deliveries bowled including extras.
        fair_balls: Legal deliveries bowled.
        wickets: Wickets taken.
        runs_off_bat: Runs conceded off the bat on fair balls.
        extras: Wides plus no balls bowled.
        ages_months: Age in months at projection date, or None.
    """

    player_names: List[str]
    period_labels: List[str]
    balls_faced: np.ndarray
    outs: np.ndarray
    boundaries: np.ndarray
    non_boundary_runs: np.ndarray
    non_boundary_balls: np.ndarray
    deliveries: np.ndarray
    fair_balls: np.ndarray
    wickets: np.ndarray
    runs_off_bat: np.ndarray
    extras: np.ndarray
    ages_months: Optional[np.ndarray] = None


@dataclass
class PopulationPriors:
    """Grade-level population parameter point estimates.

    Used by the simulator and optimizer to draw synthetic
    opposition players when no opposition roster is available.

    Attributes:
        mu_out: Mean dismissal hazard per ball.
        sigma_out: Between-player SD of dismissal hazard.
        mu_bound: Mean boundary rate per ball.
        sigma_bound: Between-player SD of boundary rate.
        mu_srr: Mean non-boundary runs per ball.
        sigma_srr: Between-player SD of scoring rate.
        mu_wicket: Mean wicket rate per fair ball.
        sigma_wicket: Between-player SD of wicket rate.
        mu_econ: Mean economy (runs off bat per fair ball).
        sigma_econ: Between-player SD of economy.
        mu_extra: Mean extras rate per delivery.
        sigma_extra: Between-player SD of extras rate.
    """

    mu_out: float = PRIOR_MU_OUT
    sigma_out: float = 0.03
    mu_bound: float = PRIOR_MU_BOUND
    sigma_bound: float = 0.02
    mu_srr: float = PRIOR_MU_SRR
    sigma_srr: float = 0.15
    mu_wicket: float = PRIOR_MU_WICKET
    sigma_wicket: float = 0.03
    mu_econ: float = PRIOR_MU_ECON
    sigma_econ: float = 0.20
    mu_extra: float = PRIOR_MU_EXTRA
    sigma_extra: float = 0.06


def prepare_model_data(
    batting_agg: pd.DataFrame,
    bowling_agg: pd.DataFrame,
    period_labels: List[str],
    player_order: Optional[List[str]] = None,
    ages_months: Optional[Dict[str, float]] = None,
) -> ModelData:
    """Pivot aggregates into dense (player, period) model arrays.

    Args:
        batting_agg: Output of ``data_loader.aggregate_batting``.
        bowling_agg: Output of ``data_loader.aggregate_bowling``.
        period_labels: Period labels oldest-first; must match the
            ``period_idx`` values used during aggregation.
        player_order: Explicit player ordering; union of batting
            and bowling players when omitted.
        ages_months: Optional player -> age in months mapping for
            the relative-age effect.

    Returns:
        Populated ``ModelData`` with zeros where a player has no
        data in a period (zero-trial likelihoods contribute
        nothing, so missing periods are handled automatically).
    """
    if player_order is None:
        batters = set(batting_agg["player_name"])
        bowlers = set(bowling_agg["player_name"])
        player_order = sorted(batters | bowlers)

    n_players = len(player_order)
    n_periods = len(period_labels)
    player_index = {name: i for i, name in enumerate(player_order)}

    def _dense(
        frame: pd.DataFrame, value: str, dtype: type = float
    ) -> np.ndarray:
        array = np.zeros((n_players, n_periods), dtype=dtype)
        if frame.empty:
            return array
        for _, row in frame.iterrows():
            i = player_index[row["player_name"]]
            t = int(row["period_idx"])
            if 0 <= t < n_periods:
                array[i, t] += row[value]
        return array

    data = ModelData(
        player_names=player_order,
        period_labels=period_labels,
        balls_faced=_dense(batting_agg, "balls_faced"),
        outs=_dense(batting_agg, "outs"),
        boundaries=_dense(batting_agg, "boundaries"),
        non_boundary_runs=_dense(batting_agg, "non_boundary_runs"),
        non_boundary_balls=_dense(batting_agg, "non_boundary_balls"),
        deliveries=_dense(bowling_agg, "deliveries"),
        fair_balls=_dense(bowling_agg, "fair_balls"),
        wickets=_dense(bowling_agg, "wickets"),
        runs_off_bat=_dense(bowling_agg, "runs_off_bat"),
        extras=_dense(bowling_agg, "extras"),
    )

    if ages_months is not None:
        data.ages_months = np.array(
            [
                ages_months.get(name, np.nan) for name in player_order
            ],
            dtype=float,
        )
        if np.isnan(data.ages_months).any():
            # Missing ages fall back to the cohort mean (zero
            # relative-age effect) rather than dropping players.
            mean_age = np.nanmean(data.ages_months)
            data.ages_months = np.where(
                np.isnan(data.ages_months), mean_age, data.ages_months
            )

    return data


def build_marcel_model(data: ModelData) -> pm.Model:
    """Build the adapted MARCEL PyMC model.

    Args:
        data: Dense observed arrays from ``prepare_model_data``.

    Returns:
        A compiled ``pm.Model`` with deterministic projected
        rates (weighted across periods, age-adjusted when ages
        are supplied) ready for NUTS sampling.
    """
    coords = {
        "player": data.player_names,
        "period": data.period_labels,
    }

    with pm.Model(coords=coords) as model:
        # --- Batting: dismissal hazard (Beta-Binomial) ---
        mu_out = pm.Beta("mu_out", mu=PRIOR_MU_OUT, sigma=0.03)
        sigma_out = pm.HalfNormal("sigma_out", sigma=0.03)
        p_out = pm.Beta(
            "p_out", mu=mu_out, sigma=sigma_out, dims=("player", "period")
        )
        pm.Binomial(
            "outs_obs",
            n=data.balls_faced,
            p=p_out,
            observed=data.outs,
            dims=("player", "period"),
        )

        # --- Batting: boundary rate (Beta-Binomial) ---
        mu_bound = pm.Beta("mu_bound", mu=PRIOR_MU_BOUND, sigma=0.02)
        sigma_bound = pm.HalfNormal("sigma_bound", sigma=0.02)
        p_bound = pm.Beta(
            "p_bound",
            mu=mu_bound,
            sigma=sigma_bound,
            dims=("player", "period"),
        )
        pm.Binomial(
            "boundaries_obs",
            n=data.balls_faced,
            p=p_bound,
            observed=data.boundaries,
            dims=("player", "period"),
        )

        # --- Batting: non-boundary scoring rate (Gamma-Poisson) ---
        mu_srr = pm.Gamma("mu_srr", mu=PRIOR_MU_SRR, sigma=0.15)
        sigma_srr = pm.HalfNormal("sigma_srr", sigma=0.15)
        srr = pm.Gamma(
            "srr", mu=mu_srr, sigma=sigma_srr, dims=("player", "period")
        )
        pm.Poisson(
            "srr_obs",
            mu=srr * data.non_boundary_balls,
            observed=data.non_boundary_runs,
            dims=("player", "period"),
        )

        # --- Bowling: wicket rate (Beta-Binomial) ---
        mu_wicket = pm.Beta("mu_wicket", mu=PRIOR_MU_WICKET, sigma=0.03)
        sigma_wicket = pm.HalfNormal("sigma_wicket", sigma=0.03)
        p_wicket = pm.Beta(
            "p_wicket",
            mu=mu_wicket,
            sigma=sigma_wicket,
            dims=("player", "period"),
        )
        pm.Binomial(
            "wickets_obs",
            n=data.fair_balls,
            p=p_wicket,
            observed=data.wickets,
            dims=("player", "period"),
        )

        # --- Bowling: economy (Gamma-Poisson) ---
        mu_econ = pm.Gamma("mu_econ", mu=PRIOR_MU_ECON, sigma=0.20)
        sigma_econ = pm.HalfNormal("sigma_econ", sigma=0.20)
        econ = pm.Gamma(
            "econ", mu=mu_econ, sigma=sigma_econ, dims=("player", "period")
        )
        pm.Poisson(
            "econ_obs",
            mu=econ * data.fair_balls,
            observed=data.runs_off_bat,
            dims=("player", "period"),
        )

        # --- Bowling: extras rate (Beta-Binomial) ---
        mu_extra = pm.Beta("mu_extra", mu=PRIOR_MU_EXTRA, sigma=0.06)
        sigma_extra = pm.HalfNormal("sigma_extra", sigma=0.06)
        p_extra = pm.Beta(
            "p_extra",
            mu=mu_extra,
            sigma=sigma_extra,
            dims=("player", "period"),
        )
        pm.Binomial(
            "extras_obs",
            n=data.deliveries,
            p=p_extra,
            observed=data.extras,
            dims=("player", "period"),
        )

        # --- MARCEL recency weights over periods ---
        weights = pm.Dirichlet(
            "recency_weights", a=DIRICHLET_CONCENTRATION, dims="period"
        )

        # --- Weighted projections (player-level) ---
        p_out_proj = pm.Deterministic(
            "p_out_proj", pm.math.dot(p_out, weights), dims="player"
        )
        p_bound_proj = pm.Deterministic(
            "p_bound_proj", pm.math.dot(p_bound, weights), dims="player"
        )
        srr_proj = pm.Deterministic(
            "srr_proj", pm.math.dot(srr, weights), dims="player"
        )
        p_wicket_proj = pm.Deterministic(
            "p_wicket_proj", pm.math.dot(p_wicket, weights), dims="player"
        )
        econ_proj = pm.Deterministic(
            "econ_proj", pm.math.dot(econ, weights), dims="player"
        )
        p_extra_proj = pm.Deterministic(
            "p_extra_proj", pm.math.dot(p_extra, weights), dims="player"
        )

        # --- Optional relative-age effect on binary rates ---
        if data.ages_months is not None:
            age_std = pm.Data(
                "age_std", (data.ages_months - np.mean(data.ages_months))
                / 12.0,
                dims="player",
            )
            beta_age_out = pm.Normal("beta_age_out", 0.0, 0.5)
            beta_age_bound = pm.Normal("beta_age_bound", 0.0, 0.5)
            beta_age_wicket = pm.Normal("beta_age_wicket", 0.0, 0.5)

            pm.Deterministic(
                "p_out_final",
                pm.math.invlogit(
                    pm.math.logit(p_out_proj) + beta_age_out * age_std
                ),
                dims="player",
            )
            pm.Deterministic(
                "p_bound_final",
                pm.math.invlogit(
                    pm.math.logit(p_bound_proj) + beta_age_bound * age_std
                ),
                dims="player",
            )
            pm.Deterministic(
                "p_wicket_final",
                pm.math.invlogit(
                    pm.math.logit(p_wicket_proj)
                    + beta_age_wicket * age_std
                ),
                dims="player",
            )
        else:
            pm.Deterministic(
                "p_out_final", p_out_proj, dims="player"
            )
            pm.Deterministic(
                "p_bound_final", p_bound_proj, dims="player"
            )
            pm.Deterministic(
                "p_wicket_final", p_wicket_proj, dims="player"
            )

    return model


# Working-scale priors for the v2 hierarchy: (mean, between-player SD of
# skill, period-to-period drift SD). Binary rates live on the logit
# scale, Poisson rates on the log scale.
_BINARY_PRIORS = (0.7, 0.5, 0.25)
_RATE_PRIORS = (0.5, 0.4, 0.2)


def _logit(p: float) -> float:
    """Logit of a probability."""
    return float(np.log(p / (1.0 - p)))


def _observed_cells(
    trials: np.ndarray, counts: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pick the (player, period) cells that have at least one trial.

    Args:
        trials: Exposure array shaped (player, period).
        counts: Observed counts of the same shape.

    Returns:
        (player index, period index, trials, counts) for cells with
        exposure. Empty cells carry no information and are left out
        of the likelihood, so an empty period cannot break sampling.
    """
    players, periods = np.nonzero(trials > 0)
    return (
        players,
        periods,
        trials[players, periods],
        np.rint(counts[players, periods]).astype(int),
    )


def _local_level(
    name: str,
    centre: float,
    n_periods: int,
    priors: Tuple[float, float, float],
):
    """Non-centred local-level state for one rate family.

    ``theta[i, 0] = m + s * z[i]`` and ``theta[i, t] = theta[i, t-1] +
    tau * eps[i, t]``: a persistent player skill around a population
    mean, drifting a little each period. Writing the offsets as
    standard normals removes the funnel the original Beta
    parameterisation had when the between-player spread is small.

    Args:
        name: Family name used to label the variables.
        centre: Prior centre of the population mean (working scale).
        n_periods: Number of periods.
        priors: (SD of the mean, SD of skill spread, SD of drift).

    Returns:
        (theta shaped (player, period), m, s, tau, total SD of the
        state at the last period).
    """
    sd_mean, sd_skill, sd_drift = priors
    m = pm.Normal(f"m_{name}", centre, sd_mean)
    s = pm.HalfNormal(f"s_{name}", sd_skill)
    tau = pm.HalfNormal(f"tau_{name}", sd_drift)
    z = pm.Normal(f"z_{name}", 0.0, 1.0, dims="player")
    start = (m + s * z)[:, None]
    if n_periods > 1:
        eps = pm.Normal(f"eps_{name}", 0.0, 1.0, dims=("player", "step"))
        state = pt.concatenate(
            [start, start + tau * pt.cumsum(eps, axis=1)], axis=1
        )
    else:
        state = start
    theta = pm.Deterministic(
        f"theta_{name}", state, dims=("player", "period")
    )
    total_sd = pt.sqrt(s**2 + (n_periods - 1) * tau**2)
    return theta, m, s, tau, total_sd


def build_marcel_model_v2(data: ModelData) -> pm.Model:
    """Build the non-centred local-level model.

    Each family (dismissal hazard, boundary rate, wicket rate, scoring
    rate, economy) gets a population mean, a between-player spread of
    persistent skill and a per-period drift; latent rates are then the
    inverse logit (or exponential) of the resulting state. The
    projected rate for the next period is the last period's state.
    Extras are modelled only when the data record any: PlayHQ U10
    scoring records none, and forcing a rate of zero would wrongly
    switch wides off in the U11 simulation, so the population prior
    is used instead.

    Args:
        data: Dense observed arrays from ``prepare_model_data``.

    Returns:
        A ``pm.Model`` exposing ``p_out_final``, ``p_bound_final``,
        ``p_wicket_final``, ``srr_proj``, ``econ_proj``, ``p_extra_proj``
        (per player) and natural-scale ``mu_*`` / ``sigma_*`` population
        summaries, the names the optimiser reads.
    """
    n_players = len(data.player_names)
    n_periods = len(data.period_labels)
    coords = {
        "player": data.player_names,
        "period": data.period_labels,
        "step": data.period_labels[1:],
    }

    with pm.Model(coords=coords) as model:
        # --- Binary per-ball rates ---
        binary = {}
        for name, prior_mu, trials, counts in (
            ("out", PRIOR_MU_OUT, data.balls_faced, data.outs),
            ("bound", PRIOR_MU_BOUND, data.balls_faced, data.boundaries),
            ("wicket", PRIOR_MU_WICKET, data.fair_balls, data.wickets),
        ):
            theta, m, s, tau, total_sd = _local_level(
                name, _logit(prior_mu), n_periods, _BINARY_PRIORS
            )
            rate = pm.Deterministic(
                f"p_{name}", pm.math.invlogit(theta), dims=("player", "period")
            )
            i, t, n, y = _observed_cells(trials, counts)
            pm.Binomial(f"{name}_obs", n=n.astype(int), p=rate[i, t], observed=y)
            mu = pm.math.invlogit(m)
            pm.Deterministic(f"mu_{name}", mu)
            pm.Deterministic(f"sigma_{name}", mu * (1.0 - mu) * total_sd)
            binary[name] = theta

        # --- Poisson rates: non-boundary scoring rate and economy ---
        rates = {}
        for name, prior_mu, trials, counts in (
            ("srr", PRIOR_MU_SRR, data.non_boundary_balls,
             data.non_boundary_runs),
            ("econ", PRIOR_MU_ECON, data.fair_balls, data.runs_off_bat),
        ):
            theta, m, s, tau, total_sd = _local_level(
                name, float(np.log(prior_mu)), n_periods, _RATE_PRIORS
            )
            rate = pm.Deterministic(
                name, pt.exp(theta), dims=("player", "period")
            )
            i, t, n, y = _observed_cells(trials, counts)
            pm.Poisson(f"{name}_obs", mu=rate[i, t] * n, observed=y)
            mean = pt.exp(m + 0.5 * total_sd**2)
            pm.Deterministic(f"mu_{name}", mean)
            pm.Deterministic(
                f"sigma_{name}", mean * pt.sqrt(pt.exp(total_sd**2) - 1.0)
            )
            rates[name] = theta

        # --- Extras: modelled only if the data record any ---
        if data.extras.sum() > 0:
            theta, m, s, tau, total_sd = _local_level(
                "extra", _logit(PRIOR_MU_EXTRA), n_periods, _BINARY_PRIORS
            )
            rate = pm.Deterministic(
                "p_extra", pm.math.invlogit(theta), dims=("player", "period")
            )
            i, t, n, y = _observed_cells(data.deliveries, data.extras)
            pm.Binomial("extra_obs", n=n.astype(int), p=rate[i, t], observed=y)
            mu = pm.math.invlogit(m)
            pm.Deterministic("mu_extra", mu)
            pm.Deterministic("sigma_extra", mu * (1.0 - mu) * total_sd)
            pm.Deterministic(
                "p_extra_proj", rate[:, -1], dims="player"
            )
        else:
            pm.Deterministic("mu_extra", pt.as_tensor_variable(PRIOR_MU_EXTRA))
            pm.Deterministic("sigma_extra", pt.as_tensor_variable(0.06))
            pm.Deterministic(
                "p_extra_proj",
                pt.ones(n_players) * PRIOR_MU_EXTRA,
                dims="player",
            )

        # --- Projections for the next period (last state) ---
        pm.Deterministic(
            "srr_proj", pt.exp(rates["srr"][:, -1]), dims="player"
        )
        pm.Deterministic(
            "econ_proj", pt.exp(rates["econ"][:, -1]), dims="player"
        )

        # --- Optional relative-age effect on the binary rates ---
        age_std = None
        if data.ages_months is not None:
            age_std = pm.Data(
                "age_std",
                (data.ages_months - np.mean(data.ages_months)) / 12.0,
                dims="player",
            )
        for name in ("out", "bound", "wicket"):
            last = binary[name][:, -1]
            if age_std is not None:
                beta = pm.Normal(f"beta_age_{name}", 0.0, 0.5)
                last = last + beta * age_std
            pm.Deterministic(
                f"p_{name}_final", pm.math.invlogit(last), dims="player"
            )

    return model
