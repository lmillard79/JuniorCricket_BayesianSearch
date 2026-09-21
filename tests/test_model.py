"""Tests for the adapted MARCEL PyMC model."""

import numpy as np
import pymc as pm
import pytest

from junior_cricket.model import (
    ModelData,
    build_marcel_model,
    build_marcel_model_v2,
)


def make_model_data(n_players: int = 3) -> ModelData:
    """Build a tiny synthetic ModelData for smoke testing."""
    n_periods = 3
    rng = np.random.default_rng(0)
    balls = rng.integers(40, 120, size=(n_players, n_periods))
    return ModelData(
        player_names=[f"Player{i}" for i in range(n_players)],
        period_labels=["p0", "p1", "p2"],
        balls_faced=balls.astype(float),
        outs=(rng.random((n_players, n_periods)) * balls * 0.06)
        .astype(int)
        .astype(float),
        boundaries=(rng.random((n_players, n_periods)) * balls * 0.04)
        .astype(int)
        .astype(float),
        non_boundary_balls=(balls * 0.9).astype(float),
        non_boundary_runs=(balls * 0.9 * 0.3).astype(int).astype(float),
        deliveries=(balls * 1.1).astype(float),
        fair_balls=balls.astype(float),
        wickets=(rng.random((n_players, n_periods)) * balls * 0.05)
        .astype(int)
        .astype(float),
        runs_off_bat=(balls * 0.55).astype(float),
        extras=(balls * 0.12).astype(float),
    )


def test_model_builds() -> None:
    """The model graph builds without sampling errors."""
    data = make_model_data()
    model = build_marcel_model(data)
    assert model is not None
    # Projected deterministic variables exist for the simulator.
    for var in (
        "p_out_final",
        "p_bound_final",
        "p_wicket_final",
        "srr_proj",
        "econ_proj",
        "p_extra_proj",
        "recency_weights",
    ):
        assert var in [v.name for v in model.free_RVs + model.deterministics]


def test_model_prior_predictive_runs() -> None:
    """A tiny prior predictive check samples without error."""
    data = make_model_data(n_players=2)
    model = build_marcel_model(data)
    with model:
        prior = pm.sample_prior_predictive(draws=20, random_seed=1)
    assert prior is not None
    assert "p_out_final" in prior.prior


OPTIMISER_VARS = (
    "p_out_final", "p_bound_final", "p_wicket_final",
    "srr_proj", "econ_proj", "p_extra_proj",
    "mu_out", "sigma_out", "mu_bound", "sigma_bound", "mu_srr", "sigma_srr",
    "mu_wicket", "sigma_wicket", "mu_econ", "sigma_econ",
    "mu_extra", "sigma_extra",
)


def _names(model: pm.Model) -> list:
    return [v.name for v in model.free_RVs + model.deterministics]


def test_v2_exposes_the_variables_the_optimiser_reads() -> None:
    """The v2 model keeps the optimiser's variable contract."""
    model = build_marcel_model_v2(make_model_data())
    for var in OPTIMISER_VARS:
        assert var in _names(model), var


def test_v2_handles_an_empty_period_and_absent_extras() -> None:
    """An all-zero period and no recorded extras must not break the model."""
    data = make_model_data(n_players=3)
    for field in ("balls_faced", "outs", "boundaries", "non_boundary_runs",
                  "non_boundary_balls", "deliveries", "fair_balls", "wickets",
                  "runs_off_bat", "extras"):
        getattr(data, field)[:, 0] = 0.0          # empty first period
    data.extras[:] = 0.0                          # PlayHQ U10 records none
    model = build_marcel_model_v2(data)
    observed = [v.name for v in model.observed_RVs]
    assert "extra_obs" not in observed            # not fitted, prior used
    for var in OPTIMISER_VARS:
        assert var in _names(model), var
    with model:
        prior = pm.sample_prior_predictive(draws=10, random_seed=3)
    extra = np.asarray(prior.prior["p_extra_proj"])
    assert np.allclose(extra, 0.12)               # the population prior


def test_v2_prior_predictive_shapes() -> None:
    """Final rates are per-player probabilities in (0, 1)."""
    model = build_marcel_model_v2(make_model_data(n_players=2))
    with model:
        prior = pm.sample_prior_predictive(draws=20, random_seed=4)
    p_out = np.asarray(prior.prior["p_out_final"])
    assert p_out.shape[-1] == 2 and ((p_out > 0) & (p_out < 1)).all()


def test_v2_samples_without_a_c_compiler() -> None:
    """The v2 graph converts to JAX and samples (the real failure mode)."""
    pytest.importorskip("numpyro")
    data = make_model_data(n_players=3)
    data.extras[:] = 0.0
    model = build_marcel_model_v2(data)
    with model:
        idata = pm.sample(
            draws=60, tune=100, chains=2, random_seed=5,
            progressbar=False, nuts_sampler="numpyro",
        )
    final = np.asarray(idata.posterior["p_out_final"])
    assert final.shape[-1] == 3 and np.isfinite(final).all()


def test_recency_weights_constrain() -> None:
    """The Dirichlet weights live on the 3-simplex."""
    data = make_model_data()
    model = build_marcel_model(data)
    with model:
        prior = pm.sample_prior_predictive(draws=50, random_seed=2)
    weights = np.asarray(prior.prior["recency_weights"][0])
    assert weights.shape == (50, 3)
    assert np.allclose(weights.sum(axis=1), 1.0)
