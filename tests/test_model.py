"""Tests for the adapted MARCEL PyMC model."""

import numpy as np
import pymc as pm

from junior_cricket.model import ModelData, build_marcel_model


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


def test_recency_weights_constrain() -> None:
    """The Dirichlet weights live on the 3-simplex."""
    data = make_model_data()
    model = build_marcel_model(data)
    with model:
        prior = pm.sample_prior_predictive(draws=50, random_seed=2)
    weights = np.asarray(prior.prior["recency_weights"][0])
    assert weights.shape == (50, 3)
    assert np.allclose(weights.sum(axis=1), 1.0)
