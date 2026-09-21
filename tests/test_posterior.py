"""Tests for reading simulator inputs out of a fitted posterior."""

import numpy as np
import pymc as pm
import pytest

from junior_cricket.model import build_marcel_model_v2
from junior_cricket.posterior import (
    PRIOR_VARS,
    priors_from_posterior,
    skills_from_posterior,
)
from test_model import make_model_data


@pytest.fixture(scope="module")
def idata():
    """A tiny real posterior from the v2 model (needs numpyro)."""
    pytest.importorskip("numpyro")
    data = make_model_data(n_players=4)
    data.extras[:] = 0.0
    model = build_marcel_model_v2(data)
    with model:
        return pm.sample(draws=40, tune=60, chains=2, random_seed=6,
                         progressbar=False, nuts_sampler="numpyro")


def test_skills_cover_every_player_with_valid_rates(idata) -> None:
    skills = skills_from_posterior(idata)
    assert list(skills) == ["Player0", "Player1", "Player2", "Player3"]
    for skill in skills.values():
        assert 0 < skill.p_out < 1 and 0 < skill.p_bound < 1
        assert 0 < skill.p_wicket < 1 and skill.srr > 0 and skill.econ > 0
        assert skill.p_extra == pytest.approx(0.12)     # prior, none recorded


def test_squad_and_extra_override(idata) -> None:
    skills = skills_from_posterior(idata, players=["Player2", "Player0"], p_extra=0.0)
    assert list(skills) == ["Player2", "Player0"]       # squad order kept
    assert all(s.p_extra == 0.0 for s in skills.values())


def test_unknown_player_is_an_error(idata) -> None:
    with pytest.raises(KeyError):
        skills_from_posterior(idata, players=["Nobody"])


def test_priors_read_every_population_variable(idata) -> None:
    priors = priors_from_posterior(idata)
    for var in PRIOR_VARS:
        value = getattr(priors, var)
        assert np.isfinite(value) and value > 0
    assert priors.mu_extra == pytest.approx(0.12)
