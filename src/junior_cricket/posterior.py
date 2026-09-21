"""Read simulator inputs out of a fitted posterior.

Shared by the lineup optimiser and the U10 totals check so both build
``PlayerSkills`` and ``PopulationPriors`` the same way.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional

import numpy as np

from junior_cricket.model import PopulationPriors
from junior_cricket.simulator import PlayerSkills

PRIOR_VARS = (
    "mu_out", "sigma_out", "mu_bound", "sigma_bound", "mu_srr", "sigma_srr",
    "mu_wicket", "sigma_wicket", "mu_econ", "sigma_econ", "mu_extra",
    "sigma_extra",
)


def skills_from_posterior(
    idata,
    players: Optional[Iterable[str]] = None,
    p_extra: Optional[float] = None,
) -> Dict[str, PlayerSkills]:
    """Build mean-rate ``PlayerSkills`` from a fitted posterior.

    Args:
        idata: InferenceData with the projected rate variables.
        players: Restrict to these players (the squad); all players in
            the posterior when omitted.
        p_extra: Override every player's wide/no-ball probability, for
            example 0 when the fitted rates already include extras.

    Returns:
        Player skills keyed by player name.

    Raises:
        KeyError: If a requested player is not in the posterior.
    """
    posterior = idata.posterior
    names = [str(n) for n in posterior["p_out_final"].coords["player"].values]

    def _mean(var: str) -> np.ndarray:
        return np.asarray(posterior[var].mean(dim=("chain", "draw")))

    columns = {
        "p_out": _mean("p_out_final"),
        "p_bound": _mean("p_bound_final"),
        "srr": _mean("srr_proj"),
        "p_wicket": _mean("p_wicket_final"),
        "econ": _mean("econ_proj"),
        "p_extra": _mean("p_extra_proj"),
    }
    wanted = names if players is None else list(players)
    missing = [p for p in wanted if p not in names]
    if missing:
        raise KeyError(f"Players not in the posterior: {missing}")
    skills: Dict[str, PlayerSkills] = {}
    for name in wanted:
        i = names.index(name)
        skills[name] = PlayerSkills(
            name=name,
            **{k: float(v[i]) for k, v in columns.items()},
        )
        if p_extra is not None:
            skills[name].p_extra = p_extra
    return skills


def priors_from_posterior(idata) -> PopulationPriors:
    """Extract population priors from the fitted posterior means.

    Args:
        idata: InferenceData with the population variables.

    Returns:
        Point-estimate population priors on the natural scale.
    """
    posterior = idata.posterior
    return PopulationPriors(**{
        var: float(posterior[var].mean(dim=("chain", "draw")))
        for var in PRIOR_VARS
    })
