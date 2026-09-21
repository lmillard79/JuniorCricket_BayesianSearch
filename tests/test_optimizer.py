"""Tests for the lineup optimiser's simulation entry points."""

from typing import Dict

from junior_cricket.model import PopulationPriors
from junior_cricket.optimizer import LineupOptimizer
from junior_cricket.simulator import PlayerSkills


def make_optimizer(n_sims: int = 3) -> LineupOptimizer:
    """Build a small optimiser over a uniform nine-player squad."""
    skills: Dict[str, PlayerSkills] = {
        f"P{i}": PlayerSkills(
            name=f"P{i}",
            p_out=0.05,
            p_bound=0.04,
            srr=0.30,
            p_wicket=0.05,
            econ=0.55,
            p_extra=0.10,
            is_wicketkeeper=i in (7, 8),
        )
        for i in range(9)
    }
    return LineupOptimizer(skills, PopulationPriors(), n_sims=n_sims)


def test_evaluate_batting_runs_against_population_bowlers() -> None:
    """Batting evaluation passes bowler names, not objects, on."""
    optimizer = make_optimizer()
    totals = optimizer.evaluate_batting(optimizer.seed_batting_order())
    assert totals.shape == (3,)
    assert (totals >= 0).all()


def test_evaluate_bowling_runs_against_population_batters() -> None:
    """Bowling evaluation simulates the seeded tiered rotation."""
    optimizer = make_optimizer()
    rotation, allocation = optimizer.seed_bowling()
    totals = optimizer.evaluate_bowling(rotation, allocation)
    assert totals.shape == (3,)
    assert (totals >= 0).all()
