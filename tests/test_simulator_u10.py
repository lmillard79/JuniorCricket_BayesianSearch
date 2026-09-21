"""Tests for the U10 innings engine (BNJCA Rule 16) and the U11 run-out fix."""

from typing import Dict, List

import numpy as np
import pytest

from junior_cricket import simulator
from junior_cricket.rules_u11 import U11
from junior_cricket.simulator import (
    InningsResult,
    PlayerSkills,
    simulate_innings,
    simulate_innings_u10,
    team_total,
)

ALLOCATIONS = {
    9: [3, 3, 3, 3, 2, 2, 2, 1, 1],
    8: [3, 3, 3, 3, 3, 3, 1, 1],
    7: [4, 4, 4, 3, 3, 1, 1],
}


def squad(n: int = 9, **rates: float) -> List[PlayerSkills]:
    """A uniform squad whose last two players are the keepers."""
    base = dict(p_out=0.05, p_bound=0.07, srr=0.55, p_wicket=0.05,
                econ=0.8, p_extra=0.0)
    base.update(rates)
    return [PlayerSkills(name=f"P{i}", is_wicketkeeper=i >= n - 2, **base)
            for i in range(n)]


def config(players: List[PlayerSkills]) -> Dict[str, object]:
    """Bowling arguments giving each player their rule allocation."""
    allocation = dict(zip((p.name for p in players), ALLOCATIONS[len(players)]))
    return dict(
        bowling_skills={p.name: p for p in players},
        bowling_rotation=[p.name for p in players],
        bowling_allocation=allocation,
    )


def innings(n: int = 9, seed: int = 0, bat: List[PlayerSkills] = None,
            field: List[PlayerSkills] = None, **kwargs) -> InningsResult:
    field = field or squad(n)
    return simulate_innings_u10(
        bat or squad(n), rng=np.random.default_rng(seed),
        population_econ=0.8, **config(field), **kwargs,
    )


@pytest.mark.parametrize("size,expected", [
    (9, [14, 14, 14, 13, 13, 13, 13, 13, 13]),     # Rule 16.10(i): 3 x 14, 6 x 13
    (8, [15] * 8),
    (7, [18, 17, 17, 17, 17, 17, 17]),
])
def test_every_batter_faces_exactly_their_allotment(size, expected) -> None:
    for seed in range(5):
        result = innings(size, seed)
        assert [c.balls for c in result.batter_cards] == expected
        assert result.fair_balls == 120


def test_dismissed_batters_keep_batting() -> None:
    """Rule 16.10(iii): lots of dismissals, yet everyone gets their balls."""
    result = innings(bat=squad(p_out=0.9), field=squad(p_wicket=0.9))
    assert result.wickets > 60
    assert sum(c.dismissals for c in result.batter_cards) == result.wickets
    assert [c.balls for c in result.batter_cards] == [14, 14, 14] + [13] * 6


def test_bowlers_bowl_their_allocated_overs() -> None:
    result = innings()
    balls = {b.name: b.balls for b in result.bowler_cards}
    assert list(balls.values()) == [6 * o for o in ALLOCATIONS[9]]


def test_wides_use_up_balls_and_score_to_the_striker() -> None:
    """Rules 16.8(iii), 16.10(vii): no extra balls; one run to the striker."""
    result = innings(field=squad(p_extra=1.0))
    assert result.fair_balls == 120 and result.wickets == 0
    assert result.runs == 120 and result.extras == 120
    assert sum(b.deliveries for b in result.bowler_cards) == 120
    assert sum(b.runs for b in result.bowler_cards) == 120    # 16.8(vii)


def test_run_outs_are_not_credited_to_bowlers() -> None:
    """Rule 16.8(vi)."""
    result = innings(bat=squad(p_out=0.9), field=squad(p_wicket=0.9),
                     run_out_share=1.0)
    assert result.wickets > 0
    assert sum(b.wickets for b in result.bowler_cards) == 0


def test_other_dismissals_are_credited_to_the_bowler() -> None:
    """Rule 16.8(v): every non-run-out dismissal counts for the bowler."""
    result = innings(bat=squad(p_out=0.9), field=squad(p_wicket=0.9),
                     run_out_share=0.0)
    assert sum(b.wickets for b in result.bowler_cards) == result.wickets


def test_team_total_adds_four_runs_per_opposition_wicket() -> None:
    own = InningsResult(runs=100, wickets=3, overs_completed=20, fair_balls=120, extras=0)
    opposition = InningsResult(runs=90, wickets=7, overs_completed=20, fair_balls=120, extras=0)
    assert team_total(own, opposition) == 128      # 100 + 4 x 7
    assert team_total(opposition, own) == 102      # 90 + 4 x 3


def test_shortened_game_stops_early() -> None:
    result = innings(n_overs=18)
    assert result.fair_balls == 108
    assert all(c.balls <= a for c, a in zip(
        result.batter_cards, [14, 14, 14, 13, 13, 13, 13, 13, 13]))


def test_illegal_configurations_are_rejected() -> None:
    with pytest.raises(ValueError):
        innings(bat=squad(4), field=squad(9))                 # no allotment for 4
    with pytest.raises(NotImplementedError):
        simulate_innings_u10(squad(9), rng=np.random.default_rng(0),
                             conditions=U11, **config(squad(9)))


def test_u11_run_out_marks_the_dismissed_batter_and_credits_no_bowler(monkeypatch) -> None:
    """Regression: the striker was marked out whoever was run out."""
    monkeypatch.setattr(simulator, "RUN_OUT_SHARE", 1.0)
    players = squad(p_out=0.5, p_wicket=0.5)
    allocation = dict(zip((p.name for p in players), [4, 4, 3, 3, 3, 3, 3, 1, 1]))
    result = simulate_innings(
        players, {p.name: p for p in players}, [p.name for p in players],
        allocation, np.random.default_rng(1),
    )
    assert result.wickets == U11.wickets_all_out
    assert sum(c.out for c in result.batter_cards) == result.wickets
    assert sum(b.wickets for b in result.bowler_cards) == 0
