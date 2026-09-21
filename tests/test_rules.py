"""Tests for the encoded BNJCA playing conditions."""

import pytest

from junior_cricket.rules_u10 import U10
from junior_cricket.rules_u11 import U11


def test_u11_core_conditions() -> None:
    """U11 core format matches the 2026 rules booklet Section 17."""
    assert U11.overs_per_innings == 25
    assert U11.fair_balls_per_over == 6
    assert U11.max_deliveries_per_over == 8
    assert U11.wickets_all_out == 8
    assert U11.team_size_default == 9
    assert U11.retirement_optional_balls == 25
    assert U11.retirement_mandatory_balls == 35
    assert not U11.dismissed_batter_continues
    assert U11.dismissal_penalty_runs == 0
    assert not U11.sundries_to_striker
    assert not U11.lbw_applies


def test_u11_bowling_allocations_sum() -> None:
    """Every U11 bowling allocation fills exactly 25 overs."""
    for size, allocation in U11.bowling_allocations.items():
        assert sum(allocation) == 25, f"team of {size}"


def test_u11_allocation_tiers() -> None:
    """Team of 9 allocation is 4, 4, 3, 3, 3, 3, 3, 1, 1."""
    assert sorted(
        U11.bowling_allocations[9], reverse=True
    ) == [4, 4, 3, 3, 3, 3, 3, 1, 1]


def test_u10_core_conditions() -> None:
    """U10 core format matches the 2026 rules booklet Section 16."""
    assert U10.overs_per_innings == 20
    assert U10.wickets_all_out == 0
    assert U10.dismissed_batter_continues
    assert U10.dismissal_penalty_runs == 4
    assert U10.sundries_to_striker


def test_u10_ballot_allotments_sum() -> None:
    """Every U10 batting allotment totals the 120 balls on offer."""
    for size, allotment in U10.batting_ball_allotments.items():
        assert sum(allotment) == 120, f"team of {size}"


def test_validate_raises_on_bad_conditions() -> None:
    """Validation catches allocation tables that do not add up."""
    from junior_cricket.playing_conditions import PlayingConditions

    broken = PlayingConditions(
        age_group="Broken",
        overs_per_innings=25,
        fair_balls_per_over=6,
        max_deliveries_per_over=8,
        wickets_all_out=8,
        team_size_default=9,
        team_size_min=7,
        team_size_max=9,
        retirement_optional_balls=25,
        retirement_mandatory_balls=35,
        dismissed_batter_continues=False,
        dismissal_penalty_runs=0,
        sundries_to_striker=False,
        lbw_applies=False,
        pitch_length_m=16,
        bowling_allocations={9: [4, 4, 3, 3, 3, 3, 3, 1]},
    )
    with pytest.raises(ValueError):
        broken.validate()
