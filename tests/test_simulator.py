"""Tests for the ball-by-ball U11 match simulator."""

from typing import Dict, List

import numpy as np
import pytest

from junior_cricket.rules_u11 import U11
from junior_cricket.simulator import (
    BatterCard,
    PlayerSkills,
    generate_over_sequence,
    simulate_innings,
)


def make_skills(
    n: int = 9,
    p_out: float = 0.05,
    p_bound: float = 0.04,
    srr: float = 0.30,
    p_wicket: float = 0.05,
    econ: float = 0.55,
    p_extra: float = 0.10,
    keepers: tuple = (7, 8),
) -> List[PlayerSkills]:
    """Build a uniform nine-player squad for testing."""
    return [
        PlayerSkills(
            name=f"P{i}",
            p_out=p_out,
            p_bound=p_bound,
            srr=srr,
            p_wicket=p_wicket,
            econ=econ,
            p_extra=p_extra,
            is_wicketkeeper=i in keepers,
        )
        for i in range(n)
    ]


def standard_allocation(skills: List[PlayerSkills]) -> Dict[str, int]:
    """Assign the U11 team-of-9 tier structure."""
    allocation: Dict[str, int] = {}
    tier = iter([4, 4, 3, 3, 3, 3, 3, 1, 1])
    for player in skills:
        if player.is_wicketkeeper:
            allocation[player.name] = 1
    for player in skills:
        if not player.is_wicketkeeper:
            allocation[player.name] = next(tier)
    return allocation


def test_over_sequence_round_robin() -> None:
    """Round-robin expansion cycles players with overs remaining."""
    rotation = ["A", "B", "C"]
    allocation = {"A": 2, "B": 1, "C": 1}
    sequence = generate_over_sequence(rotation, allocation, 4)
    # Round 1: everyone; round 2: only A has overs left.
    assert sequence == ["A", "B", "C", "A"]


def test_over_sequence_fills_exact_overs() -> None:
    """The U11 team-of-9 sequence has exactly 25 entries."""
    skills = make_skills()
    rotation = [p.name for p in skills]
    allocation = standard_allocation(skills)
    sequence = generate_over_sequence(rotation, allocation, 25)
    assert len(sequence) == 25
    counts: Dict[str, int] = {}
    for name in sequence:
        counts[name] = counts.get(name, 0) + 1
    assert sorted(counts.values(), reverse=True) == [
        4, 4, 3, 3, 3, 3, 3, 1, 1
    ]


def test_innings_ends_at_25_overs() -> None:
    """Durable batters produce a full 25-over innings."""
    skills = make_skills(p_out=0.0, p_extra=0.0)
    batting = skills
    fielding = {p.name: p for p in skills}
    result = simulate_innings(
        batting_skills=batting,
        bowling_skills=fielding,
        bowling_rotation=[p.name for p in skills],
        bowling_allocation=standard_allocation(skills),
        rng=np.random.default_rng(42),
        conditions=U11,
        population_econ=0.55,
    )
    assert result.overs_completed == 25
    assert result.wickets == 0
    assert not result.all_out


def test_mandatory_retirement_at_35_balls() -> None:
    """Batters who never get out must retire at 35 balls."""
    skills = make_skills(p_out=0.0, p_bound=0.0, srr=0.0, p_extra=0.0)
    fielding = {p.name: p for p in skills}
    result = simulate_innings(
        batting_skills=skills,
        bowling_skills=fielding,
        bowling_rotation=[p.name for p in skills],
        bowling_allocation=standard_allocation(skills),
        rng=np.random.default_rng(1),
        conditions=U11,
        population_econ=0.55,
    )
    for card in result.batter_cards:
        assert card.balls <= 35, f"{card.name} exceeded the cap"
    # Openers should have hit the cap; some batters may be unused.
    assert any(card.balls == 35 for card in result.batter_cards)


def test_all_out_at_eight_wickets() -> None:
    """Certified dismissal produces all out at exactly 8 wickets."""
    skills = make_skills(p_out=1.0, p_extra=0.0)
    fielding = {p.name: p for p in skills}
    result = simulate_innings(
        batting_skills=skills,
        bowling_skills=fielding,
        bowling_rotation=[p.name for p in skills],
        bowling_allocation=standard_allocation(skills),
        rng=np.random.default_rng(3),
        conditions=U11,
        population_econ=0.55,
    )
    assert result.wickets == 8
    assert result.all_out
    assert result.overs_completed < 25


def test_over_never_exceeds_eight_deliveries() -> None:
    """High extras rates cannot push an over past 8 deliveries."""
    skills = make_skills(p_out=0.0, p_extra=0.95)
    fielding = {p.name: p for p in skills}
    result = simulate_innings(
        batting_skills=skills,
        bowling_skills=fielding,
        bowling_rotation=[p.name for p in skills],
        bowling_allocation=standard_allocation(skills),
        rng=np.random.default_rng(9),
        conditions=U11,
        population_econ=0.55,
    )
    for card in result.bowler_cards:
        # Each over is at most 8 deliveries; overs per bowler vary.
        assert card.deliveries <= 8 * 4
    # With 95 percent extras, many overs hit the delivery cap, so
    # the innings cannot complete 25 overs of 6 fair balls.
    assert result.fair_balls < 25 * 6


def test_runs_are_nonnegative_and_consistent() -> None:
    """Total runs equal the sum of cards plus extras."""
    skills = make_skills()
    fielding = {p.name: p for p in skills}
    result = simulate_innings(
        batting_skills=skills,
        bowling_skills=fielding,
        bowling_rotation=[p.name for p in skills],
        bowling_allocation=standard_allocation(skills),
        rng=np.random.default_rng(11),
        conditions=U11,
        population_econ=0.55,
    )
    card_runs = sum(card.runs for card in result.batter_cards)
    assert result.runs == card_runs + result.extras
    assert result.runs >= 0


def test_u10_format_rejected() -> None:
    """The engine refuses U10 (dismissed batters continue)."""
    from junior_cricket.rules_u10 import U10

    skills = make_skills()
    fielding = {p.name: p for p in skills}
    with pytest.raises(NotImplementedError):
        simulate_innings(
            batting_skills=skills,
            bowling_skills=fielding,
            bowling_rotation=[p.name for p in skills],
            bowling_allocation=standard_allocation(skills),
            rng=np.random.default_rng(5),
            conditions=U10,
        )


def test_illegal_allocation_rejected() -> None:
    """Allocations that violate the tier structure are rejected."""
    skills = make_skills()
    fielding = {p.name: p for p in skills}
    bad_allocation = {p.name: 3 for p in skills}
    with pytest.raises(ValueError):
        simulate_innings(
            batting_skills=skills,
            bowling_skills=fielding,
            bowling_rotation=[p.name for p in skills],
            bowling_allocation=bad_allocation,
            rng=np.random.default_rng(6),
        )


def test_bankable_batters_use_the_optional_threshold_others_do_not() -> None:
    """Only bankable batters may retire at 25; everyone else waits for 35."""
    skills = make_skills(p_out=0.0, p_bound=0.0, srr=0.0, p_extra=0.0, keepers=())
    fielding = {p.name: p for p in skills}
    result = simulate_innings(
        batting_skills=skills,
        bowling_skills=fielding,
        bowling_rotation=[p.name for p in skills],
        bowling_allocation=standard_allocation(skills),
        rng=np.random.default_rng(2),
        conditions=U11,
        population_econ=0.55,
        retire_at_balls=25,
        bankable=("P0", "P1"),
    )
    cards = {c.name: c for c in result.batter_cards}
    assert cards["P0"].retired and cards["P0"].balls == 25
    assert cards["P1"].retired and cards["P1"].balls == 25
    for card in result.batter_cards:
        if card.name not in ("P0", "P1") and card.retired:
            assert card.balls == 35


def _bank_and_recall_fixture():
    """A squad of two indestructible batters and seven who are out first ball.

    Built so a cheap wicket is guaranteed and the banked batters, once
    recalled, never get out themselves, making the effect of recall on
    who bats easy to read off the final cards.
    """
    from junior_cricket.simulator import PlayerSkills

    durable = [PlayerSkills(name=f"P{i}", p_out=0.0, p_bound=0.0, srr=0.0, p_wicket=1.0,
                             econ=0.55, p_extra=0.0) for i in range(2)]
    fragile = [PlayerSkills(name=f"P{i}", p_out=1.0, p_bound=0.0, srr=0.0, p_wicket=1.0,
                             econ=0.55, p_extra=0.0) for i in range(2, 9)]
    skills = durable + fragile
    fielding = {p.name: p for p in skills}
    allocation = {"P0": 4, "P1": 4, "P2": 3, "P3": 3, "P4": 3, "P5": 3, "P6": 3, "P7": 1, "P8": 1}
    return skills, fielding, allocation


def test_recall_brings_back_a_banked_batter_ahead_of_fresh_ones() -> None:
    """A cheap wicket recalls a banked batter instead of the next fresh one."""
    skills, fielding, allocation = _bank_and_recall_fixture()
    result = simulate_innings(
        batting_skills=skills,
        bowling_skills=fielding,
        bowling_rotation=[p.name for p in skills],
        bowling_allocation=allocation,
        rng=np.random.default_rng(0),
        conditions=U11,
        population_econ=0.55,
        retire_at_balls=25,
        bankable=("P0", "P1"),
        recall_within_balls=6,
        recall_below_runs=5,
    )
    cards = {c.name: c for c in result.batter_cards}
    assert cards["P0"].resumed and cards["P1"].resumed
    # Recall jumped the queue: several fresh batters were never needed.
    assert sum(card.balls == 0 for card in result.batter_cards) >= 3
    assert result.wickets < 7


def test_without_recall_thresholds_fresh_batters_come_in_as_usual() -> None:
    """Bankable alone, with no recall thresholds, only changes who retires early."""
    skills, fielding, allocation = _bank_and_recall_fixture()
    result = simulate_innings(
        batting_skills=skills,
        bowling_skills=fielding,
        bowling_rotation=[p.name for p in skills],
        bowling_allocation=allocation,
        rng=np.random.default_rng(0),
        conditions=U11,
        population_econ=0.55,
        retire_at_balls=25,
        bankable=("P0", "P1"),
    )
    # Every fragile batter gets a turn: none are skipped by an early recall.
    assert result.wickets == 7
    for card in result.batter_cards:
        if card.name not in ("P0", "P1"):
            assert card.balls >= 1 and card.out


def test_bankable_name_outside_batting_order_rejected() -> None:
    """A bankable name that is not in the batting order is an error."""
    skills = make_skills()
    fielding = {p.name: p for p in skills}
    with pytest.raises(ValueError):
        simulate_innings(
            batting_skills=skills,
            bowling_skills=fielding,
            bowling_rotation=[p.name for p in skills],
            bowling_allocation=standard_allocation(skills),
            rng=np.random.default_rng(6),
            retire_at_balls=25,
            bankable=("not-a-player",),
        )


def test_recall_threshold_without_bankable_rejected() -> None:
    """A recall threshold with no bankable list is an error."""
    skills = make_skills()
    fielding = {p.name: p for p in skills}
    with pytest.raises(ValueError):
        simulate_innings(
            batting_skills=skills,
            bowling_skills=fielding,
            bowling_rotation=[p.name for p in skills],
            bowling_allocation=standard_allocation(skills),
            rng=np.random.default_rng(6),
            recall_within_balls=6,
        )
