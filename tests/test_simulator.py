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


class ScriptedOutcomes:
    """Deterministic ball outcomes that also record who faced each fair ball."""

    def __init__(self, out_names=(), runs: int = 1) -> None:
        self.out_names = set(out_names)
        self.runs_per_ball = runs
        self.strikers: List[str] = []

    def dismissal_probability(self, striker, bowler) -> float:
        self.strikers.append(striker.name)
        return 1.0 if striker.name in self.out_names else 0.0

    def runs(self, striker, bowler, rng):
        return self.runs_per_ball, False


def _play_scripted(outcomes, conditions=U11, **kwargs):
    """Run an innings of uniform batters with no extras and the scripted outcomes."""
    skills = make_skills(p_extra=0.0, keepers=())
    fielding = {p.name: p for p in skills}
    return simulate_innings(
        batting_skills=skills,
        bowling_skills=fielding,
        bowling_rotation=[p.name for p in skills],
        bowling_allocation=standard_allocation(skills),
        rng=np.random.default_rng(0),
        conditions=conditions,
        outcomes=outcomes,
        **kwargs,
    )


def test_u11_conditions_encode_the_observed_strike_rules() -> None:
    """17.4(iv) and the first U11 game: batters cross, a new batter takes strike."""
    assert U11.batters_cross_on_runs
    assert U11.incoming_batter_takes_strike


def test_batters_swap_ends_on_odd_runs_and_again_at_the_end_of_the_over() -> None:
    """A single sends the other batter on strike; the over then rotates them as well."""
    outcomes = ScriptedOutcomes(runs=1)
    _play_scripted(outcomes)
    # Ball 6 is a single by P1, who crosses; the end of the over rotates
    # again, so P1 also faces the first ball of the next over.
    assert outcomes.strikers[:12] == ["P0", "P1", "P0", "P1", "P0", "P1",
                                      "P1", "P0", "P1", "P0", "P1", "P0"]


def test_without_crossing_strike_changes_only_at_the_end_of_the_over() -> None:
    """The earlier engine: each batter faces whole overs regardless of the runs."""
    import dataclasses

    outcomes = ScriptedOutcomes(runs=1)
    _play_scripted(outcomes, conditions=dataclasses.replace(U11, batters_cross_on_runs=False))
    assert outcomes.strikers[:18] == ["P0"] * 6 + ["P1"] * 6 + ["P0"] * 6


def test_even_runs_do_not_change_the_striker_within_an_over() -> None:
    """Dots, twos and boundaries leave the same batter on strike until the over ends."""
    outcomes = ScriptedOutcomes(runs=2)
    _play_scripted(outcomes)
    assert outcomes.strikers[:12] == ["P0"] * 6 + ["P1"] * 6


def test_new_batter_takes_strike_after_the_striker_is_out(monkeypatch) -> None:
    """With the usual convention the incoming batter, not the survivor, faces next."""
    monkeypatch.setattr("junior_cricket.simulator.RUN_OUT_SHARE", 0.0)
    outcomes = ScriptedOutcomes(out_names={"P0"}, runs=0)
    _play_scripted(outcomes)
    assert outcomes.strikers[:2] == ["P0", "P2"]


def test_survivor_faces_next_when_the_incoming_batter_rule_is_off(monkeypatch) -> None:
    """The earlier engine: the batter at the other end faces, the new one waits."""
    import dataclasses

    monkeypatch.setattr("junior_cricket.simulator.RUN_OUT_SHARE", 0.0)
    outcomes = ScriptedOutcomes(out_names={"P0"}, runs=0)
    _play_scripted(
        outcomes,
        conditions=dataclasses.replace(U11, incoming_batter_takes_strike=False),
    )
    assert outcomes.strikers[:2] == ["P0", "P1"]


def test_batter_who_has_crossed_retires_from_the_non_strikers_end() -> None:
    """A retiring batter is replaced where they stand, not where they faced from."""
    outcomes = ScriptedOutcomes(runs=1)
    result = _play_scripted(outcomes, retire_at_balls=25)
    # P0 faces the odd balls and reaches 25 on ball 49, then crosses and retires
    # as the non-striker: P1 still faces ball 50, and the replacements follow.
    assert outcomes.strikers[48:52] == ["P0", "P1", "P2", "P3"]
    cards = {c.name: c for c in result.batter_cards}
    assert cards["P0"].retired and cards["P0"].balls == 25
    assert cards["P1"].retired and cards["P1"].balls == 25


def test_time_at_the_crease_counts_every_delivery_for_both_batters() -> None:
    """With only dots, P0 and P1 swap strike each over and retire at 35 balls: 65 and 71 deliveries in."""
    outcomes = ScriptedOutcomes(runs=0)
    result = _play_scripted(outcomes)
    cards = {c.name: c for c in result.batter_cards}
    assert cards["P0"].balls == 35 and cards["P0"].crease_deliveries == 65
    assert cards["P1"].balls == 35 and cards["P1"].crease_deliveries == 71
    assert cards["P0"].longest_stint == 65 and cards["P1"].longest_stint == 71     # one unbroken stint each
    assert cards["P0"].first_stint == 65 and cards["P1"].first_stint == 71
    deliveries = sum(b.deliveries for b in result.bowler_cards)
    assert sum(c.crease_deliveries for c in result.batter_cards) == 2 * deliveries


def test_time_at_the_crease_includes_wides_and_a_second_stint() -> None:
    """Retirees who come back at the end add their second stint to the time they spend out there."""
    import dataclasses

    short_stints = dataclasses.replace(U11, retirement_optional_balls=10, retirement_mandatory_balls=10)
    skills = make_skills(p_extra=0.3, p_out=0.0, keepers=())
    fielding = {p.name: p for p in skills}
    result = simulate_innings(
        batting_skills=skills, bowling_skills=fielding, bowling_rotation=[p.name for p in skills],
        bowling_allocation=standard_allocation(skills), rng=np.random.default_rng(3), conditions=short_stints,
        outcomes=ScriptedOutcomes(runs=0), retire_at_balls=10)
    deliveries = sum(b.deliveries for b in result.bowler_cards)
    assert deliveries > 25 * 6                                 # wides added deliveries
    assert sum(c.crease_deliveries for c in result.batter_cards) == 2 * deliveries
    assert any(c.resumed for c in result.batter_cards)        # all nine retired, so the first retirees came back
    # A second stint adds to the time out there but not to the longest unbroken one.
    assert any(c.crease_deliveries > c.longest_stint for c in result.batter_cards if c.resumed)
    assert all(c.longest_stint <= c.crease_deliveries for c in result.batter_cards)
    # The first stint is cut short by the retirement; the second ran to the end.
    assert all(0 <= c.first_stint <= c.longest_stint for c in result.batter_cards)
    assert any(c.first_stint < c.longest_stint for c in result.batter_cards if c.resumed)


def test_retiring_at_20_is_a_what_if_the_engine_accepts_once_the_minimum_is_lowered() -> None:
    """Dots only: P0 faces the odd overs, retires on his 20th ball (delivery 38) and P2 takes his place."""
    import dataclasses

    outcomes = ScriptedOutcomes(runs=0)
    result = _play_scripted(outcomes, conditions=dataclasses.replace(U11, retirement_optional_balls=20),
                            retire_at_balls=20)
    assert outcomes.strikers[36:40] == ["P0", "P0", "P2", "P2"]
    assert {c.name: c for c in result.batter_cards}["P0"].retired
    with pytest.raises(ValueError):                           # without lowering the minimum it is still refused
        _play_scripted(ScriptedOutcomes(runs=0), retire_at_balls=20)
