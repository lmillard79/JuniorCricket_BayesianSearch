"""Shared playing-conditions container for BNJCA junior formats.

Each age group module (``rules_u10``, ``rules_u11``) exposes a
single frozen ``PlayingConditions`` instance extracted from the
BNJCA rules booklet (2026 edition). The simulator consumes these
values directly so that rule changes require a config edit, not
engine surgery.
"""

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass(frozen=True)
class PlayingConditions:
    """Encoded playing conditions for one BNJCA age group.

    Attributes:
        age_group: Display name, e.g. ``"Under 11"``.
        overs_per_innings: Maximum overs per team per innings.
        fair_balls_per_over: Legal balls required to complete an over.
        max_deliveries_per_over: Hard cap on deliveries per over
            (extras re-bowled up to this cap); ``0`` means no cap
            beyond the fair-ball count.
        wickets_all_out: Dismissals that end an innings; ``0`` means
            the concept does not apply (dismissed batters continue).
        team_size_default: Standard game-day team size.
        team_size_min: Minimum players to constitute a team.
        team_size_max: Maximum game-day players.
        retirement_optional_balls: Balls faced after which a batter
            may retire Not out; ``None`` if not applicable.
        retirement_mandatory_balls: Balls faced at which a batter
            must retire; ``None`` if not applicable.
        dismissed_batter_continues: True when a dismissed batter
            keeps batting their allotted balls (U10 pairs-style).
        dismissal_penalty_runs: Runs credited to the opposition per
            dismissal while batting.
        sundries_to_striker: True when byes/leg byes count toward
            the striker's score (U10 convention).
        lbw_applies: True when LBW is a valid dismissal.
        pitch_length_m: Pitch length in metres.
        bowling_allocations: Team size -> per-player over counts,
            sorted descending. The two lowest allocations are the
            wicketkeepers' when keepers bowl fewer overs.
        batting_ball_allotments: Team size -> per-player ball
            allotments (U10 only); empty for U11 (retirement rules
            govern ball caps instead).
    """

    age_group: str
    overs_per_innings: int
    fair_balls_per_over: int
    max_deliveries_per_over: int
    wickets_all_out: int
    team_size_default: int
    team_size_min: int
    team_size_max: int
    retirement_optional_balls: int
    retirement_mandatory_balls: int
    dismissed_batter_continues: bool
    dismissal_penalty_runs: int
    sundries_to_striker: bool
    lbw_applies: bool
    pitch_length_m: int
    bowling_allocations: Dict[int, List[int]] = field(default_factory=dict)
    batting_ball_allotments: Dict[int, List[int]] = field(default_factory=dict)

    def validate(self) -> None:
        """Check internal consistency of the encoded conditions.

        Raises:
            ValueError: If any allocation table does not sum to the
                total balls available in an innings.
        """
        total_balls = self.overs_per_innings * self.fair_balls_per_over
        for size, allocation in self.bowling_allocations.items():
            if sum(allocation) != self.overs_per_innings:
                raise ValueError(
                    f"Bowling allocation for team of {size} sums to "
                    f"{sum(allocation)} overs, expected "
                    f"{self.overs_per_innings}"
                )
        for size, allotment in self.batting_ball_allotments.items():
            if sum(allotment) != total_balls:
                raise ValueError(
                    f"Batting allotment for team of {size} sums to "
                    f"{sum(allotment)} balls, expected {total_balls}"
                )
