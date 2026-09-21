"""BNJCA Under 10 playing conditions (2026 rules booklet).

Source: BNJCA 2026 Rules Booklet, Section 16 (pages 29-32),
https://bnjca.org.au/wp-content/uploads/2026/07/2026-RULES-BOOKLET-FINAL.pdf

Key semantics: single-day game over 40 overs in two innings of
20 overs each, 7-player teams, dismissed batters continue batting
their allotted balls (4-run penalty per dismissal credited to the
opposition), fixed ball allotments per team size, sundries are
added to the striker's score, six-ball overs with no extra balls
for no balls or wides.
"""

from junior_cricket.playing_conditions import PlayingConditions

U10 = PlayingConditions(
    age_group="Under 10",
    overs_per_innings=20,
    fair_balls_per_over=6,
    max_deliveries_per_over=6,
    wickets_all_out=0,
    team_size_default=7,
    team_size_min=5,
    team_size_max=9,
    retirement_optional_balls=0,
    retirement_mandatory_balls=0,
    dismissed_batter_continues=True,
    dismissal_penalty_runs=4,
    sundries_to_striker=True,
    lbw_applies=False,
    pitch_length_m=16,
    bowling_allocations={
        # Standard team of 7: 3 bowl 4, 2 bowl 3, 2 keepers bowl 1.
        7: [4, 4, 4, 3, 3, 1, 1],
        # Minimum team of 5: all bowl 4.
        5: [4, 4, 4, 4, 4],
        # Team of 6: 4 bowl 4, 2 keepers bowl 2.
        6: [4, 4, 4, 4, 2, 2],
        # Team of 8: 6 bowl 3, 2 keepers bowl 1.
        8: [3, 3, 3, 3, 3, 3, 1, 1],
        # Maximum team of 9: 4 bowl 3, 3 bowl 2, 2 keepers bowl 1.
        9: [3, 3, 3, 3, 2, 2, 2, 1, 1],
    },
    batting_ball_allotments={
        # 120 balls on offer (20 overs x 6 balls) split per team size.
        5: [24, 24, 24, 24, 24],
        6: [20, 20, 20, 20, 20, 20],
        # Standard team of 7: 1 player receives 18, 6 receive 17.
        7: [18, 17, 17, 17, 17, 17, 17],
        8: [15, 15, 15, 15, 15, 15, 15, 15],
        # Maximum team of 9: 3 receive 14, 6 receive 13.
        9: [14, 14, 14, 13, 13, 13, 13, 13, 13],
    },
)

U10.validate()
