"""BNJCA Under 11 boys' playing conditions (2026 rules booklet).

Source: BNJCA 2026 Rules Booklet, Section 17 (pages 32-35),
https://bnjca.org.au/wp-content/uploads/2026/07/2026-RULES-BOOKLET-FINAL.pdf

Key semantics: one-day, one innings each, 25 overs per team, 9
players, all out at 8 dismissals, dismissed batters leave (real
survival process), optional retirement at 25 balls and mandatory
retirement at 35 balls, every player bowls including both
wicketkeepers, overs are 6 fair balls capped at 8 deliveries.
"""

from junior_cricket.playing_conditions import PlayingConditions

U11 = PlayingConditions(
    age_group="Under 11",
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
    bowling_allocations={
        # 2 non-keepers bowl 4, 5 non-keepers bowl 3, 2 keepers bowl 1.
        9: [4, 4, 3, 3, 3, 3, 3, 1, 1],
        # 3 non-keepers bowl 4, 3 non-keepers bowl 3, 2 keepers bowl 2.
        8: [4, 4, 4, 3, 3, 3, 2, 2],
        # 3 non-keepers bowl 5, 2 non-keepers bowl 4, 2 keepers bowl 1.
        7: [5, 5, 5, 4, 4, 1, 1],
    },
    batting_ball_allotments={},
)

U11.validate()
