"""Ball-by-ball match simulator for BNJCA Under 11 cricket.

Implements the encoded Section 17 playing conditions: 25 overs of
6 fair balls (maximum 8 deliveries per over), 9 batters, all out
at 8 dismissals, optional retirement at 25 balls, mandatory
retirement at 35 balls, retired batters resuming in order, and the
mandatory all-player round-robin bowling rotation.

Simplifications (documented, low aggregate impact):
* Boundary sixes are drawn as 15 percent of boundaries.
* Non-boundary runs are drawn from a two-point distribution
  calibrated to the batter's scoring rate.
* Wide/no-ball extras score one run.
* End changes follow standard cricket conventions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from junior_cricket.playing_conditions import PlayingConditions
from junior_cricket.rules_u11 import U11

LOGGER = logging.getLogger(__name__)

BOUNDARY_SIX_SHARE = 0.15
RUN_OUT_SHARE = 0.10
ECON_CLIP = (0.5, 2.0)


@dataclass
class PlayerSkills:
    """Per-player rates used by the simulator.

    Attributes:
        name: Player label.
        p_out: Dismissal hazard per ball faced.
        p_bound: Boundary probability per non-dismissal ball.
        srr: Expected runs per non-boundary ball.
        p_wicket: Wicket probability per fair ball bowled.
        econ: Expected runs off the bat per fair ball bowled.
        p_extra: Wide/no-ball probability per delivery.
        is_wicketkeeper: Keeper flag for allocation checks.
    """

    name: str
    p_out: float
    p_bound: float
    srr: float
    p_wicket: float
    econ: float
    p_extra: float
    is_wicketkeeper: bool = False


@dataclass
class BatterCard:
    """One batter's innings line.

    Attributes:
        name: Player label.
        balls: Balls faced (extras included per U11 rules).
        runs: Runs scored off the bat.
        boundaries: Fours plus sixes.
        out: Whether dismissed.
        retired: Whether retired Not out.
    """

    name: str
    balls: int = 0
    runs: int = 0
    boundaries: int = 0
    out: bool = False
    retired: bool = False
    resumed: bool = False


@dataclass
class BowlerCard:
    """One bowler's innings line.

    Attributes:
        name: Player label.
        balls: Fair balls bowled.
        runs: Runs conceded off the bat.
        wickets: Wickets taken.
        extras: Wides and no balls bowled.
        deliveries: Total deliveries including extras.
    """

    name: str
    balls: int = 0
    runs: int = 0
    wickets: int = 0
    extras: int = 0
    deliveries: int = 0


@dataclass
class InningsResult:
    """Aggregate result of one simulated innings.

    Attributes:
        runs: Total runs (off the bat plus extras).
        wickets: Dismissals.
        overs_completed: Completed overs.
        fair_balls: Legal balls bowled.
        extras: Wide/no-ball runs.
        batter_cards: Per-batter lines.
        bowler_cards: Per-bowler lines.
        all_out: Whether the innings ended on dismissals.
    """

    runs: int
    wickets: int
    overs_completed: int
    fair_balls: int
    extras: int
    batter_cards: List[BatterCard] = field(default_factory=list)
    bowler_cards: List[BowlerCard] = field(default_factory=list)
    all_out: bool = False


@dataclass
class TeamLineup:
    """One team's tactical configuration for a simulated innings.

    Attributes:
        batting_order: Player names, first to bat first.
        bowling_rotation: Player names in round-robin order.
        bowling_allocation: Player name -> overs to bowl.
        retire_at_balls: Optional coach retirement threshold
            (>= 25); None means only the mandatory 35-ball cap.
    """

    batting_order: List[str]
    bowling_rotation: List[str]
    bowling_allocation: Dict[str, int]
    retire_at_balls: Optional[int] = None

    def validate(self, conditions: PlayingConditions) -> None:
        """Check the lineup against the encoded conditions.

        Args:
            conditions: Playing conditions to validate against.

        Raises:
            ValueError: On illegal orders or allocations.
        """
        size = len(self.batting_order)
        if size not in conditions.bowling_allocations:
            raise ValueError(
                f"Team size {size} has no bowling allocation table"
            )
        if sorted(self.bowling_allocation.values(), reverse=True) != sorted(
            conditions.bowling_allocations[size], reverse=True
        ):
            raise ValueError(
                f"Bowling allocation {sorted(self.bowling_allocation.values(), reverse=True)} "
                f"does not match conditions "
                f"{conditions.bowling_allocations[size]}"
            )
        if set(self.batting_order) != set(self.bowling_rotation):
            raise ValueError(
                "Batting order and bowling rotation must contain the "
                "same players"
            )
        if set(self.bowling_allocation) != set(self.batting_order):
            raise ValueError(
                "Bowling allocation keys must match the batting order"
            )
        if self.retire_at_balls is not None and (
            self.retire_at_balls < conditions.retirement_optional_balls
            or self.retire_at_balls > conditions.retirement_mandatory_balls
        ):
            raise ValueError(
                f"retire_at_balls must be between "
                f"{conditions.retirement_optional_balls} and "
                f"{conditions.retirement_mandatory_balls}"
            )


def generate_over_sequence(
    rotation: Sequence[str], allocation: Dict[str, int], n_overs: int
) -> List[str]:
    """Expand the round-robin rotation into the over-by-over sequence.

    Players bowl one over each in rotation order; anyone with overs
    remaining keeps cycling until the innings allocation is used.

    Args:
        rotation: Player names in rotation order.
        allocation: Player name -> total overs.
        n_overs: Total overs in the innings.

    Returns:
        Bowler name for each over, length ``n_overs``.

    Raises:
        ValueError: If the allocation cannot fill the overs.
    """
    remaining = dict(allocation)
    sequence: List[str] = []
    guard = 0
    while len(sequence) < n_overs:
        progressed = False
        for player in rotation:
            if len(sequence) >= n_overs:
                break
            if remaining.get(player, 0) > 0:
                sequence.append(player)
                remaining[player] -= 1
                progressed = True
        if not progressed:
            raise ValueError(
                "Bowling allocation exhausted before filling "
                f"{n_overs} overs (remaining: {remaining})"
            )
        guard += 1
        if guard > 10 * n_overs:
            raise ValueError("Bowling rotation failed to converge")
    return sequence


def draw_population_player(
    priors, rng: np.random.Generator, name: str = "opponent"
) -> PlayerSkills:
    """Draw one synthetic opposition player from population priors.

    Args:
        priors: Population parameter estimates (see model module).
        rng: NumPy random generator.
        name: Label for the drawn player.

    Returns:
        A PlayerSkills draw from the grade population.
    """
    p_out = rng.beta(
        *mu_sigma_to_alpha_beta(priors.mu_out, priors.sigma_out)
    )
    p_bound = rng.beta(
        *mu_sigma_to_alpha_beta(priors.mu_bound, priors.sigma_bound)
    )
    p_wicket = rng.beta(
        *mu_sigma_to_alpha_beta(priors.mu_wicket, priors.sigma_wicket)
    )
    p_extra = rng.beta(
        *mu_sigma_to_alpha_beta(priors.mu_extra, priors.sigma_extra)
    )
    shape_srr = (priors.mu_srr / priors.sigma_srr) ** 2
    srr = rng.gamma(shape_srr, priors.sigma_srr**2 / priors.mu_srr)
    shape_econ = (priors.mu_econ / priors.sigma_econ) ** 2
    econ = rng.gamma(shape_econ, priors.sigma_econ**2 / priors.mu_econ)
    return PlayerSkills(
        name=name,
        p_out=float(np.clip(p_out, 1e-4, 0.5)),
        p_bound=float(np.clip(p_bound, 1e-4, 0.5)),
        srr=float(np.clip(srr, 0.01, 2.0)),
        p_wicket=float(np.clip(p_wicket, 1e-4, 0.5)),
        econ=float(np.clip(econ, 0.05, 3.0)),
        p_extra=float(np.clip(p_extra, 1e-4, 0.5)),
    )


def mu_sigma_to_alpha_beta(mu: float, sigma: float) -> Tuple[float, float]:
    """Convert a Beta mean/SD to the alpha/beta parameterisation.

    Args:
        mu: Distribution mean.
        sigma: Distribution standard deviation.

    Returns:
        (alpha, beta) shape parameters.

    Raises:
        ValueError: If the mean/SD combination is not a valid Beta.
    """
    variance = sigma * sigma
    cap = mu * (1.0 - mu)
    if variance >= cap:
        raise ValueError(
            f"Beta SD {sigma} too large for mean {mu}"
        )
    alpha = mu * (mu * (1 - mu) / variance - 1)
    beta = (1 - mu) * (mu * (1 - mu) / variance - 1)
    return alpha, beta


def _combined_dismissal(p_out: float, p_wicket: float) -> float:
    """Geometric-mean blend of batter hazard and bowler wicket rate."""
    return float(np.sqrt(max(p_out * p_wicket, 1e-8)))


def _bowler_multiplier(econ: float, population_econ: float) -> float:
    """Batter scoring multiplier from bowler economy vs population."""
    return float(np.clip(econ / population_econ, *ECON_CLIP))


def simulate_innings(
    batting_skills: Sequence[PlayerSkills],
    bowling_skills: Dict[str, PlayerSkills],
    bowling_rotation: Sequence[str],
    bowling_allocation: Dict[str, int],
    rng: np.random.Generator,
    conditions: PlayingConditions = U11,
    population_econ: float = 0.55,
    retire_at_balls: Optional[int] = None,
) -> InningsResult:
    """Simulate one innings ball by ball under the encoded rules.

    Args:
        batting_skills: Batting team's players in order (index 0
            opens).
        bowling_skills: Fielding team's players by name.
        bowling_rotation: Fielding team's round-robin order.
        bowling_allocation: Fielding team's player -> overs.
        rng: NumPy random generator.
        conditions: Playing conditions (U11 by default).
        population_econ: Grade mean economy for bowler multipliers.
        retire_at_balls: Batting team's optional retirement
            threshold (>= 25); None means only the mandatory cap.

    Returns:
        The completed innings result.

    Raises:
        NotImplementedError: For formats where dismissed batters
            continue (U10), which this engine does not model.
        ValueError: On invalid bowling configuration.
    """
    if conditions.dismissed_batter_continues:
        raise NotImplementedError(
            "This engine models the U11 survival format only; U10 "
            "conditions are used for historical data parsing"
        )

    fielding_size = len(bowling_skills)
    if fielding_size not in conditions.bowling_allocations:
        raise ValueError(
            f"Fielding team size {fielding_size} has no bowling "
            "allocation table"
        )
    if sorted(bowling_allocation.values(), reverse=True) != sorted(
        conditions.bowling_allocations[fielding_size], reverse=True
    ):
        raise ValueError(
            f"Bowling allocation {sorted(bowling_allocation.values(), reverse=True)} "
            f"does not match conditions "
            f"{conditions.bowling_allocations[fielding_size]}"
        )
    if set(bowling_allocation) != set(bowling_skills):
        raise ValueError(
            "Bowling allocation keys must match the fielding players"
        )
    if sorted(bowling_rotation) != sorted(bowling_skills):
        raise ValueError(
            "Bowling rotation must contain exactly the fielding players"
        )
    if retire_at_balls is not None and (
        retire_at_balls < conditions.retirement_optional_balls
        or retire_at_balls > conditions.retirement_mandatory_balls
    ):
        raise ValueError(
            f"retire_at_balls must be between "
            f"{conditions.retirement_optional_balls} and "
            f"{conditions.retirement_mandatory_balls}"
        )

    n_batters = len(batting_skills)
    cards = {p.name: BatterCard(name=p.name) for p in batting_skills}
    bowler_cards = {
        name: BowlerCard(name=name)
        for name in bowling_allocation
    }
    over_sequence = generate_over_sequence(
        bowling_rotation,
        bowling_allocation,
        conditions.overs_per_innings,
    )

    striker = batting_skills[0]
    non_striker = batting_skills[1]
    next_batter = 2
    retired_queue: List[PlayerSkills] = []
    wickets = 0
    extras_runs = 0
    fair_balls_total = 0
    overs_completed = 0

    def _next_available() -> Optional[PlayerSkills]:
        """Fetch the next batter: fresh, else front of retired queue."""
        nonlocal next_batter
        if next_batter < n_batters:
            player = batting_skills[next_batter]
            next_batter += 1
            return player
        if retired_queue:
            player = retired_queue.pop(0)
            # Resumed batters cannot retire again this innings.
            cards[player.name].resumed = True
            return player
        return None

    def _try_retire(player: PlayerSkills) -> bool:
        """Retire a batter when the policy thresholds are hit."""
        card = cards[player.name]
        if card.out or card.retired or card.resumed:
            return False
        mandatory = conditions.retirement_mandatory_balls
        optional = retire_at_balls or mandatory
        if card.balls >= mandatory or (
            optional < mandatory and card.balls >= optional
        ):
            card.retired = True
            retired_queue.append(player)
            return True
        return False

    for over in range(conditions.overs_per_innings):
        if wickets >= conditions.wickets_all_out:
            break
        bowler = bowling_skills[over_sequence[over]]
        bowler_card = bowler_cards[bowler.name]
        fair_in_over = 0
        deliveries = 0

        while (
            fair_in_over < conditions.fair_balls_per_over
            and deliveries < conditions.max_deliveries_per_over
        ):
            deliveries += 1
            bowler_card.deliveries += 1

            if rng.random() < bowler.p_extra:
                # Wide or no ball: one run, counts in batter's balls.
                extras_runs += 1
                bowler_card.extras += 1
                cards[striker.name].balls += 1
                if _try_retire(striker):
                    replacement = _next_available()
                    if replacement is None:
                        striker = None
                        break
                    striker = replacement
                continue

            fair_in_over += 1
            fair_balls_total += 1
            bowler_card.balls += 1
            card = cards[striker.name]
            card.balls += 1

            p_dismissal = _combined_dismissal(
                striker.p_out, bowler.p_wicket
            )
            if rng.random() < p_dismissal:
                wickets += 1
                card.out = True
                bowler_card.wickets += 1
                if wickets >= conditions.wickets_all_out:
                    break
                replacement = _next_available()
                if replacement is None:
                    break
                # Surviving batter faces the next ball; the new
                # batter takes the vacant end (rules 16.10 iv-v
                # convention carried into the U11 format).
                if rng.random() < RUN_OUT_SHARE:
                    non_striker = replacement
                else:
                    striker, non_striker = non_striker, replacement
                continue

            multiplier = _bowler_multiplier(bowler.econ, population_econ)
            bound_prob = striker.p_bound * multiplier
            if rng.random() < bound_prob:
                # Boundary: four, or six with a small share.
                is_six = rng.random() < BOUNDARY_SIX_SHARE
                card.boundaries += 1
                card.runs += 6 if is_six else 4
            else:
                # Non-boundary runs calibrated to the scoring rate.
                single_prob = min(striker.srr * multiplier * 0.9, 0.9)
                two_prob = min(striker.srr * multiplier * 0.08, 0.08)
                run_roll = rng.random()
                if run_roll < single_prob:
                    card.runs += 1
                elif run_roll < single_prob + two_prob:
                    card.runs += 2

            if _try_retire(striker):
                replacement = _next_available()
                if replacement is None:
                    striker = None
                    break
                striker = replacement

        if striker is None or non_striker is None:
            break
        if wickets >= conditions.wickets_all_out:
            break

        # End of over: batters rotate (one-end batting format).
        if _try_retire(non_striker):
            replacement = _next_available()
            if replacement is None:
                break
            non_striker = replacement
        striker, non_striker = non_striker, striker
        overs_completed += 1

    total_runs = sum(c.runs for c in cards.values()) + extras_runs
    return InningsResult(
        runs=total_runs,
        wickets=wickets,
        overs_completed=overs_completed,
        fair_balls=fair_balls_total,
        extras=extras_runs,
        batter_cards=list(cards.values()),
        bowler_cards=list(bowler_cards.values()),
        all_out=wickets >= conditions.wickets_all_out,
    )
