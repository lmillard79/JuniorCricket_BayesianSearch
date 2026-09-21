"""Lineup optimiser for BNJCA Under 11 cricket.

Searches batting orders and bowling configurations against
opposition drawn from the fitted grade population, respecting the
mandatory participation constraints (all players bowl; tiered
allocations; round-robin rotation).

The optimisation is decomposed: the batting order only affects
our innings, and the bowling configuration only affects the
opposition innings, so each is evaluated with dedicated
simulations and the match outcome is assembled from the two
independent totals distributions.
"""

from __future__ import annotations

import logging
import zlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from junior_cricket.playing_conditions import PlayingConditions
from junior_cricket.simulator import (
    PlayerSkills,
    TeamLineup,
    draw_population_player,
    simulate_innings,
)
from junior_cricket.rules_u11 import U11

LOGGER = logging.getLogger(__name__)


@dataclass
class OptimizerResult:
    """Recommendation bundle from a completed optimisation.

    Attributes:
        batting_order: Recommended batting order (first bats first).
        bowling_rotation: Recommended round-robin bowling order.
        bowling_allocation: Player name -> overs.
        retire_at_balls: Recommended retirement policy, or None.
        our_totals: Simulated distribution of our totals.
        their_totals: Simulated distribution of opposition totals.
        win_probability: Fraction of matches won.
        tie_probability: Fraction of matches tied.
        expected_run_diff: Mean run differential.
        run_diff_ci: 90 percent interval of the run differential.
        batting_history: (order, mean total) per evaluated candidate.
        bowling_history: (rotation, mean total) per evaluated candidate.
    """

    batting_order: List[str]
    bowling_rotation: List[str]
    bowling_allocation: Dict[str, int]
    retire_at_balls: Optional[int]
    our_totals: np.ndarray
    their_totals: np.ndarray
    win_probability: float
    tie_probability: float
    expected_run_diff: float
    run_diff_ci: Tuple[float, float]
    batting_history: List[Tuple[List[str], float]] = field(
        default_factory=list
    )
    bowling_history: List[Tuple[List[str], float]] = field(
        default_factory=list
    )


class LineupOptimizer:
    """Hill-climbing lineup search against population opposition.

    Args:
        player_skills: Our players by name (rates from posterior).
        priors: Grade population priors for opposition draws.
        conditions: Playing conditions (U11 by default).
        n_sims: Simulations per candidate evaluation.
        seed: Base random seed; candidate evaluations derive
            stable sub-seeds for low-variance comparisons.
        outcomes: Optional joint batter-by-bowler per-ball rule (see
            ``ball_model.JointOutcomes``). When given, it decides every
            ball; ``player_skills`` then only seed the search and supply
            each player's extras rate. Opposition players are registered
            with it as they are drawn.
    """

    def __init__(
        self,
        player_skills: Dict[str, PlayerSkills],
        priors,
        conditions: PlayingConditions = U11,
        n_sims: int = 400,
        seed: int = 20260921,
        outcomes=None,
    ) -> None:
        self.skills = dict(player_skills)
        self.priors = priors
        self.conditions = conditions
        self.n_sims = n_sims
        self.base_seed = seed
        self.outcomes = outcomes

    # ------------------------------------------------------------------
    # Seed heuristics
    # ------------------------------------------------------------------

    def batting_strength(self, name: str) -> float:
        """Expected runs per dismissal - higher means bat higher.

        Args:
            name: Player name.

        Returns:
            Approximate expected runs scored before dismissal.
        """
        player = self.skills[name]
        runs_per_ball = player.p_bound * 4.6 + (1 - player.p_bound) * (
            player.srr
        )
        return runs_per_ball / max(player.p_out, 1e-4)

    def bowling_strength(self, name: str) -> float:
        """Bowling quality score - higher means stronger bowler.

        Wicket rate up, economy down.

        Args:
            name: Player name.

        Returns:
            Composite bowling strength.
        """
        player = self.skills[name]
        return player.p_wicket / max(player.econ, 1e-4)

    def seed_batting_order(self) -> List[str]:
        """Skill-sorted batting order (strongest opens)."""
        names = list(self.skills)
        names.sort(key=self.batting_strength, reverse=True)
        return names

    def seed_bowling(self) -> Tuple[List[str], Dict[str, int]]:
        """Skill-sorted rotation with tiered allocation.

        Keepers are forced into the minimum tier; the two
        strongest remaining bowlers receive the top tier.

        Returns:
            (rotation order, allocation dict).
        """
        names = list(self.skills)
        tiers = self.conditions.bowling_allocations[len(names)]
        keepers = [n for n in names if self.skills[n].is_wicketkeeper]
        non_keepers = [n for n in names if not self.skills[n].is_wicketkeeper]

        # Assign tiers: keepers take the smallest allocations first.
        allocation: Dict[str, int] = {}
        tier_list = sorted(tiers, reverse=True)
        for keeper in keepers:
            allocation[keeper] = tier_list.pop()

        # Strongest bowlers get the biggest remaining tiers.
        ordered = sorted(non_keepers, key=self.bowling_strength,
                         reverse=True)
        for player in ordered:
            allocation[player] = tier_list.pop(0)

        # Rotation interleaves tiers so strength is spread across
        # the innings: sort by tier then strength within tier.
        rotation = sorted(
            names,
            key=lambda n: (allocation[n], self.bowling_strength(n)),
            reverse=True,
        )
        return rotation, allocation

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def _candidate_rng(self, tag: str, *parts) -> np.random.Generator:
        """Stable per-candidate RNG for common-random-number compare."""
        payload = f"{self.base_seed}|{tag}|{'|'.join(map(str, parts))}"
        return np.random.default_rng(zlib.crc32(payload.encode()) & 0x7FFFFFFF)

    def _draw_opposition(
        self, rng: np.random.Generator
    ) -> List[PlayerSkills]:
        """Draw nine opposition players from the population."""
        opponents = [
            draw_population_player(self.priors, rng, name=f"opp_{i}")
            for i in range(9)
        ]
        if self.outcomes is not None:
            # A new ground and new opponents for every simulated match.
            self.outcomes.begin_game(rng)
            for player in opponents:
                self.outcomes.register(player.name, rng)
        return opponents

    def evaluate_batting(
        self,
        order: Sequence[str],
        retire_at_balls: Optional[int] = None,
    ) -> np.ndarray:
        """Simulate our batting innings against population bowling.

        Args:
            order: Batting order to evaluate.
            retire_at_balls: Retirement policy for our batters.

        Returns:
            Array of our simulated innings totals.
        """
        rng = self._candidate_rng("bat", tuple(order), retire_at_balls)
        totals = np.empty(self.n_sims, dtype=int)
        for sim in range(self.n_sims):
            opponents = self._draw_opposition(rng)
            # Opposition bowling: best drawn bowlers get the top
            # tiers; rotation is a random permutation each match.
            tier_list = sorted(
                self.conditions.bowling_allocations[9], reverse=True
            )
            by_strength = sorted(
                opponents, key=lambda p: p.p_wicket / p.econ, reverse=True
            )
            opp_allocation = {
                p.name: tier for p, tier in zip(by_strength, tier_list)
            }
            opp_rotation = [p.name for p in opponents]
            rng.shuffle(opp_rotation)
            result = simulate_innings(
                batting_skills=[self.skills[n] for n in order],
                bowling_skills={p.name: p for p in opponents},
                bowling_rotation=opp_rotation,
                bowling_allocation=opp_allocation,
                rng=rng,
                conditions=self.conditions,
                population_econ=self.priors.mu_econ,
                retire_at_balls=retire_at_balls,
                outcomes=self.outcomes,
            )
            totals[sim] = result.runs
        return totals

    def evaluate_bowling(
        self,
        rotation: Sequence[str],
        allocation: Dict[str, int],
    ) -> np.ndarray:
        """Simulate the opposition innings against our bowling.

        Args:
            rotation: Our round-robin bowling order.
            allocation: Our player name -> overs.

        Returns:
            Array of opposition simulated innings totals.
        """
        rng = self._candidate_rng("bowl", tuple(rotation))
        totals = np.empty(self.n_sims, dtype=int)
        for sim in range(self.n_sims):
            opponents = self._draw_opposition(rng)
            opp_rotation = [p.name for p in opponents]
            result = simulate_innings(
                batting_skills=opponents,
                bowling_skills=self.skills,
                bowling_rotation=rotation,
                bowling_allocation=allocation,
                rng=rng,
                conditions=self.conditions,
                population_econ=self.priors.mu_econ,
                outcomes=self.outcomes,
            )
            totals[sim] = result.runs
        return totals

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def _hill_climb(
        self,
        start: List[str],
        score_fn,
        max_evaluations: int,
    ) -> Tuple[List[str], List[Tuple[List[str], float]]]:
        """Adjacent-swap hill climb over a permutation.

        Args:
            start: Initial permutation.
            score_fn: Mapping from permutation to score (maximised).
            max_evaluations: Evaluation budget.

        Returns:
            (best permutation, evaluation history).
        """
        current = list(start)
        best_score = score_fn(current)
        history: List[Tuple[List[str], float]] = [
            (list(current), best_score)
        ]
        evaluations = 1
        improved = True
        while improved and evaluations < max_evaluations:
            improved = False
            for i in range(len(current) - 1):
                if evaluations >= max_evaluations:
                    break
                candidate = list(current)
                candidate[i], candidate[i + 1] = (
                    candidate[i + 1],
                    candidate[i],
                )
                score = score_fn(candidate)
                evaluations += 1
                history.append((candidate, score))
                if score > best_score + 1e-9:
                    current = candidate
                    best_score = score
                    improved = True
                    break
        return current, history

    def optimise(
        self,
        batting_evaluations: int = 60,
        bowling_evaluations: int = 60,
        retire_at_balls: Optional[int] = None,
    ) -> OptimizerResult:
        """Run the decomposed lineup optimisation.

        Args:
            batting_evaluations: Candidate batting orders to test.
            bowling_evaluations: Candidate bowling configs to test.
            retire_at_balls: Fixed retirement policy for our batters.

        Returns:
            The optimised recommendation with match outcome stats.
        """
        names = list(self.skills)

        # --- Batting order search (maximise our total) ---
        seed_order = self.seed_batting_order()
        batting_history: List[Tuple[List[str], float]] = []

        def batting_score(order: List[str]) -> float:
            totals = self.evaluate_batting(order, retire_at_balls)
            mean_total = float(np.mean(totals))
            batting_history.append((list(order), mean_total))
            return mean_total

        best_order, _ = self._hill_climb(
            seed_order, batting_score, batting_evaluations
        )
        # Also test the reverse order as a structural alternative.
        reverse = list(reversed(seed_order))
        reverse_score = batting_score(reverse)
        best_totals = self.evaluate_batting(best_order, retire_at_balls)
        reverse_totals = self.evaluate_batting(reverse, retire_at_balls)
        if np.mean(reverse_totals) > np.mean(best_totals):
            best_order = reverse
            best_totals = reverse_totals
        LOGGER.info(
            "Batting search done: order=%s mean=%.1f (reverse=%.1f)",
            best_order,
            float(np.mean(best_totals)),
            reverse_score,
        )

        # --- Bowling search (minimise their total) ---
        seed_rotation, seed_allocation = self.seed_bowling()
        bowling_history: List[Tuple[List[str], float]] = []

        def bowling_score(rotation: List[str]) -> float:
            totals = self.evaluate_bowling(rotation, seed_allocation)
            mean_conceded = float(np.mean(totals))
            bowling_history.append((list(rotation), mean_conceded))
            return -mean_conceded

        best_rotation, _ = self._hill_climb(
            seed_rotation, bowling_score, bowling_evaluations
        )

        # Tier reassignment: try promoting each non-keeper into the
        # top bowling tier in place of the weakest current holder.
        best_rotation, best_allocation = self._search_tiers(
            best_rotation, seed_allocation, bowling_history
        )
        their_totals = self.evaluate_bowling(
            best_rotation, best_allocation
        )
        LOGGER.info(
            "Bowling search done: rotation=%s allocation=%s mean_conceded=%.1f",
            best_rotation,
            best_allocation,
            float(np.mean(their_totals)),
        )

        # --- Combine into match outcome ---
        n = min(len(best_totals), len(their_totals))
        run_diff = best_totals[:n].astype(float) - their_totals[
            :n
        ].astype(float)
        win_prob = float(np.mean(best_totals[:n] > their_totals[:n]))
        tie_prob = float(np.mean(best_totals[:n] == their_totals[:n]))

        return OptimizerResult(
            batting_order=best_order,
            bowling_rotation=best_rotation,
            bowling_allocation=best_allocation,
            retire_at_balls=retire_at_balls,
            our_totals=best_totals,
            their_totals=their_totals,
            win_probability=win_prob,
            tie_probability=tie_prob,
            expected_run_diff=float(np.mean(run_diff)),
            run_diff_ci=(
                float(np.percentile(run_diff, 5)),
                float(np.percentile(run_diff, 95)),
            ),
            batting_history=batting_history,
            bowling_history=bowling_history,
        )

    def _search_tiers(
        self,
        rotation: List[str],
        allocation: Dict[str, int],
        history: List[Tuple[List[str], float]],
    ) -> Tuple[List[str], Dict[str, int]]:
        """Try swapping players between bowling tiers.

        Keeps the mandatory tier multiset intact (participation
        rules) while letting the strongest bowlers hold the top
        tier.

        Args:
            rotation: Current rotation order.
            allocation: Current allocation to adjust.
            history: Bowling evaluation history to append to.

        Returns:
            (best rotation, best allocation).
        """
        best_rotation = list(rotation)
        best_allocation = dict(allocation)
        best_mean = float(
            np.mean(self.evaluate_bowling(best_rotation, best_allocation))
        )

        tiers = sorted(set(allocation.values()), reverse=True)
        if len(tiers) < 2:
            return best_rotation, best_allocation

        top_tier, next_tier = tiers[0], tiers[1]
        top_holders = [
            n for n, t in best_allocation.items() if t == top_tier
        ]
        candidates = [
            n
            for n, t in best_allocation.items()
            if t == next_tier and not self.skills[n].is_wicketkeeper
        ]

        for promoted in candidates:
            for demoted in top_holders:
                if self.skills[demoted].is_wicketkeeper:
                    continue
                trial_allocation = dict(best_allocation)
                trial_allocation[promoted] = top_tier
                trial_allocation[demoted] = next_tier
                trial_mean = float(
                    np.mean(
                        self.evaluate_bowling(
                            best_rotation, trial_allocation
                        )
                    )
                )
                history.append((list(best_rotation), trial_mean))
                if trial_mean < best_mean - 1e-9:
                    best_mean = trial_mean
                    best_allocation = trial_allocation
                    top_holders = [
                        n
                        for n, t in best_allocation.items()
                        if t == top_tier
                    ]
                    break

        return best_rotation, best_allocation
