"""Check the U10 simulator against real match totals.

Replays every completed U10 game from the cached PlayHQ data: our
batters in the order they batted and our bowlers in the order and overs
they bowled. It then compares the simulated distribution of each
quantity with what was recorded: runs off the bat, wickets, and final
team totals (which include 4 penalty runs per dismissal taken, Rule
16.10(vi)).

Three ways to drive the ball outcomes:

* default: the per-player rates from ``fit_model.py`` (hand-built
  combination rules);
* ``--ball-posterior``: the joint batter-by-bowler model from
  ``fit_ball_model.py``.

The opposition is either drawn from the grade population
(``--opposition population``, what the optimiser does) or replayed with
its own fitted effects (``--opposition real``, which tests the
mechanism alone).

This is an in-sample posterior predictive check, not a forecast: the fitted
rates have already seen these games.

Examples:
    python scripts/check_u10_totals.py --team-id 75cdae66 --team-id fafdb4c4
    python scripts/check_u10_totals.py --team-id 75cdae66 --team-id fafdb4c4 \
        --ball-posterior data/outputs/ball_posterior.nc --opposition real
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import arviz as az
import numpy as np
import pandas as pd

from junior_cricket.ball_model import JointOutcomes
from junior_cricket.logging_setup import setup_logging
from junior_cricket.playhq_parse import Appearance, parse_scorecard
from junior_cricket.playhq_scorebook import PlayerAliases, completed_games
from junior_cricket.posterior import priors_from_posterior, skills_from_posterior
from junior_cricket.replay import key_to_alias, stub
from junior_cricket.rules_u10 import U10
from junior_cricket.simulator import (
    PlayerSkills,
    draw_population_player,
    simulate_innings_u10,
    team_total,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "raw" / "playhq"
OUTPUT_DIR = REPO_ROOT / "data" / "outputs"
LOGGER_NAME = "check_u10_totals"

QUANTITIES = {
    "our_runs": "Our runs off the bat (sundries included)",
    "their_runs": "Opposition runs off the bat",
    "our_wickets": "Wickets we took (opposition dismissals)",
    "their_wickets": "Wickets they took (our dismissals)",
    "our_total": "Our final total (incl. 4 per wicket we took)",
    "their_total": "Opposition final total",
    "margin": "Margin (our total minus theirs)",
}


def lineup(
    lines, aliases: Dict[str, str], skills: Dict[str, PlayerSkills]
) -> Optional[Dict[str, object]]:
    """Batting order, bowling rotation and overs as played, with skills."""
    def skill(a: Appearance) -> Optional[PlayerSkills]:
        alias = aliases.get(PlayerAliases.key_for(a))
        return skills.get(alias) if alias else None

    batters = sorted((a for a in lines if a.batting and a.batting.balls_faced > 0),
                     key=lambda a: a.batting.order)
    bowlers = sorted((a for a in lines if a.bowling and a.bowling.balls > 0),
                     key=lambda a: a.bowling.order)
    bat_skills = [skill(a) for a in batters]
    bowl_skills = [skill(a) for a in bowlers]
    if any(s is None for s in bat_skills + bowl_skills):
        return None
    return {
        "bat": bat_skills,
        "bowl": {s.name: s for s in bowl_skills},
        "rotation": [s.name for s in bowl_skills],
        "allocation": {s.name: -(-a.bowling.balls // 6)
                       for s, a in zip(bowl_skills, bowlers)},
    }


def legal(lu: Optional[Dict[str, object]]) -> bool:
    """A full-length innings by a team size the rules allow.

    The bowling split itself is not required to match the rule table: real
    teams sometimes bowl a different split, and a replay should use what
    was actually bowled.
    """
    if lu is None:
        return False
    return (sum(lu["allocation"].values()) == U10.overs_per_innings
            and len(lu["bat"]) in U10.batting_ball_allotments)


def simulate_game(
    ours, n_opp_bat, n_opp_bowl, priors, rng, n_sims, outcomes=None, theirs=None
) -> Dict[str, np.ndarray]:
    """Simulate one game ``n_sims`` times; return arrays per quantity."""
    out = {q: np.empty(n_sims) for q in QUANTITIES}
    tiers = sorted(U10.bowling_allocations.get(n_opp_bowl, []), reverse=True)
    for k in range(n_sims):
        if outcomes is not None:
            outcomes.begin_game(rng)
        if theirs is not None:
            opp_bat, opp_bowl = theirs["bat"], theirs["bowl"]
            rotation, allocation = theirs["rotation"], theirs["allocation"]
        else:
            n = max(n_opp_bat, n_opp_bowl)
            if outcomes is not None:
                opp = [stub(f"opp_{i}") for i in range(n)]
                for p in opp:
                    outcomes.register(p.name, rng)
            else:
                opp = [draw_population_player(priors, rng, name=f"opp_{i}")
                       for i in range(n)]
                for p in opp:
                    p.p_extra = 0.0        # PlayHQ U10 records no separate extras
            bowlers = opp[:n_opp_bowl]
            allocation = {p.name: t for p, t in zip(rng.permutation(bowlers), tiers)}
            rotation = [p.name for p in rng.permutation(bowlers)]
            opp_bat = list(rng.permutation(opp[:n_opp_bat]))
            opp_bowl = {p.name: p for p in bowlers}
        ours_inn = simulate_innings_u10(
            ours["bat"], opp_bowl, rotation, allocation, rng,
            population_econ=priors.mu_econ, outcomes=outcomes,
            enforce_bowling_table=False)
        theirs_inn = simulate_innings_u10(
            opp_bat, ours["bowl"], ours["rotation"], ours["allocation"], rng,
            population_econ=priors.mu_econ, outcomes=outcomes,
            enforce_bowling_table=False)
        our_total, their_total = team_total(ours_inn, theirs_inn), team_total(theirs_inn, ours_inn)
        out["our_runs"][k], out["their_runs"][k] = ours_inn.runs, theirs_inn.runs
        out["our_wickets"][k], out["their_wickets"][k] = theirs_inn.wickets, ours_inn.wickets
        out["our_total"][k], out["their_total"][k] = our_total, their_total
        out["margin"][k] = our_total - their_total
    return out


def run_check(
    posterior: str,
    team_ids: List[str],
    sims: int,
    seed: int,
    ball_posterior: Optional[str] = None,
    opposition: str = "population",
) -> "tuple[pd.DataFrame, List[str]]":
    """Replay the real games and compare with the recorded values.

    Args:
        posterior: Per-player posterior (netcdf); supplies priors and,
            without ``ball_posterior``, the skills.
        team_ids: PlayHQ team IDs whose cached fixtures to replay.
        sims: Simulations per game.
        seed: Random seed.
        ball_posterior: Joint-model posterior; when given the joint
            per-ball rule drives the innings.
        opposition: ``population`` to draw opponents from the grade, or
            ``real`` to replay their fitted effects (joint model only).

    Returns:
        (one row per game and quantity, notes on skipped games).
    """
    idata = az.from_netcdf(posterior)
    priors = priors_from_posterior(idata)
    rng = np.random.default_rng(seed)
    aliases = key_to_alias(RAW_DIR / "player_map.csv")
    outcomes = None
    if ball_posterior:
        outcomes = JointOutcomes.from_posterior(az.from_netcdf(ball_posterior))
        skills = {name: stub(name) for name in outcomes.bat}
    else:
        skills = skills_from_posterior(idata, p_extra=0.0)
    opp_aliases = None
    if opposition == "real":
        if outcomes is None:
            raise ValueError("--opposition real needs --ball-posterior")
        opp_aliases = key_to_alias(RAW_DIR / "opposition_map.csv")

    rows: List[Dict[str, object]] = []
    skipped: List[str] = []
    for team_id in team_ids:
        fixture = json.loads((RAW_DIR / "fixtures" / f"team_{team_id}.json").read_text())
        for game in completed_games(fixture):
            path = RAW_DIR / "games" / game.game_id / "scorecard.json"
            if not path.exists() or game.our_total is None or game.their_total is None:
                skipped.append(f"{game.game_id}: no scorecard or totals")
                continue
            card = parse_scorecard(json.loads(path.read_text()))
            ours_lines, theirs_lines = card.side(game.our_side), card.side(
                "AWAY" if game.our_side == "HOME" else "HOME")
            ours = lineup(ours_lines, aliases, skills)
            theirs = lineup(theirs_lines, opp_aliases, skills) if opp_aliases else None
            n_bat = sum(1 for a in theirs_lines if a.batting and a.batting.balls_faced > 0)
            n_bowl = sum(1 for a in theirs_lines if a.bowling and a.bowling.balls > 0)
            if (not legal(ours) or (opp_aliases and not legal(theirs))
                    or n_bowl not in U10.bowling_allocations):
                skipped.append(f"{game.game_id}: shortened game or unmatched lineup")
                continue
            simulated = simulate_game(ours, n_bat, n_bowl, priors, rng, sims,
                                      outcomes=outcomes, theirs=theirs)
            our_bat_runs = sum(a.batting.runs for a in ours_lines if a.batting)
            their_bat_runs = sum(a.batting.runs for a in theirs_lines if a.batting)
            observed = {
                "our_runs": our_bat_runs,
                "their_runs": their_bat_runs,
                "our_wickets": (game.our_total - our_bat_runs) / 4,
                "their_wickets": (game.their_total - their_bat_runs) / 4,
                "our_total": game.our_total,
                "their_total": game.their_total,
                "margin": game.our_total - game.their_total,
            }
            for q, values in simulated.items():
                lo, hi = np.percentile(values, [5, 95])
                rows.append({
                    "game": game.game_id, "quantity": q, "observed": observed[q],
                    "sim_mean": values.mean(), "sim_sd": values.std(),
                    "lo": lo, "hi": hi,
                    "inside90": bool(lo <= observed[q] <= hi),
                    "pit": float(np.mean(values <= observed[q])),
                })
    return pd.DataFrame(rows), skipped


def report(frame: pd.DataFrame, sims: int, title: str) -> List[str]:
    """Markdown lines summarising one check."""
    lines = [
        f"# {title}", "",
        f"{frame['game'].nunique()} games replayed, {sims} simulations each. "
        "In-sample: the fitted rates have seen these games. "
        "`inside 90%` is the share of games whose recorded value fell inside "
        "the simulated 90% interval (about 0.9 is calibrated).", "",
        "| quantity | observed mean | simulated mean | sim / observed | inside 90% |",
        "| --- | --- | --- | --- | --- |",
    ]
    for q, label in QUANTITIES.items():
        part = frame[frame["quantity"] == q]
        obs, sim = part["observed"].mean(), part["sim_mean"].mean()
        ratio = f"{sim / obs:.2f}" if abs(obs) > 1e-9 else ""
        lines.append(f"| {label} | {obs:.1f} | {sim:.1f} | {ratio} | "
                     f"{part['inside90'].mean():.2f} |")
    return lines


def main() -> None:
    """Replay the real games and write the comparison."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--team-id", action="append", required=True)
    parser.add_argument("--posterior", default=str(OUTPUT_DIR / "posterior_model.nc"))
    parser.add_argument("--ball-posterior", default=None,
                        help="Joint-model posterior from fit_ball_model.py")
    parser.add_argument("--opposition", choices=["population", "real"],
                        default="population")
    parser.add_argument("--sims", type=int, default=400)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    logger = setup_logging(LOGGER_NAME, OUTPUT_DIR)
    logger.info("Input arguments: %s", vars(args))
    frame, skipped = run_check(args.posterior, args.team_id, args.sims, args.seed,
                               args.ball_posterior, args.opposition)
    tag = ("joint_" + args.opposition) if args.ball_posterior else "player"
    frame.to_csv(OUTPUT_DIR / f"u10_totals_check_{tag}.csv", index=False)
    for note in skipped:
        logger.warning("Skipped %s", note)
    title = ("U10 simulator check against real totals ("
             + ("joint batter-by-bowler model, " + args.opposition + " opposition"
                if args.ball_posterior else "per-player rates") + ")")
    lines = report(frame, args.sims, title)
    (OUTPUT_DIR / f"u10_totals_check_{tag}.md").write_text("\n".join(lines), encoding="utf-8")
    logger.info("Report: %s", OUTPUT_DIR / f"u10_totals_check_{tag}.md")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
