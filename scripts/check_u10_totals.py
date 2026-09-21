"""Check the U10 simulator against real match totals.

Replays every completed U10 game from the cached PlayHQ data: our
batters in the order they batted and our bowlers in the order and overs
they bowled, using their fitted rates, against opposition drawn from the
grade population (so the fit should include the opposition rows). It then
compares the simulated distribution of each quantity with what was
recorded: runs off the bat, wickets, and final team totals (which include
4 penalty runs per dismissal taken, Rule 16.10(vi)).

This is an in-sample posterior predictive check of the simulator, not a
forecast: the fitted rates have already seen these games.

Example:
    python scripts/check_u10_totals.py --team-id 75cdae66 --team-id fafdb4c4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import arviz as az
import numpy as np
import pandas as pd

from junior_cricket.logging_setup import setup_logging
from junior_cricket.playhq_parse import Appearance, parse_scorecard
from junior_cricket.playhq_scorebook import PlayerAliases, completed_games
from junior_cricket.posterior import priors_from_posterior, skills_from_posterior
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


def key_to_alias(path: Path) -> Dict[str, str]:
    """Map PlayHQ profile keys to our pseudonyms."""
    frame = pd.read_csv(path)
    return dict(zip(frame["key"], frame["alias"]))


def our_lineup(
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
        "n_opposition": None,
    }


def simulate_game(lineup, n_opp_bat, n_opp_bowl, priors, rng, n_sims) -> Dict[str, np.ndarray]:
    """Simulate one game ``n_sims`` times; return arrays per quantity."""
    out = {q: np.empty(n_sims) for q in QUANTITIES}
    tiers = sorted(U10.bowling_allocations[n_opp_bowl], reverse=True)
    for k in range(n_sims):
        opp = [draw_population_player(priors, rng, name=f"opp_{i}")
               for i in range(max(n_opp_bat, n_opp_bowl))]
        for p in opp:
            p.p_extra = 0.0            # PlayHQ U10 records no separate extras
        bowlers = opp[:n_opp_bowl]
        allocation = {p.name: t for p, t in zip(rng.permutation(bowlers), tiers)}
        rotation = [p.name for p in rng.permutation(bowlers)]
        batters = list(rng.permutation(opp[:n_opp_bat]))
        ours = simulate_innings_u10(
            lineup["bat"], {p.name: p for p in bowlers}, rotation, allocation,
            rng, population_econ=priors.mu_econ)
        theirs = simulate_innings_u10(
            batters, lineup["bowl"], lineup["rotation"], lineup["allocation"],
            rng, population_econ=priors.mu_econ)
        our_total, their_total = team_total(ours, theirs), team_total(theirs, ours)
        out["our_runs"][k], out["their_runs"][k] = ours.runs, theirs.runs
        out["our_wickets"][k], out["their_wickets"][k] = theirs.wickets, ours.wickets
        out["our_total"][k], out["their_total"][k] = our_total, their_total
        out["margin"][k] = our_total - their_total
    return out


def run_check(
    posterior: str, team_ids: List[str], sims: int, seed: int
) -> "tuple[pd.DataFrame, List[str]]":
    """Replay the real games and compare with the recorded values.

    Args:
        posterior: Path to the fitted posterior (netcdf).
        team_ids: PlayHQ team IDs whose cached fixtures to replay.
        sims: Simulations per game.
        seed: Random seed.

    Returns:
        (one row per game and quantity, notes on skipped games).
    """
    idata = az.from_netcdf(posterior)
    priors = priors_from_posterior(idata)
    skills = skills_from_posterior(idata, p_extra=0.0)
    aliases = key_to_alias(RAW_DIR / "player_map.csv")
    rng = np.random.default_rng(seed)

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
            ours, theirs = card.side(game.our_side), card.side(
                "AWAY" if game.our_side == "HOME" else "HOME")
            lineup = our_lineup(ours, aliases, skills)
            n_bat = sum(1 for a in theirs if a.batting and a.batting.balls_faced > 0)
            n_bowl = sum(1 for a in theirs if a.bowling and a.bowling.balls > 0)
            alloc = sorted(lineup["allocation"].values(), reverse=True) if lineup else []
            our_bowlers = len(alloc)
            if (lineup is None
                    or sorted(U10.bowling_allocations.get(our_bowlers, [])) != sorted(alloc)
                    or n_bowl not in U10.bowling_allocations
                    or len(lineup["bat"]) not in U10.batting_ball_allotments):
                skipped.append(f"{game.game_id}: shortened game or unmatched lineup")
                continue
            simulated = simulate_game(lineup, n_bat, n_bowl, priors, rng, sims)
            our_bat_runs = sum(a.batting.runs for a in ours if a.batting)
            their_bat_runs = sum(a.batting.runs for a in theirs if a.batting)
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


def main() -> None:
    """Replay the real games and write the comparison."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--team-id", action="append", required=True)
    parser.add_argument("--posterior", default=str(OUTPUT_DIR / "posterior_model.nc"))
    parser.add_argument("--sims", type=int, default=400)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    logger = setup_logging(LOGGER_NAME, OUTPUT_DIR)
    logger.info("Input arguments: %s", vars(args))
    frame, skipped = run_check(args.posterior, args.team_id, args.sims, args.seed)
    frame.to_csv(OUTPUT_DIR / "u10_totals_check.csv", index=False)
    for note in skipped:
        logger.warning("Skipped %s", note)

    lines = [
        "# U10 simulator check against real totals", "",
        f"{frame['game'].nunique()} games replayed, {args.sims} simulations each. "
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
    (OUTPUT_DIR / "u10_totals_check.md").write_text("\n".join(lines), encoding="utf-8")
    logger.info("Report: %s", OUTPUT_DIR / "u10_totals_check.md")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
