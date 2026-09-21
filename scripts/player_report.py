"""Plain-language player profiles from the joint model.

For each squad player: what a typical 13-ball batting stint is worth, and
what an over of their bowling is worth, against an average opponent, with
the uncertainty and how much data sits behind it. Reads the joint
posterior from ``fit_ball_model.py``.

The report uses aliases. ``--named`` also writes a copy with real names
for your own use; that file goes under ``data/outputs/`` (gitignored) and
must not be committed or shared.

Example:
    python scripts/player_report.py --squad P01,P03,P04,P05,P06,P07,P08,P09,P10 --named
"""

from __future__ import annotations

import argparse
from pathlib import Path

import arviz as az
import pandas as pd

from junior_cricket.ball_model import player_profiles
from junior_cricket.logging_setup import setup_logging
from junior_cricket.playhq_scorebook import alias_names, translate_aliases

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "data" / "outputs"
RAW_DIR = REPO_ROOT / "data" / "raw" / "playhq"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
LOGGER_NAME = "player_report"


def main() -> None:
    """Write the profile report."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--squad", required=True, help="Comma-separated aliases")
    parser.add_argument("--ball-posterior", default=str(OUTPUT_DIR / "ball_posterior.nc"))
    parser.add_argument("--named", action="store_true",
                        help="Also write a copy with real names (private file)")
    args = parser.parse_args()
    logger = setup_logging(LOGGER_NAME, OUTPUT_DIR)
    logger.info("Input arguments: %s", vars(args))

    squad = [s.strip() for s in args.squad.split(",") if s.strip()]
    profiles = player_profiles(az.from_netcdf(args.ball_posterior), squad)
    batting = pd.read_csv(PROCESSED_DIR / "playhq_batting.csv")
    bowling = pd.read_csv(PROCESSED_DIR / "playhq_bowling.csv")
    balls = batting.groupby("player_name")["balls_faced"].sum()
    bowled = bowling.groupby("player_name")["balls_bowled"].sum()
    profiles["balls_faced"] = profiles["player"].map(balls).fillna(0).astype(int)
    profiles["balls_bowled"] = profiles["player"].map(bowled).fillna(0).astype(int)
    profiles = profiles.sort_values("net", ascending=False)

    lines = [
        "# Player profiles (joint batter-by-bowler model)", "",
        "Figures are against an average opponent on an average ground, from the "
        "2025/26 U10 games. **Batting** is one 13-ball stint (a U10 batter's "
        "allotment). **Net** is runs scored minus 4 x dismissals, because under "
        "U10 Rule 16.10(vi) each dismissal gifts the other side 4 runs. "
        "**Bowling** is one over. Brackets are 90% intervals; wide brackets "
        "mean little data.", "",
        "## Batting", "",
        "| player | balls faced | boundaries | runs | dismissals | NET runs | vs average batter |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in profiles.itertuples():
        lines.append(
            f"| {r.player} | {r.balls_faced} | {r.boundaries:.1f} | {r.runs:.1f} | "
            f"{r.dismissals:.2f} | {r.net:.1f} [{r.net_lo:.1f}, {r.net_hi:.1f}] | "
            f"{r.net_vs_average:+.1f} |")
    lines += ["", "## Bowling", "",
              "| player | balls bowled | wickets per over | runs per over | edge over an average bowler (runs per over) |",
              "| --- | --- | --- | --- | --- |"]
    for r in profiles.sort_values("bowling_edge", ascending=False).itertuples():
        lines.append(
            f"| {r.player} | {r.balls_bowled} | {r.wickets:.2f} [{r.wickets_lo:.2f}, "
            f"{r.wickets_hi:.2f}] | {r.runs_conceded:.2f} | {r.bowling_edge:+.2f} |")
    lines += ["", "Bowling edge counts each wicket as 4 runs plus the runs saved. "
              "Bowlers differ less than batters, so read small gaps as noise."]
    text = "\n".join(lines) + "\n"
    (OUTPUT_DIR / "player_report.md").write_text(text, encoding="utf-8")
    logger.info("Report: %s", OUTPUT_DIR / "player_report.md")
    if args.named:
        names = alias_names(RAW_DIR / "player_map.csv", RAW_DIR / "opposition_map.csv")
        named = OUTPUT_DIR / "player_report_named.md"
        named.write_text(translate_aliases(text, names), encoding="utf-8")
        logger.info("Named copy (private, gitignored): %s", named)
    print(text)


if __name__ == "__main__":
    main()
