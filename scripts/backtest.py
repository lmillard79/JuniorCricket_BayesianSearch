"""Backtest the skill model on real games (plan section 6).

Two tests, both fitted only on earlier games and scored on later ones:

* the period split from the plan: train before Christmas, predict
  post-Christmas;
* a rolling-origin test over blocks of games, which gives several test
  windows instead of one noisy one.

For five rate families the model is compared with two plug-in
baselines: everyone at the pooled rate, and each player's raw rate.
Writes ``backtest_scores.csv`` and ``backtest_report.md`` under
``data/outputs/``. Player labels are aliases.

Example:
    python scripts/backtest.py --batting data/processed/playhq_batting.csv \
        --bowling data/processed/playhq_bowling.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd

from junior_cricket.backtest import game_splits, run_split, summarise
from junior_cricket.data_loader import load_batting_csv, load_bowling_csv
from junior_cricket.logging_setup import setup_logging

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "data" / "outputs"
LOGGER_NAME = "backtest"

FAMILY_LABELS = {
    "out": "Dismissal hazard (per ball faced)",
    "bound": "Boundary rate (per ball faced)",
    "wicket": "Bowler wicket rate (per ball)",
    "srr": "Non-boundary scoring rate (runs per ball)",
    "econ": "Bowler economy (runs off the bat per ball)",
}


def markdown_table(frame: pd.DataFrame, floats: int = 2) -> str:
    """Render a DataFrame as a GitHub-flavoured markdown table."""
    def cell(value) -> str:
        if isinstance(value, float):
            return "" if pd.isna(value) else f"{value:.{floats}f}"
        return str(value)

    header = "| " + " | ".join(frame.columns) + " |"
    rule = "| " + " | ".join("---" for _ in frame.columns) + " |"
    body = ["| " + " | ".join(cell(v) for v in row) + " |"
            for row in frame.itertuples(index=False)]
    return "\n".join([header, rule, *body])


def verdict(summary: pd.DataFrame) -> pd.DataFrame:
    """One line per family: did the model beat each baseline?"""
    rows = []
    model = summary[summary["method"] == "model"].set_index("family")
    for family, row in model.iterrows():
        rows.append({
            "family": FAMILY_LABELS[family],
            "vs pooled": f"{row['model_minus_pooled']:+.2f}",
            "vs raw": f"{row['model_minus_raw']:+.2f}",
            "corr (pred vs actual)": row["corr"],
            "90% interval coverage": row["coverage90"],
        })
    return pd.DataFrame(rows)


def main() -> None:
    """Run both backtests and write the report."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--batting", required=True)
    parser.add_argument("--bowling", required=True)
    parser.add_argument("--split-date", default="2026-01-01",
                        help="First post-Christmas date for the period split")
    parser.add_argument("--min-train", type=int, default=6,
                        help="Games in the first rolling training window")
    parser.add_argument("--test-games", type=int, default=3,
                        help="Games per rolling test block")
    parser.add_argument("--draws", type=int, default=1000)
    args = parser.parse_args()

    logger = setup_logging(LOGGER_NAME, OUTPUT_DIR)
    logger.info("Input arguments: %s", vars(args))
    batting = load_batting_csv(Path(args.batting))
    bowling = load_bowling_csv(Path(args.bowling))
    players = sorted(set(batting["player_name"]) | set(bowling["player_name"]))
    dates = sorted({d.strftime("%Y-%m-%d") for d in
                    pd.concat([batting["date"], bowling["date"]])})
    logger.info("%d players, %d game dates (%s to %s)",
                len(players), len(dates), dates[0], dates[-1])

    before = [d for d in dates if d < args.split_date]
    after = [d for d in dates if d >= args.split_date]
    frames: List[pd.DataFrame] = []
    period = pd.DataFrame()
    if before and after:
        logger.info("Period split: train %s..%s, test %s..%s",
                    before[0], before[-1], after[0], after[-1])
        period = run_split(batting, bowling, players,
                           (before[0], before[-1]), (after[0], after[-1]),
                           seed=1, draws=args.draws)
        period["test_type"] = "period split"
        frames.append(period)

    rolling: List[pd.DataFrame] = []
    for i, (train, test) in enumerate(
        game_splits(dates, args.min_train, args.test_games), start=1
    ):
        logger.info("Rolling split %d: train %s..%s, test %s..%s",
                    i, *train, *test)
        part = run_split(batting, bowling, players, train, test,
                         seed=10 + i, draws=args.draws)
        part["test_type"] = "rolling"
        rolling.append(part)
    frames.extend(rolling)

    scores = pd.concat(frames, ignore_index=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    scores.to_csv(OUTPUT_DIR / "backtest_scores.csv", index=False)

    lines = [
        "# Backtest of the skill model on real U10 games",
        "",
        "Each test fits the model on earlier games only, then scores the "
        "counts a later block of games actually produced (log predictive "
        "density, higher is better). The two baselines use only the "
        "training games: **pooled** treats every player as identical; "
        "**raw** trusts each player's own record with no pooling.",
        "",
        "`vs pooled` and `vs raw` are the model's total log-score minus the "
        "baseline's: positive means the model predicted the held-out games "
        "better. `corr` is the correlation between predicted and actual "
        "rates across players; `coverage` is the share of players whose "
        "actual count fell inside the model's 90% predictive interval "
        "(about 0.9 is well calibrated).",
        "",
    ]
    for label, test_type in (("Period split (plan section 6)", "period split"),
                             ("Rolling game blocks", "rolling")):
        part = scores[scores["test_type"] == test_type]
        if part.empty:
            continue
        summary = summarise(part)
        n_splits = part[["train", "test"]].drop_duplicates().shape[0]
        lines += [f"## {label}", "",
                  f"{n_splits} test window(s); players scored per family vary "
                  "with who faced or bowled a ball.", "",
                  markdown_table(verdict(summary)), ""]
    (OUTPUT_DIR / "backtest_report.md").write_text("\n".join(lines), encoding="utf-8")
    logger.info("Report: %s", OUTPUT_DIR / "backtest_report.md")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
