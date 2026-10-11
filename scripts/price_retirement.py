"""Price the retirement options for U11: runs, time in the heat, and who gets a bat.

Rule 17.9(iv) and (v): a batter may retire not out from 25 balls (wides and no-balls count)
and must retire at 35, and a retired batter may resume "once all other players have been
dismissed or retired". So the choices a coach really has are who retires and when, with the
retirees returning at the end. This plays the conventional order (strongest to weakest) with
each such choice through the same simulated worlds and reports, against everyone batting to 35:

* the runs it costs;
* how long the busiest batter's first stint is, from walking in to being out or retiring (average,
  and the 90th percentile, which is a batter who lasts), and their total time out there including a
  second stint after returning, as minutes of a 90-minute innings (a batter is "at the crease"
  whenever they are at either end);
* how often the weakest batter gets in at all, and how often all nine bat;
* the balls the best and the weakest batters get.

Retiring below 25 balls is not allowed by the booklet (it would need both coaches to agree,
for example in the heat; the booklet allows extra drinks breaks in oppressive weather), and
neither is bringing a banked batter back ahead of a fresh one. Both are shown, flagged, as
what-ifs.

Two worlds are run: the U10 scoring rates as fitted, and a U11-like world with the shifts seen
in the first U11 game (see docs/BATTING_STRATEGY_METHOD.md, section 4b).

Example:
    python scripts/price_retirement.py
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SQUAD = "P01,P03,P04,P05,P06,P07,P08,P09,P10"
LOGGER_NAME = "price_retirement"
INNINGS_MINUTES = 90          # 25 overs in 90 minutes (rule 17.2(i))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--ball-posterior", default=None,
                        help="Default: data/<data-dir>/outputs/ball_posterior.nc")
    parser.add_argument("--squad", default=DEFAULT_SQUAD, help="Comma-separated batter aliases")
    parser.add_argument("--worlds", type=int, default=4000)
    parser.add_argument("--reps", type=int, default=2)
    parser.add_argument("--pool", type=int, default=300)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--data-dir", default=None, help="Use data/<name>/ instead of data/")
    parser.add_argument("--out", default=None, help="Default: data/<data-dir>/outputs/retirement_options")
    args = parser.parse_args()

    # Heavy imports stay inside main: worker processes on Windows re-import this module.
    import arviz as az
    import pandas as pd

    from junior_cricket import replay as R
    from junior_cricket import strategies as S
    from junior_cricket.ball_model import JointOutcomes
    from junior_cricket.logging_setup import setup_logging

    base_dir = REPO_ROOT / "data" / args.data_dir if args.data_dir else REPO_ROOT / "data"
    out = Path(args.out) if args.out else base_dir / "outputs" / "retirement_options"
    out.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(LOGGER_NAME, out)
    logger.info("Input arguments: %s", vars(args))

    squad = [p.strip() for p in args.squad.split(",")]
    idata = az.from_netcdf(args.ball_posterior or base_dir / "outputs" / "ball_posterior.nc")
    pool = JointOutcomes.pool(idata, args.pool, np.random.default_rng(args.seed))
    means = JointOutcomes.from_posterior(idata)
    value = {p: S.stint_value(means, p) for p in squad}
    ranking = sorted(squad, key=lambda p: -value[p])
    order = tuple(ranking)
    worlds = (0, args.worlds)
    scenarios = {
        "U10-like (as fitted)": S.Scenario(),
        "U11-like (first-game shifts)": S.Scenario(boundary_shift=-0.3, scoring_shift=-1.1, extras_rate=0.18),
    }

    rows = []
    with R.Runner(pool, workers=args.workers or R.default_workers()) as runner:
        for label, base in scenarios.items():
            at = lambda n: dataclasses.replace(base, retire_at=n)          # noqa: E731
            # name: (scenario, bankable, recall within, recall below, allowed as written)
            variants = {
                "everyone bats to 35 (conventional)": (base, (), None, None, True),
                "everyone retires at 30": (at(30), (), None, None, True),
                "everyone retires at 25": (at(25), (), None, None, True),
                "everyone retires at 20 (below the minimum: needs agreement)": (at(20), (), None, None, False),
                "top 4 retire at 25": (at(25), tuple(ranking[:4]), None, None, True),
                "top 4 retire at 20 (below the minimum: needs agreement)": (at(20), tuple(ranking[:4]), None, None, False),
                "top 4 retire at 25, recalled on a cheap wicket (needs agreement)": (at(25), tuple(ranking[:4]), 6, 5, False),
            }
            baseline = None
            for name, (scenario, bankable, within, below, allowed) in variants.items():
                result = S.evaluate_one(runner, order, ranking, scenario, worlds, args.reps, args.seed,
                                        bankable=bankable, recall_within_balls=within, recall_below_runs=below)
                baseline = result if baseline is None else baseline
                diff, se, better = S.paired(result, baseline)
                balls = result["balls"].mean(axis=(0, 1))
                deliveries = result["deliveries"][..., None]                     # (worlds, reps, 1)
                stint = (result["first_stint"] / deliveries).max(axis=2) * INNINGS_MINUTES   # longest first stint
                total = (result["crease"] / deliveries).max(axis=2) * INNINGS_MINUTES  # busiest batter, all stints
                bats = result["balls"] > 0
                rows.append({
                    "world": label, "option": name, "allowed_as_written": allowed,
                    "mean_total": float(S.world_means(result).mean()), "runs_vs_conventional": diff, "se": se,
                    "best_batter_balls": balls[0], "weakest_batter_balls": balls[-1],
                    "weakest_three_mean_balls": balls[-3:].mean(),
                    "first_stint_minutes": float(stint.mean()), "first_stint_minutes_p90": float(np.percentile(stint, 90)),
                    "busiest_total_minutes": float(total.mean()), "busiest_total_minutes_p90": float(np.percentile(total, 90)),
                    "weakest_bats": float(bats[:, :, -1].mean()), "all_nine_bat": float(bats.all(axis=2).mean()),
                })
                logger.info("%s | %s | %+.1f ± %.1f | stint %.0f min (p90 %.0f) | weakest bats %.0f%% | all nine %.0f%%",
                            label, name, diff, se, rows[-1]["first_stint_minutes"], rows[-1]["first_stint_minutes_p90"],
                            100 * rows[-1]["weakest_bats"], 100 * rows[-1]["all_nine_bat"])

    table = pd.DataFrame(rows)
    table.to_csv(out / "retirement_options.csv", index=False)
    lines = ["# Retirement options (rule 17.9: retire from 25 balls, retirees resume once all others are out or retired)", "",
             f"Conventional order, {args.worlds} worlds x {args.reps} innings each, the same worlds for every option. "
             "Runs are against everyone batting to 35. Balls include wides and no-balls. \"First stint\" is the longest first stint among the nine batters "
             "(from walking in to being out or retiring, at either end), as a share of the innings' deliveries "
             f"(minutes assume a {INNINGS_MINUTES}-minute innings); the 90th percentile is a batter who lasts. "
             "\"Total\" adds a second stint after returning.", ""]
    for label in scenarios:
        part = table[table["world"] == label]
        lines += [f"## {label}", "",
                  "| Option | Runs vs conventional | First stint, minutes (average, 90th percentile) | Busiest batter's total, minutes (average, 90th percentile) | Weakest batter bats | All nine bat | Weakest batter's balls |",
                  "| --- | --- | --- | --- | --- | --- | --- |"]
        for _, r in part.iterrows():
            lines.append(f"| {r['option']} | {r['runs_vs_conventional']:+.1f} ± {r['se']:.1f} | "
                         f"{r['first_stint_minutes']:.0f} ({r['first_stint_minutes_p90']:.0f}) | "
                         f"{r['busiest_total_minutes']:.0f} ({r['busiest_total_minutes_p90']:.0f}) | "
                         f"{100 * r['weakest_bats']:.0f}% | {100 * r['all_nine_bat']:.0f}% | "
                         f"{r['weakest_batter_balls']:.1f} |")
        lines.append("")
    (out / "retirement_options.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
