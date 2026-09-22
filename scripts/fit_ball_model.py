"""Fit the joint batter-by-bowler ball model on real deliveries.

Reads ``playhq_balls.csv`` (written by ``fetch_scorecards.py``), fits the
model in ``junior_cricket.ball_model`` and writes, under ``data/outputs/``:

* ``ball_posterior.nc``: the posterior, for ``check_u10_totals.py``;
* ``ball_model_summary.md``: baselines and how much batters and bowlers
  actually differ, with sampler diagnostics;
* ``ball_effects.csv``: every player's batting and bowling effects.

Players are aliases (P01, ..., O01, ...).

Example:
    python scripts/fit_ball_model.py --balls data/processed/playhq_balls.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import arviz as az
import numpy as np

from junior_cricket.ball_model import (
    COMPONENTS,
    LABELS,
    effect_table,
    fit_ball_model,
    load_balls,
)
from junior_cricket.logging_setup import setup_logging

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "data" / "outputs"
LOGGER_NAME = "fit_ball_model"


def _flat(idata, var: str) -> np.ndarray:
    return np.asarray(idata.posterior[var]).reshape(-1)


def _interval(x: np.ndarray) -> str:
    return f"{np.median(x):.2f} [{np.percentile(x, 5):.2f}, {np.percentile(x, 95):.2f}]"


def main() -> None:
    """Fit, diagnose and write the summary."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--balls", default=str(REPO_ROOT / "data" / "processed" / "playhq_balls.csv"))
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--tune", type=int, default=1000)
    args = parser.parse_args()

    logger = setup_logging(LOGGER_NAME, OUTPUT_DIR)
    logger.info("Input arguments: %s", vars(args))
    data, _ = load_balls(Path(args.balls))
    logger.info("%d deliveries, %d players, %d games", len(data.bat),
                len(data.players), len(data.games))
    idata = fit_ball_model(data, draws=args.draws, tune=args.tune, seed=20260921)

    scalars = [f"{p}_{c}" for c in COMPONENTS for p in ("a", "sb", "sw")] + ["sg_b", "sg_s"]
    scalars += [v for v in ("mu_f_d", "sf_d") if v in idata.posterior]
    summary = az.summary(idata, var_names=scalars)
    divergences = int(np.asarray(idata.sample_stats["diverging"]).sum())
    logger.info("Diagnostics: max R-hat %.3f, min bulk ESS %.0f, divergences %d",
                summary["r_hat"].max(), summary["ess_bulk"].min(), divergences)

    idata.to_netcdf(OUTPUT_DIR / "ball_posterior.nc")
    effect_table(idata).to_csv(OUTPUT_DIR / "ball_effects.csv", index=False)

    inv = lambda x: 1.0 / (1.0 + np.exp(-x))                    # noqa: E731
    lines = [
        "# Joint batter-by-bowler ball model", "",
        f"{len(data.bat)} deliveries, {len(data.players)} players, "
        f"{len(data.games)} games. Max R-hat {summary['r_hat'].max():.3f}, "
        f"min bulk ESS {summary['ess_bulk'].min():.0f}, divergences {divergences}.", "",
        "## Baseline (an average batter against an average bowler)", "",
        "| outcome | probability |", "| --- | --- |",
    ]
    for c in COMPONENTS:
        lines.append(f"| {LABELS[c]} | {inv(_flat(idata, 'a_' + c)).mean():.3f} |")
    lines += [
        "", "## How much players differ", "",
        "Spread (standard deviation, logit scale) of persistent player effects; "
        "larger means bigger differences between players. Median [90% interval].", "",
        "| outcome | batters | bowlers |", "| --- | --- | --- |",
    ]
    for c in COMPONENTS:
        lines.append(f"| {LABELS[c]} | {_interval(_flat(idata, 'sb_' + c))} | "
                     f"{_interval(_flat(idata, 'sw_' + c))} |")
    lines += ["", f"Ground and conditions (per game): boundary "
              f"{_interval(_flat(idata, 'sg_b'))}, scoring {_interval(_flat(idata, 'sg_s'))}."]
    if "mu_f_d" in idata.posterior:
        mu_f = _flat(idata, "mu_f_d")
        base_d = inv(_flat(idata, "a_d"))
        with_field = inv(_flat(idata, "a_d") + mu_f)
        share_positive = float((mu_f > 0).mean())
        lines += [
            "", "## Our team's fielding edge",
            "",
            f"Our own fielding (catches, run outs, general sharpness) adds "
            f"{_interval(mu_f)} to the dismissal log-odds on top of the bowler's own "
            f"effect, day to day spread {_interval(_flat(idata, 'sf_d'))}. On an average "
            f"ball that shifts the dismissal chance from {base_d.mean():.3f} to "
            f"{with_field.mean():.3f}. Posterior probability the edge is positive: "
            f"{share_positive:.0%}. Descriptive, not fitted: catches and run outs credited "
            f"to individual fielders (`fielding_credit_table`) are not part of this model.",
        ]
    (OUTPUT_DIR / "ball_model_summary.md").write_text("\n".join(lines), encoding="utf-8")
    logger.info("Summary: %s", OUTPUT_DIR / "ball_model_summary.md")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
