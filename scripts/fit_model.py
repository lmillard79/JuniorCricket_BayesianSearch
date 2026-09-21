"""Fit the Bayesian MARCEL model to team scorebook data.

Loads manual batting/bowling CSVs (and the player registry),
aggregates them into per-player, per-period totals, fits the
hierarchical PyMC model, and writes the posterior (netcdf), a
summary CSV and a shrinkage figure to ``data/outputs/``.

Use ``--demo`` to run the full pipeline on synthetic data when
no scorebook CSVs exist yet (pipeline verification).

Examples:
    python scripts/fit_model.py --demo
    python scripts/fit_model.py --batting data/processed/batting.csv \
        --bowling data/processed/bowling.csv
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm

from junior_cricket.data_loader import (
    aggregate_batting,
    aggregate_bowling,
    load_batting_csv,
    load_bowling_csv,
    load_player_registry,
    player_months_old,
)
from junior_cricket.figures import plot_shrinkage
from junior_cricket.logging_setup import setup_logging
from junior_cricket.model import (
    ModelData,
    build_marcel_model,
    build_marcel_model_v2,
    prepare_model_data,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "data" / "outputs"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
TEMPLATES_DIR = REPO_ROOT / "data" / "templates"

LOGGER_NAME = "fit_model"
RANDOM_SEED = 20260921

PERIOD_LABELS = [
    "2024-25 U10 season",
    "2025-26 U10 pre-Xmas",
    "2025-26 U10 post-Xmas",
]

SYNTHETIC_PLAYERS = [
    "Archie", "Ben", "Charlie", "Declan", "Ethan",
    "Finn", "Gus", "Harvey", "Ike",
]

SUMMARY_VARS = [
    "mu_out",
    "mu_bound",
    "mu_srr",
    "mu_wicket",
    "mu_econ",
    "mu_extra",
    "recency_weights",
    "p_out_final",
    "p_bound_final",
    "p_wicket_final",
    "srr_proj",
    "econ_proj",
    "p_extra_proj",
]

# v2: working-scale population mean (m), between-player skill spread (s)
# and period-to-period drift (tau) per family, plus the projections.
SUMMARY_VARS_V2 = [
    f"{prefix}_{family}"
    for family in ("out", "bound", "wicket", "srr", "econ")
    for prefix in ("m", "s", "tau")
] + [
    "p_out_final",
    "p_bound_final",
    "p_wicket_final",
    "srr_proj",
    "econ_proj",
]


def generate_demo_data(seed: int = 7) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate a plausible synthetic season for pipeline testing.

    Args:
        seed: Random seed for reproducibility.

    Returns:
        (batting_df, bowling_df) matching the manual CSV schemas.
    """
    rng = np.random.default_rng(seed)
    n_players = len(SYNTHETIC_PLAYERS)

    # Latent skill spread across the cohort.
    p_out = rng.uniform(0.03, 0.10, n_players)
    p_bound = rng.uniform(0.02, 0.08, n_players)
    srr = rng.uniform(0.20, 0.45, n_players)
    p_wicket = rng.uniform(0.03, 0.08, n_players)
    econ = rng.uniform(0.40, 0.80, n_players)
    p_extra = rng.uniform(0.08, 0.18, n_players)

    batting_rows = []
    bowling_rows = []
    # Three periods, eight rounds each, all Saturdays.
    period_starts = pd.to_datetime(
        ["2024-10-12", "2025-10-10", "2026-01-31"]
    )
    opponents = ["Northside", "Southbank", "Eastgrove", "Westend"]

    for period in range(3):
        for round_no in range(8):
            date = period_starts[period] + pd.Timedelta(days=7 * round_no)
            opponent = opponents[round_no % len(opponents)]
            for i, name in enumerate(SYNTHETIC_PLAYERS):
                # A couple of absences per player per period.
                if rng.random() < 0.15:
                    continue
                balls = int(rng.integers(10, 26))
                outs = int(rng.random() < p_out[i] * balls)
                boundaries = int(
                    rng.binomial(max(balls - outs, 0), p_bound[i])
                )
                non_boundary_balls = max(balls - boundaries, 0)
                runs_nb = int(rng.poisson(srr[i] * non_boundary_balls))
                batting_rows.append(
                    {
                        "date": date.strftime("%Y-%m-%d"),
                        "opponent": opponent,
                        "player_name": name,
                        "balls_faced": balls,
                        "runs_scored": runs_nb + 4 * boundaries,
                        "fours": boundaries,
                        "sixes": 0,
                        "dismissed": outs,
                        "retired": int(balls >= 25 and rng.random() < 0.3),
                        "source": "demo",
                    }
                )

                overs = int(rng.integers(2, 5))
                deliveries = overs * 6
                extras = int(rng.binomial(deliveries, p_extra[i]))
                fair = deliveries - extras
                wickets = int(rng.binomial(fair, p_wicket[i]))
                runs_off_bat = int(rng.poisson(econ[i] * fair))
                bowling_rows.append(
                    {
                        "date": date.strftime("%Y-%m-%d"),
                        "opponent": opponent,
                        "player_name": name,
                        "balls_bowled": deliveries,
                        "runs_conceded": runs_off_bat + extras,
                        "wickets": wickets,
                        "wides": extras,
                        "no_balls": 0,
                        "source": "demo",
                    }
                )

    return pd.DataFrame(batting_rows), pd.DataFrame(bowling_rows)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batting", help="Batting scorebook CSV path")
    parser.add_argument("--bowling", help="Bowling scorebook CSV path")
    parser.add_argument("--registry", help="Player registry CSV path")
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run on synthetic data for pipeline verification",
    )
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--tune", type=int, default=1000)
    parser.add_argument(
        "--model",
        choices=["v2", "v1"],
        default="v2",
        help="'v2' non-centred local-level model (default); 'v1' is the "
        "original specification, which does not sample on real data",
    )
    parser.add_argument(
        "--sampler",
        choices=["nuts", "numpyro"],
        default="nuts",
        help="NUTS backend: 'nuts' (PyTensor) or 'numpyro' (JAX; "
        "recommended on Windows without a C compiler)",
    )
    return parser


def main() -> None:
    """Run the full model-fitting pipeline."""
    args = build_parser().parse_args()
    logger = setup_logging(LOGGER_NAME, OUTPUT_DIR)
    logger.info("Input arguments: %s", vars(args))

    if args.demo:
        logger.info("Generating synthetic demo data")
        batting_raw, bowling_raw = generate_demo_data()
    else:
        batting_path = Path(args.batting) if args.batting else (
            TEMPLATES_DIR / "batting_scorebook_template.csv"
        )
        bowling_path = Path(args.bowling) if args.bowling else (
            TEMPLATES_DIR / "bowling_scorebook_template.csv"
        )
        batting_raw = load_batting_csv(batting_path)
        bowling_raw = load_bowling_csv(bowling_path)
        logger.info(
            "Loaded %d batting and %d bowling records",
            len(batting_raw),
            len(bowling_raw),
        )

    batting_agg = aggregate_batting(batting_raw)
    bowling_agg = aggregate_bowling(bowling_raw)
    logger.info(
        "Aggregated to %d batting and %d bowling player-period rows",
        len(batting_agg),
        len(bowling_agg),
    )

    # Persist interim aggregates (glass-box audit trail).
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    batting_agg.to_csv(
        PROCESSED_DIR / "batting_player_period.csv", index=False
    )
    bowling_agg.to_csv(
        PROCESSED_DIR / "bowling_player_period.csv", index=False
    )
    logger.info("Interim aggregates written to %s", PROCESSED_DIR)

    ages = None
    if args.registry:
        registry = load_player_registry(Path(args.registry))
        ages = player_months_old(registry)
        logger.info("Registry loaded with %d players", len(registry))

    data: ModelData = prepare_model_data(
        batting_agg,
        bowling_agg,
        period_labels=PERIOD_LABELS,
        ages_months=ages,
    )
    logger.info(
        "Model data: %d players x %d periods, %d balls faced total",
        len(data.player_names),
        len(data.period_labels),
        int(data.balls_faced.sum()),
    )

    if args.model == "v2":
        model = build_marcel_model_v2(data)
        summary_vars = SUMMARY_VARS_V2
    else:
        model = build_marcel_model(data)
        summary_vars = SUMMARY_VARS
    logger.info(
        "Model %s built; starting NUTS sampling (%s)", args.model, args.sampler
    )
    if args.sampler == "numpyro":
        try:
            import numpyro  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "numpyro sampler requested but numpyro is not "
                "installed. Run: pip install numpyro jax"
            ) from exc
        with model:
            idata = pm.sample(
                draws=args.draws,
                tune=args.tune,
                chains=4,
                target_accept=0.9,
                random_seed=RANDOM_SEED,
                progressbar=False,
                nuts_sampler="numpyro",
            )
    else:
        with model:
            idata = pm.sample(
                draws=args.draws,
                tune=args.tune,
                chains=4,
                target_accept=0.9,
                random_seed=RANDOM_SEED,
                progressbar=False,
            )
    logger.info("Sampling complete")

    summary = az.summary(idata, var_names=summary_vars)
    divergences = int(idata.sample_stats["diverging"].sum())
    rhat_max = float(summary["r_hat"].max())
    ess_min = float(summary["ess_bulk"].min())
    logger.info(
        "Diagnostics: max R-hat %.3f, min bulk ESS %.0f, divergences %d",
        rhat_max, ess_min, divergences,
    )
    if rhat_max >= 1.05 or divergences > 0:
        logger.warning(
            "Convergence concerns: R-hat >= 1.05 or divergences present"
        )

    posterior_path = OUTPUT_DIR / "posterior_model.nc"
    idata.to_netcdf(posterior_path)
    logger.info("Posterior saved: %s", posterior_path)

    summary_path = OUTPUT_DIR / "model_summary.csv"
    summary.to_csv(summary_path)
    logger.info("Summary saved: %s", summary_path)

    # Shrinkage figure: raw vs posterior dismissal hazard.
    raw_rates = []
    sample_sizes = []
    for name in data.player_names:
        mask = batting_agg["player_name"] == name
        balls = int(batting_agg.loc[mask, "balls_faced"].sum())
        outs = int(batting_agg.loc[mask, "outs"].sum())
        sample_sizes.append(balls)
        raw_rates.append(outs / balls if balls else 0.0)

    posterior_means = np.asarray(
        idata.posterior["p_out_final"].mean(dim=("chain", "draw"))
    )
    population_mean = float(
        idata.posterior["mu_out"].mean(dim=("chain", "draw"))
    )
    shrinkage_path = OUTPUT_DIR / "shrinkage_p_out.png"
    plot_shrinkage(
        raw_rates=np.array(raw_rates),
        posterior_means=posterior_means,
        population_mean=population_mean,
        sample_sizes=np.array(sample_sizes, dtype=float),
        player_names=data.player_names,
        metric_label="Dismissal hazard per ball",
        output_path=shrinkage_path,
    )
    logger.info("Shrinkage figure saved: %s", shrinkage_path)
    logger.info("Fit complete")


if __name__ == "__main__":
    main()
