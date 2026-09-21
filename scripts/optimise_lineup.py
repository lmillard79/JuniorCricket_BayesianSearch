"""Optimise the U11 lineup from a fitted posterior.

Loads the model posterior (netcdf from ``fit_model.py``), builds
per-player skill rates from posterior means, runs the decomposed
hill-climbing search against population-drawn opposition, and
writes a markdown recommendation, a CSV of the search history and
a run-differential figure to ``data/outputs/``.

Use ``--demo`` to run against synthetic skill rates when no
posterior exists yet (pipeline verification).

Examples:
    python scripts/optimise_lineup.py --posterior data/outputs/posterior_model.nc
    python scripts/optimise_lineup.py --demo
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict

import arviz as az
import numpy as np
import pandas as pd

from junior_cricket.figures import plot_run_differential
from junior_cricket.logging_setup import setup_logging
from junior_cricket.model import PopulationPriors
from junior_cricket.optimizer import LineupOptimizer
from junior_cricket.simulator import PlayerSkills

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "data" / "outputs"

LOGGER_NAME = "optimise_lineup"

DEMO_SKILLS = {
    "Archie": dict(p_out=0.032, p_bound=0.078, srr=0.44,
                   p_wicket=0.075, econ=0.41, p_extra=0.08),
    "Ben": dict(p_out=0.045, p_bound=0.060, srr=0.38,
               p_wicket=0.065, econ=0.48, p_extra=0.10),
    "Charlie": dict(p_out=0.052, p_bound=0.052, srr=0.34,
                    p_wicket=0.058, econ=0.53, p_extra=0.11),
    "Declan": dict(p_out=0.058, p_bound=0.047, srr=0.31,
                   p_wicket=0.052, econ=0.57, p_extra=0.12),
    "Ethan": dict(p_out=0.064, p_bound=0.042, srr=0.29,
                  p_wicket=0.048, econ=0.60, p_extra=0.13),
    "Finn": dict(p_out=0.070, p_bound=0.038, srr=0.27,
                 p_wicket=0.044, econ=0.64, p_extra=0.14),
    "Gus": dict(p_out=0.076, p_bound=0.034, srr=0.25,
                p_wicket=0.040, econ=0.68, p_extra=0.15),
    "Harvey": dict(p_out=0.084, p_bound=0.030, srr=0.23,
                   p_wicket=0.036, econ=0.73, p_extra=0.16),
    "Ike": dict(p_out=0.092, p_bound=0.026, srr=0.21,
                p_wicket=0.032, econ=0.78, p_extra=0.17),
}


def skills_from_posterior(
    idata, logger: logging.Logger
) -> Dict[str, PlayerSkills]:
    """Build mean-rate PlayerSkills from a fitted posterior.

    Args:
        idata: InferenceData with the projected rate variables.
        logger: Logger for the audit trail.

    Returns:
        Player skills keyed by player name.
    """
    posterior = idata.posterior
    players = list(posterior["p_out_final"].coords["player"].values)

    def _mean(var: str) -> np.ndarray:
        return np.asarray(posterior[var].mean(dim=("chain", "draw")))

    p_out = _mean("p_out_final")
    p_bound = _mean("p_bound_final")
    p_wicket = _mean("p_wicket_final")
    srr = _mean("srr_proj")
    econ = _mean("econ_proj")
    p_extra = _mean("p_extra_proj")

    skills: Dict[str, PlayerSkills] = {}
    for i, name in enumerate(players):
        skills[str(name)] = PlayerSkills(
            name=str(name),
            p_out=float(p_out[i]),
            p_bound=float(p_bound[i]),
            srr=float(srr[i]),
            p_wicket=float(p_wicket[i]),
            econ=float(econ[i]),
            p_extra=float(p_extra[i]),
        )
    logger.info("Loaded %d players from posterior", len(skills))
    return skills


def priors_from_posterior(idata) -> PopulationPriors:
    """Extract population priors from the fitted posterior means.

    Args:
        idata: InferenceData with the population variables.

    Returns:
        Point-estimate population priors.
    """
    posterior = idata.posterior

    def _mean(var: str) -> float:
        return float(posterior[var].mean(dim=("chain", "draw")))

    return PopulationPriors(
        mu_out=_mean("mu_out"),
        sigma_out=_mean("sigma_out"),
        mu_bound=_mean("mu_bound"),
        sigma_bound=_mean("sigma_bound"),
        mu_srr=_mean("mu_srr"),
        sigma_srr=_mean("sigma_srr"),
        mu_wicket=_mean("mu_wicket"),
        sigma_wicket=_mean("sigma_wicket"),
        mu_econ=_mean("mu_econ"),
        sigma_econ=_mean("sigma_econ"),
        mu_extra=_mean("mu_extra"),
        sigma_extra=_mean("sigma_extra"),
    )


def write_report(
    result,
    skills: Dict[str, PlayerSkills],
    n_sims: int,
    path: Path,
) -> None:
    """Write the markdown recommendation report.

    Args:
        result: OptimizerResult from the search.
        skills: Player skills for the context table.
        n_sims: Simulations per candidate.
        path: Output markdown path.
    """
    lines = [
        "# U11 Lineup Optimisation Report",
        "",
        f"Simulations per candidate: {n_sims}",
        "",
        "## Recommended batting order",
        "",
        "| Position | Player | Dismissal hazard | Boundary rate |"
        " Scoring rate |",
        "| --- | --- | --- | --- | --- |",
    ]
    for pos, name in enumerate(result.batting_order, start=1):
        s = skills[name]
        lines.append(
            f"| {pos} | {name} | {s.p_out:.3f} | {s.p_bound:.3f} "
            f"| {s.srr:.2f} |"
        )

    lines += [
        "",
        f"Retirement policy: retire at "
        f"{result.retire_at_balls or 35} balls",
        "",
        "## Recommended bowling",
        "",
        "Round-robin rotation (over-by-over cycles through this list):",
        "",
        "1. " + " -> ".join(result.bowling_rotation),
        "",
        "| Player | Overs | Wicket rate | Economy |",
        "| --- | --- | --- | --- |",
    ]
    for name in result.bowling_rotation:
        s = skills[name]
        lines.append(
            f"| {name} | {result.bowling_allocation[name]} "
            f"| {s.p_wicket:.3f} | {s.econ:.2f} |"
        )

    lines += [
        "",
        "## Match outlook vs unknown opposition",
        "",
        f"- Expected our total: **{result.our_totals.mean():.1f}**"
        f" (sd {result.our_totals.std():.1f})",
        f"- Expected their total: **{result.their_totals.mean():.1f}**"
        f" (sd {result.their_totals.std():.1f})",
        f"- Win probability: **{result.win_probability:.0%}**"
        f" (tie {result.tie_probability:.0%})",
        f"- Expected run differential: **{result.expected_run_diff:+.1f}**"
        f" (90% interval {result.run_diff_ci[0]:+.0f} to "
        f"{result.run_diff_ci[1]:+.0f})",
        "",
        "Participation note: all players bat and bowl under the "
        "BNJCA U11 rules; the optimisation searches within those "
        "mandatory constraints, not around them.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="ascii")
    return None


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--posterior",
        default=str(OUTPUT_DIR / "posterior_model.nc"),
        help="Posterior netcdf from fit_model.py",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run on synthetic skill rates for pipeline verification",
    )
    parser.add_argument("--n-sims", type=int, default=400)
    parser.add_argument("--batting-evals", type=int, default=60)
    parser.add_argument("--bowling-evals", type=int, default=60)
    parser.add_argument(
        "--retire-at", type=int, default=None,
        help="Retirement policy (25-35); default is the 35-ball cap",
    )
    return parser


def main() -> None:
    """Run the lineup optimisation and write recommendations."""
    args = build_parser().parse_args()
    logger = setup_logging(LOGGER_NAME, OUTPUT_DIR)
    logger.info("Input arguments: %s", vars(args))

    if args.demo:
        logger.info("Using synthetic demo skills")
        skills = {
            name: PlayerSkills(name=name, **rates)
            for name, rates in DEMO_SKILLS.items()
        }
        priors = PopulationPriors()
    else:
        posterior_path = Path(args.posterior)
        if not posterior_path.exists():
            raise FileNotFoundError(
                f"Posterior not found: {posterior_path}. "
                "Run fit_model.py first, or use --demo."
            )
        idata = az.from_netcdf(posterior_path)
        skills = skills_from_posterior(idata, logger)
        priors = priors_from_posterior(idata)
        logger.info(
            "Population priors: mu_out=%.3f mu_econ=%.2f",
            priors.mu_out,
            priors.mu_econ,
        )

    optimizer = LineupOptimizer(
        player_skills=skills,
        priors=priors,
        n_sims=args.n_sims,
    )
    logger.info(
        "Starting optimisation: %d sims per candidate, "
        "%d batting + %d bowling evaluations",
        args.n_sims,
        args.batting_evals,
        args.bowling_evals,
    )
    result = optimizer.optimise(
        batting_evaluations=args.batting_evals,
        bowling_evaluations=args.bowling_evals,
        retire_at_balls=args.retire_at,
    )

    report_path = OUTPUT_DIR / "lineup_recommendation.md"
    write_report(result, skills, args.n_sims, report_path)
    logger.info("Recommendation report: %s", report_path)

    history_path = OUTPUT_DIR / "search_history.csv"
    history_rows = []
    for order, score in result.batting_history:
        history_rows.append(
            {"phase": "batting", "config": " > ".join(order), "score": score}
        )
    for rotation, score in result.bowling_history:
        history_rows.append(
            {
                "phase": "bowling",
                "config": " > ".join(rotation),
                "score": score,
            }
        )
    pd.DataFrame(history_rows).to_csv(history_path, index=False)
    logger.info("Search history: %s", history_path)

    figure_path = OUTPUT_DIR / "run_differential.png"
    plot_run_differential(result.our_totals, result.their_totals,
                           figure_path)
    logger.info("Run differential figure: %s", figure_path)

    logger.info(
        "Optimisation complete: win probability %.0f, "
        "expected run diff %+.1f",
        result.win_probability,
        result.expected_run_diff,
    )


if __name__ == "__main__":
    main()
