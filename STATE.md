# STATE.md - Session Close Handoff

Session: 2026-09-21 (cerulean-trapezoid)
Project: JuniorCricket_BayesianSearch (VDCC Under 11 lineup optimiser)
Durable project doc: PROJECT_STATUS.md (pushed to repo, supersedes this
file for full detail; this file is the session-level handoff).

## 1. Current Status

* Full implementation pipeline built, tested (25/25 tests passing) and
  pushed to https://github.com/lmillard79/JuniorCricket_BayesianSearch.git
  (commit 57125c9).
* PlayHQ public GraphQL client verified live against real BNJCA data with
  no API key needed (endpoint https://api.playhq.com/graphql with
  tenant header "ca").
* BNJCA U11 boys' playing conditions (2026 rules booklet Section 17)
  encoded and validated: 25 overs, 9 players, all out at 8 dismissals,
  retire at 25-35 balls, mandatory all-player tiered bowling.
* PyMC MARCEL model builds and passes prior-predictive checks.
* Ball-by-ball U11 simulator passes all rule-enforcement tests.
* Decomposed lineup optimizer (batting order + bowling tiers/rotation)
  demo-verified on synthetic skills.

## 2. Active Blockers & Errors

* PyMC default NUTS sampler crashes inside PyTensor's Python fallback
  linker on this work machine ("'Scratchpad' object has no attribute
  'ufunc'") because no C compiler (g++) is available.
* llvmlite.dll is blocked on this machine (WinError 5), so numba and
  llvmlite were uninstalled; arviz and PyMC run fine without them.
* Net effect: the end-to-end demo fit (scripts/fit_model.py --demo) has
  not completed a sampling run on this machine. Environment restriction,
  not a code bug.

## 3. Immediate Next Steps (Start Here)

1. On the personal PC: clone the repo, create a venv, install
   requirements-dev.txt and `pip install numpyro jax`, then run
   `python scripts/fit_model.py --demo --sampler numpyro`.
2. Check MCMC diagnostics (R-hat < 1.05, no divergences) and review the
   shrinkage figure in data/outputs/.
3. Enter the U10 Whites scorebook history (2024/25 and 2025/26 seasons)
   into copies of the data/templates CSVs (batting, bowling, registry).
4. When BNJCA publishes the 2026/27 draw (Round 1: Sat 10 Oct 2026),
   get the U11 grade ID from the team's PlayHQ page URL, then run
   `python scripts/fetch_playhq_data.py --grade-id <id>` and
   `--stats-grade <id>` for grade-wide baselines.
5. Fit on real data, then run
   `python scripts/optimise_lineup.py --posterior data/outputs/posterior_model.nc`.

## 4. Working Files & Context

* Entry points (in run order):
  * scripts/fetch_playhq_data.py -> pulls PlayHQ data to data/raw and
    data/processed
  * scripts/fit_model.py -> fits MARCEL model, writes posterior to
    data/outputs/posterior_model.nc
  * scripts/optimise_lineup.py -> writes recommendation report, search
    history and run-differential figure to data/outputs
* Known PlayHQ identifiers (verified live):
  * game f52ca224 -> grade 2bb24d79 -> season 7ba7b60f
    (BNJCA Girls Summer 2024/25)
  * gradePlayerStatistics confirmed working and returns real VDCC teams
* Core library: src/junior_cricket/ (rules, client, loader, model,
  simulator, optimizer, figures, logging)
* Tests: tests/ (run with `python -m pytest tests/`)
* Venv: .venv (Python 3.11, package installed editable via pyproject.toml)
