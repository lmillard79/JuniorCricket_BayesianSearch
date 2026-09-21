# PROJECT_STATUS.md

## Junior Cricket Bayesian MARCEL - VDCC Under 11 Lineup Optimiser

Last updated: 2026-09-21 (session: cerulean-trapezoid)

## Current state

**Phase: pipeline built and unit-verified; first real-data fit pending.**

The full architecture from the plan (README sections 1-5) is implemented:

| Component | Status | File |
| --- | --- | --- |
| U10/U11 playing conditions encoded from BNJCA 2026 rules booklet | Done, validated | `src/junior_cricket/rules_u10.py`, `rules_u11.py` |
| PlayHQ public GraphQL client (no API key needed) | Done, live-verified vs BNJCA data | `src/junior_cricket/playhq_client.py` |
| Scorebook CSV loading, validation, period aggregation | Done, tested | `src/junior_cricket/data_loader.py` |
| PyMC MARCEL model (Beta-Binomial + Gamma-Poisson hierarchy, Dirichlet recency weights, optional relative-age effect) | Done, builds and passes prior-predictive tests | `src/junior_cricket/model.py` |
| Ball-by-ball U11 simulator (25 overs, 8 wickets, retire 25-35 balls, round-robin bowling) | Done, 10 rule tests passing | `src/junior_cricket/simulator.py` |
| Decomposed lineup optimizer (hill-climb batting order + bowling tiers/rotation vs population opposition) | Done, demo-verified | `src/junior_cricket/optimizer.py` |
| WRM-branded figures (shrinkage, forest, run differential) | Done | `src/junior_cricket/figures.py` |
| Scripts (fetch / fit / optimise) with mandatory logging protocol | Done | `scripts/` |
| Test suite | 25/25 passing | `tests/` |

## Verified facts

- PlayHQ public GraphQL (`https://api.playhq.com/graphql`, `tenant: ca`
  header) works without an API key; game `f52ca224` and grade `2bb24d79`
  fetched live during this session (BNJCA Girls Summer 2024/25, season
  `7ba7b60f`).
- `gradePlayerStatistics` works and returns real VDCC teams (e.g. "Post
  Xmas VDCC Girls Stage 2 Holmes"), confirming VDCC data is reachable
  through the public API. Statistics arrays were empty for that grade.
- BNJCA 2026 rules booklet extracted; U11 boys' Section 17 semantics are
  encoded in `rules_u11.py` (see README Section 2 for the full summary).

## Current blocker (work machine only)

`fit_model.py --demo` runs through data generation, aggregation and model
build, but PyMC's default NUTS sampler crashes inside PyTensor's Python
fallback linker (`'Scratchpad' object has no attribute 'ufunc'`) because
this machine has no C compiler (g++ unavailable) and llvmlite.dll is
blocked (WinError 5). This is an environment restriction, not a code bug:

- **Fix option A (recommended on the personal PC):** install numpyro and
  run `python scripts/fit_model.py --demo --sampler numpyro`
  (`pip install numpyro jax`).
- **Fix option B:** install a C compiler (VS Build Tools / MinGW) so
  PyTensor compiles C code and use the default sampler.
- A `PYTENSOR_FLAGS=optimizer=fast_compile` test was queued but the
  session paused before it ran; worth one retry on the personal PC.

## Next steps (in order)

1. **Unblock sampling** on the personal PC (option A above), then run
   `python scripts/fit_model.py --demo --sampler numpyro` end-to-end and
   check diagnostics (R-hat, divergences).
2. **Enter real U10 Whites history** from the paper scorebooks into
   copies of the `data/templates/` CSVs (2024/25 and 2025/26 seasons).
   The default period table in `data_loader.py` already splits history
   into the three MARCEL periods.
3. **Find the 2026/27 U11 grade ID** once BNJCA publishes the draw
   (Round 1: Sat 10 Oct 2026): navigate the team's PlayHQ page and copy
   the grade ID from the URL, then
   `python scripts/fetch_playhq_data.py --grade-id <id>` and
   `--stats-grade <id>` to pull grade-wide baselines for the population
   prior.
4. **Fit on real data** (`fit_model.py --batting ... --bowling ...`),
   review the shrinkage figure, then
   `optimise_lineup.py --posterior data/outputs/posterior_model.nc`.
5. **Backtest** (plan Section 6): hold out post-Xmas 2025/26 games,
   predict from earlier periods, check posterior predictive calibration.

## Key decisions log

- Simulator models U11 only (survival format); U10 conditions are used
  for historical data parsing (sundries conventions differ).
- Optimizer decomposes the match: batting order affects only our innings,
  bowling config only theirs; win prob assembled from two independent
  totals distributions.
- Opposition drawn per simulated match from the fitted population
  (mu/sigma do double duty: shrinkage target and unknown-opponent prior).
- Posterior means feed the optimizer (documented simplification);
  opposition and match randomness still propagate into the CIs.
- Round-robin bowling expansion is strict per Rule 17.7(v); the tier
  multiset is fixed, the optimizer searches tier assignment + rotation.

## Environment notes

- Python 3.11, venv at `.venv`, package installed editable
  (`pip install -e .`).
- numba/llvmlite were uninstalled (llvmlite.dll blocked on this machine);
  arviz and PyMC do not require them.
- Data dirs (`data/raw`, `data/processed`, `data/outputs`), `log/` and
  `*.nc` are gitignored.
