# PROJECT_STATUS.md

## Junior Cricket Bayesian MARCEL - VDCC Lineup Optimiser

Last updated: 2026-09-21 (personal PC, real-data validation)

## Current state

**Phase: real U10 data ingested and validated; simulator calibration is the
open problem before trusting lineup output.**

| Component | Status | Where |
| --- | --- | --- |
| BNJCA U10 and U11 playing conditions | Done. U10 checked against the Sept 2025 rule book, Section 16 | `rules_u10.py`, `rules_u11.py` |
| PlayHQ client (fixtures, results, scorecards, ball-by-ball) | Done, live-verified. Polite: spaced, stops on 403 or 429 | `playhq_client.py` |
| Scorecard and event parsing, name linking, reconciliation | Done, tested on 26 real innings | `playhq_parse.py` |
| Pseudonymised scorebook builder (ours and opposition) | Done | `playhq_scorebook.py`, `scripts/fetch_scorecards.py` |
| Model v2 (non-centred local level) | Done. Samples real data | `model.py` |
| Model v1 (original) | Kept behind `--model v1`; does not sample real data | `model.py` |
| Out-of-sample backtest | Done | `backtest.py`, `scripts/backtest.py` |
| U10 innings engine and totals check | Done; check shows calibration gaps | `simulator.py`, `scripts/check_u10_totals.py` |
| U11 simulator and optimiser | Works; run-out bookkeeping fixed; **not yet calibrated** | `simulator.py`, `optimizer.py` |
| Tests | 87 passing | `tests/` |

## Real data

VDCC U10 White, BNJCA Summer 2025/26 (season `5a7c9a42`): pre-Christmas team
`75cdae66` (grade U10 Walters), post-Christmas team `fafdb4c4` (grade U10
Kasprowicz Post). 17 games, 15 completed, 2 abandoned. Squad: 10 players (9
with PlayHQ profiles, plus a fill-in). Opposition: 87 players.

* Per-player scorecards for all 15 games; ball-by-ball for 13. Games
  `e7967782` and `67b87a4c` have no events (CloudFront 403 stopped the pull),
  so their batting rows are skipped and their bowling rows kept.
* Rows: 112 batting and 129 bowling for our squad; 107 and 125 for the
  opposition.
* Data quality: events reconcile exactly with the scorecard in 23 of 26
  innings (three differ by one ball or run; once the scorecard shows 121
  balls where the rules allow 120, and the events are right). Team total =
  batter runs + 4 x dismissals taken holds in 25 of 26 innings.
* U10 facts measured: 0.0556 dismissals per ball (6.6 per innings); 24% of
  dismissals are run outs and the striker is out in 83% of those; no wide or
  no-ball events are recorded; no stumpings.

## Model findings (real data)

Original model: 4000 of 4000 draws diverged, R-hat infinite. It also cannot
learn its recency weights (they only appear in a deterministic projection).
v2: max R-hat 1.007, min ESS 620, no divergences (squad only); R-hat 1.024
with the whole grade (96 players).

Between-player differences (skill spread on the logit or log scale, squad
only): boundary rate 0.85 (strong), dismissal hazard 0.26, wicket rate 0.20,
scoring rate 0.14, bowler economy 0.06 (none). Grade-wide, bowler economy
does differ (0.26). The grade population differs from White's own squad
(mean dismissal hazard 0.053 vs 0.043; economy 1.0 vs 0.82 runs per ball), so
the opposition prior must come from the whole grade.

## Backtest (fit on earlier games, score later ones)

Total log predictive density of the model minus each baseline (positive =
model better). 10 players; the period split is pre-Christmas to post-Christmas.

| Family | Period split vs pooled | vs raw | Rolling vs pooled | vs raw | corr |
| --- | --- | --- | --- | --- | --- |
| Boundary rate | +22.4 | +5.8 | +31.2 | +0.4 | 0.76-0.81 |
| Dismissal hazard | +5.4 | +3.6 | +3.3 | +2.1 | 0.23-0.25 |
| Scoring rate | +4.7 | -1.3 | +5.5 | -3.8 | 0.64-0.65 |
| Wicket rate | +0.8 | +1.3 | +0.2 | +2.8 | 0.26-0.36 |
| Bowler economy | -1.9 | +0.1 | +1.1 | +5.6 | 0.07-0.32 |

Read: boundary hitting is a stable, predictable skill and the model captures
it. Dismissal hazard carries a small real signal. Wicket rate is marginal and
bowler economy is not predictable within one team. Scoring rate has signal
but the raw record beat the model: runs per ball has variance 0.70 x its mean
so the Poisson likelihood over-states noise and over-shrinks. Calibration
(90% interval coverage) was 0.78-1.00. Only 10 players and 15 games, so the
small differences are inside the noise; the boundary result is not.

## U10 simulator check (13 games replayed, in-sample)

| Quantity | Simulated / real | Inside 90% |
| --- | --- | --- |
| Our runs off the bat | 0.89 | 0.62 |
| Opposition runs | 0.75 | 0.38 |
| Wickets we took | 0.72 | 0.69 |
| Wickets they took | 1.05 | 0.92 |
| Our final total | 0.86 | 0.54 |
| Opposition final total | 0.80 | 0.46 |
| **Margin** | **1.01** | **0.85** |

The margin is right but totals are low, largely by luck of cancelling errors.
Switching off the bowler-economy multiplier lifts opposition runs to 0.94;
replacing the geometric-mean dismissal rule with a relative one fixes wickets
we take (0.98) but pushes wickets against us to 1.30. Each piece is
individually miscalibrated. Cause: a bowler's economy and a batter's scoring
rate are estimated separately, so each absorbs the other's opposition quality
(confounding), and the combination rules are ad hoc. **Fix: a joint ball-level
batter-by-bowler model** using the ball-by-ball data.

## Next steps (in order)

1. Joint ball-level model (dismissal, boundary, runs given neither) with
   batter and bowler effects; refit; re-run the totals check.
2. Replace the scoring-rate Poisson likelihood; re-run the backtest.
3. Fetch the two missing games (later, or with an API key) and 2024/25.
4. When BNJCA publishes the 2026/27 U11 draw (Round 1 Sat 10 Oct 2026), get
   the grade ID, fetch, fit, and optimise with `--squad`.
5. Ask PlayHQ for an API key.

## Key decisions log

- Two model versions coexist so the failure of v1 stays reproducible.
- v2 uses a random-walk drift between periods, so recency weighting is
  learned instead of fixed.
- Opposition rows are fitted with our squad so the population prior reflects
  the grade (`playhq_all_*.csv`).
- Extras are not modelled when none are recorded; the population prior is used
  so U11 wides are not switched off.
- Games without events skip batting rows rather than record zero dismissals.
- Player names never enter the model CSVs; aliases only.
- The 4-run U10 penalty goes to the fielding side's total (Rule 16.10(vi)),
  which is how PlayHQ reports U10 scores.

## Environment notes

- Python 3.14.7 in `.venv`; PyMC 6.3.2, PyTensor 3.3.2, ArviZ 1.3, numpyro
  0.22, jax 0.11.2, h5netcdf and h5py (ArviZ 1.x has no bundled netCDF
  backend). `requirements.txt` lists them.
- No C compiler: use `--sampler numpyro`. The PyTensor "g++ not available"
  line is harmless then.
- `data/raw/`, `data/processed/`, `data/outputs/`, `log/` and `*.nc` are
  gitignored on `main`.
