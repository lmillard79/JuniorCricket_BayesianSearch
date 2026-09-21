# STATE.md - Session Close Handoff

Session: 2026-09-21, personal PC (real-data bring-up and validation)
Project: JuniorCricket_BayesianSearch (VDCC lineup optimiser)
Durable detail: PROJECT_STATUS.md. This file replaces the earlier
work-machine STATE.md, whose blockers are resolved.

## 1. Current status

* Runs end to end on the personal PC (Windows 11, Python 3.14.7, PyMC
  6.3.2, ArviZ 1.x). `python -m pytest tests/` gives 87 passed. No C
  compiler is needed: sampling uses numpyro/JAX (`--sampler numpyro`).
* Real data is in. VDCC U10 White, BNJCA Summer 2025/26: teams `75cdae66`
  (pre-Christmas) and `fafdb4c4` (post-Christmas), 17 games, 15 completed.
  Per-player scorecards for all 15; ball-by-ball events for 13.
* The original model did not sample real data (4000 of 4000 draws
  diverged). `build_marcel_model_v2` does: R-hat 1.007, no divergences.
* Backtest (fit early games, score later ones) and a U10 simulator check
  against real totals are done. Results are in PROJECT_STATUS.md.
* Nothing is pushed. Work is in local commits on `main`.

## 2. Open items, in priority order

1. **Simulator calibration.** The U10 simulator reproduces the winning
   margin (1.01 x real) but its absolute totals run 14 to 20% low and
   wickets we take run 28% low. Switching off the bowler-economy
   multiplier or the geometric-mean dismissal rule each fixes one number
   and breaks another, so the fix is a joint batter-by-bowler ball-level
   model on the ball-by-ball data, not a constant tweak. Treat optimiser
   output as indicative until then.
2. **Two games lack ball-by-ball** (`e7967782`, `67b87a4c`). A burst of
   requests earned a CloudFront 403, so fetching stopped. Do not retry
   soon. Re-run `fetch_scorecards.py` later (it resumes) or use an
   official PlayHQ API key. Batting rows for those games are skipped.
3. **Scoring-rate likelihood.** Runs per ball has variance 0.70 x its
   mean, so the Poisson model over-shrinks scoring rate (the raw record
   predicted better in the backtest). Use a likelihood that fits.
4. **Third period.** Pull 2024/25 (season `2470e549`) for the model's
   oldest period, politely and in small batches.
5. **2026/27.** Round 1 is Sat 10 Oct 2026 (season `077d0fa1`). Get the U11
   grade ID from the team's PlayHQ page, then fetch and refit.
6. **Ask PlayHQ for an API key** (help@playhq.com). The scripts use the
   website's undocumented endpoints; the documented API needs a key.

## 3. How to run (from the repo root)

```
python scripts/fetch_scorecards.py --team-id 75cdae66 --team-id fafdb4c4 --offline
python scripts/fit_model.py --batting data/processed/playhq_all_batting.csv --bowling data/processed/playhq_all_bowling.csv --sampler numpyro
python scripts/backtest.py --batting data/processed/playhq_batting.csv --bowling data/processed/playhq_bowling.csv
python scripts/check_u10_totals.py --team-id 75cdae66 --team-id fafdb4c4
python scripts/optimise_lineup.py --posterior data/outputs/posterior_model.nc --squad P01,P03,P04,P05,P06,P07,P08,P09,P10
```

`--offline` builds from the cache in `data/raw/playhq/` and makes no
requests. Drop it (and set `--max-games 3`) to fetch, slowly.

## 4. Things that will bite

* **Privacy.** The GitHub repo is public. Raw PlayHQ pulls hold children's
  names (`data/raw/`, gitignored on `main`). The CSVs the model reads use
  aliases (P01..; opposition O01..). The alias-to-name maps are
  `data/raw/playhq/player_map.csv` and `opposition_map.csv`. Never commit
  them, and stage files explicitly, never `git add -A`.
* **Tenant headers differ by service.** `api.playhq.com` needs
  `tenant: cricket-australia` (with `ca` the innings scores come back
  empty); the spectator service needs `x-phq-tenant: ca`.
* **One working tree, one session.** Two sessions plus a terminal in one
  folder moved branches under each other today. Give a second session its
  own clone or `git worktree`.
* **U10 is not U11.** The real data is U10 (dismissed batters carry on,
  4 penalty runs per wicket to the other side, fixed ball allotments).
  The optimiser runs U11 rules, so its output on U10 skills is a wiring
  check, not advice.
* Rules: the September 2025 BNJCA rule book (Section 16) was read and
  matches `rules_u10.py`. `rules_u11.py` follows the 2026 booklet.
