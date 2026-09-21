# STATE.md - Session Close Handoff

Session: 2026-09-21, personal PC (real-data bring-up, validation, joint model)
Project: JuniorCricket_BayesianSearch (VDCC lineup optimiser)
Durable detail and all evidence: PROJECT_STATUS.md.

## 1. Current status

* Runs end to end on the personal PC (Windows 11, Python 3.14.7, PyMC
  6.3.2, ArviZ 1.x). `python -m pytest tests/` gives 104 passed. No C
  compiler needed: sampling uses numpyro/JAX.
* Real data: VDCC U10 White 2025/26 (teams `75cdae66`, `fafdb4c4`), 15
  completed games, ball-by-ball for 13 (3,095 deliveries), plus 87 opposition
  players.
* A joint batter-by-bowler ball model is fitted and drives the optimiser
  (`--ball-posterior`). It fixes most of the simulator's bias: opposition
  totals now 0.98 to 1.04 of real (were 0.80). It still understates White's own
  strength, so treat margins as conservative.
* Local commits on `main`; the push needs the user's GitHub sign-in (git
  credential manager cannot prompt from the agent shell).

## 2. Open items, in priority order

1. **Team fielding effect.** White took more wickets than its individual
   bowlers explain (0.80 x). Add a team fielding effect (and try ball-count and
   position effects), then re-run `check_u10_totals.py`.
2. **Wicketkeeper flags.** The optimiser cannot tell who keeps; the two players
   given one over are the keepers by rule. Wire flags into the pipeline.
3. **U10 to U11 transfer.** U11 boundaries are 45 m (U10 30 to 35 m). The U10
   boundary rate will not carry over at face value. First U11 games are the
   first direct test.
4. **Two games lack ball-by-ball** (`e7967782`, `67b87a4c`). A burst of
   requests earned a CloudFront 403. Do not retry soon; use an API key (the
   user is requesting one) or re-run `fetch_scorecards.py` later (it resumes).
5. **2024/25** (season `2470e549`) as an older period; **2026/27** (season
   `077d0fa1`, Round 1 Sat 10 Oct 2026): get the U11 grade ID from the team
   page, then fetch, refit and optimise.

## 3. How to run (from the repo root)

```
python scripts/fetch_scorecards.py --team-id 75cdae66 --team-id fafdb4c4 --offline
python scripts/fit_ball_model.py
python scripts/fit_model.py --batting data/processed/playhq_all_batting.csv --bowling data/processed/playhq_all_bowling.csv --sampler numpyro
python scripts/check_u10_totals.py --team-id 75cdae66 --team-id fafdb4c4 --ball-posterior data/outputs/ball_posterior.nc --opposition real
python scripts/optimise_lineup.py --posterior data/outputs/posterior_model.nc --ball-posterior data/outputs/ball_posterior.nc --squad P01,P03,P04,P05,P06,P07,P08,P09,P10
python scripts/player_report.py --squad P01,P03,P04,P05,P06,P07,P08,P09,P10 --named
python scripts/name_report.py data/outputs/lineup_recommendation.md
```

`--offline` builds from the cache in `data/raw/playhq/` and makes no requests.
Drop it (and set `--max-games 3`) to fetch, slowly.

## 4. Things that will bite

* **Privacy.** The GitHub repo is public. Raw pulls hold children's names
  (`data/raw/`, gitignored on `main`). Model CSVs use aliases (P01..;
  opposition O01..); the alias-to-name maps are `data/raw/playhq/player_map.csv`
  and `opposition_map.csv`. `player_report.py --named` and `name_report.py`
  write private named copies under `data/outputs/` (gitignored). Never commit
  those; stage files explicitly, never `git add -A`.
* **Tenant headers differ by service.** `api.playhq.com` needs
  `tenant: cricket-australia` (with `ca` scores come back empty); the
  spectator service needs `x-phq-tenant: ca`.
* **One working tree, one session.** Two sessions plus a terminal in one
  folder moved branches under each other. Give a second session its own clone
  or `git worktree`.
* **U10 is not U11.** The optimiser runs U11 rules on skills learned from
  U10 games, so its output is indicative, not advice.
* Rules: the September 2025 BNJCA rule book (Section 16) was read and matches
  `rules_u10.py`. `rules_u11.py` follows the 2026 booklet.
