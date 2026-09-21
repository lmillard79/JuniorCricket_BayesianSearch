# STATE.md - Session Close Handoff

Session: 2026-09-22, personal PC (game replays, U11 batting-strategy method)
Earlier: 2026-09-21 (real-data bring-up, validation, joint model)
Project: JuniorCricket_BayesianSearch (VDCC lineup optimiser)
Durable detail and all evidence: PROJECT_STATUS.md. How to read the outputs:
docs/GUIDE.md. The U11 batting-order method and pathway: docs/BATTING_STRATEGY_METHOD.md.

## 1. Current status

* Runs end to end on the personal PC (Windows 11, Python 3.14.7, PyMC
  6.3.2, ArviZ 1.x). `python -m pytest tests/` gives 142 passed. No C
  compiler needed: sampling uses numpyro/JAX.
* **Game replays** (`scripts/replay_games.py`): each real U10 game is rebuilt and
  replayed thousands of times; recorded totals are placed against the replay
  median (above for our total in 7 of 12 games, average percentile 61st: the
  known understatement of our side); batting order is worth about +0.8 runs of
  margin at best and the bowling split +0.7 (keepers as played), against a
  28-run luck spread. Over-by-over Manhattan and worm charts per game.
* **U11 batting strategies** (`scripts/compare_batting_strategies.py`): named orders
  played through the same thousands of "worlds" of unknown opposition, tactics and
  ground. Strongest to weakest was not beaten by anything tried, including
  strong-weak pairings and a swap search, in 36 scenarios. Strong-weak alternating
  costs about 7 runs; leading with the weakest about 19. Indicative: U10 skills.
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
   bowlers explain (0.80 x), and the replays put our totals above the median
   in most games. Add a team fielding effect (and try ball-count and position
   effects), then re-run `check_u10_totals.py` and `replay_games.py`.
2. **Per-batter retirement in the U11 engine.** `simulate_innings` takes one
   retirement threshold for the whole team; the rule seems to let the coach
   retire batters individually between 25 and 35 balls. Retiring everyone at 25
   instead of 35 costs about 5 runs, so a per-batter policy is the next lever.
   Confirm the rule first.
3. **Wicketkeeper flags.** The optimiser cannot tell who keeps; the two players
   given one over are the keepers by rule. Wire flags into the pipeline.
4. **U10 to U11 transfer.** U11 boundaries are 45 m (U10 30 to 35 m). The U10
   boundary rate will not carry over at face value; the strategy grid tests
   shifts of 0, -0.7 and -1.4 on the boundary log-odds. First U11 games are the
   first direct test: refit with a U11 offset per outcome, and compare balls
   faced by batting position and the boundary rate with the model's.
5. **Two games lack ball-by-ball** (`e7967782`, `67b87a4c`). A burst of
   requests earned a CloudFront 403. Do not retry soon; use an API key (the
   user is requesting one) or re-run `fetch_scorecards.py` later (it resumes).
   One more game (`ee765d0d`) was shortened to 18 overs and is left out of the replays.
6. **2024/25** (season `2470e549`) as an older period; **2026/27** (season
   `077d0fa1`, Round 1 Sat 10 Oct 2026): get the U11 grade ID from the team
   page, then fetch, refit, and run `compare_batting_strategies.py --squad ...`
   for the real squad.

## 3. How to run (from the repo root)

```
python scripts/fetch_scorecards.py --team-id 75cdae66 --team-id fafdb4c4 --offline
python scripts/fit_ball_model.py
python scripts/fit_model.py --batting data/processed/playhq_all_batting.csv --bowling data/processed/playhq_all_bowling.csv --sampler numpyro
python scripts/check_u10_totals.py --team-id 75cdae66 --team-id fafdb4c4 --ball-posterior data/outputs/ball_posterior.nc --opposition real
python scripts/optimise_lineup.py --posterior data/outputs/posterior_model.nc --ball-posterior data/outputs/ball_posterior.nc --squad P01,P03,P04,P05,P06,P07,P08,P09,P10
python scripts/player_report.py --squad P01,P03,P04,P05,P06,P07,P08,P09,P10 --named
python scripts/name_report.py data/outputs/lineup_recommendation.md
python scripts/replay_games.py --team-id 75cdae66 --team-id fafdb4c4
python scripts/compare_batting_strategies.py --search
```

The replay takes about 10 minutes and the strategy comparison about 5 (both use
all but two cores). Outputs: `data/outputs/replay/replay_report.html` and
`data/outputs/strategies/strategy_report.html` (gitignored; aliases and public
team names only).

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
* **Use the paired comparison for order questions.** `optimise_lineup.py` gives
  each candidate order its own random opposition, so differences of a few runs
  between neighbouring orders are noise. `compare_batting_strategies.py` plays
  every order through the same worlds.
* **Worker processes on Windows re-import the script.** `replay_games.py` and
  `compare_batting_strategies.py` start worker processes that re-import the
  module; keep heavy imports (arviz, matplotlib) inside functions, and keep
  PyMC out of `ball_model.py`'s top level (it is imported where it is used).
* **In-sample.** The replay percentiles and the totals check have seen these
  games; they flag odd days and model gaps, they are not forecasts.
* Rules: the September 2025 BNJCA rule book (Section 16) was read and matches
  `rules_u10.py`. `rules_u11.py` follows the 2026 booklet.
