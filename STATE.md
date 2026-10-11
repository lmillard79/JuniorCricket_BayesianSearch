# STATE.md - Session Close Handoff

Session: 2026-10-11, personal PC (Round 1 of 2026/27: first U11 data, strike rules
corrected, U11-like scenario dials)
Earlier: 2026-09-22 (game replays, U11 batting-strategy method, bank-and-recall,
team fielding effect, VDCC Navy), 2026-09-21 (real-data bring-up, validation, joint model)
Project: JuniorCricket_BayesianSearch (VDCC lineup optimiser)
Durable detail and all evidence: PROJECT_STATUS.md. How to read the outputs:
docs/GUIDE.md. The U11 batting-order method and pathway: docs/BATTING_STRATEGY_METHOD.md.

## 1. Current status

* Runs end to end on the personal PC (Windows 11, Python 3.14.7, PyMC
  6.3.2, ArviZ 1.x). `python -m pytest tests/` gives 196 passed. No C
  compiler needed: sampling uses numpyro/JAX.
* **First U11 data (Round 1, Sat 10 Oct 2026; docs/BATTING_STRATEGY_METHOD.md section
  4b).** Navy 142/7 beat Orange 56/8 (Hartley pool), complete on PlayHQ with ball-by-ball.
  White 131/4 beat Turquoise 91/7 (Smith pool, 22-over game): PlayHQ has only totals and a
  line-up (scorecard LIVE, events empty), so nothing player-level. Fetched into
  `data/u11_white/` and `data/u11_navy/`. Findings: (1) the engine's strike rule was wrong,
  batters cross on odd runs (rule 17.4(iv); 30 of 33 odd-run balls) and a new batter
  usually takes strike after a wicket; fixed on `rules_u11.py` (`batters_cross_on_runs`,
  `incoming_batter_takes_strike`), conclusions unchanged; (2) U11 scoring differs from U10:
  scoring shots 48% to 24% of non-boundary balls, wides and no-balls 18% of deliveries,
  boundaries slightly rarer, dismissals the same (one game: scenario dials
  `--scoring-shift`, `--extras-rate`, `--boundary-shift`, not fitted effects); (3) in that
  U11-like world strongest to weakest still wins among fixed orders and bank-and-recall
  is worth about +3 to +4 runs (best in all 37 scenarios), BUT see the next bullet.
* **Bank-and-recall's recall step is not allowed as written (found 11 Oct 2026).** Rule
  17.9(v): a retiree may resume only "once all other players have been dismissed or retired",
  so recalling a banked batter ahead of a fresh one needs both coaches to agree. I did not
  check this when it was built on 22 Sep. In Round 1 there were 16 retirements and 10
  resumptions across six innings, every resumption with no fresh batter left. What the rules
  allow is priced by `scripts/price_retirement.py` (method note section 4c): in a U11-like world
  retiring everyone at 25 costs about 3.1 runs and gives the weakest batter 4 more balls
  (8.0 to 12.2); retiring the top four only costs 4.1 for 3 more.
* **Smith-pool scouting (11 Oct 2026).** The other pool teams' finished games are cached in
  `data/smith_pool/`; `scripts/scout_pool.py` builds a private report that names children
  (`data/smith_pool/outputs/scouting/scouting_report.md`, gitignored, local only). Round 1:
  Bulls White 156/6 and Bulls Yellow 135 won; WNJCC Purple and Teal lost; Ashgrove (White's
  Round 2 opponent) had the bye so there is no data yet. Last season Bulls teams beat White or
  Navy 2 of 3 times, WNJCC teams lost 11 of 11. The report also has a team-strength model
  (`scouting.fit_team_strengths`, `matchup`): after adjusting for the weak attacks they faced and
  starting each club from last season's record, Bulls White is rated about level with White (White
  57% to win, needing about 117 in 22 overs for an even chance and 138 for three in four), and both
  Bulls sides are top-four batting teams (the other five batters made 7 and 9 runs).
* **Heat and retiring at 20 (11 Oct 2026).** `price_retirement.py` now reports time at the crease
  (`BatterCard.crease_deliveries`, `longest_stint`, `first_stint`) and how often the weakest batter
  bats. A first stint is about 43 minutes of a 90-minute innings at the 35 cap, 34 at 25, 29 at 20, but
  the total time out there is unchanged because retirees return; retire at 20 costs about the same runs
  as 25 and needs both coaches' agreement (minimum 25 balls, wides and no-balls counting).
* **Team fielding effect** (new this session): the user corrected an earlier wrong
  claim that PlayHQ's data does not name fielders — it does, in both the scorecard
  stats and the ball-by-ball events feed, the second of which `playhq_parse.py` was
  already parsing and just not carrying further. Added a per-game effect on
  dismissal specific to our own team's fielding (`ball_model.py`: `mu_f_d`, `sf_d`,
  `field_d`), refit: **0.17 [-0.15, 0.47]** on the logit scale, 81% posterior
  probability positive — real but modest, narrows the "10% stingy on our side" gap
  without closing it. Individual catches/run-outs are now recorded per ball too
  (`playhq_scorebook.py`'s `fielder` column) and reported descriptively (not
  fitted) in `player_report.md`'s new Fielding table.
* **Game replays** (`scripts/replay_games.py`): each real U10 game is rebuilt and
  replayed thousands of times; recorded totals are placed against the replay
  median (above for our total in 7 of 12 games, average percentile 59th, was 61st
  before the fielding effect); batting order is worth about +0.9 runs of
  margin at best and the bowling split +0.7 (keepers as played), against a
  29-run luck spread. Over-by-over Manhattan and worm charts per game.
* **U11 batting strategies** (`scripts/compare_batting_strategies.py`): named orders
  played through the same thousands of "worlds" of unknown opposition, tactics and
  ground. Among fixed orders, strongest to weakest was not beaten by anything tried,
  including strong-weak pairings and a swap search, in 36 scenarios. Strong-weak
  alternating costs about 7 runs; leading with the weakest about 19. Indicative:
  U10 skills.
* **Bank-and-recall retirement policy** (new this session): confirmed with the user
  that per-batter retirement (25 to 35 balls) is allowed but unused, and that the
  objective includes participation. Built into `simulator.py`/`strategies.py`: the
  top 4 batters may retire not out at 25 balls; the strongest banked one is recalled
  immediately if the next batter is out within 6 balls or for under 5 runs. Result:
  +1.4 ± 0.2 runs against strongest to weakest, and the best alternative in all 37
  scenarios tested (the only one that gained runs rather than lost them). Caveat: it
  shifts a couple of balls toward the best batter and away from the weakest, so it is
  not by itself a fairness win for the weakest batters specifically.
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

1. **White's Round 1 record.** PlayHQ has only the totals and a line-up of eight (one new
   to the data). Ask for the batting order, runs and balls, bowlers and keepers, or re-check
   the scorecard later with `fetch_scorecards.py --team-id 2320bb54 --data-dir u11_white
   --refresh` (about four requests: it re-reads the fixture, so new rounds appear, and
   re-asks only about recent games that were not FINAL or had empty events; checked at
   12:20 on 11 Oct 2026 and still LIVE with no events). Ask whoever scores White's home
   games to use live ball-by-ball scoring.
2. **Weekly routine after each Saturday.** `fetch_scorecards.py --refresh` for the pool
   teams (`--data-dir smith_pool`, team ids 44faa9cd ba4abc43 2ca59b41 050b5216 a7cd7e29) and
   for White and Navy, then `scout_pool.py` (see section 3). Ashgrove's first scorecard arrives
   after Round 2. The Hartley pool (Navy's) is not pulled; ask first. The pool's ball-by-ball
   should later feed the joint model as the U11 opposition prior.
3. **U11 refit.** After a few more U11 games, refit the joint model with a U11 offset on each
   outcome and a wides rate instead of the fixed scenario dials, and compare predicted with
   actual balls by batting position. A U11 version of `replay_games.py` does not exist yet
   (the replay engine is U10-only).
4. **Retirement policy.** Agree with the coach what run cost is acceptable for the weakest
   batters' balls and the strong batters' rest in the heat (method note section 4c has the price
   list, including retire at 20 and stint lengths), whether to ask the opposing coach to agree to
   a below-25 minimum in hot weather, and whether to ask about early resumption of a retiree.
   Not modelled: a returning batter retiring again after a further 25 balls, and a tired
   retiree choosing not to resume (the booklet says "may"). The engine's early-recall thresholds
   (6 balls, 5 runs) were set at U10 scoring; re-tune them only if early recall is on the table.
5. **Wicketkeeper flags.** `is_wicketkeeper` already exists in the player registry and
   `PlayerSkills`, but only feeds bowling-tier allocation, not dismissal modelling. PlayHQ
   does not record who kept (checked properly: it does record catches and run outs, now
   used); a keeper flag would need the coach's own per-game record. Rule 17.2(ii): a change
   of wicketkeeper at the drinks break after over 13. Ask before building.
6. **Two U10 games lack ball-by-ball** (`e7967782`, `67b87a4c`). A burst of requests earned
   a CloudFront 403. Do not retry soon; use an API key (the user has asked PlayHQ, see the
   access-notes memory) or re-run `fetch_scorecards.py` later (it resumes). One more game
   (`ee765d0d`) was shortened to 18 overs and is left out of the replays. 2024/25 (season
   `2470e549`) is not fetched.

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
python scripts/compare_batting_strategies.py --search --boundary-shift -0.3 --scoring-shift -1.1 --extras-rate 0.18 --out data/outputs/strategies_u11_like
python scripts/price_retirement.py                  # what the retirement rules allow, priced (under a minute)
# scouting the pool (after each Saturday); the report names children, keep it local
python scripts/fetch_scorecards.py --data-dir smith_pool --refresh --team-id 44faa9cd --team-id ba4abc43 --team-id 2ca59b41 --team-id 050b5216 --team-id a7cd7e29
python scripts/fetch_scorecards.py --data-dir u11_white --refresh --team-id 2320bb54
python scripts/scout_pool.py
# 2026/27 U11 pulls (each in its own folder; seeded with the U10 alias maps); --refresh
# picks up new rounds and unfinished games, and is what to run after each Saturday
python scripts/fetch_scorecards.py --team-id 2320bb54 --data-dir u11_white --refresh
python scripts/fetch_scorecards.py --team-id 10d73647 --data-dir u11_navy --refresh
```

The replay takes about 10 minutes and the strategy comparison about 4 (both use
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
* **A second team runs the whole pipeline with `--data-dir <name>`**
  (`fetch_scorecards.py`, `fit_ball_model.py`, `check_u10_totals.py`,
  `player_report.py`, `replay_games.py`, `compare_batting_strategies.py`,
  `name_report.py`), keeping its raw cache, alias numbering, processed CSVs
  and outputs under `data/<name>/` instead of `data/`, fully separate so
  it cannot overwrite the default team's (`.gitignore` covers `data/*/{raw,
  processed,outputs}/` generically). Built 22 Sep 2026 for VDCC Navy
  (U10 Ponting grade, a different grade to White's, teams `7b1eba2e`
  pre-Christmas and `e3fff983` post; see the Navy note in
  playhq-access-notes.md). A *named* variant (real names used as the alias
  itself, so charts render with names, not just tables) is also possible:
  copy the raw cache to a new data-dir, rewrite that copy's `player_map.csv`/
  `opposition_map.csv` so `alias == name` (checking for real name collisions
  first), then run the pipeline `--offline` against it. `load_balls` needs
  `our_names` explicitly for this (real names have no P/O prefix to read
  "who's ours" from); `fit_ball_model.py` detects a non-P0x alias map and
  passes it automatically. Never commit or share a named variant's outputs
  beyond the family who asked for it.
* **The two U11 data dirs share alias numbering with the U10 ones only by copy.** They were
  seeded with copies of `player_map.csv` and `opposition_map.csv` so returning players keep
  their P-numbers; new players get new numbers in that folder's map only. If a pull is ever
  merged back into `data/` or `data/navy/`, copy the maps back first, or the numbering
  diverges. U11 events carry wides and no-balls as their own events (`icon` WD or NB, no
  score), which `playhq_parse.py` files under "not understood": harmless for the ball model.
* **The scouting report names children.** `data/smith_pool/outputs/scouting/` is under the
  gitignored `data/`; never commit or post it. Its processed CSVs (`data/smith_pool/processed/`)
  are NOT clean model inputs: `fetch_scorecards.py` treats each team id as "ours", so a game
  between two pool teams is processed twice with different aliases. Scouting reads the raw
  cache only; dedupe before ever modelling from those CSVs.
* **Strike options.** `U11` has `batters_cross_on_runs` and `incoming_batter_takes_strike`
  both True. `compare_batting_strategies.py --strike-rules earlier` (or
  `dataclasses.replace(U11, ...)` with both False) reproduces the engine as it was before
  11 Oct 2026, exactly (verified). A new `PlayingConditions` field must default so the U10
  rules still build.
* **Reading the BNJCA rules PDF.** WebFetch saves the PDF under the session's tool-results
  folder; the Read tool's PDF mode needs `pdftoppm` (not installed), but `pdftotext -layout`
  (mingw) works. Section 17 is the U11 boys' rules; 17.4(iv) and 17.9 matter most.
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
