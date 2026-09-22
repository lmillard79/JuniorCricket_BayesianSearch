# PROJECT_STATUS.md

## Junior Cricket Bayesian MARCEL - VDCC Lineup Optimiser

Last updated: 2026-09-22 (personal PC, game replays, U11 batting-strategy method)

How to read every output: `docs/GUIDE.md`. U11 batting-order method, pilot
results and pathway: `docs/BATTING_STRATEGY_METHOD.md`.

## Current state

**Phase: real U10 data ingested and validated. A joint batter-by-bowler
ball model fixes most of the simulator's bias and now drives the optimiser.
Real games can be replayed and reshuffled, and U11 batting-order strategies
are compared against an unknown opposition. What remains is a team fielding
effect and the U10-to-U11 transfer.**

| Component | Status | Where |
| --- | --- | --- |
| BNJCA U10 and U11 playing conditions | Done. U10 checked against the Sept 2025 rule book, Section 16 | `rules_u10.py`, `rules_u11.py` |
| PlayHQ client (fixtures, results, scorecards, ball-by-ball) | Done, live-verified. Polite: spaced, stops on 403 or 429 | `playhq_client.py` |
| Scorecard and event parsing, name linking, reconciliation | Done, tested on 26 real innings | `playhq_parse.py` |
| Pseudonymised scorebook and ball rows (ours and opposition) | Done | `playhq_scorebook.py`, `scripts/fetch_scorecards.py` |
| Per-player model v2 (non-centred local level) | Done. Samples real data | `model.py` |
| Per-player model v1 (original) | Kept behind `--model v1`; does not sample real data | `model.py` |
| **Joint batter-by-bowler ball model** | Done. Drives the optimiser via `--ball-posterior` | `ball_model.py`, `scripts/fit_ball_model.py` |
| Out-of-sample backtests | Done (per-player and per-ball) | `backtest.py`, `scripts/backtest.py` |
| U10 innings engine and totals check | Done | `simulator.py`, `scripts/check_u10_totals.py` |
| U11 simulator and optimiser | Works; run-out bookkeeping fixed; accepts the joint model | `simulator.py`, `optimizer.py` |
| Player report and private name translation | Done | `scripts/player_report.py`, `scripts/name_report.py` |
| **Game replays and decision studies** (U10) | Done. Over-by-over charts, percentile of each recorded result, batting-order and bowling-split studies | `replay.py`, `replay_plots.py`, `scripts/replay_games.py` |
| **U11 batting-order strategies** | Done as a pilot. Paired worlds, named strategies, swap search on fresh worlds, 36-scenario stress grid | `strategies.py`, `scripts/compare_batting_strategies.py` |
| Tests | 143 passing | `tests/` |

## Real data

VDCC U10 White, BNJCA Summer 2025/26 (season `5a7c9a42`): pre-Christmas team
`75cdae66` (grade U10 Walters), post-Christmas team `fafdb4c4` (grade U10
Kasprowicz Post). 17 games, 15 completed, 2 abandoned. Squad: 10 players (9
with PlayHQ profiles, plus a fill-in). Opposition: 87 players.

* Per-player scorecards for all 15 games; ball-by-ball for 13. Games
  `e7967782` and `67b87a4c` have no events (a CloudFront 403 stopped the pull),
  so their batting rows and balls are skipped and their bowling rows kept.
* 3,095 deliveries in `playhq_balls.csv` (88 batters, 88 bowlers).
* Data quality: events reconcile exactly with the scorecard in 23 of 26
  innings (three differ by one ball or run; once the scorecard shows 121
  balls where the rules allow 120, and the events are right). Team total =
  batter runs + 4 x dismissals taken holds in 25 of 26 innings.
* U10 facts measured: 0.0556 dismissals per ball (6.6 per innings); 24% of
  dismissals are run outs and the striker is out in 83% of those; no wide or
  no-ball events are recorded; no stumpings.

## Per-player model findings

Original model: 4000 of 4000 draws diverged. It also cannot learn its recency
weights. v2: max R-hat 1.007 (squad) and 1.024 (whole grade), no divergences.
Between-player spread of skill (logit or log scale, squad only): boundary rate
0.85 (strong), dismissal hazard 0.26, wicket rate 0.20, scoring rate 0.14,
bowler economy 0.06. The grade differs from White's own squad (mean dismissal
hazard 0.053 vs 0.043; economy 1.0 vs 0.82 runs per ball), so the opposition
prior must come from the whole grade.

Backtest of that model (fit early games, score later ones), model minus
baseline in log predictive density, period split then rolling blocks:

| Family | vs pooled | vs raw | vs pooled | vs raw | corr |
| --- | --- | --- | --- | --- | --- |
| Boundary rate | +22.4 | +5.8 | +31.2 | +0.4 | 0.76-0.81 |
| Dismissal hazard | +5.4 | +3.6 | +3.3 | +2.1 | 0.23-0.25 |
| Scoring rate | +4.7 | -1.3 | +5.5 | -3.8 | 0.64-0.65 |
| Wicket rate | +0.8 | +1.3 | +0.2 | +2.8 | 0.26-0.36 |
| Bowler economy | -1.9 | +0.1 | +1.1 | +5.6 | 0.07-0.32 |

Scoring rate lost to the raw record because runs per ball has variance 0.70 x
its mean, so the Poisson likelihood over-shrinks. Small samples: only the
boundary result is clearly outside the noise.

## Joint ball model (the current basis for player and order statements)

Each ball's probability of a dismissal, a boundary (given not out) and a
scoring shot (given neither) depends on batter and bowler together, additively
on the logit scale, plus a per-game effect on boundary and scoring. It fits
cleanly: max R-hat 1.008, min ESS 608, no divergences.

Baseline (average batter vs average bowler): dismissal 0.051 per ball, boundary
0.075 per non-dismissal ball, scoring shot 0.514 per non-boundary ball.

Player spread (SD on the logit scale, median and 90% interval):

| Outcome | Batters | Bowlers |
| --- | --- | --- |
| Dismissal | 0.39 [0.14, 0.62] | 0.31 [0.05, 0.56] |
| Boundary | **0.84 [0.65, 1.07]** | 0.22 [0.02, 0.46] |
| Scoring shot | 0.39 [0.27, 0.52] | **0.36 [0.23, 0.50]** |

Boundary hitting is a batter trait; bowlers differ in wicket-taking and in the
scoring shots they concede. Ground and conditions vary game to game (boundary
0.26, scoring 0.30). Held-out balls, three windows of two games (1,440 balls):
batter effects add +26.6 log-score units for boundaries (+0.2 dismissal, +0.6
scoring); bowler effects add a further +1.4, +1.1 and +2.7. Small but
consistent.

## U10 simulator check (in-sample; simulated / real)

| Quantity | Per-player rates (13 games) | Joint, real opposition (13) | Joint, population opposition (14) |
| --- | --- | --- | --- |
| Our runs off the bat | 0.89 | 0.93 | 0.92 |
| Opposition runs | 0.75 | **0.97** | **1.03** |
| Wickets we took | 0.72 | 0.80 | 0.85 |
| Wickets they took | 1.05 | 1.04 | 1.11 |
| Our final total | 0.86 | 0.90 | 0.90 |
| Opposition final total | 0.80 | **0.98** | **1.04** |
| Margin | 1.01 (luck) | 0.65 | 0.50 |

The old model's totals were low because a bowler's economy and a batter's
scoring were estimated separately and double-counted opposition quality; its
margin was right only because errors cancelled. The joint model fixes the
opposition side. It still understates White's own strength (our total 0.90 x,
wickets we take 0.80 x), so the simulated margin is about half the real one.
Likely cause: a team fielding effect (White's wickets exceed what its
individual bowlers explain), plus possible ball-count and position effects.

## Order stakes (joint model; U11 rules; indicative)

* U10: best-first vs worst-first batting order differs by about 1 net run
  (noise). Everyone faces 13 or 14 balls whatever the order.
* U11 batting order: best vs an average order +7.1 runs, worst -8.2 (single
  innings SD about 39), so roughly 3 to 4 win-probability points. The two
  power hitters score the same anywhere in slots 1 to 6; burying them in 8 and
  9 costs about 15 runs. (Superseded in detail by the paired comparison below.)
* U11 bowling split: the best allocation concedes about 4.5 fewer runs than an
  average split (about 10 fewer than the worst).

## U10 game replays (12 full-length games; `scripts/replay_games.py`)

Each game is rebuilt from its scorecard and ball rows (batting orders, the
bowler of every over) and replayed 4,000 times, each replay drawing a fresh
plausible set of player skills from the joint posterior (300 draws) and a typical
ground. In-sample.

* **Above or below the median?** Our recorded total was above the median replay
  in 7 of 12 games (average percentile 61st); theirs in 6 of 12 (44th); the margin
  in 9 of 12 (64th). A calibrated model sits near 50th, so the model is stingy
  about our side (the known gap of about 10%); it is calibrated for the opposition.
  Extremes: +116 recorded margin against a replay median of +41 (100th
  percentile) on 2025-10-18, and -35 against +10 (6th) on 2026-01-31.
* **Luck.** A single game's margin has a spread (1 SD) of about 28 runs with the
  lineups fixed.
* **Batting order.** Best of 40 random and 4 rule-of-thumb orders, screened on
  2,000 replays and judged on 10,000 fresh ones: **+0.8 runs** of margin on average
  (95% interval about +0.6 to +1.0). Random orders differ by only about 0.4 runs
  (SD, corrected for simulation noise).
* **Bowling split** (same overs, same bowlers, different shares; round-robin
  sequence): **+0.7 runs** with the two keepers left as played; **+3.0** if the
  keepers could also be chosen freely, which is optimistic because keeper skill is
  not modelled. Random reassignments differ by about 0.8 runs (SD).
* **Together** the best batting order and bowling split found would have added
  about 0.18 expected wins across the 12 games.
* **Shape of an innings.** Averaged over the games, recorded runs per over run above
  the replay average in our innings from about over 10 to 18, and scatter around it
  in theirs: a hint of something the model misses (a team effect or a
  within-innings pattern), to be checked once the team effect is added.

Data: the over number is now carried in `playhq_balls.csv` (one game has an over
with a missing ball, which the stamp keeps aligned). Games left out: `ee765d0d`
(shortened to 18 overs) and the two without ball-by-ball.

## U11 batting-order strategies (`scripts/compare_batting_strategies.py`)

Full method, table and pathway: `docs/BATTING_STRATEGY_METHOD.md`. In brief: each
strategy is played through the same thousands of worlds (true skills drawn from
the posterior, nine unknown opposition players from the grade, random or smart
tactics, ground), so differences are paired; a swap search is judged on fresh worlds.

Main scenario (retire at 35, U10 boundary rates, random attack), runs against
strongest to weakest: strong-middle alternating -0.3 ± 0.3, balanced pairs (top
six) -0.8 ± 0.4, strong-weak alternating -7.4 ± 0.4, weakest to strongest
-18.7 ± 0.5, random orders -9.1 on average. The swap search found nothing better
than strongest to weakest, and among these *fixed orders*, none of the 35 other
scenarios (retirement 25, 30 or 35; boundary shift 0, -0.7, -1.4; skill drift 0 or
0.3; random or smart attack) found an alternative order beating it by more than 0.2
runs. Split by opposition attack strength (thirds of the worlds), strong-weak
alternating loses -7.3 ± 0.6 against the strongest attacks and -8.0 ± 0.7 against
the weakest, so pairing does not shield the best batters from the best bowling. The
mechanism is who gets the balls: the six best batters face 17 to 20 balls each
under the conventional order.

**Per-batter retirement: "bank top 4, recall on a cheap wicket".** Confirmed with
the user (22 Sep 2026): coaches may retire batters individually between 25 and 35
balls (allowed, not yet used on the day), and the objective includes participation,
not only total runs. Built as a new engine feature (`simulator.py`: `bankable`,
`recall_within_balls`, `recall_below_runs` on `simulate_innings`; `strategies.py`:
`StrategyTask` gains the same fields, plus `evaluate_one` for a strategy that needs
its own retirement rule rather than the one `evaluate` shares across a batch) and
tested via `scripts/compare_batting_strategies.py` alongside the fixed orders, using
the user's own framing of the policy: the top 4 batters may retire not out at 25
balls instead of batting on; the strongest currently-banked batter is recalled
immediately, ahead of the next fresh batter, if whoever comes in next is out within
6 balls or for under 5 runs (both thresholds are illustrative defaults, not fitted).
Result: **+1.3 ± 0.2 runs** against strongest to weakest in the main scenario, and
the best alternative in **all 37 scenarios tested** (main plus the 36-scenario
grid), ranging +0.7 to +7.4. It helped slightly more against the strongest third
of opposition attacks (+1.8) than the weakest (+1.2). Balls-by-rank shows the gain
comes from the best batter facing about 2 more balls (20.1 to 22.1) at the expense
of about 1 fewer for the weakest (5.3 to 4.1): the runs-maximising version of the
policy is not, by itself, a direct answer to "everyone should get a go"; a
fairness-weighted recall rule (protect the weakest batters' balls rather than
maximise runs) has not been built. Retiring *every* batter at 25 rather than 35
(no bankable subset) still costs about 5 runs at the current ground size.

**Team fielding effect: design, not yet built.** The user approved adding this
(22 Sep 2026) but has no view on mechanism. Plan: `ball_model.py`'s joint model
already gives each component (`d` dismissal, `b` boundary, `s` scoring) a batter
effect, a bowler effect, and (for `b`/`s` only) a per-game effect shared by both
sides batting that day. Add one more term to `d` only: a per-game effect specific
to *our* team's fielding (a `HalfNormal` scale plus a `Normal` group mean, non-centred
by game, mirroring the existing `game_b`/`game_s` pattern), added to a ball's dismissal
log-odds only when the bowler is one of ours. Checked: PlayHQ's raw scorecards do not
tag who kept or fielded well (grepped a sample `scorecard.json`, no match), so this
cannot be a per-fielder effect; it is scoped to a team-level, per-game nudge, which
needs no new data collection (only which side bowled each ball, already known from
the alias convention: `P0x` ours, `O0x` opposition). This directly targets the
diagnosed bias (section "U10 simulator check" below): our own bowlers currently take
more wickets than their individual skills explain. Not yet built: needs a refit
(`fit_ball_model.py`) and will shift the headline replay and strategy numbers above,
so it is being kept as its own step rather than folded into this session's changes.
Wicketkeeper flags (`is_wicketkeeper` already exists in the player registry and
`PlayerSkills`, but is only wired into bowling-tier allocation, not dismissal
modelling) would need a real per-game record of who kept, which PlayHQ does not
provide; blocked on the coach supplying it.

## Caveats that matter for 2026/27

* **Field size and ball.** U11 boundaries are 45 m (U10: 30 to 35 m) and U11
  may use a 130 g ball. The U10 boundary rate (about 7.5% of balls, up to 22%
  for the best hitters) will not carry over at face value. Rankings probably
  persist; absolute boundary value will fall, raising the weight on scoring
  and staying in.
* **Wicketkeepers.** The optimiser does not know who keeps. The two players
  given one over are by rule the keepers; flag them or the bowling advice is
  unreliable.
* **Skill transfer** U10 to U11 is an assumption. Only 10 players and 15 games.
* The check is in-sample; the backtests are the out-of-sample evidence.

## Next steps (in order)

1. Add the team fielding effect (design above) to the joint model; refit; re-run
   the totals check, the replays and the strategy comparison (numbers above will
   move).
2. Wire wicketkeeper flags into the pipeline, if the coach can supply who kept
   each game.
3. Consider a fairness-weighted variant of bank-and-recall that protects the
   weakest batters' balls specifically, if the current runs-maximising version
   (see above) is not what is wanted.
4. Fetch the two missing games and 2024/25 (API key, or slowly later).
5. When BNJCA publishes the 2026/27 U11 draw (Round 1 Sat 10 Oct 2026), get
   the grade ID, fetch, fit, and rerun the strategy comparison and optimiser for
   the real squad. First U11 games give the first direct test of the transfer
   assumption (balls faced by batting position, boundary rate).

## Key decisions log

- Two per-player model versions coexist so v1's failure stays reproducible.
- Opposition rows are fitted with our squad so priors reflect the grade.
- The joint model replaces the hand-built combination rules (geometric-mean
  dismissal, economy multiplier) through an optional `outcomes` hook, so the
  legacy behaviour and its tests are untouched.
- Games without events skip batting rows rather than record zero dismissals.
- Player names never enter model CSVs; aliases only. `player_report.py --named`
  and `name_report.py` write private, gitignored named copies on request.
- The 4-run U10 penalty goes to the fielding side's total (Rule 16.10(vi)).
- Extras are not modelled when none are recorded; the population prior is used.
- Order and bowling alternatives are compared on the same worlds (paired), screened
  on one set of replays and judged on a fresh set, because picking the best of many
  noisy estimates flatters it. The older optimiser's unpaired search is not used for
  order questions.
- Replays draw skills from a pool of posterior draws (not the posterior mean), so
  intervals include uncertainty about each player as well as ball-to-ball luck.
- A batter's value for ordering is the expected runs in a 30-ball stint with a
  dismissal ending it.
- Bank-and-recall's recall priority is the caller-supplied `bankable` order
  (strongest first); the engine does not rank players itself, so the same
  mechanism works for a synthetic test squad or the real one.
- A recalled batter cannot retire again this innings (`resumed`, pre-existing
  flag reused): without this, an indestructible recalled batter and a
  still-batting partner can both become un-dismissable, freezing the wicket
  count for the rest of the innings.

## Environment notes

- Python 3.14.7 in `.venv`; PyMC 6.3.2, PyTensor 3.3.2, ArviZ 1.3, numpyro
  0.22, jax 0.11.2, h5netcdf and h5py. `requirements.txt` lists them.
- No C compiler: use `--sampler numpyro` (the joint model always does). The
  PyTensor "g++ not available" line is harmless.
- `data/raw/`, `data/processed/`, `data/outputs/`, `log/` and `*.nc` are
  gitignored on `main`.
