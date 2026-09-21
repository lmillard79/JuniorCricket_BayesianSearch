# PROJECT_STATUS.md

## Junior Cricket Bayesian MARCEL - VDCC Lineup Optimiser

Last updated: 2026-09-21 (personal PC, real-data validation, joint model)

## Current state

**Phase: real U10 data ingested and validated. A joint batter-by-bowler
ball model fixes most of the simulator's bias and now drives the optimiser.
What remains is a team fielding effect and the U10-to-U11 transfer.**

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
| Tests | 104 passing | `tests/` |

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
  9 costs about 15 runs.
* U11 bowling split: the best allocation concedes about 4.5 fewer runs than an
  average split (about 10 fewer than the worst).

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

1. Add a team fielding effect (and test ball-count and position effects) to
   the joint model; re-run the totals check.
2. Wire wicketkeeper flags into the pipeline.
3. Fetch the two missing games and 2024/25 (API key, or slowly later).
4. When BNJCA publishes the 2026/27 U11 draw (Round 1 Sat 10 Oct 2026), get
   the grade ID, fetch, fit, and optimise with `--squad` and `--ball-posterior`.
   First U11 games give the first direct test of the transfer assumption.

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

## Environment notes

- Python 3.14.7 in `.venv`; PyMC 6.3.2, PyTensor 3.3.2, ArviZ 1.3, numpyro
  0.22, jax 0.11.2, h5netcdf and h5py. `requirements.txt` lists them.
- No C compiler: use `--sampler numpyro` (the joint model always does). The
  PyTensor "g++ not available" line is harmless.
- `data/raw/`, `data/processed/`, `data/outputs/`, `log/` and `*.nc` are
  gitignored on `main`.
