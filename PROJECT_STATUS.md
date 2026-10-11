# PROJECT_STATUS.md

## Junior Cricket Bayesian MARCEL - VDCC Lineup Optimiser

Last updated: 2026-10-11 (Round 1 of 2026/27: first U11 data, strike rules corrected)

How to read every output: `docs/GUIDE.md`. U11 batting-order method, pilot
results and pathway: `docs/BATTING_STRATEGY_METHOD.md`.

## Current state

**Phase: real U10 data ingested and validated. A joint batter-by-bowler
ball model fixes most of the simulator's bias and now drives the optimiser.
Real games can be replayed and reshuffled, and U11 batting-order strategies
are compared against an unknown opposition. The team fielding effect is in.
The first U11 game (Round 1, 10 Oct 2026) is the first direct look at the
U10-to-U11 transfer; what remains is wicketkeeper flags, a proper U11 refit as
games arrive, and White's own Round 1 record (PlayHQ has only its totals).**

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
| **U11 batting-order strategies** | Done as a pilot, re-run 11 Oct 2026 with corrected strike rules and dials for the first U11 game's scoring and extras. Paired worlds, named strategies, swap search on fresh worlds, 36-scenario stress grid | `strategies.py`, `scripts/compare_batting_strategies.py` |
| Pool scouting from public scorecards (private, names children) | Done. Per-player lines, shrunk ratings, U10 history, White's draw | `scouting.py`, `scripts/scout_pool.py` |
| Retirement options priced (rule 17.9) | Done | `scripts/price_retirement.py` |
| Tests | 182 passing | `tests/` |

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
* 2026/27 U11 (pulled 11 Oct 2026, local only, in `data/u11_white/` and `data/u11_navy/`,
  alias maps seeded from the U10 ones so returning players keep their P-numbers): White
  (team `2320bb54`, grade U11 Smith) Round 1 has only team totals and a line-up on PlayHQ
  (scorecard status LIVE, both events feeds empty); Navy (team `10d73647`, grade U11
  Hartley) Round 1 is complete, 223 fair balls. See "The first U11 data" below.
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
on the logit scale, plus a per-game effect on boundary and scoring, plus (new,
22 Sep 2026) a per-game effect on dismissal specific to our own side's
fielding (see below). It fits cleanly: max R-hat 1.012, min ESS 665, no
divergences.

Baseline (average batter vs average bowler): dismissal 0.046 per ball, boundary
0.074 per non-dismissal ball, scoring shot 0.516 per non-boundary ball.

Player spread (SD on the logit scale, median and 90% interval):

| Outcome | Batters | Bowlers |
| --- | --- | --- |
| Dismissal | 0.36 [0.10, 0.59] | 0.30 [0.06, 0.57] |
| Boundary | **0.83 [0.65, 1.06]** | 0.21 [0.03, 0.47] |
| Scoring shot | 0.38 [0.26, 0.52] | **0.36 [0.24, 0.50]** |

Boundary hitting is a batter trait; bowlers differ in wicket-taking and in the
scoring shots they concede. Ground and conditions vary game to game (boundary
0.26, scoring 0.30). Held-out balls, three windows of two games (1,440 balls):
batter effects add +26.6 log-score units for boundaries (+0.2 dismissal, +0.6
scoring); bowler effects add a further +1.4, +1.1 and +2.7. Small but
consistent.

**Team fielding effect** (`ball_model.py`: `fielding_effects`, `mu_f_d`, `sf_d`,
`field_d`; `JointOutcomes.set_fielding_ours`). Our own fielding (catches, run
outs, general sharpness, on top of the bowler's own effect) adds **0.17
[-0.15, 0.47]** to the dismissal log-odds, day-to-day spread 0.16 [0.02, 0.41].
On an average ball that shifts the dismissal chance from 0.046 to 0.054.
Posterior probability the edge is positive: **81%** — a real, modest lean, not
a strong result; the 90% interval still crosses zero. Applies only when our
bowlers are on (derived from the alias convention, `P0x` ours); the batting-
order strategy comparison never simulates our own bowling, so this leaves
those results unaffected in principle, though the bowler effects it was fit
alongside shift slightly, which does move the strategy numbers a little (see
below). Individual catches and run outs are also now recorded per ball
(`playhq_scorebook.py`'s `fielder` column, from data `playhq_parse.py` was
already parsing and linking but not carrying through) and reported
descriptively, not fitted, in `player_report.py` (`fielding_credit_table`):
PlayHQ's ball-by-ball feed names a fielder for only 67 of 172 dismissals in
this data, too few to support a credible per-player estimate.

## U10 simulator check (in-sample; simulated / real)

| Quantity | Per-player rates (13 games) | Joint, real opposition (13) | + fielding effect (13) |
| --- | --- | --- | --- |
| Our runs off the bat | 0.89 | 0.93 | 0.93 |
| Opposition runs | 0.75 | **0.97** | **0.97** |
| Wickets we took | 0.72 | 0.80 | 0.83 |
| Wickets they took | 1.05 | 1.04 | 0.99 |
| Our final total | 0.86 | 0.90 | 0.91 |
| Opposition final total | 0.80 | **0.98** | **0.97** |
| Margin | 1.01 (luck) | 0.65 | 0.73 |

The old model's totals were low because a bowler's economy and a batter's
scoring were estimated separately and double-counted opposition quality; its
margin was right only because errors cancelled. The joint model fixes the
opposition side. Adding the fielding effect (22 Sep 2026) narrows the
remaining gap without closing it: margin calibration moved from 0.65x to
0.73x and wickets we take from 0.80x to 0.83x, and wickets they take from
slightly over (1.04x) to almost exact (0.99x). White's own strength is still
somewhat understated (our total 0.91x). Candidates for the rest of the gap:
ball-count and position effects, or simply the small sample (13 games).

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

* **Above or below the median?** Re-run 22 Sep 2026 with the fielding-effect
  posterior. Our recorded total was above the median replay in 7 of 12 games
  (average percentile 59th, was 61st); theirs in 7 of 12 (46th, was 44th); the
  margin in 9 of 12 (62nd, was 64th). A calibrated model sits near 50th, so all
  three moved slightly toward calibrated, consistent with the fielding effect
  narrowing (not closing) the gap on our side. Extremes: +116 recorded margin
  against a replay median of +43 (98th percentile) on 2025-10-18, and -35
  against +13 (5th) on 2026-01-31.
* **Luck.** A single game's margin has a spread (1 SD) of about 29 runs with the
  lineups fixed.
* **Batting order.** Best of 40 random and 4 rule-of-thumb orders, screened on
  2,000 replays and judged on 10,000 fresh ones: **+0.9 runs** of margin on average
  (95% interval about +0.7 to +1.1). Random orders differ by only about 0.4 runs
  (SD, corrected for simulation noise).
* **Bowling split** (same overs, same bowlers, different shares; round-robin
  sequence): **+0.7 runs** with the two keepers left as played; **+2.8** if the
  keepers could also be chosen freely, which is optimistic because keeper skill is
  not modelled. Random reassignments differ by about 0.6 runs (SD).
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

Main scenario (retire at 35, U10 boundary and scoring rates, random attack; re-run
11 Oct 2026 with the corrected strike rules, which moved nothing by more than half a
run from the 22 Sep fielding-effect run), runs against strongest to weakest:
strong-middle alternating -0.2 ± 0.3, balanced pairs (top six) -1.2 ± 0.4,
strong-weak alternating -7.9 ± 0.4, weakest to strongest -19.9 ± 0.4, random orders
-9.6 on average. The swap search found nothing better than strongest to weakest, and
among these *fixed orders*, none of the 35 other scenarios (retirement 25, 30 or 35;
boundary shift 0, -0.7, -1.4; skill drift 0 or 0.3; random or smart attack) found an
alternative order beating it by more than 0.5 runs. Split by opposition attack
strength (thirds of the worlds), strong-weak alternating loses -8.7 ± 0.6 against
the strongest attacks and -7.6 ± 0.7 against the weakest, so pairing does not shield
the best batters from the best bowling. The mechanism is who gets the balls: the six
best batters face 17 to 20 balls each under the conventional order.

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
**Correction, 11 Oct 2026: the recall step is not allowed as written.** Rule 17.9(v)
lets a retiree resume only "once all other players have been dismissed or retired",
so bringing a banked batter back ahead of a fresh one needs both coaches to agree. I
did not check this when it was built. Round 1 bears the rule out: 16 retirements and
10 resumptions across six innings, every resumption with no fresh batter left. The
figures below are therefore the most early recall could be worth; the options the
rules allow (retire everyone, or the top few, at 25 or 30 balls, retirees returning
at the end) are priced by `scripts/price_retirement.py` and in the method note's
section 4c. Result: **+1.3 ± 0.2 runs** against strongest to weakest in the main scenario, and
the best alternative in **all 37 scenarios tested** (main plus the 36-scenario
grid), ranging +0.7 to +7.8. It stayed positive against every third of opposition
attacks (+1.1 to +1.7), with no sign it depends on how strong the attack is.
Balls-by-rank shows the gain comes from the best batter facing about 2 more balls
(20.5 to 22.3) at the expense of about 1 fewer for the weakest (5.4 to 4.4): the
runs-maximising version of the policy is not a participation lever (early recall works
against it). Retiring *every* batter at 25 rather than 35 costs about 6 runs at the
U10 scoring rates and about 3 in a U11-like world, where it lifts the weakest batter
from about 8 to 12 balls: the cheapest lever the rules allow for "everyone gets a go".

**Team fielding effect: built 22 Sep 2026** (see the joint ball model section above for
the result, 0.17 [-0.15, 0.47] on the dismissal log-odds, 81% probability positive).
An earlier version of this note planned it as a team-level nudge on the grounds that
PlayHQ does not tag fielders; that was wrong. Catches and run outs are named in both
the scorecard statistics and the ball-by-ball titles, and are now carried per ball
and reported per player (`player_report.py`'s Fielding table), though only 67 of 172
dismissals name a fielder, too few for a per-player effect. Wicketkeeper flags
(`is_wicketkeeper` already exists in the player registry and `PlayerSkills`, but is
only wired into bowling-tier allocation, not dismissal modelling) would need a real
per-game record of who kept, which PlayHQ does not provide; blocked on the coach
supplying it. Rule 17.2(ii) has a change of wicketkeeper at the drinks break after
over 13, so each side uses two keepers per game.

## The first U11 data (Round 1, Sat 10 Oct 2026)

Full write-up with tables: `docs/BATTING_STRATEGY_METHOD.md`, section 4b. In short:

* **Results.** Navy 142/7 (25 overs) beat Orange 56/8 (all out in 14.1). White 131/4
  beat Turquoise 91/7 in a game cut to 22 overs a side. Over limits were 25 and 22
  (`OVER_LIMIT`); the rules are 25, minus 2 overs per 6 minutes lost.
* **What PlayHQ holds.** Navy's game is complete (scorecard FINAL, ball-by-ball both
  innings). White's has the team totals and a line-up of eight (one new to the data)
  only: Turquoise was the home side and, apparently, did not score ball-by-ball. Worth
  re-checking later, and worth asking whoever scores White's home games to use the live
  scoring, because the analysis needs it.
* **Engine fix.** The simulator rotated strike only at the end of the over. Rule
  17.4(iv) has batters run popping crease to popping crease, and the game shows them
  crossing on odd runs (the striker changed after 30 of 33 odd-run balls, 4 of 138
  others) with a new batter usually taking strike after a wicket. Both are now on the
  U11 playing conditions and switchable. Conclusions unchanged (at most half a run).
  `--strike-rules earlier` reproduces the first pilot exactly (330 result rows
  identical), and a separate check against the committed engine found 400 of 400
  innings identical with both options off.
* **U10 to U11 transfer (one game, so scenario dials, not fitted effects).** On the
  model's logit scale, against Navy's 2,668 U10 balls: dismissal +0.12 [-0.34, +0.53],
  boundary -0.34 [-0.75, +0.03], scoring shot **-1.08 [-1.37, -0.80]** (24% of
  non-boundary balls scored against 48%), and wides or no-balls 18% [15%, 22%] of
  deliveries against 7%. The strategy simulations had no wides at all.
* **Conclusions in a U11-like world** (boundary -0.3, scoring -1.1, extras 18%): the
  squad's mean total falls from 152 to 129 (the four real innings averaged 105);
  strongest to weakest still beats every fixed alternative; strong-weak alternating
  costs 4.9 rather than 7.9 runs; bank-and-recall is worth +3.3 ± 0.2 (+4.5 on
  average over the 37 scenarios, best in all of them), though that early recall needs
  both coaches' agreement (17.9(v)), its thresholds were set for U10 scoring and it takes
  balls from the weakest batter (8 to 5). What the rules allow costs 2 to 4 runs in that
  world (retire everyone at 25: 3.1 runs, 4 more balls to the weakest batter).
* **Pools.** White is in U11 Smith with Turquoise (a VDCC sibling, met in Rounds 1 and
  8), Ashgrove Astronauts, WNJCC Purple and Teal, and Bulls White and Yellow. Navy is
  in U11 Hartley with VDCC Maroon and Orange. Different grades, so not pooled.

## The Smith pool (scouting; pulled 11 Oct 2026)

White's pool is seven teams: VDCC Turquoise and White, Ashgrove Astronauts, WNJCC Purple and
Teal, Bulls White and Yellow (Toombul District). `scripts/fetch_scorecards.py --data-dir
smith_pool` cached the other teams' fixtures and finished games (two, complete with ball-by-ball
and FINAL scorecards); `scripts/scout_pool.py` turns the cached scorecards into a private report
(`data/smith_pool/outputs/scouting/`, names children, gitignored) of per-player lines, ratings
shrunk toward the pool average, U10 history against White and Navy where a player has one
(11 of 38 players, all Bulls Yellow or Teal), White's draw and a team-by-team breakdown.

* **Round 1.** Bulls White 156/6 beat WNJCC Purple 108/4; Bulls Yellow 135 (all out, 17.1
  overs) beat WNJCC Teal 105/7; White 131/4 beat Turquoise 91/7. All three were 22-over
  games. Both Bulls sides scored 7.1 and 7.9 an over against 4 to 6 for the rest. Bulls Yellow's
  bowlers conceded 43 extras in the game: extras are a large share of U11 totals.
* **Next opponent.** Ashgrove (Round 2, White at home) had the bye, so there is nothing to
  go on until they play. The Bulls are met in Rounds 5 and 6.
* **Last season's evidence for the Bulls.** As opponents of White or Navy in U10 2025/26,
  Bulls teams won 2 of 3 (averaging 160 to our 135); WNJCC teams lost all 11 (104 to our 156).
* **Retirement is routine here.** 16 retirements and 10 resumptions across the six
  innings with ball-by-ball; Teal retired six batters in one innings. See the correction under
  "Per-batter retirement" above.
* Everyone has played one game or fewer, so the ratings are a shortlist to watch.

## Caveats that matter for 2026/27

* **Field size and ball.** U11 boundaries are 45 m (U10: 30 to 35 m) and U11
  may use a 130 g ball. The U10 boundary rate (about 7.5% of balls, up to 22%
  for the best hitters) will not carry over at face value. Rankings probably
  persist; absolute boundary value will fall, raising the weight on scoring
  and staying in.
* **Wicketkeepers.** The optimiser does not know who keeps. The two players
  given one over are by rule the keepers; flag them or the bowling advice is
  unreliable.
* **Skill transfer** U10 to U11 is an assumption. Only 10 players and 15 games. The
  first U11 game (one game, Navy v Orange) points to far fewer scoring shots, more
  wides and slightly fewer boundaries; these are scenario dials until more games arrive.
* The check is in-sample; the backtests are the out-of-sample evidence.

## A second team: VDCC Navy

Built 22 Sep 2026 at the user's request, for a friend (a Navy parent) to check
the model against what he actually watched. Navy is U10, grade "Ponting"
(pre-Christmas, team `7b1eba2e`) then "Chappell (Post)" (post-Christmas,
`e3fff983`) — confirmed a different grade to White's ("Walters"/"Kasprowicz
Post") both halves, so this is a standalone analysis, not pooled into White's
opposition prior (see the "does more data help" note above and the Navy entry
in the playhq-access-notes memory).

The whole pipeline now takes `--data-dir <name>`, keeping a second (or third)
team's raw cache, alias numbering, processed CSVs and outputs under
`data/<name>/` rather than `data/`, so nothing about White's own data is at
risk (see STATE.md). Fetched 12 of 14 Navy games (2 not electronically scored);
their ball-by-ball reconciles with the scorecard noticeably less cleanly than
White's (many more "balls faced differ"/"event title not understood" notes),
which shows up as both sides' wicket-taking being under-predicted in the
totals check (0.78x each), not just ours as with White. Fitted, replayed and
reported the same way as White throughout.

For the friend specifically, a fully named variant also exists: real names
used as the alias itself (not a find-and-replace after the fact, which cannot
reach names baked into the chart images), built by copying the raw cache into
`data/navy_named/`, rewriting its alias maps so `alias == name`, and running
the pipeline `--offline` against that copy. Never committed; never shared
beyond the family who asked for it.

Two small, real bugs surfaced and were fixed while doing this: `check_u10_totals.py`
required the older, unrelated per-player posterior even when unused whenever
`--ball-posterior` was given (two separate eager-evaluation spots); `load_balls`
could not identify "our" bowlers once real names replaced the P0x/O0x
convention, needing an explicit `our_names` argument (now auto-detected by
`fit_ball_model.py` from the alias map's shape).

## Next steps (in order)

1. White's Round 1 record. PlayHQ has only the totals; the coach's own note of the
   batting order, runs and balls, bowlers and keepers would let it enter the U11 data.
   Re-check PlayHQ's scorecard later in case the scorer finalises it.
2. After each Saturday: `fetch_scorecards.py --refresh` for the pool teams and White,
   then `scout_pool.py`. Ashgrove's first scorecard arrives after Round 2.
3. Decide the retirement policy with the coach: the price list is in the method note's
   section 4c (retire everyone at 25 costs about 3 runs in a U11-like world for 4 more balls
   to the weakest batter). Ask whether the opposing coach would agree to early resumption.
4. After a few more U11 games: refit the joint model with a U11 offset on each outcome
   (and a wides rate) instead of the fixed scenario dials, and compare predicted with actual
   balls by batting position. Feed the pool's ball-by-ball in as the U11 opposition prior.
5. Wire wicketkeeper flags into the pipeline, if the coach can supply who kept each game.
6. Fetch the two missing White U10 games and 2024/25 (API key, or slowly later).
7. If the Navy friend (or Nick) wants to keep using this, build the U11 batting-strategy
   comparison for Navy's own squad once there are a few more games.
8. Optional: parse wides and no-balls as first-class events (they currently land in
   "event titles not understood", which is harmless for the ball model).

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
- Extras: not modelled for U10 where none were recorded; for U11 they are a scenario dial
  (`--extras-rate`, one run each, counted toward the 25 and 35-ball limits).
- Retirement follows rule 17.9: optional from 25 balls, mandatory at 35, a retiree resumes only once
  all other players are out or retired. The engine's early "recall" is kept as an upper bound on what
  agreement between coaches could be worth, flagged as not allowed as written.
- Strike rules follow rule 17.4(iv) and the first U11 game (batters cross on odd runs and
  rotate at the end of the over; a new batter takes strike after a wicket), kept as two
  switchable options on the playing conditions so the first pilot can be reproduced.
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
