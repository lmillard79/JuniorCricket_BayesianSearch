# Guide: what this project does and how to read its outputs

This guide explains, in plain language, what has been built, what each output
file means, and how to read the numbers, especially the figures in square
brackets. It assumes you know cricket but not statistics. Nothing here needs you
to run code to follow it; the commands are collected in section 7.

## 1. In one minute

- **What it is.** A model that learns how each junior player bats and bowls from
  the ball-by-ball records on PlayHQ, plus a simulator that plays whole
  innings, ball by ball, under the BNJCA rules. Together they let us ask
  "what if" questions: what if this batter opened, what if these two bowled
  more overs, how did that day compare with a typical day for these players?
- **What it is for.** Understanding players and testing lineup ideas with honest
  uncertainty. It is not a crystal ball and it never overrides the
  participation rules the leagues set.
- **What it has found so far (2025/26 U10 White data, 15 games).**
  1. Boundary hitting is the one strong, persistent batting skill. Bowlers
     differ modestly in wickets and in scoring shots conceded, but not in
     boundaries.
  2. Under U10 rules the batting order barely matters (about 1 run of margin),
     because every batter is given a fixed 13 or 14 balls. A single game's margin
     swings about 28 runs on luck alone.
  3. Under U11 rules the order matters much more, and among *fixed* orders the
     conventional one (strongest to weakest) held up against every alternative
     tried, including strong-weak pairings, across a wide range of assumptions.
     The one thing that beat it was not a different order: letting the top four
     retire at 25 balls and recalling the strongest one straight back in if the
     next batter is out cheaply gained about 1.4 runs, in every scenario tested.
     See [BATTING_STRATEGY_METHOD.md](BATTING_STRATEGY_METHOD.md).

## 2. How the pieces fit together

```
PlayHQ (public scorecards and ball-by-ball)
      |   scripts/fetch_scorecards.py        (slow and polite; caches everything)
      v
data/processed/*.csv                         (aliases only: P01.. ours, O01.. opposition)
      |
      +--> scripts/fit_ball_model.py  --> ball_posterior.nc     the JOINT model:
      |                                                          every ball, batter and bowler together
      +--> scripts/fit_model.py       --> posterior_model.nc    the older per-player model
      v
Simulator (src/junior_cricket/simulator.py): plays a full innings ball by ball
under U10 or U11 rules, using the joint model to decide each ball
      |
      +--> scripts/check_u10_totals.py            does it reproduce real totals?
      +--> scripts/replay_games.py                replay each real game; shuffle order and bowling
      +--> scripts/compare_batting_strategies.py  U11 batting-order strategies, unknown opposition
      +--> scripts/optimise_lineup.py             one recommended U11 lineup
      +--> scripts/player_report.py               plain-language player profiles
```

Everything on the right is an *analysis* of the same underlying model, so a
weakness in the model shows up in all of them. That is why several of the
scripts exist to check the model itself, not to give advice.

## 3. Vocabulary you need

**Alias.** Players appear as `P01`, `P02` ... (ours) and `O01`, `O02` ...
(opposition) everywhere except in private, gitignored files. The repository is
public and the data concerns children, so real names never enter a model file, a
report or a commit. See section 8.

**Ball row.** One line per delivery: who batted, who bowled, runs, whether it was
a boundary, whether it was a dismissal, and the over number. There are about
3,100 of them, from 13 games. Everything the joint model knows comes from these.

**Probability and odds.** A probability of 0.075 is a 7.5% chance. Odds are the
chance for divided by the chance against (0.075 / 0.925 = 0.081).

**The logit scale.** The model does not add effects to probabilities directly (that
could push a probability above 100%). It adds them to the *log of the odds*, called
the logit scale. You only need three ideas:

- zero means average; positive means more of the outcome, negative means less;
- an effect of +0.5 multiplies the odds by about 1.65, +1.0 by about 2.7, and
  +0.85 by about 2.3 (the number is `e` raised to the effect);
- effects add up. A batter with +0.85 on boundaries facing a bowler with -0.2 has
  a combined +0.65.

Worked example: the average batter hits a boundary on 7.4% of balls where they
are not out (odds 0.080). A batter one standard deviation better (+0.84 on the logit
scale) has odds 0.080 x 2.3 = 0.187, which is a boundary on about 16% of balls.

**Standard deviation (SD) as "how much players differ".** When a table says
"batters, boundary: 0.84", it means that across players the boundary effect
typically sits about 0.84 above or below average on the logit scale. Big number:
players really differ. Small number (say 0.2): players are nearly
interchangeable on that skill.

**Posterior, and the brackets.** The model does not give one number per
quantity. It gives a spread of plausible values, called the posterior. When you see

```
17.1 [14.3, 20.1]
```

it means: the best single estimate is 17.1, and given the model and the data there
is a 90% chance the true value lies between 14.3 and 20.1. This is a *90% credible
interval*. Read the brackets like this:

- **narrow bracket** means plenty of data; trust the number;
- **wide bracket** means little data; the number is mostly the model's prior
  (the average player) plus a little evidence;
- **two players whose brackets overlap heavily** cannot be told apart. A ranking of
  players with overlapping brackets is partly noise;
- a bracket that excludes zero (or excludes "average") is real evidence.

In a few places the bracket is an *interval of simulations* instead (for example
"90% of replays fell in [x, y]"); the text says so.

**Percentile.** "The 92nd percentile" means that 92% of the simulated versions of
that game did worse than what was recorded. The 50th percentile is exactly the
median. Values near 0 or 100 are days that were unusually bad or good relative to
what the model expected of the players.

**Mean, median, standard error.** The mean is the average; the median is the
middle value (half above, half below). The standard error (SE) is how uncertain a
*computed average* is, purely from the number of simulations. "+0.8 ± 0.2" says the
average gain was 0.8 runs and the simulation noise is about 0.1 either side (± is
usually a 95% interval, roughly 2 SE; the report says which).

**In-sample and out-of-sample.** A model checked against the same games it learned
from is being marked on its own homework (in-sample). A backtest (out-of-sample)
learns from early games and is scored on later games it never saw. Out-of-sample
evidence is worth far more.

**Log score.** A backtest measures how surprised the model was by what later
happened, summed over every ball. Reported as "model minus baseline": positive
means the model was less surprised than the baseline. A difference of +20 or more
is strong; differences below about 2 are noise.

**Paired comparison.** To compare two batting orders fairly, both must face the
same opposition and the same luck. The strategy comparison plays every order
through the same thousands of simulated worlds, so a difference between orders is
not caused by one of them meeting a tougher attack.

**Net runs.** Under the U10 rules each dismissal gives the *other* team 4 runs
(Rule 16.10(vi)). A batter's net contribution is therefore runs scored minus 4
times dismissals. A team's final total is its own runs plus 4 for each dismissal
its bowlers take.

**Fit health (R-hat, ESS, divergences).** Three checks that the fitting routine
worked. R-hat should be 1.01 or below (1.00 is perfect); ESS (effective sample
size) should be in the hundreds or more; divergences should be zero. If any of
these look bad, do not trust that model's numbers.

## 4. Reading each output file

All of these live in `data/outputs/` (gitignored, so they exist only on the machine
that ran the scripts) unless marked otherwise.

### 4.1 `player_report.md` (from `scripts/player_report.py`)

The plain-language profile of each player, from the joint model, against an average
opponent on an average ground.

```
| player | balls faced | boundaries | runs | dismissals | NET runs         | vs average batter |
| P08    | 180         | 2.9        | 19.0 | 0.48       | 17.1 [14.3, 20.1] | +9.2              |
```

- **balls faced**: how much data sits behind the row. More balls, narrower brackets.
- **boundaries, runs, dismissals**: what the model expects in one 13-ball stint (a
  U10 batter's allotment): 2.9 boundaries, 19.0 runs, 0.48 dismissals.
- **NET runs**: runs minus 4 x dismissals: 19.0 - 4 x 0.48 = 17.1. The bracket is the
  90% interval on that net figure.
- **vs average batter**: net minus what an average batter nets in the same 13
  balls. +9.2 says this batter is worth about 9 more net runs per stint than average.

The bowling table works the same way for one over: *wickets per over* (with its
bracket), *runs per over*, and *edge over an average bowler*, which counts each
wicket as 4 runs plus the runs saved. Bowlers differ less than batters, so read
small edges (a few tenths of a run) as noise.

A third table, **Fielding**, lists catches and run outs actually credited to each
player in the ball-by-ball data. This is a plain count, not a model estimate: with
only a handful of events per player across a season, and most dismissals ("bowled")
crediting no fielder at all even though everyone else was still in the field, it is
too little to fit reliably. Players with no credited event are left off the table.

### 4.2 `ball_model_summary.md` (from `scripts/fit_ball_model.py`)

Five things, top to bottom:

1. **Health line**: "Max R-hat 1.012, min bulk ESS 665, divergences 0." This is the
   fit health check from section 3. It is good.
2. **Baseline**: the average batter against the average bowler. In this data a
   dismissal is 4.6% of balls, a boundary is 7.4% of balls that are not dismissals,
   and a scoring shot is 51.6% of the remaining balls (not boundaries, not
   dismissals). The model builds every outcome in that order: first "is it a
   dismissal", then "is it a boundary", then "is it a scoring shot".
3. **How much players differ** (the important table):

   ```
   | outcome                | batters            | bowlers           |
   | boundary (per ball...) | 0.83 [0.65, 1.06]  | 0.21 [0.03, 0.47] |
   ```

   These are SDs on the logit scale, with 90% brackets. Boundary hitting differs a
   lot between batters (0.83, bracket well above zero) and hardly at all between
   bowlers (0.21, bracket touching zero): boundaries are a batter trait. Wickets
   and scoring shots differ modestly for both.
4. **Ground and conditions**: how much a whole game's boundary and scoring rate
   moves with the ground, the ball and the weather (0.26 and 0.30).
5. **Our team's fielding edge** (new, 22 Sep 2026): how much our own fielding
   (catches, run outs, general sharpness) adds to the dismissal log-odds on top of
   the bowler's own effect, on the days we field. The 2025/26 estimate is 0.17
   [-0.15, 0.47], shifting an average ball's dismissal chance from 4.6% to 5.4%.
   The bracket still crosses zero (81% posterior probability the edge is positive),
   so read this as a real, modest lean, not a settled fact. It only ever applies to
   our own bowling, in the replay and totals-check tools; the batting-order
   strategy comparison never simulates us bowling, so it never fires there,
   though the bowler effects it was fitted alongside shift slightly too.

### 4.3 `ball_effects.csv`

One row per player, role and outcome: `player, role (bat or bowl), component (d, b or
s), effect, lo, hi`. Components are `d` dismissal, `b` boundary, `s` scoring shot.
`effect` is the posterior mean on the logit scale (positive means more of that
outcome); `lo` and `hi` are the 90% bracket. Note that for a *bowler*, a positive
`d` effect is good (more dismissals) and a positive `b` or `s` effect is bad (more
runs conceded).

### 4.4 `backtest_report.md` (from `scripts/backtest.py`)

Does the per-player model predict games it has not seen? Two tables: one splits the
season in half, one rolls forward through blocks of games.

- **vs pooled**: the model's log score minus that of a model that treats every
  player as identical. Positive means the model found real differences between
  players. Boundary rate is +22 to +31, which is strong; the others are small.
- **vs raw**: minus that of a model that simply trusts each player's own record
  with no smoothing. Positive means the smoothing helped.
- **corr**: correlation between predicted and actual rates across players. 0.76 to
  0.81 for boundaries is excellent for this little data; 0.2 for dismissals is weak.
- **coverage**: the share of players whose actual result fell inside the model's 90%
  interval. About 0.9 is well calibrated.

### 4.5 `u10_totals_check_*.md` (from `scripts/check_u10_totals.py`)

Replays every real game with the model and compares simulated with recorded
totals. Read the `sim / observed` column: 1.00 is perfect. In the joint-model file
the opposition's totals come out at 0.98 of real (good) and ours at 0.90 (the model is
about 10% stingy about our own side; a team fielding effect is the likely missing
piece). `inside 90%` should be about 0.9: the share of games whose recorded value
fell inside the simulated 90% range. This check is in-sample.

### 4.6 `lineup_recommendation.md` (from `scripts/optimise_lineup.py`)

The single recommended U11 batting order and bowling split, with an outlook
against an unknown opposition (expected totals, win probability, run differential
with a 90% interval). Two cautions:

- It uses U10 skills under U11 rules, so treat it as indicative.
- Its search compares candidate orders using *different* random opposition for each
  candidate, so small differences between neighbouring orders are noise. For any
  question about batting order use `compare_batting_strategies.py` instead (4.8), which
  compares orders on the same worlds.

### 4.7 `replay/` (from `scripts/replay_games.py`)

Open `replay_report.html`. It replays each real game thousands of times.

- **The season chart** ("Each day against the model's replays of that day"): each
  row is a game; the thick bar is the middle 50% of replays, the thin line 90%, the
  white tick the median; the dot is what actually happened, green if better for us
  than the median and red if worse; the number on the right is the percentile.
- **The per-game charts**:
  - top: histograms of our total, their total and the margin across replays, with the
    recorded value as a thick line;
  - left: the **Manhattan** (runs per over): pale bars are the replay median with the
    middle 50% as a whisker; dark narrow bars are what was recorded; black
    triangles are recorded dismissals;
  - right: the **worm** (runs accumulated): the line is the replay median, the bands hold
    50% and 90% of replays, the orange line is the recorded innings.
- **`over_profile.png`**: average runs per over across all games, replay against
  recorded. A systematic gap in certain overs would mean the model misses something
  about how an innings unfolds.
- **`decisions.png`** and the table in the report: how much a different batting order
  or bowling split could have changed the margin, in runs, with 95% intervals. "Best"
  alternatives are chosen on one set of replays and judged on a fresh set, so the
  quoted gain is not flattered by picking the best of many noisy estimates.
- **CSVs**: `replay_games.csv` (per game: recorded, replay median, 5th and 95th
  percentile, percentile, and the same given the day's fitted ground),
  `replay_overs.csv` (per game, innings and over), `replay_decisions.csv` (per game and
  alternative: gain and standard error).

### 4.8 `strategies/` (from `scripts/compare_batting_strategies.py`)

Open `strategy_report.html`. The main table has one row per batting strategy:

- **Mean total, SD**: the average and spread of our innings total across worlds.
- **Runs vs strongest-to-weakest**: the paired difference from the conventional
  order (negative means worse), with its standard error.
- **Share of worlds better**: in what share of the simulated worlds the strategy scored
  more than the conventional order. Around 50% means indistinguishable.

One row, `bank top 4, recall on a cheap wicket`, is not a different order at all: it
keeps the conventional order but lets the top four retire not out at 25 balls and
brings the strongest one straight back in, ahead of the next fresh batter, if whoever
comes in next is out within 6 balls or for under 5 runs. It is the only alternative
that gained runs rather than lost them.

A second table splits the same comparison by how strong the opposition attack was (the
strongest, middle and weakest thirds of the simulated worlds; `strategy_by_attack_strength.csv`).
If a pairing protected the best batters from strong bowling it would show as a smaller
loss in the first column. It does not, for any of the reordering strategies.

The charts: `worms.png` (median runs accumulated and per over by strategy),
`balls_by_rank.png` (who gets the balls and what they do with them, which is the
*reason* orders differ), and `sensitivity.png` (the difference from the conventional
order in every scenario tested; red is worse). Full method and findings in
[BATTING_STRATEGY_METHOD.md](BATTING_STRATEGY_METHOD.md).

### 4.8b `model_summary.csv`, `search_history.csv`, `run_differential.png`

Older outputs of the per-player model and the original optimiser. `model_summary.csv`
lists every parameter with its posterior mean, SD, 89% interval and the fit-health
columns (`ess_bulk`, `r_hat`). `search_history.csv` is every batting order and bowling
configuration the optimiser tried with its score. `run_differential.png` is the
histogram of simulated margins for the recommended lineup.

## 5. A worked example: reading one game

Take a row from the season chart: our total recorded 104, replay median 142, 5th
percentile. That reads: "If this team played that opposition over and over, it would
usually score around 142. Only about 1 replay in 20 scored as little as 104. So this
was an unusually bad day for our batting." Then look at the game's Manhattan: the
recorded bars sit below the pale median bars in most overs, with triangles (dismissals)
clustered in the middle overs. The worm shows the recorded line falling away from the
median from about over 8 and never recovering.

What you cannot conclude: *why*. The model knows about batter and bowler skill and the
ground. It does not know about fielding, nerves, the pitch, or a batter who is unwell.
A low percentile flags a day worth looking at; it is not a diagnosis.

## 6. How much to trust it

**Checked and holding up**

- The models fit cleanly (R-hat 1.012, no divergences).
- Out-of-sample, the boundary-rate differences between players are real and
  predictable (corr 0.76 to 0.81).
- The U10 simulator reproduces the *opposition's* totals within a few percent
  (0.97 to 1.04 of real) and the shape of an innings over by over.
- Event data reconcile exactly with the scorecards in 23 of 26 innings, and the
  4-run penalty rule holds in 25 of 26.
- Adding the team fielding effect (section 4.2) moved the replay percentiles for
  our totals from the 61st toward the 50th (now 59th), without fully closing it.

**Known weaknesses**

- The model is still somewhat stingy about our own side (replay percentiles for
  our totals average about the 59th, not the 50th). The fielding effect narrowed
  this gap (it was the 61st) but did not close it; ball-count and position effects
  are candidates for the rest, or it may just be the small sample.
- Everything about U11 is an extrapolation from U10 skills: U11 boundaries are 45 m
  against 30 to 35 m, the ball may be heavier, and the players are a year older.
- Only 10 players and 15 games (13 with ball-by-ball). Wide brackets on some players
  are the honest reflection of that.
- Wicketkeepers are not individually modelled; PlayHQ's data does not record who
  kept, only catches and run outs (now used, at a team level for the fielding
  effect, and shown per player for interest, not fitted, in `player_report.md`).
  The rule gives the two keepers one over each, so bowling advice that ignores who
  keeps is still unreliable.
- Bowling differences between players are small; do not over-read them.

**Rules of thumb**

1. Trust ranges over single numbers.
2. Trust big, well-supported effects (boundary hitting) over small ones.
3. Treat "the model says A beats B by a run or two" as a tie.
4. Treat U11 statements as hypotheses until U11 games have been played and the model
   has been updated with them.

## 7. Running it yourself

From the repository root (Windows shown; use `.venv/Scripts/python` or an activated
environment). The first three steps take a few minutes each; the analyses take from a
few seconds to about ten minutes.

```bash
# one-off setup
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements-dev.txt
.venv/Scripts/python -m pip install -e .
.venv/Scripts/python -m pytest tests/

# 1. build ball and scorebook CSVs from the cached PlayHQ data (no network)
python scripts/fetch_scorecards.py --team-id 75cdae66 --team-id fafdb4c4 --offline

# 2. fit the models
python scripts/fit_ball_model.py
python scripts/fit_model.py --batting data/processed/playhq_all_batting.csv --bowling data/processed/playhq_all_bowling.csv --sampler numpyro

# 3. read the players
python scripts/player_report.py --squad P01,P03,P04,P05,P06,P07,P08,P09,P10

# 4. check the simulator against real games
python scripts/check_u10_totals.py --team-id 75cdae66 --team-id fafdb4c4 --ball-posterior data/outputs/ball_posterior.nc --opposition real

# 5. replay the real games and reshuffle decisions (about 10 minutes)
python scripts/replay_games.py --team-id 75cdae66 --team-id fafdb4c4

# 6. U11 batting-order strategies (about 5 minutes)
python scripts/compare_batting_strategies.py --search
```

Drop `--offline` (and add `--max-games 3`) to fetch new games from PlayHQ, slowly. It
stops for good on the first "blocked" response; wait before trying again. After new
games arrive, rerun steps 1 and 2 and then whichever analyses you want.

## 8. Privacy rules

- The GitHub repository is public; the data concerns children. Names and PlayHQ
  profile IDs never appear in committed files.
- Raw downloads (which include names) live in `data/raw/`, which is gitignored.
- Model files use aliases. The alias-to-name maps are `data/raw/playhq/player_map.csv`
  and `opposition_map.csv`, also private.
- `player_report.py --named` and `name_report.py` write a private copy with real names
  into `data/outputs/` (gitignored). Do not commit, upload or share those copies.
- Stage files explicitly when committing; do not use `git add -A`.

## 9. Common questions

**Why is the batting order irrelevant in U10?** Rule 16.10 gives every batter a fixed
number of balls (three get 14 and six get 13 in a team of nine), so everyone faces
roughly the same number of balls whatever the order. Only who gets the extra ball and
who faces which bowler move, and both are tiny. The replay measured it: the best order
found was worth under one run of margin.

**Why does the model say we were above median in so many games?** Because it
underestimates our own side by about 10% (section 6). That is a known gap, not a
coincidence.

**What is "the day's ground" versus "a typical ground"?** Each game has its own boundary
and scoring level (ground, ball, weather). The main replays use a typical ground, so
"above the median" includes luck with conditions. The CSVs also give the percentile
when the day's own fitted conditions are used.

**What is the difference between the two models?** The per-player model
(`fit_model.py`) estimates a batter's scoring and a bowler's economy separately, so
each absorbs the quality of the opposition the other faced. The joint model
(`fit_ball_model.py`) records every ball with both players and separates the two
properly. Use the joint model for anything about players or lineups.

**Is the recommended lineup a prediction that we will win?** No. It is the best lineup
the model can find given U10 skills and the U11 rules, and the win probability in it
assumes the U11 transfer holds. The first U11 games are the first direct test.
