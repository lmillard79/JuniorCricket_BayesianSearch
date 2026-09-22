# JuniorCricket_BayesianSearch
An attempt at finding the optimal batting and bowling order for Under 11s based on varied player strength for both teams. 
https://www.pymc.io/projects/examples/en/latest/case_studies/rugby_analytics.html
https://www.pymc.io/projects/examples/en/latest/case_studies/hierarchical_partial_pooling.html
https://bnjca.org.au/wp-content/uploads/2026/07/Quick-Reference-Rules.pdf

## What this is, in plain language

A model that learns how each junior player bats and bowls from PlayHQ's ball-by-ball
records, and a simulator that plays whole innings under the BNJCA rules. Together they
answer "what if" questions about batting order and bowling, with honest uncertainty
instead of a single confident answer. It supports coaching decisions; it does not
override the leagues' participation rules and it is not a forecast.

**Start here**

- [docs/GUIDE.md](docs/GUIDE.md): a tutorial on what has been built and how to read
  every output, including what the figures in square brackets mean.
- [docs/BATTING_STRATEGY_METHOD.md](docs/BATTING_STRATEGY_METHOD.md): the method and
  pathway for testing U11 batting-order strategies (strong-weak pairing against
  strongest to weakest) when the opposition is unknown, with pilot results.
- [PROJECT_STATUS.md](PROJECT_STATUS.md) and [STATE.md](STATE.md): current status,
  evidence and next steps.

**What it has found so far** (VDCC U10 White 2025/26: 15 games, 13 with ball-by-ball)

- Boundary hitting is the one strong, persistent, predictable batting skill. Bowlers
  differ modestly in wickets and scoring shots, but not in boundaries.
- Under U10 rules the batting order barely matters (about 1 run of margin) because every
  batter gets a fixed 13 or 14 balls. Luck alone swings a game's margin by about 28 runs.
- Under U11 rules order matters more. Among *fixed* orders, strongest to weakest held up
  against every alternative tried, including strong-weak pairings, across 36 sets of
  assumptions about the move to U11; pairing a strong batter with a weak one costs runs.
  The one alternative that beat it was not a different order at all: letting the top
  four retire at 25 balls and recalling the strongest one straight back in if the next
  batter is out cheaply gained about 1.4 runs, in every scenario tested.
- The model understates our own side by about 10%; a team fielding effect is the next
  planned change. U11 results extrapolate from U10 skills and are indicative until U11
  games have been played.

**Reading a figure such as `17.1 [14.3, 20.1]`.** The model gives a spread of plausible
values, not one number. 17.1 is the best estimate; the bracket is a 90% interval, so
there is a 90% chance the true value lies between 14.3 and 20.1. A wide bracket means
little data; players whose brackets overlap cannot be told apart.

**Privacy.** The repository is public and the data concerns children. Model files, reports
and commits use aliases (P01, O01 ...) only. Raw downloads and the alias-to-name maps live
in gitignored folders; never commit or share a named copy of a report.

---
created: 2026-09-21T07:15:07Z
---
 Bayesian MARCEL for Junior Cricket - VDCC Under 11 Lineup Optimiser

Adapt the Bayesian MARCEL baseball projection framework to junior cricket for VDCC Under 11s (BNJCA): hierarchical Beta-Binomial player model on PlayHQ data, grade population as unknown-opponent prior, and a match simulator under the now-encoded BNJCA U11 playing conditions (25 overs, 9 players, all-out at 8, retire 25-35 balls, mandatory all-player bowling).

## Summary
Adapt the Bayesian MARCEL baseball projection framework to junior cricket for VDCC's team moving from Under 10 Whites to Under 11 (BNJCA): hierarchical Beta-Binomial player model on PlayHQ data, grade-wide population as unknown-opponent prior, and a ball-by-ball match simulator under the encoded BNJCA U11 playing conditions to optimise batting order and bowling allocation.

---

## 1. Why the baseball models transfer

### Source models
- **Hierarchical partial pooling** (Efron & Morris batting averages, PyMC example): players' true rates drawn from one league-level Beta distribution; small-sample players shrink toward the league mean.
- **Bayesian MARCEL** (PyMC Labs blog): adds Dirichlet-learned recency weights over 3 periods, an aging effect on the logit scale, and full posterior uncertainty per player.

### Concept mapping (baseball -> junior cricket)
- At-bat (Bernoulli trial) -> ball faced
- Batting average (hits/AB) -> dismissal hazard (outs / balls faced), Beta-Binomial, transfers unchanged
- Hard-hit rate -> boundary rate (4s+6s / balls faced), Beta-Binomial, transfers unchanged
- Pitcher hard-hit-against -> bowler wicket rate (wickets / balls bowled), Beta-Binomial
- 3 MLB seasons -> 3 history periods: U10 season(s), this season pre-Xmas, post-Xmas
- Aging curve (peak age ~28) -> replaced by monotone relative-age-within-age-group effect (no decline phase for juniors)
- League average -> BNJCA grade-wide distribution; does DOUBLE DUTY: shrinkage target for our players AND the prior for unknown opponents

### Adaptations required
1. Runs off the bat are categorical {0,1,2,3,4,6}: model boundary rate (Beta-Binomial) + non-boundary strike rate (shrunken average) separately.
2. Simulator must encode actual BNJCA rules (now extracted - see Section 2).
3. Data parsing must handle the U10 vs U11 sundries convention difference (U10: sundries added to striker's score; U11: sundries separate).

---

## 2. Encoded BNJCA Under 11 Boys' playing conditions (2026 rules booklet, Section 17)

Extracted from https://bnjca.org.au/wp-content/uploads/2026/07/2026-RULES-BOOKLET-FINAL.pdf pages 32-35:

### Match structure
- One day, one innings each, maximum **25 overs per team** (Saturday afternoons, 1:30pm)
- **Non-competitive** developmental format (PlayHQ "points" are notional)

### Team size and dismissals
- Game-day team = **9 players** (max 11 registered, min 7); 9 fielders on field
- **All out at 8 dismissals** (with 9 players, one batter can remain not out)
- **Dismissed batter leaves and cannot return** (real survival process, unlike U10)

### Batting rules
- May retire Not out after facing minimum **25 balls** (coach/optional)
- **Must retire at 35 balls** (mandatory cap)
- Retired batters resume in order of retirement after all others dismissed/retired
- No balls and Wides count in the batter's ball count
- **Sundries NOT added to striker's score** (differs from U10)
- 16 m pitch, batters bat from one end, rotate at end of each over
- LBW does not apply

### Bowling rules (the participation constraint that shapes the optimizer)
- Bowlers bowl from one end for the whole innings
- Over = **6 fair balls or max 8 deliveries** (whichever first; extras re-bowled up to the 8-delivery cap)
- **ALL players must bowl, including both wicketkeepers**, with 25 overs divided:
  - Team of 9: 2 non-WK bowl 4 overs; 5 non-WK bowl 3 overs; 2 WK bowl 1 over each
  - Team of 8: 3 non-WK bowl 4; 3 non-WK bowl 3; 2 WK bowl 2
  - Team of 7: 3 non-WK bowl 5; 2 non-WK bowl 4; 2 WK bowl 1
- Players bowl roughly one over each in sequence (round-robin rotation)
- All sundries count against the bowler

### U10 (prior season) differences - for historical data parsing
- 20 overs, 7-player teams; dismissed batter CONTINUES batting their allotted balls; dismissal = 4-run penalty to batting team; sundries ARE added to striker's score; all bowl (7 players: 3x4, 2x3, 2x1 overs)

### Model implications
1. **U11 is a survival format** (dismissal ends a batter's innings, depletes the 8-wicket resource) but with ball caps: 25 overs = 150 balls across 9 batters means average ~17 balls per batter if all 150 used - the 35-ball cap binds only for top-order batters.
2. **The optimizer's bowling decision space is constrained**: within the mandatory allocation (2x4, 5x3, 2x1 for 9 players), choose WHO gets which tier and the rotation SEQUENCE. Participation is a hard rule, not an option.
3. **The optimizer's batting decision space**: batting order 1-9 (who faces the most balls) + optional-retirement policy (retire a batter at 25+ balls to protect others' participation, or let them bat to 35).
4. **Skills transfer U10 -> U11** (per-ball rates), while the format change is handled entirely in the simulator re-parameterization. The 3-period MARCEL weighting bridges the format transition.

---

## 3. Model specification (adapted MARCEL for juniors)

### Core hierarchical structure (per player i, per period t)
Batting:
- Dismissal hazard: `p_out[i,t] ~ Beta(mu_out, sigma_out)`, observed `outs[i,t] ~ Binomial(balls_faced[i,t], p_out[i,t])`
- Boundary rate: `p_bound[i,t] ~ Beta(mu_bound, sigma_bound)`, observed `boundaries[i,t] ~ Binomial(balls_faced[i,t], p_bound[i,t])`
- Non-boundary strike rate: partially-pooled mean runs per non-boundary ball (Gamma likelihood)

Bowling:
- Wicket rate: `p_wicket[i,t] ~ Beta(mu_w, sigma_w)`, observed `wickets[i,t] ~ Binomial(balls_bowled[i,t], p_wicket[i,t])`
- Economy: partially-pooled runs conceded per fair ball (Gamma likelihood); extras rate modelled separately (8-delivery cap interaction)

### MARCEL components
- Recency weights: `{w_1, w_2, w_3} ~ Dirichlet([3,4,5])` over the 3 history periods; projected rate = weighted sum of latent rates
- Relative-age effect: `age_effect_i = beta_age * ((months_old_i - mean_months) / 12)` on the logit scale, monotone
- Opposition: opponent batters/bowlers drawn from the fitted population distributions (mu, sigma)

### Simulator and optimizer
- Ball-by-ball simulation under the encoded Section 2 rules: 25 overs/6 fair balls (8-delivery cap), 9 batters, all-out at 8, retire at 25-35 balls, WK/bowler allocation tiers, round-robin over sequence, sundries conventions
- Posterior predictive draws for our players; population draws for opponents
- Search space: batting order (1-9), retirement policy, bowler tier assignment (who gets 4/3/1 overs), rotation sequence
- Objective: expected run differential / win probability across 5,000 simulated matches
- Backtest: hold out post-Xmas games, predict from pre-Xmas data, check posterior predictive calibration

---

## 4. Data acquisition strategy

### Sources
1. PlayHQ public GraphQL API (verified working, no API key): `https://api.playhq.com/graphql` with `tenant: ca` header and web origin headers - queries: `discoverGame`, `discoverGrade`, `discoverSeason`, `discoverTeams`, `teamFixture`, `gradeLadder`, `gradePlayerStatistics`
2. PlayHQ official REST API (if a key is later obtained): `GET /v1/organisations/{id}/seasons`, `/v1/seasons/{id}/grades`, `/v2/grades/{id}/games`, `/v2/games/{id}/summary`
3. Manual scorebook ingestion: CSV template for paper-scored games (BNJCA U10/U11 e-scoring is optional)

### Known identifiers
- Club: Valley District Cricket Club (VDCC), Ashgrove, Brisbane; CA club entity `c8fd4cfc-87d8-eb11-a7ad-2818780da0cc`
- Competition: BNJCA; verified live season/grade/team access on PlayHQ tenant `ca`
- Team: Under 10 Whites (2024/25, 2025/26) -> Under 11 for 2026/27 (Round 1: Sat 10 Oct 2026)
- Rules: BNJCA 2026 rules booklet Section 17 (encoded above in Section 2)

---

## 5. Implementation steps

1. **Repo setup**: clone `https://github.com/lmillard79/JuniorCricket_BayesianSearch.git` into `D:\Code\CricketScores_BayesianModel`; create venv; `requirements.txt` with `pymc`, `arviz`, `pandas`, `requests`, `matplotlib`, `scipy`; WRM-standard layout (`src/junior_cricket/`, `scripts/`, `data/raw|processed|outputs`, `notebooks/`).
2. **Rules config**: encode Section 2 playing conditions as `src/junior_cricket/rules_u11.py` (and U10 config for historical parsing).
3. **Data pipeline**: `src/junior_cricket/playhq_client.py` (GraphQL + optional REST), `src/junior_cricket/data_loader.py` (collation + manual CSV merge, sundries convention handling), interim flat CSV outputs for auditability.
4. **PyMC model**: `src/junior_cricket/model.py` implementing the adapted MARCEL spec; `notebooks/01_model_exploration.ipynb` for fitting and diagnostics (R-hat, divergences, posterior predictive checks).
5. **Simulator**: `src/junior_cricket/simulator.py` - ball-by-ball engine under encoded U11 rules.
6. **Optimizer**: `src/junior_cricket/optimizer.py` - order/allocation search within the mandatory participation constraints; outputs recommendations with win-probability CIs.
7. **Figures**: WRM palette, 300 dpi (batting posteriors, shrinkage demo, win-prob curves).

---

## 6. Verification plan
- [ ] Verify PlayHQ GraphQL returns BNJCA/VDCC historical games
- [ ] Validate CSV manual-scorebook ingestion against a known e-scored game (U10 sundries convention)
- [ ] MCMC convergence: R-hat < 1.05, no divergences
- [ ] Shrinkage sanity check: small-sample players pulled toward grade mean
- [ ] Simulator rule checks: 8-wicket all out, 35-ball retirement, 6-fair-ball/8-delivery overs, WK bowling allocation tiers, round-robin sequence
- [ ] Simulator aggregate checks: typical U11 totals plausible vs observed grade scores
- [ ] Backtest: pre-Xmas-only model predicts post-Xmas performances within posterior predictive bands

## 7. Risks / considerations
- U10/U11 PlayHQ data may be sparse (optional e-scoring): priors carry the model; document data coverage honestly
- Small rosters mean optimizer differences may be within model noise: report uncertainty, not just the argmax order
- The format is explicitly non-competitive and developmental: the optimizer works WITHIN mandatory participation rules (all players bowl; retirement rules) and should be framed as a fun decision-support tool, not a win-maximiser that overrides participation
- U11 exact playing format (Stage 1 pairs vs Stage 2) must be read from the BNJCA rules booklet before coding the simulator - do not assume
- Small rosters mean the optimizer's differences may be within model noise: report uncertainty, not just the argmax order

---

## Implementation (status: real U10 data ingested and validated; simulator calibration open)

Current status, validation results and next steps: `PROJECT_STATUS.md` and
`STATE.md`.

### Quickstart

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements-dev.txt
.venv/Scripts/python -m pip install -e .
.venv/Scripts/python -m pytest tests/
```

### Pipeline entry points

1. **Fetch PlayHQ data** (verified live against the BNJCA Girls Summer 2024/25 season):

```bash
python scripts/fetch_playhq_data.py --game-id f52ca224
python scripts/fetch_playhq_data.py --grade-id 2bb24d79
python scripts/fetch_playhq_data.py --stats-grade 2bb24d79
```

2. **Fit the MARCEL model** (synthetic demo verifies the full pipeline):

```bash
python scripts/fit_model.py --demo
python scripts/fit_model.py --batting data/processed/batting.csv \
    --bowling data/processed/bowling.csv --registry data/templates/player_registry_template.csv
```

3. **Optimise the lineup** (demo mode runs on synthetic skill rates):

```bash
python scripts/optimise_lineup.py --demo
python scripts/optimise_lineup.py --posterior data/outputs/posterior_model.nc
```

Outputs land in `data/outputs/` (posterior netcdf, summary CSV, shrinkage
figure, markdown recommendation, run-differential figure) with an audit log
under `data/log/`.

### Real-data workflow (PlayHQ scorecards and ball-by-ball)

The public PlayHQ site publishes per-player scorecards and ball-by-ball events
for e-scored games. These scripts pull them for a team and build scorebook CSVs
in the same schema as the manual templates, with **aliases (P01, ...) instead of
names**.

```bash
python scripts/fetch_scorecards.py --team-id 75cdae66 --team-id fafdb4c4
```

Everything fetched is cached under `data/raw/playhq/` (it holds player names and
is gitignored), so re-runs resume and never request a game twice. `--offline`
builds from the cache only, and `--max-games N` limits a run. It writes
`playhq_batting.csv` and `playhq_bowling.csv` (our squad), the opposition's rows,
and `playhq_all_*.csv` (both, so the model learns the whole grade). Then:

```bash
python scripts/fit_ball_model.py
python scripts/fit_model.py --batting data/processed/playhq_all_batting.csv --bowling data/processed/playhq_all_bowling.csv --sampler numpyro
python scripts/backtest.py --batting data/processed/playhq_batting.csv --bowling data/processed/playhq_bowling.csv
python scripts/check_u10_totals.py --team-id 75cdae66 --team-id fafdb4c4 --ball-posterior data/outputs/ball_posterior.nc --opposition real
python scripts/optimise_lineup.py --posterior data/outputs/posterior_model.nc --ball-posterior data/outputs/ball_posterior.nc --squad P01,P03,P04,P05,P06,P07,P08,P09,P10
python scripts/player_report.py --squad P01,P03,P04,P05,P06,P07,P08,P09,P10 --named
python scripts/replay_games.py --team-id 75cdae66 --team-id fafdb4c4
python scripts/compare_batting_strategies.py --search
```

`replay_games.py` rebuilds every real U10 game and plays it thousands of times, showing
where each recorded result sat against the model's median replay (over-by-over Manhattan
and runs-accumulated charts, plus season and decision charts) and how far a different
batting order or bowling split could have moved the margin. Alternatives are screened on
one set of replays and judged on a fresh set. `compare_batting_strategies.py` plays named
U11 batting-order strategies (and a swap search) through the same thousands of simulated
worlds of unknown opposition, tactics and ground, then repeats over a grid of assumptions
about the move to U11. Both write HTML reports under `data/outputs/` (gitignored). How to
read them: [docs/GUIDE.md](docs/GUIDE.md).

`fit_ball_model.py` fits the **joint batter-by-bowler ball model**, which is the
recommended basis for player and lineup statements: each ball's outcome depends
on batter and bowler together, so a bowler's economy is not judged against the
weak batters he happened to face. `--ball-posterior` makes it decide every ball
in the optimiser and the totals check. `fit_model.py` fits the older per-player
model (v2 by default; `--model v1` is the original specification, which does not
sample real data); it still seeds the optimiser's search. `--squad` names the
players to optimise when the posterior also holds opposition players.

Reports use aliases. `player_report.py --named` and
`python scripts/name_report.py <report>` write a private copy with real names
under `data/outputs/` (gitignored); never commit or share it.

### Manual data entry (primary data path)

BNJCA U10/U11 e-scoring is optional, so paper scorebooks are the main
historical source. Templates live in `data/templates/`:

- `batting_scorebook_template.csv` - one row per batter per innings
- `bowling_scorebook_template.csv` - one row per bowler per innings
- `player_registry_template.csv` - names, dates of birth (drives the
  relative-age effect), wicketkeeper flags (drive bowling tiers)

### Known issues

- On machines without a C compiler, PyMC's default PyTensor NUTS sampler
  crashes in the Python fallback linker. Use `--sampler numpyro` (numpyro and
  jax are in `requirements.txt`). The "g++ not available" line is then harmless.
- PlayHQ's public endpoints rate-limit bursts: a burst earns a CloudFront 403.
  The client spaces requests (2.5 s) and stops for good on the first 403 or
  429. If it stops, wait before re-running; the fetch resumes from the cache.
  PlayHQ's documented API (https://docs.playhq.com/tech/) needs an API key
  from PlayHQ, which is the proper route for anything ongoing.
- The two PlayHQ services take different tenant headers: `api.playhq.com`
  wants `tenant: cricket-australia` (with `ca` per-innings scores come back
  empty) and the spectator service wants `x-phq-tenant: ca`.
- `gradePlayerStatistics` returns rows with empty statistics; use the
  scorecards instead.
- The GitHub repo is public and PlayHQ data holds children's names. Keep raw
  pulls and the alias maps out of git (`data/raw/` is gitignored), and stage
  files explicitly.
- With the per-player rates the simulator's absolute totals run 14 to 20% low
  against real U10 games. The joint ball model fixes the opposition side (totals
  0.98 to 1.04 of real) but still understates our own strength by about 10%,
  probably a missing team fielding effect. See PROJECT_STATUS.md.
- The models learn from U10 games; U11 has a 45 m boundary (U10: 30 to 35 m), so
  U10 boundary rates will not carry over at face value.
- The optimiser does not know who keeps wicket; the two players given one over
  are the keepers by rule.
