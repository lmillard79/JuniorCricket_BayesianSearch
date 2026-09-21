# JuniorCricket_BayesianSearch
An attempt at finding the optimal batting and bowling order for Under 11s based on varied player strength for both teams. 
https://www.pymc.io/projects/examples/en/latest/case_studies/rugby_analytics.html
https://www.pymc.io/projects/examples/en/latest/case_studies/hierarchical_partial_pooling.html
https://bnjca.org.au/wp-content/uploads/2026/07/Quick-Reference-Rules.pdf

---
created: 2026-09-21T07:15:07Z
---
# Bayesian MARCEL for Junior Cricket - VDCC Under 11 Lineup Optimiser

Adapt the Bayesian MARCEL baseball projection framework to junior cricket for VDCC Under 11s: hierarchical Beta-Binomial player model on PlayHQ BNJCA data, grade-wide population as unknown-opponent prior, and a match simulator to optimise batting and bowling orders.

## Summary
Adapt the Bayesian MARCEL baseball projection framework (PyMC Labs) and hierarchical partial pooling to junior cricket: fit a hierarchical Beta-Binomial model on PlayHQ BNJCA data for the VDCC team moving from Under 10 Whites to Under 11, then use the fitted population distribution as the prior for unknown opponents in a ball-by-ball match simulator that optimises batting order and bowling allocation.

---

## 1. Why these baseball models transfer

### Source models
- **Hierarchical partial pooling** (Efron & Morris batting averages, PyMC example): players' true rates drawn from one league-level Beta distribution; players with small samples shrink toward the league mean.
- **Bayesian MARCEL** (PyMC Labs blog): adds (a) Dirichlet-learned recency weights over 3 periods, (b) an aging effect on the logit scale, (c) full posterior uncertainty per player.

### Concept mapping (baseball -> junior cricket)
- At-bat (Bernoulli trial) -> ball faced
- Batting average (hits/AB) -> dismissal hazard (outs / balls faced), Beta-Binomial, transfers unchanged
- Hard-hit rate -> boundary rate (4s+6s / balls faced), Beta-Binomial, transfers unchanged
- Pitcher hard-hit-against -> bowler wicket rate (wickets / balls bowled), Beta-Binomial
- 3 MLB seasons -> 3 history periods: prior U10 season(s), this season pre-Xmas, post-Xmas
- Aging curve (peak age ~28) -> replaced by a monotone relative-age effect within the age group (an older-by-months U11 is stronger on average; no decline phase)
- League average -> BNJCA grade-wide distribution, and it does DOUBLE DUTY: shrinkage target for our players AND the prior for unknown opponents (solves the "don't know opposition strength" problem with one model)

### What must be adapted (cricket-specific)
1. Runs off the bat are categorical {0,1,2,3,4,6}, not Bernoulli: model boundary rate (Beta-Binomial) + non-boundary scoring rate (shrunken average) separately - keeps the model MARCEL-simple.
2. Junior format rules change dismissal semantics: Stage 1 pairs cricket makes a dismissal a -3 penalty (batter continues); BNJCA U11 rules (Section 17, rules booklet PDF) must be encoded as simulator config, not assumed.
3. The optimizer has no baseball analog: batting order determines who faces the most deliveries, and junior rules force bowling participation limits. Match simulation + order search is additional cricket-specific work.
4. Data reality: BNJCA makes U10/U11 e-scoring optional, so PlayHQ holds only some games. Pipeline must ingest API data + manual scorebook CSVs; the Bayesian prior handles the gaps.
5. Recency weighting maps perfectly to fast-developing juniors; the Dirichlet weights should recover "recent form dominates" for free.

---

## 2. Model specification (adapted MARCEL for juniors)

### Core hierarchical structure (per player i, per period t)
Batting:
- Dismissal hazard: `p_out[i,t] ~ Beta(mu_out, sigma_out)`, observed `outs[i,t] ~ Binomial(balls_faced[i,t], p_out[i,t])`
- Boundary rate: `p_bound[i,t] ~ Beta(mu_bound, sigma_bound)`, observed `boundaries[i,t] ~ Binomial(balls_faced[i,t], p_bound[i,t])`
- Non-boundary strike rate: partially-pooled mean runs per non-boundary ball (Gamma likelihood)

Bowling:
- Wicket rate: `p_wicket[i,t] ~ Beta(mu_w, sigma_w)`, observed `wickets[i,t] ~ Binomial(balls_bowled[i,t], p_wicket[i,t])`
- Economy: partially-pooled runs conceded per ball (Gamma likelihood)

### MARCEL components
- Recency weights: `{w_1, w_2, w_3} ~ Dirichlet([3,4,5])` over the 3 history periods; projected rate = weighted sum of latent rates
- Relative-age effect: `age_effect_i = beta_age * ((months_old_i - mean_months) / 12)` on the logit scale, monotone (no peak age - juniors only improve)
- Opposition: opponent batters/bowlers drawn from the fitted population distributions (mu, sigma) - the unknown-opponent prior

### Simulator and optimizer
- Ball-by-ball simulation of a full match under encoded BNJCA U11 rules (pairs vs dismissal format, max overs per bowler, participation bowling, retirement rules)
- Posterior predictive draws for our players; population draws for opponents
- Search over batting orders and bowling allocations; objective = expected run differential / win probability across 5,000 simulated matches
- Backtest: hold out post-Xmas games, predict from pre-Xmas data, check posterior predictive calibration (mirrors the MARCEL blog's projection evaluation)

---

## 3. Data acquisition strategy

### Sources
1. PlayHQ public GraphQL API (verified working, no API key): `https://api.playhq.com/graphql` with `tenant: ca` header and web origin headers - queries: `discoverGame`, `discoverGrade`, `discoverSeason`, `discoverTeams`, `teamFixture`, `gradeLadder`, `gradePlayerStatistics`
2. PlayHQ official REST API (if a key is later obtained via Cricket Australia): `GET /v1/organisations/{id}/seasons`, `/v1/seasons/{id}/grades`, `/v2/grades/{id}/games`, `/v2/games/{id}/summary`
3. Manual scorebook ingestion: CSV template for paper-scored games (BNJCA U10/U11 e-scoring is optional)

### Known identifiers
- Club: Valley District Cricket Club (VDCC), Ashgrove, Brisbane; CA club entity `c8fd4cfc-87d8-eb11-a7ad-2818780da0cc`
- Competition: Brisbane North Junior Cricket Association (BNJCA); verified live season/grade/team access on PlayHQ tenant `ca`
- Team: Under 10 Whites (2024/25, 2025/26) transitioning to Under 11 for season 2026/27 (Round 1: Sat 10 Oct 2026)
- Rules source: BNJCA rules booklet PDF (bnjca.org.au) - Section 17 for U11 playing conditions

---

## 4. Implementation steps

1. **Repo setup**: clone `https://github.com/lmillard79/JuniorCricket_BayesianSearch.git` into `D:\Code\CricketScores_BayesianModel`; create venv; `requirements.txt` with `pymc`, `arviz`, `pandas`, `requests`, `matplotlib`, `scipy`; WRM-standard layout (`src/junior_cricket/`, `scripts/`, `data/raw|processed|outputs`, `notebooks/`).
2. **Data pipeline**: `src/junior_cricket/playhq_client.py` (GraphQL + optional REST), `src/junior_cricket/data_loader.py` (collation + manual CSV merge), interim flat CSV outputs for auditability (`vdcc_player_stats.csv`, `grade_baseline_stats.csv`).
3. **PyMC model**: `src/junior_cricket/model.py` implementing the adapted MARCEL spec above; `notebooks/01_model_exploration.ipynb` for fitting and diagnostics (R-hat, divergences, posterior predictive checks).
4. **Simulator**: `src/junior_cricket/simulator.py` - ball-by-ball engine with encoded U11 rules config.
5. **Optimizer**: `src/junior_cricket/optimizer.py` - order/allocation search against opponent population draws; outputs recommendations with win-probability CIs.
6. **Figures**: WRM palette, 300 dpi (batting posteriors, shrinkage demo, win-prob curves).

---

## 5. Verification plan
- [ ] Verify PlayHQ GraphQL returns BNJCA/VDCC historical games
- [ ] Validate CSV manual-scorebook ingestion against a known e-scored game
- [ ] MCMC convergence: R-hat < 1.05, no divergences
- [ ] Shrinkage sanity check: small-sample players pulled toward grade mean (replicate the partial-pooling example's classic figure with our data)
- [ ] Simulator rule checks: overs per bowler caps, retirement rules enforced
- [ ] Backtest: pre-Xmas-only model predicts post-Xmas performances within posterior predictive bands

## 6. Risks / considerations
- U10/U11 PlayHQ data may be sparse (optional e-scoring): priors carry the model; document data coverage honestly
- U11 exact playing format (Stage 1 pairs vs Stage 2) must be read from the BNJCA rules booklet before coding the simulator - do not assume
- Small rosters mean the optimizer's differences may be within model noise: report uncertainty, not just the argmax order
