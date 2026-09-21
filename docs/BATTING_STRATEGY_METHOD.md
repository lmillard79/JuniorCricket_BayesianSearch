# Batting-order strategy for U11: method and pathway

## 1. The question

Under BNJCA U11 rules there is a wide gap between our strongest and weakest
batters. Should the order be **conventional** (strongest to weakest), or should a
strong batter be **paired with a weaker one** (strong, weak, strong ...)? The
working thought behind the pairing idea is that the opposition attack is unknown, so
our best batters will not necessarily meet their best bowlers, and a pairing might
spread the risk.

The aim is a Bayesian match simulator that can trial order strategies when the
opposition's strength and tactics are completely unknown, and say what each strategy
does to **total runs**, with honest uncertainty.

## 2. Why the U10 data alone cannot answer it (and why we replayed U10 anyway)

Under U10 rules (Rule 16.10) every batter is given a fixed 13 or 14 balls, so the
order hardly changes who faces how much. Replaying all 12 full-length U10 games with
shuffled orders confirmed it: the best order found was worth **+0.8 runs** of margin
on average (95% interval about +0.6 to +1.0), against a typical game-to-game swing of
**28 runs** from luck alone. (The bowling split was worth +0.7 runs with the keepers as
played.) So the U10 games cannot say whether order matters under U11; they can only
supply the skills.

U11 is different: there are no fixed allotments. Batters retire between 25 and 35
balls, a dismissal ends an innings, and 150 balls are shared among nine batters, so
order decides *who gets the balls*. The U10 replay was still worth doing because it
checks the simulator against reality (section 7) before we lean on it for U11.

## 3. The method in six steps

**Step 1. A model of the ball.** The joint batter-by-bowler model (`ball_model.py`)
gives, for each ball, the chance of a dismissal, a boundary and a scoring shot as a
function of the batter, the bowler and the ground. It is fitted to 3,095 real U10 balls.
Because it is Bayesian, each player's skill is a *spread* of plausible values, and we
sample from that spread.

**Step 2. A "world".** One world is one draw of everything we cannot know before the
match:

- how good each of our players *really* is (a draw from the fitted spread, optionally
  nudged to allow for a year's development, "drift");
- who the opposition are (nine players drawn from the grade, so nothing is assumed
  about them beyond what the grade looks like);
- their bowling tactics (who bowls how many overs and in what rotation: random, or
  "smart" with their best bowlers given the biggest shares and the opening overs);
- the ground (boundary and scoring level).

**Step 3. Strategies as rules.** A strategy turns *our estimates* into a batting order.
Each batter is valued by the expected runs in a stint of up to 30 balls with a
dismissal ending it (so quick scorers who get out often are not overrated). Then:

| Strategy | Order (r1 best, r9 worst) |
| --- | --- |
| strongest to weakest | r1 r2 r3 r4 r5 r6 r7 r8 r9 |
| weakest to strongest | r9 r8 ... r1 |
| strong-weak alternating | r1 r9 r2 r8 r3 r7 r4 r6 r5 (every pair holds one strong and one weak batter) |
| balanced pairs (top six) | r1 r6 r2 r5 r3 r4 r7 r8 r9 (strong with weaker, but only among the six who normally bat) |
| strong-middle alternating | r1 r4 r2 r5 r3 r6 r7 r8 r9 |
| random orders | a baseline showing what ignoring skill costs |
| searched order | the best order found by trying every swap of two batters |

The strategy only knows the posterior means; the worlds contain the truth. Estimation
error is therefore part of the test.

**Step 4. Paired evaluation.** Every strategy is played through the *same* worlds (same
true skills, same opposition, same tactics, same ground, and the same ball-level random
numbers to start with). A difference between two strategies is then not caused by one of
them meeting a tougher attack. We report the mean total, its spread, the paired
difference from strongest to weakest with its standard error, and the share of worlds
in which the strategy did better. Results can also be split by attack strength.

**Step 5. Search, judged on fresh worlds.** A swap search starts from the conventional
order, tries every pair swap on a set of worlds, and accepts a swap only if it wins by
more than two standard errors, so noise is not chased. Whatever it finds is then
re-judged on 4,000 fresh worlds it never saw. (Choosing the best of many noisy estimates
flatters it; the fresh worlds remove that flattery.)

**Step 6. Stress-test the assumptions.** The U10-to-U11 transfer is an assumption, so it
is a dial, not a fact. The scenario grid varies:

- **retirement policy**: retire every batter at 25, 30 or 35 balls;
- **ground size**: boundary log-odds shifted by 0, -0.7 and -1.4 (the 45 m U11 boundary
  against 30 to 35 m; -0.7 roughly halves the odds of a boundary);
- **skill drift**: 0 or 0.3 (a year's development, independently per player);
- **opposition tactics**: random or smart.

That is 36 scenarios. A conclusion only counts if it survives the grid.

## 4. What the pilot says

Squad: the nine 2025/26 players ranked by the model; U10 skills. Main scenario: retire at
35, no ground shift, no drift, random attack. The named strategies were played through
4,000 fresh worlds x 2 innings each (the swap search was tuned on a separate 3,000);
the random orders used a separate 6,000. All comparisons are paired; the ± is one
standard error.

| Strategy | Mean total | Runs vs strongest to weakest | Share of worlds better |
| --- | --- | --- | --- |
| strongest to weakest | 149.4 | 0 | n/a |
| strong-middle alternating | 149.0 | -0.3 ± 0.3 | 48% |
| balanced pairs (top six) | 148.5 | -0.8 ± 0.4 | 47% |
| strong-weak alternating | 142.0 | -7.4 ± 0.4 | 36% |
| weakest to strongest | 130.7 | -18.7 ± 0.5 | 24% |
| random orders (12) | about 140 | -9.1 on average (-15.1 to -3.7) | n/a |

**Findings**

1. **Order matters under U11**, far more than under U10. Ignoring skill (a random order)
   costs about 9 runs of a 149 total; leading with the weakest costs about 19.
2. **The conventional order is not beaten by any alternative tried.** The swap search
   found no order that beats it, and across all 35 other scenarios in the grid no
   alternative beat it by more than 0.2 runs.
3. **Pairing a strong batter with a weak one loses runs.** Alternating strong and weak
   across the whole squad costs about 7 runs. Pairing only within the six batters who
   normally bat (balanced pairs) or top-third with middle-third (strong-middle) is close
   to a tie (under 1 run). There is no sign that pairing spreads risk in a way that
   helps.
4. **The mechanism is who gets the balls.** In the conventional order the six best batters
   face 17 to 20 balls each and the three weakest 5 to 12. The alternating order pushes 13
   to 15 balls onto the weakest batters at the expense of the fifth-best (7 balls), which is
   where the runs are lost. (`balls_by_rank.png`)
5. **The "best batters miss the best bowlers" idea does not change the ranking.** Against
   a smart attack (best bowlers open and bowl most overs) every total falls by about 4
   runs, but the ranking of strategies is unchanged. With logit-additive batter and bowler
   effects there is little matchup to exploit even if the attack were known.
6. **The ranking is robust.** Across retirement policy, ground size, drift and opposition
   tactics the order of strategies never changed. Bigger boundaries shrink the penalty for
   a poor order (averaged over the retirement policies, leading with the weakest costs about
   13 runs at no shift and about 7 at a -1.4 shift), presumably because boundary hitting is
   where batters differ most and a bigger ground removes many boundaries.
7. **The retirement policy moves totals as much as most order choices.** With the same
   order and the current ground size, retiring every batter at 25 rather than 35 balls
   costs about 5 runs (less on a bigger ground), because the best batters are taken off.
   A per-batter policy (best batters to 35, weaker ones at 25) is not yet in the engine
   and is the next lever to test.
8. **What a participation guarantee costs.** The order that guarantees the weakest three
   at least 13 balls each (strong-weak alternating) costs about 7 runs, roughly 5% of a
   total. That is a coaching choice about fairness, not something the model can decide,
   but the model can now put a price on it.

## 5. What would change the answer

| Uncertainty | Why it matters | How we will find out |
| --- | --- | --- |
| Skill transfer from U10 to U11 | Skills are U10 estimates; a year older and a bigger ground | Refit as U11 games arrive; drift is already in the grid |
| Boundary size (45 m) | Boundary hitting is the strongest skill and drives the ranking | Measure the boundary rate in the first U11 games; it pins down the ground-shift dial |
| Per-batter retirement | The coach may be able to retire batters individually between 25 and 35 balls | Confirm the rule; extend the engine (small change) |
| Wicketkeepers and fielding | Not modelled; the model is about 10% stingy about our own side | Team fielding effect and keeper flags (planned) |
| Strike rotation | The engine changes ends only at the end of an over (per the 2026 rules); real practice may differ | Check balls per batter by position in U11 ball data |
| In-innings form | Batters are modelled as constant within an innings | Compare runs per ball by balls faced in U11 data |
| Extras | Wides and no-balls are not modelled (none were recorded in U10) | Add once U11 data shows their rate; it affects all strategies equally |
| Small sample | 10 players, 15 games | Brackets are honest about it; U11 data adds to it |

A caution on validation. A season of about 15 games can never *directly* show a 1 to 3 run
order effect against a 28 run game-to-game swing; that would need thousands of games. The
claim rests on the model being right about the *mechanism*: who faces how many balls, and
what they do with them. Both are testable in U11 data (balls faced by batting position,
runs per ball by position, the boundary rate), which is where real evidence will come
from.

## 6. Pathway

| When | What | Output |
| --- | --- | --- |
| Done (22 Sep 2026) | Joint ball model; U10 and U11 engines; U10 replay validation; strategy harness with paired worlds, search and stress grid; pilot results above | This document, `replay_report.html`, `strategy_report.html` |
| Before Round 1 (Sat 10 Oct 2026) | Confirm the exact U11 rules with the coach (per-batter retirement, extras); add the team fielding effect and wicketkeeper flags to the joint model and re-check calibration; add per-batter retirement policies to the engine; load the U11 squad (new players get the grade prior); rerun the strategy comparison for the actual squad | A pre-season recommendation with ranges, and a list of the assumptions it rests on |
| Rounds 1 to 4 | After each game, fetch the ball-by-ball (API key when granted, otherwise slowly); refit with a U11 offset for each outcome; run the replay tool on U11 games; compare predicted with actual balls by position and boundary rate | The first real test of the transfer assumption |
| After about 8 games | Narrow the ground-shift and drift dials to what the data support; rerun the grid | An updated recommendation; shrinking uncertainty |
| End of season | Out-of-sample review: did the model predict totals and the shape of innings? | A go or no-go on relying on it for 2027 |

**Decision rule.** Keep strongest to weakest as the default. Change it only if an alternative
beats it by more than two standard errors in at least 80% of the scenarios, or for
participation reasons the coach chooses, using the model's price for that choice.

## 7. How this was validated so far (in-sample)

Replaying the 12 full-length U10 games with the joint model, the recorded results sat
above the model's median replay in 7 of 12 games for our total (average percentile 61st),
6 of 12 for theirs (44th) and 9 of 12 for the margin (64th). A well-calibrated model would
sit near the 50th. So the model is somewhat stingy about our side, consistent with the known
gap of about 10% (a missing team fielding effect is the likeliest cause); it is well
calibrated for the opposition. This does not affect the ranking of strategies (which is
about our batters relative to each other) but it does affect absolute totals, and it is why
the next model step is the team effect.

## 8. Decisions needed from you

1. Can the coach retire an individual batter at any point between 25 and 35 balls (per-batter
   choice), as we assume, or must the whole team follow one threshold?
2. Who keeps wicket, and can the choice change game to game?
3. What is the objective: maximise total runs, or maximise runs subject to a fairness rule (for
   example, every batter faces at least 10 balls)? The model can price the second.
4. The 2026/27 U11 squad and any new players, so the comparison is run on the real group.
5. Whether to add the team fielding effect before Round 1 (recommended: it is the biggest known
   calibration gap).

## 9. How to reproduce

```bash
python scripts/compare_batting_strategies.py --search
```

Writes `data/outputs/strategies/` (gitignored): `strategy_report.html`, `strategy_results.csv`,
`strategy_balls_by_rank.csv`, `worms.png`, `balls_by_rank.png`, `sensitivity.png`. Read the
outputs with [GUIDE.md](GUIDE.md), section 4.8.
