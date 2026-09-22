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
| bank top 4, recall on a cheap wicket | r1 r2 r3 ... r9 (the conventional order), but r1 to r4 may retire not out at 25 balls, and the strongest retired batter comes straight back in, ahead of the next fresh batter, if whoever comes in next is out within 6 balls or for under 5 runs |
| random orders | a baseline showing what ignoring skill costs |
| searched order | the best order found by trying every swap of two batters |

The strategy only knows the posterior means; the worlds contain the truth. Estimation
error is therefore part of the test.

**Step 4. Paired evaluation.** Every strategy is played through the *same* worlds (same
true skills, same opposition, same tactics, same ground, and the same ball-level random
numbers to start with). A difference between two strategies is then not caused by one of
them meeting a tougher attack. We report the mean total, its spread, the paired
difference from strongest to weakest with its standard error, and the share of worlds
in which the strategy did better. Results are also split by attack strength (the worlds
grouped into the strongest, middle and weakest thirds of opposition attacks), which tests
directly whether a pairing helps when the bowling is strong.

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
| **bank top 4, recall on a cheap wicket** | **153.5** | **+1.4 ± 0.2** | **49%** |
| strongest to weakest | 152.2 | 0 | n/a |
| strong-middle alternating | 151.5 | -0.6 ± 0.3 | 48% |
| balanced pairs (top six) | 150.9 | -1.2 ± 0.3 | 47% |
| strong-weak alternating | 144.6 | -7.6 ± 0.4 | 36% |
| weakest to strongest | 132.1 | -20.0 ± 0.5 | 22% |
| random orders (12) | about 142 | -10.0 on average | n/a |

**Findings**

1. **Order matters under U11**, far more than under U10. Ignoring skill (a random order)
   costs about 10 runs of a 152 total; leading with the weakest costs about 20.
2. **Among the fixed orders, the conventional one is not beaten by any alternative
   tried.** The swap search found no reordering that beats it, and across the grid no
   fixed-order alternative beat it by more than 0.4 runs.
3. **Pairing a strong batter with a weak one loses runs.** Alternating strong and weak
   across the whole squad costs about 7 runs. Pairing only within the six batters who
   normally bat (balanced pairs) or top-third with middle-third (strong-middle) is close
   to a tie (under 1 run). There is no sign that pairing spreads risk in a way that
   helps.
4. **Changing the retirement tactics, not the order, is what actually gains runs.**
   "Bank top 4, recall on a cheap wicket" keeps the conventional order but lets the top
   four retire not out at 25 balls and come straight back in, strongest first, if
   whoever replaces them is out cheaply or quickly. It is the only alternative tried
   that beat strongest to weakest, by 1.4 runs on average (about 7 standard errors, so
   a real if modest effect), and it was the *best* alternative in all 37 scenarios
   tested (main plus the 36-scenario grid), ranging from +0.7 to +7.4 runs. It stayed
   positive against every third of opposition attacks (+0.8 to +1.7), with no sign it
   depends on how strong the attack is.
5. **That gain is not obviously a participation win.** Balls faced per innings barely
   move for most of the order, but the best batter gains about 2 balls (20.4 to 22.5)
   and the weakest loses about 1 (5.3 to 4.0): a recalled top batter uses overs that
   would otherwise eventually have reached the tail. If the coach's aim is specifically
   to guarantee the weakest batters more balls (rather than more team runs), this
   version of the policy is not that; a version that recalls to *protect the weakest*
   rather than *maximise runs* would need a different recall rule and has not been built.
6. **The mechanism for the fixed orders is who gets the balls.** In the conventional
   order the six best batters face 17 to 20 balls each and the three weakest 5 to 12.
   The alternating order pushes 13 to 15 balls onto the weakest batters at the expense
   of the fifth-best (7 balls), which is where the runs are lost. (`balls_by_rank.png`)
7. **The "best batters miss the best bowlers" idea does not help a fixed order.**
   Splitting the worlds into thirds by how strong the opposition attack was, strong-weak
   alternating loses about the same against the strongest attacks (-7.4 ± 0.6) as against
   the weakest (-8.3 ± 0.7), and balanced pairs and strong-middle are within about a run
   of a tie in every third (`strategy_by_attack_strength.csv`). A smart attack (best
   bowlers open and bowl most overs) lowers every total by about 4 runs but does not
   change the ranking. Under this model, where batter and bowler effects add on the
   logit scale, there is little matchup to exploit even if the attack were known.
8. **The ranking is robust.** Across retirement policy, ground size, drift and opposition
   tactics the order of strategies never changed, and bank-and-recall stayed on top
   throughout. Bigger boundaries shrink the penalty for a poor fixed order (averaged
   over the retirement policies, leading with the weakest costs about 13 runs at no
   shift and about 7 at a -1.4 shift), presumably because boundary hitting is where
   batters differ most and a bigger ground removes many boundaries.
9. **A blanket retirement policy moves totals as much as most order choices.** With the
   same order and the current ground size, retiring *every* batter at 25 rather than 35
   balls costs about 5 runs (less on a bigger ground), because the best batters are taken
   off too. Bank-and-recall is the per-batter alternative to that blanket policy: only the
   top four retire early, and only long enough to see off a cheap wicket.
10. **What a participation guarantee costs, if runs are what matters.** The order that
    guarantees the weakest three at least 13 balls each (strong-weak alternating) costs
    about 7 runs, roughly 5% of a total. That is a coaching choice about fairness, not
    something the model can decide, but the model can now put a price on it, and
    bank-and-recall shows that a *different* kind of fairness lever (giving the coach the
    retirement choice) does not have to cost runs at all, even though this particular
    version of it does not target fairness for the weakest batters specifically.

## 5. What would change the answer

| Uncertainty | Why it matters | How we will find out |
| --- | --- | --- |
| Skill transfer from U10 to U11 | Skills are U10 estimates; a year older and a bigger ground | Refit as U11 games arrive; drift is already in the grid |
| Boundary size (45 m) | Boundary hitting is the strongest skill and drives the ranking | Measure the boundary rate in the first U11 games; it pins down the ground-shift dial |
| Per-batter retirement | Confirmed allowed by the rules, though not yet used on the day | Done: "bank top 4, recall on a cheap wicket" in the engine and the comparison above |
| Wicketkeepers and fielding | The model was about 10% stingy about our own side | Team fielding effect done (22 Sep 2026): a real but modest edge, 0.17 [-0.15, 0.47] on the dismissal log-odds (81% posterior probability positive), narrowing the gap without closing it (PROJECT_STATUS.md). Individual catches and run outs are recorded per ball too (`player_report.md`'s Fielding table), but only 67 of 172 dismissals name a fielder, too few to fit a per-player effect. A true wicketkeeper flag would still need the coach's own per-game record; PlayHQ does not tag it |
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
| Done (22 Sep 2026) | Joint ball model; U10 and U11 engines; U10 replay validation; strategy harness with paired worlds, search and stress grid; pilot results above; per-batter bank-and-recall retirement policy in the engine and comparison | This document, `replay_report.html`, `strategy_report.html` |
| Before Round 1 (Sat 10 Oct 2026) | Add the team fielding effect to the joint model and re-check calibration; wicketkeeper flags if a real per-game signal can be sourced; load the U11 squad (new players get the grade prior); rerun the replay and strategy comparison for the actual squad | A pre-season recommendation with ranges, and a list of the assumptions it rests on |
| Rounds 1 to 4 | After each game, fetch the ball-by-ball (API key when granted, otherwise slowly); refit with a U11 offset for each outcome; run the replay tool on U11 games; compare predicted with actual balls by position and boundary rate | The first real test of the transfer assumption |
| After about 8 games | Narrow the ground-shift and drift dials to what the data support; rerun the grid | An updated recommendation; shrinking uncertainty |
| End of season | Out-of-sample review: did the model predict totals and the shape of innings? | A go or no-go on relying on it for 2027 |

**Decision rule.** Keep strongest to weakest as the default. Change it only if an alternative
beats it by more than two standard errors in at least 80% of the scenarios, or for
participation reasons the coach chooses, using the model's price for that choice.

## 7. How this was validated so far (in-sample)

Replaying the 12 full-length U10 games with the joint model, the recorded results sat
above the model's median replay in 7 of 12 games for our total (average percentile 59th),
7 of 12 for theirs (46th) and 9 of 12 for the margin (62nd). A well-calibrated model would
sit near the 50th. The model is still somewhat stingy about our side, but less than before
the team fielding effect was added (was 61st/44th/64th; PROJECT_STATUS.md): the effect is
real but modest, so it narrows the gap rather than closing it. This does not affect the
ranking of strategies (which is about our batters relative to each other) but it does affect
absolute totals.

## 8. Decisions needed from you

Answered so far (2026-09-22):

1. ~~Can the coach retire an individual batter at any point between 25 and 35 balls?~~ Yes,
   allowed by the rules, though not yet used on the day. Built as "bank top 4, recall on a
   cheap wicket" above.
2. ~~What is the objective: pure runs, or a fairness rule?~~ Participation matters: "everyone
   should get a go", with a preference for banking a strong batter at 25 and bringing them
   back if a weaker batter is dismissed quickly or cheaply. Built as above, with the caveat
   in finding 5: the runs-maximising version of this idea shifts a couple of balls *away*
   from the weakest batter, not toward them, so it is not yet a direct answer to the
   fairness half of the question.
3. ~~Add the team fielding effect and wicketkeeper flags?~~ Yes, approved, and built
   (22 Sep 2026): the raw PlayHQ data turned out to name the fielder on catches and run
   outs directly (checked properly this time, PROJECT_STATUS.md), so the fielding effect
   uses that rather than a blanket team guess. It came out real but modest: 0.17
   [-0.15, 0.47] on the dismissal log-odds, 81% posterior probability positive. It
   narrows the "10% stingy on our side" gap without closing it.

Still open:

4. The 2026/27 U11 squad and any new players, so the comparison is run on the real group.
5. Who keeps wicket, and can the choice change game to game? PlayHQ's data does not record
   it, so a true keeper flag would need the coach's own per-game note.
6. Is the runs-maximising bank-and-recall policy (finding 5) an acceptable trade against its
   fairness goal, or should a version that specifically protects the weakest batters' balls
   be built instead, even if it costs a little more than +1.4 runs?

## 9. How to reproduce

```bash
python scripts/compare_batting_strategies.py --search
```

Writes `data/outputs/strategies/` (gitignored): `strategy_report.html`, `strategy_results.csv`,
`strategy_balls_by_rank.csv`, `strategy_by_attack_strength.csv`, `worms.png`,
`balls_by_rank.png`, `sensitivity.png`. Read the outputs with [GUIDE.md](GUIDE.md), section 4.8.
