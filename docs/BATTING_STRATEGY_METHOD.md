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

Three more dials were added on 11 Oct 2026, after the first U11 game, and are held fixed
across the grid by command-line options: the **strike rules** (`--strike-rules`), a
**scoring shift** on the log-odds that a non-boundary ball scores (`--scoring-shift`) and an
**extras rate**, the chance a delivery is a wide or no-ball (`--extras-rate`). Section 4b
uses them to describe a "U11-like" world.

## 4. What the pilot says

Squad: the nine 2025/26 players ranked by the model; U10 skills and U10 scoring rates.
Main scenario: retire at 35, no ground shift, no drift, random attack. The named
strategies were played through 4,000 fresh worlds x 2 innings each (the swap search was
tuned on a separate 3,000); the random orders used a separate 6,000. All comparisons are
paired; the ± is one standard error. Figures are with the strike rules corrected on
11 Oct 2026 (batters cross on odd runs, a new batter takes strike after a wicket;
section 4b). They moved by at most half a run from the first pilot, which rotated strike
at the end of the over only, and nothing changed sign or rank.

| Strategy | Mean total | Runs vs strongest to weakest | Share of worlds better |
| --- | --- | --- | --- |
| **bank top 4, recall on a cheap wicket** | **153.1** | **+1.3 ± 0.2** | **48%** |
| strongest to weakest | 151.8 | 0 | n/a |
| strong-middle alternating | 151.6 | -0.2 ± 0.3 | 49% |
| balanced pairs (top six) | 150.6 | -1.2 ± 0.4 | 46% |
| strong-weak alternating | 143.9 | -7.9 ± 0.4 | 35% |
| weakest to strongest | 131.9 | -19.9 ± 0.4 | 22% |
| random orders (12) | about 142 | -9.6 on average | n/a |

**Findings**

1. **Order matters under U11**, far more than under U10. Ignoring skill (a random order)
   costs about 10 runs of a 152 total; leading with the weakest costs about 20.
2. **Among the fixed orders, the conventional one is not beaten by any alternative
   tried.** The swap search found no reordering that beats it, and across the grid no
   fixed-order alternative beat it by more than 0.5 runs.
3. **Pairing a strong batter with a weak one loses runs.** Alternating strong and weak
   across the whole squad costs about 8 runs. Pairing only within the six batters who
   normally bat (balanced pairs) or top-third with middle-third (strong-middle) is close
   to a tie (about a run or less). There is no sign that pairing spreads risk in a way that
   helps.
4. **Changing the retirement tactics, not the order, is what actually gains runs.**
   "Bank top 4, recall on a cheap wicket" keeps the conventional order but lets the top
   four retire not out at 25 balls and come straight back in, strongest first, if
   whoever replaces them is out cheaply or quickly. It is the only alternative tried
   that beat strongest to weakest, by 1.3 runs on average (about 6 standard errors, so
   a real if modest effect), and it was the *best* alternative in all 37 scenarios
   tested (main plus the 36-scenario grid), ranging from +0.7 to +7.8 runs. It stayed
   positive against every third of opposition attacks (+1.1 to +1.7), with no sign it
   depends on how strong the attack is.
5. **That gain is not obviously a participation win.** Balls faced per innings barely
   move for most of the order, but the best batter gains about 2 balls (20.5 to 22.3)
   and the weakest loses about 1 (5.4 to 4.4): a recalled top batter uses overs that
   would otherwise eventually have reached the tail. If the coach's aim is specifically
   to guarantee the weakest batters more balls (rather than more team runs), this
   version of the policy is not that; a version that recalls to *protect the weakest*
   rather than *maximise runs* would need a different recall rule and has not been built.
6. **The mechanism for the fixed orders is who gets the balls.** In the conventional
   order the six best batters face 17 to 20 balls each and the three weakest 5 to 12.
   The alternating order pushes 14 to 15 balls onto the weakest batters at the expense
   of the fifth-best (7 balls), which is where the runs are lost. (`balls_by_rank.png`)
7. **The "best batters miss the best bowlers" idea does not help a fixed order.**
   Splitting the worlds into thirds by how strong the opposition attack was, strong-weak
   alternating loses about the same against the strongest attacks (-8.7 ± 0.6) as against
   the weakest (-7.6 ± 0.7), and balanced pairs and strong-middle are within about a run
   of a tie in every third (`strategy_by_attack_strength.csv`). A smart attack (best
   bowlers open and bowl most overs) lowers every total by about 6 runs but does not
   change the ranking. Under this model, where batter and bowler effects add on the
   logit scale, there is little matchup to exploit even if the attack were known.
8. **The ranking is robust.** Across retirement policy, ground size, drift and opposition
   tactics the order of strategies never changed, and bank-and-recall stayed on top
   throughout. Bigger boundaries shrink the penalty for a poor fixed order (averaged
   over the retirement policies and the other dials, leading with the weakest costs about
   15 runs at no shift and about 7 at a -1.4 shift), presumably because boundary hitting is where
   batters differ most and a bigger ground removes many boundaries.
9. **A blanket retirement policy moves totals as much as most order choices.** With the
   same order and the current ground size, retiring *every* batter at 25 rather than 35
   balls costs about 6 runs (less on a bigger ground), because the best batters are taken
   off too. Bank-and-recall is the per-batter alternative to that blanket policy: only the
   top four retire early, and only long enough to see off a cheap wicket.
10. **What a participation guarantee costs, if runs are what matters.** The order that
    guarantees the weakest three at least 14 balls each (strong-weak alternating) costs
    about 8 runs, roughly 5% of a total. That is a coaching choice about fairness, not
    something the model can decide, but the model can now put a price on it, and
    bank-and-recall shows that a *different* kind of fairness lever (giving the coach the
    retirement choice) does not have to cost runs at all, even though this particular
    version of it does not target fairness for the weakest batters specifically.

## 4b. After the first U11 game (Round 1, Sat 10 Oct 2026)

Navy v Orange (Hartley pool) came through PlayHQ complete, with ball-by-ball for both
innings: 223 fair balls, Navy 142/7 in 25 overs against Orange 56/8, all out in 14.1.
White's game against Turquoise (Smith pool, cut to 22 overs a side) has only the team
totals on PlayHQ so far, 131/4 against 91/7, so it says nothing about individual batters.
**This is one game between two sides: a first look to test the model's assumptions against,
not an estimate.**

**1. The engine's strike rule was wrong, and the fix barely matters.** The engine rotated
strike only at the end of the over. Rule 17.4(iv) has batters bat from one end, rotate at the
end of each over *and* "run from popping crease to popping crease", so they also cross on odd
runs. The game agrees: the striker changed after 30 of 33 odd-run balls and after 4 of 138
other balls inside an over, and standard crossing predicts 94% of striker identities against
80% for rotating at the over's end only. After a wicket the incoming batter faced the next
ball in 6 of the 8 clear cases (the engine had the surviving batter face it). Both are now
on the U11 playing conditions (`batters_cross_on_runs`, `incoming_batter_takes_strike`),
tested, and switchable (`--strike-rules earlier` reproduces the first pilot exactly). Main
scenario, runs against strongest to weakest, at the same U10 scoring rates:

| Strike rules | bank top 4 | strong-weak alternating | weakest to strongest |
| --- | --- | --- | --- |
| earlier (end of over only) | +1.4 ± 0.2 | -7.6 ± 0.4 | -20.0 ± 0.4 |
| crossing only | +1.6 ± 0.2 | -7.5 ± 0.4 | -19.4 ± 0.4 |
| crossing and new batter takes strike (now the default) | +1.3 ± 0.2 | -7.9 ± 0.4 | -19.9 ± 0.4 |

Across all 37 scenarios bank-and-recall stays best in every one, and strong-weak alternating
stays worse in every one.

**2. U11 scoring looks quite different from U10.** Navy's first U11 game against Navy's own
2,668 U10 balls, on the model's own logit scale (median and 90% interval, from a Beta
posterior on one game):

| Quantity | U10 (Navy) | U11 Round 1 | Shift |
| --- | --- | --- | --- |
| Dismissal, per fair ball | 6.2% | 6.7% | +0.12 [-0.34, +0.53] |
| Boundary, per ball not out | 12.7% | 9.1% | -0.34 [-0.75, +0.03] |
| Scoring shot, per non-boundary ball not out | 48.2% | 23.8% | **-1.08 [-1.37, -0.80]** |
| Wide or no-ball, per delivery | 6.8% | 18.3% [14.8%, 22.5%] | n/a |

Dismissals are unchanged and boundaries are a bit rarer (inside the grid's 0 to -0.7), but
far fewer balls are worked for runs: dot balls rose from 42% to 65% of balls, and extras were
54 of Navy's 142. One plausible reason, not tested, is that a dismissal now ends a batter's
innings (in U10 the batter carried on at a four-run cost), so batters run less. Neither the
scoring-shot rate nor the extras rate was a dial before; the strategy simulations had no
wides at all.

**3. The conclusions in a "U11-like" world.** `--boundary-shift -0.3 --scoring-shift -1.1
--extras-rate 0.18` (the grid still sweeps its own boundary shift). The squad's mean total
falls from 152 to 129 (the four real Round 1 innings averaged 105, from 56 to 142, with 5.9
simulated wickets against 6.5 real), and:

| Strategy | Mean total | U11-like: runs vs strongest to weakest | U10-like (section 4) |
| --- | --- | --- | --- |
| **bank top 4, recall on a cheap wicket** | **132.2** | **+3.3 ± 0.2** | +1.3 |
| strongest to weakest | 128.9 | 0 | 0 |
| strong-middle alternating | 128.7 | -0.2 ± 0.3 | -0.2 |
| balanced pairs (top six) | 128.1 | -0.8 ± 0.3 | -1.2 |
| strong-weak alternating | 124.0 | -4.9 ± 0.3 | -7.9 |
| weakest to strongest | 116.7 | -12.2 ± 0.4 | -19.9 |
| random orders (12) | | -6.0 on average | -9.6 |

Across the 37 scenarios: bank-and-recall is best in all 37 (+4.5 on average, +2.1 to +7.9);
strong-weak alternating is worse in all 37 (-2.9 on average); no fixed-order alternative
beats the conventional order by more than 0.5 runs. So **the ranking does not change, the
penalties for reordering shrink (singles are rarer, so order decides fewer balls), and
bank-and-recall is worth more.** Two cautions: its thresholds (out within 6 balls or for under
5 runs) were set for U10 scoring, where 5 runs was cheap; at U11 scoring they fire on most weak
dismissals and are worth re-tuning. And the participation cost grows: the best batter gains
4 balls (23.1 to 27.3) and the weakest loses 3 (8.1 to 5.4).

## 5. What would change the answer

| Uncertainty | Why it matters | How we will find out |
| --- | --- | --- |
| Skill transfer from U10 to U11 | Skills are U10 estimates; a year older and a bigger ground | Refit as U11 games arrive; drift is already in the grid |
| Boundary size (45 m) | Boundary hitting is the strongest skill and drives the ranking | First U11 game: shift -0.34 [-0.75, +0.03] (section 4b), inside the grid's 0 to -0.7; more games will narrow it |
| Per-batter retirement | Confirmed allowed by the rules, though not yet used on the day | Done: "bank top 4, recall on a cheap wicket" in the engine and the comparison above |
| Wicketkeepers and fielding | The model was about 10% stingy about our own side | Team fielding effect done (22 Sep 2026): a real but modest edge, 0.17 [-0.15, 0.47] on the dismissal log-odds (81% posterior probability positive), narrowing the gap without closing it (PROJECT_STATUS.md). Individual catches and run outs are recorded per ball too (`player_report.md`'s Fielding table), but only 67 of 172 dismissals name a fielder, too few to fit a per-player effect. A true wicketkeeper flag would still need the coach's own per-game record; PlayHQ does not tag it |
| Strike rotation | The engine changed ends only at the end of the over | Done (11 Oct 2026): rule 17.4(iv) and the first U11 game both have batters crossing on odd runs and a new batter taking strike after a wicket; engine fixed, conclusions unchanged (section 4b) |
| In-innings form | Batters are modelled as constant within an innings | Compare runs per ball by balls faced in U11 data |
| Scoring shots | The first U11 game had 24% of non-boundary balls scoring against 48% in U10; one game | Dial added (`--scoring-shift`); the U11-like world uses -1.1 (section 4b); refit with a U11 offset as games arrive |
| Extras | Wides and no-balls were not modelled (none were recorded in White's U10 feed) | Dial added (`--extras-rate`); the first U11 game had 18% [15%, 22%] of deliveries, and they count toward the 25 and 35-ball limits |
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
| Before Round 1 (Sat 10 Oct 2026) | Team fielding effect added and calibration re-checked (done 22 Sep 2026); wicketkeeper flags if a real per-game signal can be sourced; load the U11 squad (new players get the grade prior); rerun the replay and strategy comparison for the actual squad | A pre-season recommendation with ranges, and a list of the assumptions it rests on |
| Round 1 (done, 11 Oct 2026) | Fetched both U11 games; Navy's came through complete, White's has team totals only; engine strike rules corrected; first-game shifts turned into scenario dials | Section 4b |
| Rounds 2 to 4 | After each game, fetch the ball-by-ball (API key when granted, otherwise slowly); refit with a U11 offset for each outcome; run the replay tool on U11 games; compare predicted with actual balls by position and boundary rate | The real test of the transfer assumption |
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
   be built instead, even if it costs a little more than +1.3 runs? At U11-like scoring the
   weakest batter's balls fall from about 8 to 5 under the current thresholds (section 4b).
7. White's Round 1 record: batting order, runs and balls, who bowled and who kept. PlayHQ has
   only the totals for that game (131/4 against 91/7); if whoever scored it did not use
   ball-by-ball scoring, your own record is the only source.

## 9. How to reproduce

```bash
python scripts/compare_batting_strategies.py --search
# the first pilot's strike rules, for comparison (reproduces its numbers exactly)
python scripts/compare_batting_strategies.py --search --strike-rules earlier --out data/outputs/strategies_earlier_strike
# a U11-like world (section 4b)
python scripts/compare_batting_strategies.py --search --boundary-shift -0.3 --scoring-shift -1.1 --extras-rate 0.18 --out data/outputs/strategies_u11_like
```

Writes `data/outputs/strategies/` (gitignored): `strategy_report.html`, `strategy_results.csv`,
`strategy_balls_by_rank.csv`, `strategy_by_attack_strength.csv`, `worms.png`,
`balls_by_rank.png`, `sensitivity.png`. Read the outputs with [GUIDE.md](GUIDE.md), section 4.8.
