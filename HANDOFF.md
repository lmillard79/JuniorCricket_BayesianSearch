# Handoff notes: batting order strategy + PlayHQ data check

Session on branch `claude/batting-order-strategy-2341yc`, picking up
from a cloud Claude Code session. Written so a fresh local session
has enough context to continue without re-reading the whole thread.

## 1. Batting order strategy discussion (stack strong batters vs balance)

Worked through whether to stack the two strongest batters at 1-2 or
spread strength through the order, using the existing `model.py` /
`simulator.py` / `optimizer.py`.

Key findings, reasoned from the actual code, not yet simulated:

- No partner-protection mechanic exists. Dismissal probability is
  `sqrt(striker.p_out * bowler.p_wicket)` (`_combined_dismissal`,
  `simulator.py:293`), using only the striker's own hazard. A weak
  batter is exactly as exposed with a strong non-striker as a weak
  one.
- Strike only rotates at over-end (`simulator.py:513`), not by runs
  scored mid-over, so the simulator cannot currently represent
  strike-farming to shield a weak partner even if we wanted to test
  that as a strategy.
- The tiered bowling allocation (`[4,4,3,3,3,3,3,1,1]` for 9
  players) plus round-robin cycling (`generate_over_sequence`,
  `simulator.py:187`) guarantees the two 4-over bowlers bowl the
  last two overs of the innings (24-25), regardless of rotation
  order, because everyone else's allocation runs out first. This
  applies symmetrically to the simulated opposition's bowlers too
  (`optimizer.py` tiers by strength for both sides).
- Net implication: stacking your best batters at the top likely
  wins on mean runs (they get first claim on the 150-ball pool, and
  can resume a second stint via the retirement queue), but risks
  feeding weaker/resumed batters to the opposition's best bowler
  late in the innings. That's a real, quantifiable tradeoff, not
  just an intuition, and simulation should settle it rather than
  argument.

**Possible bug, flagged but not verified**: `simulator.py:472-475`,
in the 10 percent `RUN_OUT_SHARE` branch of the dismissal handler,
only `non_striker` is reassigned to the replacement batter.
`striker` is not reset off the just-dismissed player, which reads
like it could let a given-out batter keep facing balls in that
branch. Worth a proper look before relying on simulated output.

**Proposed next step, not yet built, waiting on user decisions**:
compare named batting-order archetypes head to head (skill-stacked,
alternating strong/weak, strongest-in-middle, hill-climbed optimum,
a "protect the tail" variant using moderate retirement thresholds),
via `LineupOptimizer.evaluate_batting` against repeated
`draw_population_player` draws, reporting full distributions (SD,
10th/90th percentile, win probability) broken down by innings third
(overs 1-8, 9-16, 17-25), not just the mean. Proposed as
`notebooks/01_batting_order_strategy.ipynb`, extending `figures.py`
with a per-strategy distribution plot.

Open questions the user has not yet answered:
- [ ] Notebook, or a script/report instead?
- [ ] What skill spread to assume for the synthetic squad (no real
      match data exists yet; season starts 10 Oct 2026), or should
      Claude propose one from typical U11 variance?

## 2. PlayHQ data availability check

User's core question: can the data the model needs (balls faced,
outs, boundaries, balls bowled, wickets, runs conceded, extras, all
per player per match) actually be pulled from PlayHQ's public API?

The cloud sandbox cannot reach playhq.com or its subdomains at all
(egress policy blocks both direct requests and WebFetch), so this
has been worked via WebSearch plus live calls run by the user on
their own machine.

Confirmed so far:
- The API works with no key, as the README claimed. `discover_game`
  returned real data for game `a481b130` (VDCC U11 Aqua v BULLS U11
  GREEN, 2023/24 R6): team names/ids, round/grade/season chain. No
  score or player data came back, because the query as written
  (`DISCOVER_GAME_QUERY` in `playhq_client.py`) doesn't ask for any.
- The README's club entity id
  (`c8fd4cfc-87d8-eb11-a7ad-2818780da0cc`) does not match PlayHQ's
  own short-form id scheme. The real VDCC org id is `c5c60ce3`.
- PlayHQ cricket has two separate data pathways: manual post-game
  aggregate entry (overs, maidens, wides, no balls, wickets, runs
  conceded for bowling; runs and dismissal for batting, no mention
  of balls faced) versus live ball-by-ball e-scoring (which should
  capture balls faced). BNJCA U10/U11 e-scoring is optional per the
  README, so which pathway applies to any given VDCC game is still
  unknown, and it matters a lot since balls faced is the key input
  the dismissal-hazard model needs.

Just written, **not yet run**: `scripts/probe_playhq_schema.py
--game-id a481b130`. Tests, in order: (0) resolves the real grade id
from the game; (1) whether schema introspection is actually blocked
(the existing client docstring claims this but it was never tested
live); (2) if introspection works, dumps fields on any score,
innings, batting, bowling, performance, result, stat, wicket, over,
delivery or ball -shaped types; (3) tries a short list of plausibly
named extra fields directly on `discoverGame` regardless of (1); (4)
calls `grade_player_statistics` for the resolved grade id and prints
the raw response, to see whether it returns real rows for BNJCA U11
and what the currently-unlabelled `statistics.details.value` shape
actually contains.

**Immediate next step**: run it and report back what came out of
each of the four parts. That answers the user's actual question
(can this data be obtained from the API) with evidence instead of
inference.

## 3. Environment notes

- `pip install -e .` is required after cloning; `pyproject.toml` is
  already configured for the `src` layout
  (`packages.find where = ["src"]`).
- The cloud sandbox this work originated in cannot reach
  playhq.com; the user's local machine can, which is why live
  PlayHQ testing has moved there.
- User preferences carried over from the cloud session: propose an
  approach and wait for confirmation before writing full
  implementations; concise, well-structured responses; Australian
  English; no em dashes; no emoji (ticks/crosses fine).
