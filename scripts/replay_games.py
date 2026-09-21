"""Replay real U10 games under the joint ball model and reshuffle our decisions.

For every completed game with ball-by-ball data this script:

1. replays the game as played (same batting orders, same bowler in each
   over) thousands of times, and reports where the recorded totals sat in
   that spread: above or below the median replay;
2. redoes the game with our batting order shuffled, and with our overs
   shared out differently among the same bowlers, to see how far either
   decision could have moved the margin;
3. draws over-by-over charts (runs per over, and runs accumulated) with the
   recorded innings on top of the replay median and bands.

Batting order can only change our runs and how many dismissals we suffer;
the bowling split only changes theirs (Rule 16.10(vi) turns every dismissal
into 4 runs for the other side), so each decision is judged on its own part
of the margin. Alternatives are screened on one set of replays and judged on
fresh ones, so picking the best of many noisy estimates does not flatter it.

In-sample: the model has seen these games. Read the results as what the model
thinks a typical version of each day looks like.

Outputs (all under ``data/outputs/replay/``, gitignored; aliases and public
team names only):
    replay_report.html      one page with every chart and table
    replay_games.csv        per game: recorded vs replay totals and percentiles
    replay_overs.csv        per game, innings and over: replay spread and record
    replay_decisions.csv    per game and alternative: gain in margin, with error
    season.png, over_profile.png, decisions.png, game_<date>_<id>.png

Example:
    python scripts/replay_games.py --team-id 75cdae66 --team-id fafdb4c4
"""

from __future__ import annotations

import argparse
import base64
import html
import math
from collections import Counter
from datetime import date as _date
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from junior_cricket import replay as R
from junior_cricket.logging_setup import setup_logging

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "raw" / "playhq"
OUTPUT_DIR = REPO_ROOT / "data" / "outputs"
LOGGER_NAME = "replay_games"


def _phi(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def win_chance(mean: float, sd: float) -> float:
    """Chance the margin is above zero, treating it as roughly normal."""
    return _phi(mean / sd)


def bowling_candidates(game: R.RealGame, value: Dict[str, float], rng, n_random: int,
                       keepers_fixed: bool) -> Dict[str, tuple]:
    """Over-by-over bowler sequences to compare, including the one played."""
    allocation = dict(Counter(game.our_overs))
    rotation = R.rotation_of(game.our_overs)
    out = {"as played": game.our_overs,
           "as played (round robin)": R.sequence_for(allocation, rotation)}
    for label, alloc in R.allocation_candidates(
            allocation, value, rng, n_random, keepers_fixed).items():
        out[label] = R.sequence_for(alloc, rotation)
    return out


def analyse(args, logger) -> "tuple[list, list[str]]":
    """Replay every game and study the decisions; returns (GameReplay list, notes)."""
    import arviz as az

    from junior_cricket.ball_model import JointOutcomes, player_profiles
    from junior_cricket.replay_plots import GameReplay

    games, notes = R.load_real_games(RAW_DIR, Path(args.balls), args.team_id)
    for note in notes:
        logger.warning("Left out %s", note)
    logger.info("%d games to replay", len(games))
    idata = az.from_netcdf(args.ball_posterior)
    rng = np.random.default_rng(args.seed)
    pool = JointOutcomes.pool(idata, args.pool, rng)
    ours = sorted({p for g in games for p in g.ours + g.our_overs})
    profiles = player_profiles(idata, ours)
    bat_value = dict(zip(profiles["player"], profiles["net"]))
    bowl_value = dict(zip(profiles["player"], profiles["bowling_edge"]))

    replays = []
    with R.Runner(pool, workers=args.workers) as runner:
        for i, g in enumerate(games):
            seed = (args.seed, i)
            typical, day = runner.run([R.match_task(g, args.sims, (*seed, 0)),
                                       R.match_task(g, args.sims, (*seed, 1), day=True)])
            orders = {"as played": g.ours, **R.ranked_orders(g.ours, bat_value)}
            for j, order in enumerate(R.random_orders(g.ours, args.random, rng), 1):
                orders[f"random {j}"] = order
            batting = R.study_decision(runner, g, "bat", orders, "as played", (*seed, 2),
                                       args.screen, args.confirm)
            fixed_options = bowling_candidates(g, bowl_value, rng, args.random, True)
            fixed = R.study_decision(runner, g, "bowl", fixed_options, "as played",
                                     (*seed, 3), args.screen, args.confirm)
            free = R.study_decision(
                runner, g, "bowl", bowling_candidates(g, bowl_value, rng, args.random, False),
                "as played", (*seed, 4), args.screen, args.confirm)
            best_bat, best_bowl = batting.best_label(), fixed.best_label()
            [best] = runner.run([R.match_task(
                g, args.sims, (*seed, 5), our_bat=orders[best_bat],
                our_overs=fixed_options[best_bowl])])
            replays.append(GameReplay(g, typical, day, best, batting, fixed, free,
                                      (best_bat, best_bowl)))
            t = R.totals(typical)
            logger.info(
                "%s v %s: recorded margin %+d, replay median %+.0f (%s percentile); "
                "best order %+.1f, best bowling split %+.1f runs",
                g.date, g.opponent, g.our_total - g.their_total, np.median(t["margin"]),
                f"{R.percentile_rank(t['margin'], g.our_total - g.their_total) * 100:.0f}th",
                batting.gain(best_bat)[0], fixed.gain(best_bowl)[0])
    return replays, notes


def games_table(replays) -> pd.DataFrame:
    """One row per game: recorded against replay, with percentiles."""
    rows = []
    for gr in replays:
        g, t, d = gr.game, R.totals(gr.typical), R.totals(gr.day)
        margin = g.our_total - g.their_total
        row = {"game_id": g.game_id, "date": g.date, "opponent": g.opponent,
               "our_recorded": g.our_total, "their_recorded": g.their_total,
               "margin_recorded": margin}
        for key, actual in (("our_total", g.our_total), ("their_total", g.their_total),
                            ("margin", margin)):
            row[f"{key}_median"] = float(np.median(t[key]))
            row[f"{key}_q05"], row[f"{key}_q95"] = (float(x) for x in np.percentile(t[key], [5, 95]))
            row[f"{key}_pct"] = R.percentile_rank(t[key], actual)
            row[f"{key}_pct_given_day"] = R.percentile_rank(d[key], actual)
        row["margin_sd"] = float(t["margin"].std())
        row["win_chance_replay"] = float(np.mean(t["margin"] > 0) + 0.5 * np.mean(t["margin"] == 0))
        rows.append(row)
    return pd.DataFrame(rows)


def overs_table(replays) -> pd.DataFrame:
    frames = []
    for gr in replays:
        for key, actual in (("our", gr.game.our_innings), ("their", gr.game.their_innings)):
            frame = R.over_summary(gr.typical[f"{key}_runs"], gr.typical[f"{key}_wkts"],
                                   actual.runs, actual.wickets)
            frame.insert(0, "innings", "ours" if key == "our" else "theirs")
            frame.insert(0, "game_id", gr.game.game_id)
            frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def decisions_table(replays) -> pd.DataFrame:
    rows = []
    for gr in replays:
        for name, study in (("batting order", gr.batting),
                            ("bowling split, keepers as played", gr.bowling_fixed),
                            ("bowling split, keepers free", gr.bowling_free)):
            for label, (mean, se) in study.confirm.items():
                gain, gse = study.gain(label)
                rows.append({"game_id": gr.game.game_id, "decision": name, "alternative": label,
                             "confirm_mean": mean, "confirm_se": se, "gain_vs_played": gain,
                             "gain_se": gse, "screen_mean": study.screen[label][0],
                             "spread_of_random_alternatives": study.true_sd})
    return pd.DataFrame(rows)


def _b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def _table_html(frame: pd.DataFrame) -> str:
    return frame.to_html(index=False, escape=True, border=0, classes="t")


def summarise(replays) -> Dict[str, float]:
    """Season-level numbers quoted in the report."""
    games = games_table(replays)
    out = {"n": len(games)}
    for key in ("our_total", "their_total", "margin"):
        out[f"{key}_above"] = int((games[f"{key}_pct"] > 0.5).sum())
        out[f"{key}_mean_pct"] = float(games[f"{key}_pct"].mean())
    out["luck_sd"] = float(games["margin_sd"].mean())
    bat = [(gr.batting.gain(gr.batting.best_label())) for gr in replays]
    fixed = [gr.bowling_fixed.gain(gr.bowling_fixed.best_label()) for gr in replays]
    free = [gr.bowling_free.gain(gr.bowling_free.best_label()) for gr in replays]
    for name, vals in (("bat", bat), ("fixed", fixed), ("free", free)):
        out[f"{name}_gain"] = float(np.mean([v[0] for v in vals]))
        out[f"{name}_se"] = float(math.sqrt(sum(v[1] ** 2 for v in vals)) / len(vals))
    out["order_sd"] = float(np.nanmean([gr.batting.true_sd for gr in replays]))
    out["bowl_sd"] = float(np.nanmean([gr.bowling_fixed.true_sd for gr in replays]))
    wins = 0.0
    for gr, b, f in zip(replays, bat, fixed):
        t = R.totals(gr.typical)["margin"]
        mu, sd = float(t.mean()), float(t.std())
        wins += win_chance(mu + b[0] + f[0], sd) - win_chance(mu, sd)
    out["extra_wins"] = wins
    return out


def write_report(replays, out_dir: Path, summary: Dict[str, float], args) -> Path:
    """One self-contained HTML page with every chart and table."""
    games = games_table(replays)
    decisions = decisions_table(replays)
    s = summary
    show = games.copy()
    show["date"] = [_date.fromisoformat(d).strftime("%d %b %Y") for d in show["date"]]
    view = pd.DataFrame({
        "Date": show["date"], "Opponent": show["opponent"],
        "Our total": [f"{a} (median {m:.0f}, {p * 100:.0f}th)" for a, m, p in zip(
            show["our_recorded"], show["our_total_median"], show["our_total_pct"])],
        "Their total": [f"{a} (median {m:.0f}, {p * 100:.0f}th)" for a, m, p in zip(
            show["their_recorded"], show["their_total_median"], show["their_total_pct"])],
        "Margin": [f"{a:+d} (median {m:+.0f}, {p * 100:.0f}th)" for a, m, p in zip(
            show["margin_recorded"], show["margin_median"], show["margin_pct"])],
        "Model win chance": [f"{w * 100:.0f}%" for w in show["win_chance_replay"]],
    })
    rows = []
    for gr in replays:
        b_label = gr.batting.best_label()
        b_gain, b_se = gr.batting.gain(b_label)
        f_label = gr.bowling_fixed.best_label()
        f_gain, f_se = gr.bowling_fixed.gain(f_label)
        r_label = gr.bowling_free.best_label()
        r_gain, r_se = gr.bowling_free.gain(r_label)
        rows.append({
            "Date": _date.fromisoformat(gr.game.date).strftime("%d %b"),
            "Opponent": gr.game.opponent,
            "Best batting order found": f"{b_label}: {b_gain:+.1f} ± {1.96 * b_se:.1f}",
            "Best bowling split (keepers as played)": f"{f_label}: {f_gain:+.1f} ± {1.96 * f_se:.1f}",
            "Best bowling split (keepers free)": f"{r_label}: {r_gain:+.1f} ± {1.96 * r_se:.1f}",
        })
    decision_view = pd.DataFrame(rows)
    lean = s["our_total_mean_pct"]
    if lean > 0.56:
        calibration = ("so the model is stingy about our side: it expects us to score less than "
                       "we did. That is the known gap (about 10%) that a team fielding effect is "
                       "meant to close.")
    elif lean < 0.44:
        calibration = "so the model is generous about our side."
    else:
        calibration = "so the model is roughly balanced about our side."
    biggest = max(abs(s["bat_gain"]), abs(s["fixed_gain"]), abs(s["free_gain"]))
    luck_lead = "Luck dwarfs lineup choices. " if s["luck_sd"] > 5 * biggest else ""
    figures = "".join(
        f'<h3>{html.escape(gr.game.opponent)}, '
        f'{_date.fromisoformat(gr.game.date).strftime("%d %b %Y")}</h3>'
        f'<img src="data:image/png;base64,{_b64(out_dir / f"game_{gr.game.date}_{gr.game.game_id}.png")}">'
        for gr in replays)
    body = f"""
<h1>Replaying the U10 season</h1>
<p class="lead">Every completed game with ball-by-ball data ({s['n']} of them) was rebuilt from the
scorecard (who batted in what order, who bowled each over) and played again
{args.sims:,} times with the joint batter-by-bowler model. Each replay draws a fresh
plausible set of player skills from the fitted model, so the spread includes
uncertainty about the players as well as ball-to-ball luck. It is in-sample: the model has
already seen these games.</p>

<h2>The short answer</h2>
<ul>
<li><b>Above or below the median?</b> Our recorded total was above the model's median replay in
{s['our_total_above']} of {s['n']} games (average percentile {s['our_total_mean_pct'] * 100:.0f}th); the
opposition's was above its median in {s['their_total_above']} of {s['n']} (average
{s['their_total_mean_pct'] * 100:.0f}th); the margin was above the median in
{s['margin_above']} of {s['n']} (average {s['margin_mean_pct'] * 100:.0f}th). A fair model averages
about 50th, {calibration}</li>
<li><b>{luck_lead}</b>A single game's margin has a spread of about
{s['luck_sd']:.0f} runs (one standard deviation) even with the lineups fixed.</li>
<li><b>Batting order</b> is worth about {s['bat_gain']:+.1f} runs of margin at best
(±{1.96 * s['bat_se']:.1f}), and random orders differ by only about {s['order_sd']:.1f} runs. The U10 rules give
every batter a fixed 13 or 14 balls, so order barely matters.</li>
<li><b>Bowling split</b> is worth about {s['fixed_gain']:+.1f} runs with the keepers left as
played (±{1.96 * s['fixed_se']:.1f}) and {s['free_gain']:+.1f} if the two keepers could also be chosen
freely (±{1.96 * s['free_se']:.1f}); keeper skill is not modelled, so the second is optimistic.</li>
<li>Together the best batting order and bowling split found would have added about
<b>{s['extra_wins']:.1f} expected wins</b> across the {s['n']} games.</li>
</ul>

<h2>1. Was each day above or below the median?</h2>
<img src="data:image/png;base64,{_b64(out_dir / 'season.png')}">
{_table_html(view)}
<p class="note">Percentile is the share of replays below the recorded result: 50th means dead on the
median, 95th means only 5% of replays did better. Percentiles far from the middle mark days that were
unusually good or bad relative to what the model expected of these players.</p>

<h2>2. Does the model get the shape of an innings right?</h2>
<img src="data:image/png;base64,{_b64(out_dir / 'over_profile.png')}">
<p class="note">Bars are the replay average for each over, the line is the recorded average across the
{s['n']} games. A systematic gap in particular overs would mean the model misses something about how
an innings unfolds (for example batters settling in).</p>

<h2>3. Could a different order or bowling split have helped?</h2>
<img src="data:image/png;base64,{_b64(out_dir / 'decisions.png')}">
{_table_html(decision_view)}
<p class="note">Gains are in runs of margin against what was actually done, from fresh replays of the
alternative that screened best, ± 95% interval. Grey ticks in the chart are random alternatives.</p>

<h2>4. Every game, over by over</h2>
{figures}

<h2>How to read the charts</h2>
<ul>
<li><b>Histograms (top of each game):</b> the spread of totals across replays; dashed line is the median;
shaded band the middle 90%; the thick line is what happened (green if better for us than the median,
red if worse). The purple outline shows the same game with the best batting order and bowling
split found; it overlaps almost completely, which is the point.</li>
<li><b>Manhattan (bars):</b> runs per over. Wide pale bars are the replay median with the middle 50% as a
whisker; narrow dark bars are the recorded overs; black triangles are recorded dismissals.</li>
<li><b>Worm (lines):</b> runs accumulated. Bands hold 50% and 90% of replays; the orange line is the
recorded innings.</li>
</ul>
<h2>Caveats</h2>
<ul>
<li>In-sample: the fitted skills have seen these results, which pulls percentiles towards the middle.</li>
<li>No team-level effect is modelled (fielding, keeping, coaching, team form), so day-to-day swings that
come from the whole team look like luck, and our own side is understated by about 10%.</li>
<li>Wicketkeeper skill is not modelled; who keeps is a fairness and skill decision the model cannot judge.</li>
<li>These are U10 rules. Under U11 (no fixed ball allotments) batting order matters more.</li>
</ul>
"""
    style = """
body{font-family:system-ui,Segoe UI,Arial,sans-serif;max-width:1100px;margin:24px auto;padding:0 16px;color:#222}
img{max-width:100%;height:auto;margin:8px 0}
table.t{border-collapse:collapse;font-size:13px;margin:8px 0}
table.t th,table.t td{border-bottom:1px solid #ddd;padding:4px 10px;text-align:left}
table.t th{background:#f3f6fa}
.lead{font-size:15px}.note{color:#555;font-size:13px}
h1{margin-bottom:4px}h2{margin-top:32px;border-bottom:2px solid #6baed6;padding-bottom:4px}
"""
    page = f"<!doctype html><html><head><meta charset='utf-8'><title>Replaying the U10 season</title><style>{style}</style></head><body>{body}</body></html>"
    path = out_dir / "replay_report.html"
    path.write_text(page, encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--team-id", action="append", required=True)
    parser.add_argument("--ball-posterior", default=str(OUTPUT_DIR / "ball_posterior.nc"))
    parser.add_argument("--balls", default=str(REPO_ROOT / "data" / "processed" / "playhq_balls.csv"))
    parser.add_argument("--sims", type=int, default=4000, help="Replays of each game as played")
    parser.add_argument("--pool", type=int, default=300, help="Posterior draws to sample skills from")
    parser.add_argument("--random", type=int, default=40, help="Random alternatives per decision")
    parser.add_argument("--screen", type=int, default=2000, help="Replays per alternative when screening")
    parser.add_argument("--confirm", type=int, default=10000, help="Fresh replays per finalist")
    parser.add_argument("--workers", type=int, default=R.default_workers())
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", default=str(OUTPUT_DIR / "replay"))
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(LOGGER_NAME, OUTPUT_DIR)
    logger.info("Input arguments: %s", vars(args))

    from junior_cricket import replay_plots as plots

    replays, _ = analyse(args, logger)
    games_table(replays).to_csv(out_dir / "replay_games.csv", index=False)
    overs_table(replays).to_csv(out_dir / "replay_overs.csv", index=False)
    decisions_table(replays).to_csv(out_dir / "replay_decisions.csv", index=False)
    summary = summarise(replays)
    for gr in replays:
        plots.plot_game(gr, out_dir / f"game_{gr.game.date}_{gr.game.game_id}.png")
    plots.plot_season(replays, out_dir / "season.png")
    plots.plot_over_profile(replays, out_dir / "over_profile.png")
    plots.plot_decisions(replays, out_dir / "decisions.png", summary["luck_sd"])
    report = write_report(replays, out_dir, summary, args)
    logger.info("Report: %s", report)
    print(f"Report: {report}")
    print({k: round(v, 3) if isinstance(v, float) else v for k, v in summary.items()})


if __name__ == "__main__":
    main()
