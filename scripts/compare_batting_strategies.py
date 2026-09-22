"""Compare U11 batting-order strategies against an unknown opposition.

Does the batting order matter under U11 rules, and does it pay to pair a
strong batter with a weaker one (strong, weak, strong ...) rather than bat
strongest to weakest? Every strategy is played through the same thousands of
random "worlds" (a draw of our players' true skills, nine unknown opposition
players, their bowling tactics and the ground), so differences between
strategies are paired and not muddied by luck of the draw.

Steps:

1. Value each batter from the joint ball model (expected runs in a stint of
   up to 30 balls, a dismissal ending it) and build the named strategies.
2. Play them, twelve random orders, and (with ``--search``) the best order
   found by a swap search, through the same worlds for a main scenario.
3. Judge the searched order on fresh worlds it was not tuned on.
4. Repeat for a grid of assumptions about the U11 world (retirement policy,
   bigger boundaries, skill drift since U10, smarter opposition) to see
   whether the ranking of strategies survives.

Skills come from U10 games, so results are indicative until U11 data arrives.

Outputs (under ``data/outputs/strategies/``, gitignored; aliases only):
    strategy_report.html, strategy_results.csv, strategy_balls_by_rank.csv,
    strategy_by_attack_strength.csv, worms.png, balls_by_rank.png, sensitivity.png

Example:
    python scripts/compare_batting_strategies.py --search
"""

from __future__ import annotations

import argparse
import base64
import dataclasses
import itertools
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from junior_cricket import replay as R
from junior_cricket import strategies as S
from junior_cricket.logging_setup import setup_logging
from junior_cricket.rules_u11 import U11

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "data" / "outputs"
LOGGER_NAME = "compare_batting_strategies"
DEFAULT_SQUAD = "P01,P03,P04,P05,P06,P07,P08,P09,P10"
BASELINE = "strongest to weakest"
MAIN = S.Scenario()
TEST_WORLDS = (100_000, 104_000)                 # never used for tuning

# The bank-and-recall strategy: the conventional order, but the top BANK_SIZE
# batters may retire (not out) at 25 balls instead of batting on, and come
# straight back in, strongest first, ahead of the next fresh batter, if
# whoever comes in next is out within RECALL_WITHIN_BALLS balls or for under
# RECALL_BELOW_RUNS runs. Everyone else bats on as usual.
BANK_NAME = "bank top 4, recall on a cheap wicket"
BANK_SIZE = 4
RECALL_WITHIN_BALLS = 6
RECALL_BELOW_RUNS = 5


def scenario_grid() -> List[S.Scenario]:
    """Assumptions to test: retirement, ground size, drift and opposition tactics."""
    return [S.Scenario(retire_at=r, boundary_shift=b, drift_sd=d, tactics=t)
            for r, b, d, t in itertools.product((25, 30, None), (0.0, -0.7, -1.4),
                                                (0.0, 0.3), ("random", "smart"))]


def summarise(scenario: S.Scenario, results: Dict[str, Dict[str, np.ndarray]]) -> List[dict]:
    """One row per strategy: level, spread and paired difference from the baseline."""
    base = results[BASELINE]
    rows = []
    for name, r in results.items():
        diff, se, better = S.paired(r, base)
        rows.append({
            "scenario": scenario.label(), "retire_at": scenario.retire_at or 35,
            "boundary_shift": scenario.boundary_shift, "drift_sd": scenario.drift_sd,
            "tactics": scenario.tactics, "strategy": name,
            "mean_total": float(S.world_means(r).mean()), "sd_total": float(r["total"].std()),
            "mean_wickets": float(r["wickets"].mean()),
            "diff_vs_baseline": diff, "se": se, "share_of_worlds_better": better,
        })
    return rows


def attack_table(results: Dict[str, Dict[str, np.ndarray]]) -> pd.DataFrame:
    """Each strategy against strongest to weakest, split by how strong the opposition attack was.

    Worlds are grouped into thirds by the attack's rating (runs an average batter scores per
    ball off it), so this shows whether a pairing helps when the bowling is strong, weak or
    in between. Every strategy met the same worlds, so the groups are the same for all.
    """
    base = results[BASELINE]
    groups = S.by_attack(base)
    rows = []
    for name, r in results.items():
        if name == BASELINE:
            continue
        diff = S.world_means(r) - S.world_means(base)
        row = {"Strategy": name}
        for label, idx in zip(("Strongest third of attacks", "Middle third", "Weakest third"), groups):
            d = diff[idx]
            row[label] = f"{d.mean():+.1f} ± {d.std(ddof=1) / np.sqrt(len(d)):.1f}"
        rows.append(row)
    return pd.DataFrame(rows)


def with_bank_recall(scenario: S.Scenario) -> S.Scenario:
    """``scenario``, pinned to the optional-retirement balls bank-and-recall needs.

    Its bankable batters always retire at 25 (the earliest the rules allow),
    which is the strategy's own rule, not a scenario assumption to sweep, so
    this ignores whatever the grid's ``retire_at`` happens to be.
    """
    return dataclasses.replace(scenario, retire_at=U11.retirement_optional_balls)


def bank_recall_result(
    runner: R.Runner, order, ranking, scenario: S.Scenario, worlds, reps: int, seed: int,
):
    """Play the bank-and-recall strategy through the same worlds as everyone else."""
    return S.evaluate_one(runner, order, ranking, with_bank_recall(scenario), worlds, reps, seed,
                          bankable=ranking[:BANK_SIZE], recall_within_balls=RECALL_WITHIN_BALLS,
                          recall_below_runs=RECALL_BELOW_RUNS)


def bank_recall_row(scenario: S.Scenario, baseline, result) -> dict:
    """Summary row for bank-and-recall, grouped under ``scenario`` like its siblings.

    Its own batters retire at 25 regardless of ``scenario.retire_at``, so
    ``retire_at`` here is set to that literal figure rather than the
    scenario's, even though the "scenario" label text matches the outer
    grid point (so the sensitivity chart groups it with the right column).
    """
    diff, se, better = S.paired(result, baseline)
    return {
        "scenario": scenario.label(), "retire_at": U11.retirement_optional_balls,
        "boundary_shift": scenario.boundary_shift, "drift_sd": scenario.drift_sd,
        "tactics": scenario.tactics, "strategy": BANK_NAME,
        "mean_total": float(S.world_means(result).mean()), "sd_total": float(result["total"].std()),
        "mean_wickets": float(result["wickets"].mean()),
        "diff_vs_baseline": diff, "se": se, "share_of_worlds_better": better,
    }


def plot_worms(results, path: Path, names: List[str], manhattan: List[str]) -> None:
    """Median runs accumulated (with the middle 50%) and runs per over for each strategy."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colours = dict(zip(names, ["#08519c", "#cb181d", "#238b45", "#6a3d9a", "#e6550d", "#636363"]))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    x = np.arange(0, S.OVERS + 1)
    width = 0.8 / len(manhattan)
    for name in names:
        runs = results[name]["over_runs"].reshape(-1, S.OVERS).astype(float)
        cum = np.concatenate([np.zeros((len(runs), 1)), np.cumsum(runs, axis=1)], axis=1)
        q25, q50, q75 = np.percentile(cum, [25, 50, 75], axis=0)
        axes[0].fill_between(x, q25, q75, color=colours[name], alpha=0.12)
        axes[0].plot(x, q50, color=colours[name], lw=2.2, label=name)
    for k, name in enumerate(manhattan):
        runs = results[name]["over_runs"].reshape(-1, S.OVERS).astype(float)
        axes[1].bar(np.arange(1, S.OVERS + 1) + (k - (len(manhattan) - 1) / 2) * width,
                    np.median(runs, axis=0), width=width, color=colours[name], label=name)
    axes[1].legend(fontsize=8, frameon=False)
    axes[0].set_title("Runs accumulated: median (line) and middle 50% (band)", fontsize=10)
    axes[0].set_xlabel("over")
    axes[0].set_ylabel("runs so far")
    axes[1].set_title("Runs per over: median across worlds", fontsize=10)
    axes[1].set_xlabel("over")
    axes[1].set_ylabel("runs in the over")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(fontsize=8, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_balls(results, ranking: List[str], path: Path, names: List[str]) -> None:
    """How the balls (top) and runs (bottom) are shared out, by batter rank."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colours = ["#08519c", "#cb181d", "#238b45", "#6a3d9a", "#e6550d"]
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    n = len(ranking)
    width = 0.8 / len(names)
    for ax, key, label in ((axes[0], "balls", "balls faced per innings"),
                           (axes[1], "runs", "runs scored per innings")):
        for k, (name, colour) in enumerate(zip(names, colours)):
            ax.bar(np.arange(n) + (k - (len(names) - 1) / 2) * width,
                   results[name][key].mean(axis=(0, 1)), width=width, color=colour, label=name)
        ax.set_ylabel(label, fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
    axes[1].set_xticks(range(n))
    axes[1].set_xticklabels([f"{r + 1}\n{p}" for r, p in enumerate(ranking)])
    axes[1].set_xlabel("batter, ranked best (1) to worst (%d) by the model" % n, fontsize=9)
    axes[0].legend(fontsize=8, frameon=False)
    axes[0].set_title("Who gets the balls, and what they do with them", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_sensitivity(table: pd.DataFrame, path: Path) -> None:
    """Paired difference from the baseline for every strategy in every scenario."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    strategies = [s for s in table["strategy"].unique() if s != BASELINE and not s.startswith("random")]
    scenarios = list(dict.fromkeys(table["scenario"]))
    grid = np.array([[table[(table["strategy"] == s) & (table["scenario"] == sc)]["diff_vs_baseline"].iloc[0]
                      for sc in scenarios] for s in strategies])
    fig, ax = plt.subplots(figsize=(13, 0.55 * len(strategies) + 3.6))
    limit = max(abs(grid).max(), 1.0)
    image = ax.imshow(grid, aspect="auto", cmap="RdBu", vmin=-limit, vmax=limit)
    ax.set_yticks(range(len(strategies)))
    ax.set_yticklabels(strategies, fontsize=8)
    first = table.drop_duplicates("scenario").set_index("scenario")
    codes = [f"{int(first.loc[s, 'retire_at'])} | {first.loc[s, 'boundary_shift']:+.1f} | "
             f"{first.loc[s, 'drift_sd']:.1f} | {str(first.loc[s, 'tactics'])[0]}" for s in scenarios]
    ax.set_xticks(range(len(scenarios)))
    ax.set_xticklabels(codes, rotation=90, fontsize=7)
    ax.set_xlabel("each column: retire at (balls) | boundary shift | skill drift | opposition tactics "
                  "(r random, s smart)", fontsize=8)
    fig.colorbar(image, ax=ax, label="runs against strongest to weakest (red: worse)")
    ax.set_title("Does the ranking of strategies survive different assumptions about U11?", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--ball-posterior", default=str(OUTPUT_DIR / "ball_posterior.nc"))
    parser.add_argument("--squad", default=DEFAULT_SQUAD, help="Comma-separated batter aliases")
    parser.add_argument("--worlds", type=int, default=6000, help="Worlds for the main scenario")
    parser.add_argument("--grid-worlds", type=int, default=3000, help="Worlds per grid scenario")
    parser.add_argument("--reps", type=int, default=2, help="Innings per world")
    parser.add_argument("--random", type=int, default=12, help="Random orders to include")
    parser.add_argument("--search", action="store_true", help="Also search for a better order")
    parser.add_argument("--pool", type=int, default=300)
    parser.add_argument("--workers", type=int, default=R.default_workers())
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--out", default=str(OUTPUT_DIR / "strategies"))
    args = parser.parse_args()

    import arviz as az

    from junior_cricket.ball_model import JointOutcomes

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(LOGGER_NAME, OUTPUT_DIR)
    logger.info("Input arguments: %s", vars(args))

    squad = [p.strip() for p in args.squad.split(",")]
    idata = az.from_netcdf(args.ball_posterior)
    pool = JointOutcomes.pool(idata, args.pool, np.random.default_rng(args.seed))
    means = JointOutcomes.from_posterior(idata)
    value = {p: S.stint_value(means, p) for p in squad}
    ranking = sorted(squad, key=lambda p: -value[p])
    orders: Dict[str, tuple] = dict(S.named_orders(squad, value))
    orders.update(S.random_orders(squad, args.random, args.seed))
    logger.info("Batters ranked by 30-ball stint value: %s",
                ", ".join(f"{p} {value[p]:.1f}" for p in ranking))

    rows: List[dict] = []
    searched = None
    with R.Runner(pool, workers=args.workers) as runner:
        main_results = S.evaluate(runner, orders, ranking, MAIN, (0, args.worlds), args.reps, args.seed)
        if args.search:
            logger.info("Searching for a better order on worlds 0 to %d ...", args.worlds // 2)
            found, history = S.hill_climb(runner, orders[BASELINE], ranking, MAIN,
                                          (0, args.worlds // 2), args.reps, args.seed)
            searched = found
            logger.info("Search done after %d accepted swaps: %s", len(history) - 1, " ".join(found))
            if found != orders[BASELINE]:
                orders["searched order"] = found
        logger.info("Main scenario: %s", MAIN.label())
        fresh = S.evaluate(runner, {k: v for k, v in orders.items() if not k.startswith("random")},
                           ranking, MAIN, TEST_WORLDS, args.reps, args.seed)
        rows += summarise(MAIN, fresh)
        bank_result = bank_recall_result(runner, orders[BASELINE], ranking, MAIN, TEST_WORLDS,
                                         args.reps, args.seed)
        rows.append(bank_recall_row(MAIN, fresh[BASELINE], bank_result))
        fresh[BANK_NAME] = bank_result
        main_named = fresh
        rand_diffs = [S.paired(main_results[k], main_results[BASELINE])[0]
                      for k in main_results if k.startswith("random")]
        # The grid: every named order (and the searched one) in every scenario.
        grid_orders = {k: v for k, v in orders.items() if not k.startswith("random")}
        grid_orders.update(S.random_orders(squad, 3, args.seed + 1))
        for scenario in scenario_grid():
            res = S.evaluate(runner, grid_orders, ranking, scenario, (0, args.grid_worlds),
                             args.reps, args.seed)
            rows += summarise(scenario, res)
            grid_bank = bank_recall_result(runner, orders[BASELINE], ranking, scenario,
                                           (0, args.grid_worlds), args.reps, args.seed)
            rows.append(bank_recall_row(scenario, res[BASELINE], grid_bank))
            logger.info("Grid: %s", scenario.label())

    table = pd.DataFrame(rows)
    table.to_csv(out / "strategy_results.csv", index=False)
    balls = pd.DataFrame({name: r["balls"].mean(axis=(0, 1)) for name, r in main_named.items()},
                         index=[f"{i + 1} {p}" for i, p in enumerate(ranking)])
    balls.to_csv(out / "strategy_balls_by_rank.csv")

    shown = [BASELINE, "strong-weak alternating", "balanced pairs (top six)", "weakest to strongest",
             BANK_NAME]
    if searched and searched != orders[BASELINE]:
        shown.append("searched order")
    plot_worms(main_named, out / "worms.png", shown,
               [BASELINE, "strong-weak alternating", "weakest to strongest"])
    plot_balls(main_named, ranking, out / "balls_by_rank.png", shown)
    plot_sensitivity(table[table["strategy"].isin(list(grid_orders) + [BASELINE, BANK_NAME])],
                     out / "sensitivity.png")

    main_rows = table[table["scenario"] == MAIN.label()].iloc[: len(main_named)]
    if searched and searched == orders[BASELINE]:
        main_rows = main_rows[main_rows["strategy"] != "searched order"]
    view = main_rows[["strategy", "mean_total", "sd_total", "mean_wickets", "diff_vs_baseline", "se",
                      "share_of_worlds_better"]].copy()
    view.columns = ["Strategy", "Mean total", "SD", "Mean wickets", "Runs vs strongest-to-weakest",
                    "± SE", "Share of worlds better"]
    grid_view = table[~table["strategy"].str.startswith("random")].pivot_table(
        index="strategy", columns=["retire_at", "boundary_shift"], values="diff_vs_baseline",
        aggfunc="mean").round(1)
    order_of = [k for k in main_named]
    grid_view = grid_view.reindex(order_of)
    spread = np.array(rand_diffs)
    attack_view = attack_table({k: v for k, v in main_named.items()})
    attack_view.to_csv(out / "strategy_by_attack_strength.csv", index=False, encoding="utf-8-sig")
    alternatives = table[(table["strategy"] != BASELINE) & ~table["strategy"].str.startswith("random")]
    best_alternative = alternatives.groupby("scenario")["diff_vs_baseline"].max()
    ranges = alternatives.groupby("strategy")["diff_vs_baseline"].agg(["min", "max"])
    answer = "".join(
        f"<li><b>{name}</b>: between {lo:+.1f} and {hi:+.1f} runs across the {len(best_alternative)} scenarios.</li>"
        for name, (lo, hi) in ranges.iterrows())
    if not args.search:
        search_note = "No order search was run (use --search)."
    elif searched == orders[BASELINE]:
        search_note = ("A swap search (every pair of batters, repeated while it helped by more than two "
                       "standard errors) found nothing better than strongest to weakest.")
    else:
        search_note = "The swap search moved away from strongest to weakest: " + " ".join(searched)
    text = f"""
<h1>U11 batting-order strategies against an unknown opposition</h1>
<p>Each strategy was played through {2 * (TEST_WORLDS[1] - TEST_WORLDS[0]):,} innings ({TEST_WORLDS[1] - TEST_WORLDS[0]:,} worlds
x {args.reps}). A world is a draw of our players' true skills, nine unknown opposition players, their tactics
and the ground, and every strategy meets the same worlds. Skills are U10 estimates, so this is indicative.</p>
<h2>The short answer</h2>
<p>Across {len(best_alternative)} sets of assumptions (retire at 25, 30 or 35 balls; boundary shifts for the bigger
U11 ground; skill drift; smarter opposition), no alternative beat strongest to weakest by more than
{max(best_alternative.max(), 0):.1f} runs. Against strongest to weakest, each alternative scored:</p>
<ul>{answer}</ul>
<p>Negative means fewer runs than the conventional order. Comparisons are paired: every strategy met the same
worlds, so a difference is not down to one of them meeting a tougher attack.</p>
<h3>A different kind of alternative: banking and recalling</h3>
<p><b>{BANK_NAME}</b> keeps the conventional order, but changes the retirement tactics instead: the top
{BANK_SIZE} batters may retire (not out) once they reach 25 balls rather than batting on, and if whoever comes
in next is out within {RECALL_WITHIN_BALLS} balls or for under {RECALL_BELOW_RUNS} runs, the strongest retired
batter comes straight back in, ahead of the next fresh batter. Everyone else bats until out or the mandatory
35-ball cap, as usual. Its "retire at" figure in the tables below is always 25, regardless of what the column
or scenario says for the other strategies, since that is the rule being tested, not an assumption to sweep.</p>
<h3>How to read the tables and charts</h3>
<ul>
<li><b>Mean total, SD</b>: the average and spread of our innings total across worlds.</li>
<li><b>Runs vs strongest-to-weakest</b>: the paired difference (negative is worse) and its standard error.
About two standard errors or more is a real difference; smaller is a tie.</li>
<li><b>Share of worlds better</b>: in what share of worlds the strategy out-scored the conventional order;
around 50% means indistinguishable.</li>
<li><b>Worm and Manhattan</b>: median runs accumulated (line, with the middle 50% as a band) and median runs per
over. <b>Balls by rank</b>: who faces the balls and what they do with them, which is why orders differ.
<b>Sensitivity</b>: the same comparison under every assumption tested (columns: retire at | boundary shift |
skill drift | random or smart opposition).</li>
</ul>
<h2>Main scenario: {MAIN.label()}</h2>
{view.round(2).to_html(index=False, border=0, classes='t')}
<p>Twelve random batting orders scored {spread.mean():+.1f} runs on average against strongest to weakest
(from {spread.min():+.1f} to {spread.max():+.1f}).</p>
<p>{search_note}</p>
<h3>Does it depend on how strong the opposition attack is?</h3>
{attack_view.to_html(index=False, border=0, classes='t')}
<p>Runs against strongest to weakest (± standard error), for the worlds with the strongest third, middle third
and weakest third of opposition attacks. If a pairing helped against strong bowling it would show as a
smaller loss (or a gain) in the first column.</p>
<img src="data:image/png;base64,{_b64(out / 'worms.png')}">
<img src="data:image/png;base64,{_b64(out / 'balls_by_rank.png')}">
<h2>Do the conclusions survive other assumptions?</h2>
<p>Runs against strongest to weakest, averaged over drift and opposition tactics, for each retirement policy
and boundary shift. Negative means worse than strongest to weakest.</p>
{grid_view.to_html(border=0, classes='t')}
<img src="data:image/png;base64,{_b64(out / 'sensitivity.png')}">
"""
    style = ("body{font-family:system-ui,Segoe UI,Arial,sans-serif;max-width:1100px;margin:24px auto;"
             "padding:0 16px;color:#222}img{max-width:100%}table.t{border-collapse:collapse;font-size:13px}"
             "table.t th,table.t td{border-bottom:1px solid #ddd;padding:4px 10px;text-align:left}")
    (out / "strategy_report.html").write_text(
        f"<!doctype html><html><head><meta charset='utf-8'><title>U11 batting strategies</title>"
        f"<style>{style}</style></head><body>{text}</body></html>", encoding="utf-8")
    logger.info("Report: %s", out / "strategy_report.html")
    print(view.round(2).to_string(index=False))
    if searched:
        print("searched order:", " ".join(searched), "| ranking:", " ".join(ranking))


if __name__ == "__main__":
    main()
