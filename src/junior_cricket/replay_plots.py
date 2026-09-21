"""Charts for the game replays: histograms, Manhattans, worms, decision scale.

Every chart shows the model's replays beside the recorded game. Only aliases
(P01, O01) and public team names appear.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date as _date
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402

from junior_cricket.replay import (      # noqa: E402
    DecisionStudy,
    RealGame,
    ordinal as _ordinal,
    over_summary,
    percentile_rank,
    totals,
)

SIM = "#6baed6"
SIM_DARK = "#08519c"
ACTUAL = "#d94801"
GOOD = "#238b45"
BAD = "#cb181d"
GREY = "#636363"
PLAN = "#6a3d9a"


@dataclass
class GameReplay:
    """Everything the charts and tables need for one game.

    Attributes:
        game: The recorded game.
        typical: Replays of the game as played, on a typical ground.
        day: Replays using the game's own fitted ground and conditions.
        best: Replays with the best batting order and bowling split found,
            or None.
        batting: Batting-order study.
        bowling_fixed: Bowling-split study with the keepers left as played.
        bowling_free: Bowling-split study with the keepers free to move.
        best_labels: The (batting, bowling) alternatives used for ``best``.
    """

    game: RealGame
    typical: Dict[str, np.ndarray]
    day: Dict[str, np.ndarray]
    best: Optional[Dict[str, np.ndarray]]
    batting: DecisionStudy
    bowling_fixed: DecisionStudy
    bowling_free: DecisionStudy
    best_labels: tuple = ("", "")


def _title(game: RealGame) -> str:
    when = _date.fromisoformat(game.date).strftime("%d %b %Y").lstrip("0")
    won = game.our_total - game.their_total
    result = f"won by {won}" if won > 0 else (f"lost by {-won}" if won < 0 else "tied")
    return f"{when} v {game.opponent}: {result} ({game.our_total} to {game.their_total})"


def _histogram(ax, values: np.ndarray, actual: float, label: str, good_high: bool,
               extra: Optional[np.ndarray] = None, bin_width: int = 4) -> None:
    lo, hi = np.floor(values.min() / bin_width) * bin_width, values.max() + bin_width
    bins = np.arange(lo, hi + bin_width, bin_width)
    ax.hist(values, bins=bins, density=True, color=SIM, alpha=0.85, label="as played")
    if extra is not None:
        ax.hist(extra, bins=bins, density=True, histtype="step", color=PLAN, lw=1.6,
                label="best lineup found")
    q05, q50, q95 = np.percentile(values, [5, 50, 95])
    ax.axvspan(q05, q95, color=SIM, alpha=0.12)
    ax.axvline(q50, color=SIM_DARK, ls="--", lw=1.4)
    higher = actual > q50
    colour = GOOD if higher == good_high else BAD
    ax.axvline(actual, color=colour, lw=2.6)
    ax.set_title(f"{label}: recorded {actual:.0f}, median {q50:.0f}\n"
                 f"{_ordinal(percentile_rank(values, actual))} percentile "
                 f"({'above' if higher else 'below'} median)", fontsize=9)
    ax.set_yticks([])
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(labelsize=8)


def _manhattan(ax, table, our_bowlers: Optional[List[str]], title: str) -> None:
    x = table["over"].to_numpy()
    ax.bar(x, table["sim_median"], width=0.8, color=SIM, alpha=0.8,
           label="replay median (bar) and middle 50%")
    ax.vlines(x, table["sim_q25"], table["sim_q75"], color=SIM_DARK, lw=1.4)
    ax.bar(x, table["actual_runs"], width=0.36, color=ACTUAL, label="recorded")
    for over, wkts, runs in zip(x, table["actual_wickets"], table["actual_runs"]):
        for k in range(int(wkts)):
            ax.plot(over, runs + 0.6 + 0.9 * k, marker="v", color="black", ms=4, ls="")
    ax.set_xlim(0.4, len(x) + 0.6)
    ax.set_xticks(x)
    labels = [f"{o}\n{b}" for o, b in zip(x, our_bowlers)] if our_bowlers else [str(o) for o in x]
    ax.set_xticklabels(labels, fontsize=6.5 if our_bowlers else 8)
    ax.set_ylabel("runs in the over", fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="y", labelsize=8)


def _worm(ax, table, title: str) -> None:
    x = np.concatenate([[0], table["over"].to_numpy()])
    pad = lambda col: np.concatenate([[0.0], table[col].to_numpy()])          # noqa: E731
    ax.fill_between(x, pad("cum_q05"), pad("cum_q95"), color=SIM, alpha=0.25, label="90% of replays")
    ax.fill_between(x, pad("cum_q25"), pad("cum_q75"), color=SIM, alpha=0.5, label="middle 50%")
    ax.plot(x, pad("cum_median"), color=SIM_DARK, lw=2, label="replay median")
    ax.plot(x, pad("actual_cum"), color=ACTUAL, lw=2.6, label="recorded")
    ax.set_xlim(0, x[-1])
    ax.set_xlabel("over", fontsize=8)
    ax.set_ylabel("runs so far", fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)


def plot_game(gr: GameReplay, path: Path) -> None:
    """One game: where the recorded totals sit, and both innings over by over."""
    g, sims = gr.game, totals(gr.typical)
    fig = plt.figure(figsize=(13, 11))
    grid = fig.add_gridspec(3, 6, hspace=0.55, wspace=1.2)
    extra = totals(gr.best) if gr.best is not None else None
    _histogram(fig.add_subplot(grid[0, 0:2]), sims["our_total"], g.our_total,
               "Our total", True, extra["our_total"] if extra else None)
    _histogram(fig.add_subplot(grid[0, 2:4]), sims["their_total"], g.their_total,
               "Their total", False, extra["their_total"] if extra else None)
    ax = fig.add_subplot(grid[0, 4:6])
    _histogram(ax, sims["margin"], g.our_total - g.their_total, "Margin", True,
               extra["margin"] if extra else None, bin_width=6)
    ax.legend(fontsize=7, frameon=False, loc="upper right")
    for row, (key, actual, name) in enumerate((
            ("our", g.our_innings, "Our innings"), ("their", g.their_innings, "Their innings")), 1):
        table = over_summary(gr.typical[f"{key}_runs"], gr.typical[f"{key}_wkts"],
                             actual.runs, actual.wickets)
        bowlers = actual.bowlers if key == "their" else None
        m_ax, w_ax = fig.add_subplot(grid[row, 0:3]), fig.add_subplot(grid[row, 3:6])
        _manhattan(m_ax, table, bowlers,
                   f"{name}: runs per over" + (" (our bowler under each over)" if bowlers else ""))
        _worm(w_ax, table, f"{name}: runs accumulated")
        if row == 1:
            m_ax.legend(fontsize=7, frameon=False, loc="upper right")
            w_ax.legend(fontsize=7, frameon=False, loc="upper left")
    fig.suptitle(_title(g), fontsize=13, y=0.98)
    fig.text(0.5, 0.005, "Black triangles: dismissals in the recorded innings. Replays use the joint "
             "ball model on a typical ground and are in-sample.", ha="center", fontsize=8, color=GREY)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def plot_season(replays: List[GameReplay], path: Path) -> None:
    """Every game against its own replays: was the day above or below the median?"""
    n = len(replays)
    fig, axes = plt.subplots(n, 3, figsize=(13, 0.62 * n + 1.8), sharey=True)
    columns = (("our_total", "Our total", True), ("their_total", "Their total", False),
               ("margin", "Margin (ours minus theirs)", True))
    for i, gr in enumerate(replays):
        sims, g = totals(gr.typical), gr.game
        real = {"our_total": g.our_total, "their_total": g.their_total,
                "margin": g.our_total - g.their_total}
        for j, (key, label, good_high) in enumerate(columns):
            ax, v = axes[i, j], sims[key]
            q05, q25, q50, q75, q95 = np.percentile(v, [5, 25, 50, 75, 95])
            ax.hlines(0, q05, q95, color=SIM, lw=2)
            ax.hlines(0, q25, q75, color=SIM_DARK, lw=7)
            ax.plot([q50], [0], marker="|", color="white", ms=10, mew=2)
            higher = real[key] > q50
            ax.plot([real[key]], [0], marker="o", ms=9, color=GOOD if higher == good_high else BAD,
                    mec="black", mew=0.6)
            span = max(q95, real[key]) - min(q05, real[key])
            ax.set_xlim(min(q05, real[key]) - 0.08 * span, max(q95, real[key]) + 0.08 * span)
            ax.annotate(_ordinal(percentile_rank(v, real[key])), xy=(1.0, 0.5),
                        xycoords="axes fraction", xytext=(5, 0), textcoords="offset points",
                        ha="left", va="center", fontsize=8, color=GREY, annotation_clip=False)
            ax.set_yticks([])
            ax.xaxis.set_major_locator(plt.MaxNLocator(5))
            ax.spines[["top", "right", "left"]].set_visible(False)
            ax.tick_params(labelsize=8)
            if key == "margin":
                ax.axvline(0, color=GREY, lw=0.6)
            if i == 0:
                ax.set_title(label, fontsize=10)
        axes[i, 0].set_ylabel(f"{_date.fromisoformat(g.date).strftime('%d %b')}\n{g.opponent[:16]}",
                              rotation=0, ha="right", va="center", fontsize=8)
    fig.suptitle("Each day against the model's replays of that day", fontsize=13)
    fig.text(0.5, 0.0, "Bar: middle 50% of replays (thick) and 90% (thin), white tick = median. "
             "Dot: recorded result, green if better for us than the median, red if worse. "
             "Right-hand number: percentile of the recorded result.", ha="center", fontsize=8, color=GREY)
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def plot_over_profile(replays: List[GameReplay], path: Path) -> None:
    """Season average runs per over: recorded against replay, for each innings."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), sharey=True)
    for ax, key, name in zip(axes, ("our", "their"), ("Our innings", "Their innings")):
        actual = np.mean([getattr(gr.game, f"{key}_innings").runs for gr in replays], axis=0)
        sim = np.mean([gr.typical[f"{key}_runs"].mean(axis=0) for gr in replays], axis=0)
        x = np.arange(1, len(actual) + 1)
        ax.bar(x, sim, width=0.8, color=SIM, alpha=0.85, label="replay average")
        ax.plot(x, actual, color=ACTUAL, lw=2.4, marker="o", ms=4, label="recorded average")
        ax.set_title(f"{name}: average runs per over across {len(replays)} games", fontsize=10)
        ax.set_xticks(x)
        ax.set_xlabel("over", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=8)
    axes[0].set_ylabel("runs per over", fontsize=9)
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle("Does the model get the shape of an innings right?", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def _best_gain(study: DecisionStudy):
    """(label, gain, se) of the alternative that screened best, judged on fresh runs."""
    best = study.best_label()
    return (best, *study.gain(best))


def plot_decisions(replays: List[GameReplay], path: Path, luck_sd: float) -> None:
    """How much can order and bowling split move the margin, beside luck?"""
    fig, axes = plt.subplots(1, 3, figsize=(14, 0.5 * len(replays) + 2.6),
                             gridspec_kw={"width_ratios": [1, 1, 0.9], "wspace": 0.35})
    rows = [(f"{_date.fromisoformat(gr.game.date).strftime('%d %b')} {gr.game.opponent[:14]}", gr)
            for gr in replays]
    y = np.arange(len(rows))[::-1]
    for ax, attr, title in ((axes[0], "batting", "Batting order"),
                            (axes[1], "bowling_fixed", "Bowling split (keepers as played)")):
        for yi, (name, gr) in zip(y, rows):
            study = getattr(gr, attr)
            randoms = [v for k, v in study.screen.items() if k.startswith("random")]
            base_mean = study.screen[study.baseline][0]
            if randoms:
                ax.plot([m - base_mean for m, _ in randoms], [yi] * len(randoms), "|", color=GREY,
                        alpha=0.5, ms=7)
            for label, marker, colour in (("best first", "o", GOOD), ("worst first", "o", BAD)):
                if label in study.confirm:
                    gain, se = study.gain(label)
                    ax.errorbar([gain], [yi], xerr=[1.96 * se], fmt=marker, color=colour, ms=5, lw=1)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels([n for n, _ in rows] if ax is axes[0] else [], fontsize=8)
        ax.set_xlabel("runs of margin against the order or split as played", fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=8)
    bat = [_best_gain(gr.batting)[1] for gr in replays]
    bowl = [_best_gain(gr.bowling_fixed)[1] for gr in replays]
    free = [_best_gain(gr.bowling_free)[1] for gr in replays]
    values = [luck_sd, np.mean(free), np.mean(bowl), np.mean(bat)]
    names = ["typical spread of a game's margin (1 SD)", "best bowling split, keepers free",
             "best bowling split, keepers fixed", "best batting order"]
    ax = axes[2]
    ax.barh(range(4), values, color=[GREY, GOOD, GOOD, GOOD])
    for i, v in enumerate(values):
        ax.text(v + 0.5, i, f"{v:.1f}", va="center", fontsize=8)
    ax.set_yticks(range(4))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("runs of margin", fontsize=8)
    ax.set_title("For scale (average of the games)", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Grey ticks: random alternatives (each carries about half a run of simulation noise). "
                 "Dots: best-first (green) and worst-first (red) with 95% intervals from fresh replays",
                 fontsize=10)
    fig.subplots_adjust(left=0.13, right=0.97, top=0.9, bottom=0.12)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
