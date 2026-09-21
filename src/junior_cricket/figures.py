"""WRM-branded publication figures for the junior cricket model.

Sets the WRM corporate palette as the matplotlib default at
import time and provides the three core figures:

* shrinkage demonstration (raw rates vs posterior),
* player posterior forest plot,
* optimizer run-differential and win-probability summary.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

WRM_COLOUR_NAMES = [
    "WRM Green",
    "WRM Teal",
    "WRM Blue",
    "Bright Blue",
    "Bright Teal",
    "Charcoal",
    "Citrus",
    "Light orange",
    "Sky",
]
WRM = [
    "#8dc63f",
    "#00928f",
    "#1e4164",
    "#00539b",
    "#00b49d",
    "#485253",
    "#d7df23",
    "#d3bd2a",
    "#6dcff6",
]
WRM_COLOUR = dict(zip(WRM_COLOUR_NAMES, WRM))

plt.rcParams["axes.prop_cycle"] = plt.cycler(color=WRM)


def set_wrm_style() -> None:
    """Apply WRM corporate styling to the current matplotlib state."""
    plt.rcParams["axes.prop_cycle"] = plt.cycler(color=WRM)
    plt.rcParams["font.family"] = "sans-serif"


def plot_shrinkage(
    raw_rates: np.ndarray,
    posterior_means: np.ndarray,
    population_mean: float,
    sample_sizes: np.ndarray,
    player_names: list,
    metric_label: str,
    output_path,
) -> None:
    """Plot raw vs posterior rates showing partial-pooling shrinkage.

    Args:
        raw_rates: Observed rates per player.
        posterior_means: Posterior mean rates per player.
        population_mean: Estimated population mean rate.
        sample_sizes: Balls faced (or bowled) per player; drives
            the marker size.
        player_names: Player labels for the x axis.
        metric_label: Axis label for the plotted rate.
        output_path: Destination file (saved at 300 dpi).
    """
    set_wrm_style()
    fig, ax = plt.subplots(figsize=(10, 5))
    positions = np.arange(len(player_names))
    sizes = 20 + 80 * (
        sample_sizes / max(sample_sizes.max(), 1)
    )
    ax.scatter(
        positions,
        raw_rates,
        s=sizes,
        color=WRM_COLOUR["Bright Blue"],
        label="Raw rate",
        zorder=3,
    )
    ax.scatter(
        positions,
        posterior_means,
        s=sizes,
        color=WRM_COLOUR["WRM Teal"],
        label="Posterior (shrunk)",
        zorder=3,
    )
    for i in range(len(player_names)):
        ax.annotate(
            "",
            xy=(positions[i], posterior_means[i]),
            xytext=(positions[i], raw_rates[i]),
            arrowprops={"arrowstyle": "->", "color": "#9aa0a6",
                        "lw": 1},
        )
    ax.axhline(
        population_mean,
        color=WRM_COLOUR["Charcoal"],
        linestyle="--",
        label="Population mean",
    )
    ax.set_xticks(positions)
    ax.set_xticklabels(player_names, rotation=45, ha="right")
    ax.set_ylabel(metric_label)
    ax.set_title("Partial pooling: small samples shrink to the mean")
    ax.legend(frameon=False)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_posterior_forest(
    posterior_draws: np.ndarray,
    player_names: list,
    metric_label: str,
    output_path,
) -> None:
    """Forest plot of per-player posterior distributions.

    Args:
        posterior_draws: Array (draws, players) of projected rates.
        player_names: Player labels in draw-column order.
        metric_label: Metric being plotted.
        output_path: Destination file (saved at 300 dpi).
    """
    set_wrm_style()
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(player_names) + 1.5))
    means = posterior_draws.mean(axis=0)
    order = np.argsort(means)
    for rank, idx in enumerate(order):
        draws = posterior_draws[:, idx]
        lo, hi = np.percentile(draws, [5, 95])
        ax.plot(
            [lo, hi],
            [rank, rank],
            color=WRM_COLOUR["Bright Blue"],
            lw=2,
        )
        ax.scatter(
            means[idx],
            rank,
            color=WRM_COLOUR["WRM Green"],
            zorder=3,
        )
    ax.set_yticks(range(len(player_names)))
    ax.set_yticklabels([player_names[i] for i in order])
    ax.invert_yaxis()
    ax.set_xlabel(metric_label)
    ax.set_title("Posterior projected rates (90% intervals)")
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_run_differential(
    our_totals: np.ndarray,
    their_totals: np.ndarray,
    output_path,
) -> None:
    """Histogram of simulated match run differentials.

    Args:
        our_totals: Our simulated innings totals.
        their_totals: Opposition simulated innings totals.
        output_path: Destination file (saved at 300 dpi).
    """
    set_wrm_style()
    diff = our_totals[: len(their_totals)] - their_totals[
        : len(our_totals)
    ]
    win = float(np.mean(diff > 0))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(
        diff,
        bins=30,
        color=WRM_COLOUR["Bright Teal"],
        edgecolor="white",
    )
    ax.axvline(0, color=WRM_COLOUR["Charcoal"], linestyle="--")
    ax.set_xlabel("Run differential (us minus them)")
    ax.set_ylabel("Simulated matches")
    ax.set_title(f"Match simulations: win probability {win:.0%}")
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
