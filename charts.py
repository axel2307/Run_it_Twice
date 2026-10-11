"""Comparable histograms and exact empirical steps, including point masses."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter


STRATEGY_COLORS = ("#0072B2", "#E69F00")


def empirical_cdf(values):
    """Aggregate duplicates; return exact cumulative mass at each outcome."""
    outcomes, counts = np.unique(values, return_counts=True)
    return outcomes, np.cumsum(counts) / values.size


def plot_bounds(once, multiple, ev_total=0.0):
    low = min(float(np.min(once)), float(np.min(multiple)), 0.0, ev_total)
    high = max(float(np.max(once)), float(np.max(multiple)), 0.0, ev_total)
    padding = max((high - low) * 0.05, 0.5)
    return low - padding, high + padding


def prepare_axes():
    # Small source dimensions preserve readable text when scaled to a phone.
    fig, ax = plt.subplots(figsize=(4.8, 3.8), layout="constrained")
    ax.tick_params(labelsize=11)
    ax.xaxis.label.set_size(12)
    ax.yaxis.label.set_size(12)
    ax.spines[["top", "right"]].set_visible(False)
    return fig, ax


def distribution_figure(once, multiple, labels, ev_total=0.0):
    fig, ax = prepare_axes()
    low, high = plot_bounds(once, multiple, ev_total)
    edges = np.linspace(low, high, 81)
    for values, label, color in zip([once, multiple], labels, STRATEGY_COLORS):
        ax.hist(values, bins=edges, weights=np.full(values.size, 1 / values.size),
                alpha=0.5, color=color, label=label)
    if ev_total == 0:
        ax.axvline(0, color="#333333", linestyle="--", linewidth=1.5,
                   label="EV = break-even")
    else:
        ax.axvline(0, color="#333333", linestyle=":", linewidth=1.5,
                   label="Break-even (0)")
        ax.axvline(ev_total, color="#008050", linestyle="--", linewidth=1.5,
                   label="Theoretical EV")
    ax.set(xlabel="Net profit (BI)", ylabel="Sessions per bin", xlim=(low, high))
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.legend(fontsize=10, loc="lower center", bbox_to_anchor=(0.5, 1.01),
              ncol=2, frameon=False, columnspacing=0.8, handlelength=1.8)
    return fig


def ecdf_figure(once, multiple, labels):
    fig, ax = prepare_axes()
    low, high = plot_bounds(once, multiple)
    for values, label, color in zip([once, multiple], labels, STRATEGY_COLORS):
        outcomes, probability = empirical_cdf(values)
        ax.step(np.r_[low, outcomes, high], np.r_[0, probability, 1],
                where="post", color=color, label=label, linewidth=1.7)
    ax.axvline(0, color="gray", linestyle="--", linewidth=1)
    ax.set(xlabel="Net profit X (BI)", ylabel="P(profit ≤ X)",
           xlim=(low, high), ylim=(0, 1.02))
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.legend(fontsize=11)
    return fig
