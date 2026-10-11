"""Comparable histograms and exact empirical steps, including point masses."""

import matplotlib.pyplot as plt
import numpy as np


def empirical_cdf(values):
    """Aggregate duplicates; return exact cumulative mass at each outcome."""
    outcomes, counts = np.unique(values, return_counts=True)
    return outcomes, np.cumsum(counts) / values.size


def plot_bounds(once, multiple):
    low = min(float(np.min(once)), float(np.min(multiple)))
    high = max(float(np.max(once)), float(np.max(multiple)))
    padding = max((high - low) * 0.05, 0.5)
    return low - padding, high + padding


def distribution_figure(once, multiple, labels):
    fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
    low, high = plot_bounds(once, multiple)
    edges = np.linspace(low, high, 81)
    for values, label in zip([once, multiple], labels):
        ax.hist(values, bins=edges, weights=np.full(values.size, 1 / values.size),
                alpha=0.5, label=label)
    ax.axvline(0, color="gray", linestyle="--", linewidth=1)
    ax.set(xlabel="Session net profit (BI)", ylabel="Probability per bin",
           title="Distribution of session results", xlim=(low, high))
    ax.legend()
    return fig


def ecdf_figure(once, multiple, labels):
    fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
    low, high = plot_bounds(once, multiple)
    for values, label in zip([once, multiple], labels):
        outcomes, probability = empirical_cdf(values)
        ax.step(np.r_[low, outcomes, high], np.r_[0, probability, 1],
                where="post", label=label)
    ax.axvline(0, color="gray", linestyle="--", linewidth=1)
    ax.set(xlabel="Session net profit X (BI)", ylabel="P(session net profit ≤ X)",
           title="Cumulative probability of session results", xlim=(low, high), ylim=(0, 1.02))
    ax.legend()
    return fig
