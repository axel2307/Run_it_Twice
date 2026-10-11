"""Check discrete probability semantics, including deterministic outcomes."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from charts import distribution_figure, ecdf_figure, empirical_cdf


@pytest.fixture(autouse=True)
def close_test_figures():
    before = set(plt.get_fignums())
    yield
    for number in set(plt.get_fignums()) - before:
        plt.close(number)


def test_ecdf_groups_ties_and_includes_break_even_sessions():
    outcomes, probabilities = empirical_cdf(np.array([-1.0, 0.0, 0.0, 1.0]))
    np.testing.assert_array_equal(outcomes, [-1.0, 0.0, 1.0])
    np.testing.assert_array_equal(probabilities, [0.25, 0.75, 1.0])
    # P(X <= 0) includes the two break-even samples; P(X < 0) does not.
    assert probabilities[np.searchsorted(outcomes, 0)] == 0.75


def test_ecdf_renders_post_steps_with_zero_and_one_tails():
    values = np.array([-1.0, 0.0, 0.0, 1.0])
    figure = ecdf_figure(values, values, ["Once", "Twice"])
    axes = figure.axes[0]
    assert "\u2264" in axes.get_ylabel()
    for line in axes.lines[:2]:
        assert line.get_drawstyle() == "steps-post"
        x, y = line.get_data()
        assert x[0] < -1 and x[-1] > 1
        np.testing.assert_array_equal(x[1:-1], [-1.0, 0.0, 1.0])
        np.testing.assert_array_equal(y, [0.0, 0.25, 0.75, 1.0, 1.0])
        assert np.all(np.diff(x) > 0)
        assert np.all(np.diff(y) >= 0)


def test_histograms_share_bin_edges_and_each_has_unit_probability_mass():
    once = np.array([-3.0, -3.0, 0.0, 1.0])
    multiple = np.array([-0.5, 0.0, 0.0, 0.5, 2.0])
    figure = distribution_figure(once, multiple, ["Once", "Multiple"])
    containers = figure.axes[0].containers
    assert len(containers) == 2
    left_edges = [[bar.get_x() for bar in container] for container in containers]
    widths = [[bar.get_width() for bar in container] for container in containers]
    np.testing.assert_array_equal(left_edges[0], left_edges[1])
    np.testing.assert_array_equal(widths[0], widths[1])
    for container in containers:
        masses = np.array([bar.get_height() for bar in container])
        assert np.isfinite(masses).all()
        assert np.all(masses >= 0)
        assert masses.sum() == pytest.approx(1.0)
    assert left_edges[0][0] < min(once.min(), multiple.min())
    assert left_edges[0][-1] + widths[0][-1] > max(once.max(), multiple.max())


@pytest.mark.parametrize("value", [-1.0, 0.0, 0.6])
def test_constant_samples_have_one_point_mass_and_valid_plot_limits(value):
    samples = np.full(1000, value)
    outcomes, probabilities = empirical_cdf(samples)
    np.testing.assert_array_equal(outcomes, [value])
    np.testing.assert_array_equal(probabilities, [1.0])
    for maker in [distribution_figure, ecdf_figure]:
        figure = maker(samples, samples, ["Once", "Multiple"])
        axes = figure.axes[0]
        left, right = axes.get_xlim()
        assert np.isfinite([left, right]).all()
        assert left < value < right
        if maker is distribution_figure:
            for container in axes.containers:
                masses = np.array([bar.get_height() for bar in container])
                assert np.count_nonzero(masses) == 1
                assert masses.sum() == pytest.approx(1.0)
        else:
            for line in axes.lines[:2]:
                x, y = line.get_data()
                np.testing.assert_array_equal(x[1:-1], [value])
                np.testing.assert_array_equal(y, [0.0, 1.0, 1.0])
