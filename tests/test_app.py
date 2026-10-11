"""Exercise the actual Streamlit script through its public widget interface."""

from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest


APP = Path(__file__).resolve().parents[1] / "flips.py"
NONFINITE = re.compile(r"(?<![A-Za-z])(?:nan|[+-]?inf(?:inity)?)(?![A-Za-z])", re.I)


def assert_clean_ui(app):
    assert not list(app.exception), [item.message for item in app.exception]
    assert not list(app.error), [item.value for item in app.error]
    text = []
    for kind in ["caption", "markdown", "info", "warning", "text"]:
        text.extend(str(item.value) for item in getattr(app, kind))
    text.extend(str(item.value) for item in app.metric)
    for dataframe in app.dataframe:
        text.extend(str(value) for value in dataframe.value.to_numpy().ravel())
    assert not NONFINITE.search("\n".join(text)), "Nonfinite value rendered in the UI"


def submit(app):
    app.button[0].click().run(timeout=60)
    assert_clean_ui(app)
    return app.session_state["result"]


def submitted_caption(app):
    return next(item.value for item in app.caption if item.value.startswith("Submitted scenario:"))


@pytest.fixture
def app():
    instance = AppTest.from_file(str(APP), default_timeout=60).run()
    assert_clean_ui(instance)
    return instance


def test_initial_form_and_default_submission_have_analytic_reference_values(app):
    assert len(app.metric) == 0
    assert len(app.dataframe) == 0
    assert app.number_input(key="trials").value == 50_000
    assert app.number_input(key="allins").value == 200
    result = submit(app)
    assert result.once.shape == result.multiple.shape == (50_000,)
    metrics = {item.label: item.value for item in app.metric}
    assert metrics["Theoretical EV / session (BI)"] == "0.000"
    assert metrics["Theoretical variance ratio"] == "0.500"
    assert metrics["Theoretical SD ratio"] == "0.707"
    table = app.dataframe[0].value
    assert table["Theoretical SD (BI)"].tolist() == ["14.142", "10.000"]
    assert table["MC standard error (BI)"].tolist() == ["0.06325", "0.04472"]
    assert any("\u2264" in caption.value for caption in app.caption)


def test_same_seed_reproduces_samples_after_a_different_seed(app):
    app.number_input(key="trials").set_value(1000)
    original = submit(app)
    original_table = app.dataframe[0].value.copy(deep=True)
    app.number_input(key="seed").set_value(43)
    changed = submit(app)
    assert not np.array_equal(original.once, changed.once)
    assert not np.array_equal(original.multiple, changed.multiple)
    app.number_input(key="seed").set_value(42)
    repeated = submit(app)
    np.testing.assert_array_equal(repeated.once, original.once)
    np.testing.assert_array_equal(repeated.multiple, original.multiple)
    pd.testing.assert_frame_equal(app.dataframe[0].value, original_table)


def test_single_run_uses_identical_samples_and_reports_ratio_one(app):
    app.number_input(key="trials").set_value(1000)
    app.selectbox(key="run_times").select(1)
    result = submit(app)
    np.testing.assert_array_equal(result.once, result.multiple)
    assert app.metric[1].value == app.metric[2].value == "1.000"
    table = app.dataframe[0].value.drop(columns="Strategy")
    assert table.iloc[0].equals(table.iloc[1])


@pytest.mark.parametrize("equity, expected_profit", [(0.0, -1.0), (1.0, 0.6)])
def test_deterministic_extremes_show_zero_sd_and_undefined_ratios(app, equity, expected_profit):
    app.number_input(key="trials").set_value(1000)
    app.number_input(key="allins").set_value(1)
    app.slider(key="equity").set_value(equity)
    app.slider(key="rake_pct").set_value(20.0)
    result = submit(app)
    for samples in [result.once, result.multiple]:
        np.testing.assert_allclose(samples, expected_profit, rtol=0, atol=1e-15)
        assert np.all(samples == samples[0])
    assert app.metric[1].value == app.metric[2].value == "N/A"
    table = app.dataframe[0].value
    assert table["Sample SD (BI)"].tolist() == ["0.000", "0.000"]
    assert table["Theoretical SD (BI)"].tolist() == ["0.000", "0.000"]
    assert table["MC standard error (BI)"].tolist() == ["0.00000", "0.00000"]
    assert table["MC minus EV (BI)"].tolist() == ["0.00000", "0.00000"]
    observed = next(item.value for item in app.caption if item.value.startswith("Observed variance ratio:"))
    assert "Observed variance ratio: N/A" in observed
    assert "Observed SD ratio: N/A" in observed


def test_unsubmitted_widget_changes_preserve_result_and_its_original_labels(app):
    app.number_input(key="trials").set_value(1000)
    original = submit(app)
    caption = submitted_caption(app)
    table = app.dataframe[0].value.copy(deep=True)
    app.run()
    assert_clean_ui(app)
    assert submitted_caption(app) == caption
    pd.testing.assert_frame_equal(app.dataframe[0].value, table)
    # Form edits stay on the client until submit. AppTest discards pending form
    # edits on an explicit non-submit run, so stage the browser interaction
    # after the independent rerun/persistence check above.
    app.number_input(key="allins").set_value(17)
    app.number_input(key="pot").set_value(0.01)
    app.slider(key="equity").set_value(0.75)
    app.selectbox(key="run_times").select(4)
    app.number_input(key="seed").set_value(123)
    assert_clean_ui(app)
    assert submitted_caption(app) == caption
    assert app.session_state["result"].config == original.config
    pd.testing.assert_frame_equal(app.dataframe[0].value, table)
    updated = submit(app)
    assert updated.config.allins == 17
    assert updated.config.run_times == 4
    assert updated.config.pot == 0.01
    assert updated.config.equity == 0.75
    assert updated.config.seed == 123
    assert submitted_caption(app) != caption
    assert "17 all-ins per session, 4 run(s)" in submitted_caption(app)


def test_minimum_and_maximum_pot_leave_buy_in_results_identical(app):
    app.number_input(key="trials").set_value(1000)
    app.number_input(key="pot").set_value(0.01)
    minimum = submit(app)
    assert "gross pot 0.01" in submitted_caption(app)
    table = app.dataframe[0].value.copy(deep=True)
    app.number_input(key="pot").set_value(1e12)
    maximum = submit(app)
    assert maximum.config.pot == 1e12
    assert "gross pot 1,000,000,000,000.00" in submitted_caption(app)
    np.testing.assert_array_equal(maximum.once, minimum.once)
    np.testing.assert_array_equal(maximum.multiple, minimum.multiple)
    pd.testing.assert_frame_equal(app.dataframe[0].value, table)


def test_maximum_supported_session_horizon_and_run_count_remain_finite(app):
    app.number_input(key="trials").set_value(500_000)
    app.number_input(key="allins").set_value(50_000)
    app.number_input(key="pot").set_value(1e12)
    app.number_input(key="seed").set_value(2**32 - 1)
    app.selectbox(key="run_times").select(4)
    app.slider(key="rake_pct").set_value(20.0)
    app.slider(key="equity").set_value(0.99)
    result = submit(app)
    assert result.config.trials == 500_000
    assert result.config.allins == 50_000
    assert result.config.run_times == 4
    for samples in [result.once, result.multiple]:
        assert samples.shape == (500_000,)
        assert np.isfinite(samples).all()
        assert samples.min() >= -50_000
        assert samples.max() <= 30_000
    assert len(app.dataframe) == 2


def test_repeated_submissions_and_reruns_do_not_leak_matplotlib_figures(app):
    app.number_input(key="trials").set_value(1000)
    initial_figures = set(plt.get_fignums())
    for seed in [0, 1, 2]:
        app.number_input(key="seed").set_value(seed)
        submit(app)
        assert set(plt.get_fignums()) == initial_figures
        app.run()
        assert_clean_ui(app)
        assert set(plt.get_fignums()) == initial_figures
