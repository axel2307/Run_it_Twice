"""Exercise the actual Streamlit script through its public widget interface."""

from pathlib import Path
import re
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from simulation import SimulationConfig, simulate


APP = Path(__file__).resolve().parents[1] / "flips.py"
NONFINITE = re.compile(r"(?<![A-Za-z])(?:nan|[+-]?inf(?:inity)?)(?![A-Za-z])", re.I)


def assert_clean_ui(app):
    assert not list(app.exception), [item.message for item in app.exception]
    assert not list(app.error), [item.value for item in app.error]
    text = []
    for kind in ["caption", "markdown", "info", "warning", "text"]:
        text.extend(str(item.value) for item in getattr(app, kind))
    text.extend(str(item.value) for item in app.metric)
    for table in [*app.table, *app.dataframe]:
        text.extend(str(value) for value in table.value.to_numpy().ravel())
    assert not NONFINITE.search("\n".join(text)), "Nonfinite value rendered in the UI"


def submit(app):
    app.button[0].click().run(timeout=60)
    assert_clean_ui(app)
    return app.session_state["result"]


def submitted_caption(app):
    return next(item.value for item in app.caption if item.value.startswith("Submitted scenario:"))


def statistics(app):
    return [
        table.value.set_index("Measure")["BI"].to_dict()
        for table in app.table if "Measure" in table.value.columns
    ]


def probabilities(app):
    return [
        table.value["Probability"].str.rstrip("%").astype(float).to_numpy() / 100
        for table in app.table if "Probability" in table.value.columns
    ]


def snapshot_tables(app):
    return [table.value.copy(deep=True) for table in app.table]


def assert_tables_unchanged(app, expected):
    assert len(app.table) == len(expected)
    for table, original in zip(app.table, expected):
        pd.testing.assert_frame_equal(table.value, original)


@pytest.fixture
def app():
    instance = AppTest.from_file(str(APP), default_timeout=60).run()
    assert_clean_ui(instance)
    return instance


def test_initial_form_and_default_submission_have_analytic_reference_values(app):
    assert len(app.metric) == 0
    assert len(app.table) == 0
    assert len(app.dataframe) == 0
    assert app.number_input(key="trials").value == 50_000
    assert app.number_input(key="allins").value == 200
    result = submit(app)
    assert result.once.shape == result.multiple.shape == (50_000,)
    metrics = {item.label: item.value for item in app.metric}
    assert metrics["EV / session (BI)"] == "0.000"
    assert metrics["Variance reduction"] == "50.0%"
    assert metrics["SD reduction"] == "29.3%"
    tables = statistics(app)
    assert [table["Theoretical EV"] for table in tables] == ["0.000", "0.000"]
    assert [table["Theoretical SD"] for table in tables] == ["14.142", "10.000"]
    assert [table["MC standard error"] for table in tables] == ["0.06325", "0.04472"]
    assert any("\u2264" in caption.value for caption in app.caption)
    assert len(app.dataframe) == 0
    assert len(app.table) == 4
    # Main results stay visible; the two narrow statistics tables and CDF are
    # the only results hidden in the initially collapsed details expander.
    details = next(item for item in app.expander if item.label == "Statistical details")
    assert details.proto.expanded is False
    assert len(details.table) == 2
    assert all(list(table.value.columns) == ["Measure", "BI"] for table in details.table)
    assert all(table.value.shape == (8, 2) for table in details.table)
    assert sum(len(item.metric) for item in app.expander) == 0
    assert all("Probability" not in table.value.columns for item in app.expander for table in item.table)
    assert len(app.get("image")) == 2
    assert len(details.get("image")) == 1
    for table in app.table[:2]:
        assert list(table.value.columns) == ["Result", "Probability"]
        assert table.value["Result"].tolist() == ["Loss (< 0)", "Break-even (= 0)", "Profit (> 0)"]


def test_same_seed_reproduces_samples_after_a_different_seed(app):
    app.number_input(key="trials").set_value(1000)
    original = submit(app)
    original_tables = snapshot_tables(app)
    app.number_input(key="seed").set_value(43)
    changed = submit(app)
    assert not np.array_equal(original.once, changed.once)
    assert not np.array_equal(original.multiple, changed.multiple)
    app.number_input(key="seed").set_value(42)
    repeated = submit(app)
    np.testing.assert_array_equal(repeated.once, original.once)
    np.testing.assert_array_equal(repeated.multiple, original.multiple)
    assert repeated.once_outcomes == original.once_outcomes
    assert repeated.multiple_outcomes == original.multiple_outcomes
    assert_tables_unchanged(app, original_tables)


def test_single_run_uses_identical_samples_and_reports_zero_reduction(app):
    app.number_input(key="trials").set_value(1000)
    app.selectbox(key="run_times").select(1)
    result = submit(app)
    np.testing.assert_array_equal(result.once, result.multiple)
    assert app.metric[1].value == app.metric[2].value == "0.0%"
    assert statistics(app)[0] == statistics(app)[1]
    np.testing.assert_array_equal(probabilities(app)[0], probabilities(app)[1])


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
    for table in statistics(app):
        assert table["Sample SD"] == "0.000"
        assert table["Theoretical SD"] == "0.000"
        assert table["MC standard error"] == "0.00000"
        assert table["MC minus EV"] == "0.00000"
    expected_probabilities = [1, 0, 0] if equity == 0 else [0, 0, 1]
    for probability in probabilities(app):
        np.testing.assert_array_equal(probability, expected_probabilities)
    observed = next(item.value for item in app.caption if item.value.startswith("Observed variance ratio:"))
    assert "Observed variance ratio: N/A" in observed
    assert "Observed SD ratio: N/A" in observed


def test_one_hand_risk_probabilities_distinguish_strict_loss_from_break_even(app):
    app.number_input(key="allins").set_value(1)
    result = submit(app)
    # For one fair all-in, once has outcomes -1/+1 and twice -1/0/+1.
    # Fixed seed plus a >6-SE tolerance avoids checking random coincidences.
    expected = ([0.5, 0.0, 0.5], [0.25, 0.5, 0.25])
    for displayed, exact, samples in zip(probabilities(app), expected, [result.once, result.multiple]):
        np.testing.assert_allclose(displayed, exact, rtol=0, atol=0.015)
        independent_counts = np.array([
            np.count_nonzero(samples < 0),
            np.count_nonzero(samples == 0),
            np.count_nonzero(samples > 0),
        ])
        np.testing.assert_allclose(displayed, independent_counts / samples.size, rtol=0, atol=0.00005)
        assert displayed.sum() == pytest.approx(1.0, abs=0.00015)
    assert probabilities(app)[1][0] < 0.3
    assert sum(probabilities(app)[1][:2]) > 0.7


def test_decimal_rake_break_even_uses_exact_classification_in_probability_table(app):
    app.number_input(key="trials").set_value(1000)
    app.number_input(key="allins").set_value(31)
    app.selectbox(key="run_times").select(3)
    app.slider(key="equity").set_value(0.54)
    app.slider(key="rake_pct").set_value(7.0)
    result = submit(app)
    # 50 wins across 93 runs returns 2*.93*50/3 == 31 BI exactly in
    # decimal arithmetic. Floating-point profit may have a tiny residual.
    break_even = np.count_nonzero(np.isclose(result.multiple, 0, rtol=0, atol=1e-10))
    assert break_even > 0
    assert probabilities(app)[1][1] == pytest.approx(break_even / 1000, abs=0.00005)
    assert probabilities(app)[0][1] == 0
    # Both strategies have EV 31 * (1.86 * .54 - 1) = .1364 BI.
    assert app.metric[0].value == "0.136"
    assert [table["Theoretical EV"] for table in statistics(app)] == ["0.136", "0.136"]


def test_unsubmitted_widget_changes_preserve_result_and_its_original_labels(app):
    app.number_input(key="trials").set_value(1000)
    original = submit(app)
    caption = submitted_caption(app)
    tables = snapshot_tables(app)
    app.run()
    assert_clean_ui(app)
    assert submitted_caption(app) == caption
    assert_tables_unchanged(app, tables)
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
    assert_tables_unchanged(app, tables)
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
    tables = snapshot_tables(app)
    app.number_input(key="pot").set_value(1e12)
    maximum = submit(app)
    assert maximum.config.pot == 1e12
    assert "gross pot 1,000,000,000,000.00" in submitted_caption(app)
    np.testing.assert_array_equal(maximum.once, minimum.once)
    np.testing.assert_array_equal(maximum.multiple, minimum.multiple)
    assert_tables_unchanged(app, tables)


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
    assert len(app.table) == 4
    for displayed in probabilities(app):
        assert np.all((0 <= displayed) & (displayed <= 1))
        assert displayed.sum() == pytest.approx(1.0, abs=0.00015)


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


def test_legacy_session_snapshot_migrates_without_submitting_default_form_inputs():
    config = SimulationConfig(
        trials=1000, allins=17, pot=125.0, run_times=3,
        equity=0.61, rake_pct=4.5, seed=123,
    )
    expected = simulate(config)
    legacy = SimpleNamespace(config=config, once=expected.once, multiple=expected.multiple)
    app = AppTest.from_file(str(APP), default_timeout=60)
    app.session_state["result"] = legacy
    app.run()
    assert_clean_ui(app)

    migrated = app.session_state["result"]
    assert migrated is not legacy
    assert migrated.config == config
    assert migrated.once_outcomes == expected.once_outcomes
    assert migrated.multiple_outcomes == expected.multiple_outcomes
    np.testing.assert_array_equal(migrated.once, expected.once)
    np.testing.assert_array_equal(migrated.multiple, expected.multiple)
    assert "17 all-ins per session, 3 run(s)" in submitted_caption(app)
    assert "win probability 61%, rake 4.5%" in submitted_caption(app)
    assert "seed 123" in submitted_caption(app)
    assert len(app.table) == 4
    # The fresh form has different defaults; migration must use the submitted
    # snapshot, not silently replace it with those unsubmitted widget values.
    assert app.number_input(key="allins").value == 200
    assert app.slider(key="equity").value == 0.5
    assert app.number_input(key="seed").value == 42
