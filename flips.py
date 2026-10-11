"""Streamlit presentation for the independent all-in model."""

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from charts import distribution_figure, ecdf_figure
from simulation import SimulationConfig, simulate, theory


def number(value, digits=3):
    """Keep undefined ratios readable; never display NaN/Inf."""
    if value is None or not np.isfinite(value):
        return "N/A"
    return f"{value:,.{digits}f}"


def sample_variance(values):
    # Floating-point accumulation can give a constant sample a spurious tiny SD.
    return 0.0 if np.all(values == values[0]) else float(np.var(values, ddof=1))


def show_results(result):
    config = result.config
    once = theory(config, 1)
    multiple = theory(config, config.run_times)
    labels = ["Run once", f"Run {config.run_times} times"]
    st.subheader("Session results")
    st.caption(
        f"Submitted scenario: {config.trials:,} simulated sessions, "
        f"{config.allins:,} all-ins per session, {config.run_times} run(s), "
        f"win probability {config.equity:.0%}, rake {config.rake_pct:g}%, "
        f"gross pot {config.pot:,.2f}, seed {config.seed}. "
        "Results below describe this scenario until you submit another."
    )
    st.caption(
        f"1 buy-in (BI) = own contribution per all-in = pot / 2 = "
        f"{config.stake:,.3f} monetary units; this is not necessarily a table buy-in. "
        "Changing the pot scales monetary results, but leaves BI results unchanged."
    )
    columns = st.columns(3)
    columns[0].metric("Theoretical EV / session (BI)", number(once.ev_total))
    columns[1].metric("Theoretical variance ratio", number(multiple.variance_ratio))
    columns[2].metric("Theoretical SD ratio", number(multiple.sd_ratio))
    st.caption(
        f"Theoretical EV per all-in: {number(once.ev_per_hand, 5)} BI. "
        "Ratios compare multiple runs with run once; variance ratio = 1/r "
        "and SD ratio = 1/sqrt(r) when variance is positive."
    )
    if once.variance_total == 0:
        st.info("This scenario is deterministic. Both variances are zero, so their ratios are undefined (N/A).")
    if config.run_times == 1:
        st.info("One run selected: both series are the same sample and represent the same strategy.")

    rows = []
    samples = [result.once, result.multiple]
    for label, values, exact in zip(labels, samples, [once, multiple]):
        mean = float(values[0]) if exact.variance_total == 0 else float(np.mean(values))
        rows.append({
            "Strategy": label,
            "Theoretical EV (BI)": number(exact.ev_total),
            "MC mean (BI)": number(mean),
            "MC minus EV (BI)": number(mean - exact.ev_total, 5),
            "MC standard error (BI)": number(exact.mc_se, 5),
            "Theoretical SD (BI)": number(exact.sd_total),
            "Sample SD (BI)": number(np.sqrt(sample_variance(values))),
        })
    st.dataframe(rows, hide_index=True, width="stretch")
    st.caption(
        "All values in the table refer to the full session. MC = Monte Carlo. "
        "MC standard error = theoretical session SD / sqrt(simulated sessions): "
        "it measures simulation error in the mean, not poker risk. "
        "About ±1.96 standard errors is a normal-approximation 95% sampling band "
        "around EV for the MC mean; it can be inaccurate for very rare outcomes. "
        "Sample SD uses the sample variance denominator T−1."
    )
    variance_once = sample_variance(result.once)
    variance_multiple = sample_variance(result.multiple)
    observed_ratio = variance_multiple / variance_once if variance_once > 0 else None
    st.caption(
        f"Observed variance ratio: {number(observed_ratio)} · "
        f"Observed SD ratio: {number(np.sqrt(observed_ratio) if observed_ratio is not None else None)}. "
        "Observed ratios are N/A when the run-once sample has zero variance."
    )

    for make_figure in [distribution_figure, ecdf_figure]:
        figure = make_figure(*samples, labels)
        try:
            st.pyplot(figure, width="stretch")
        finally:
            plt.close(figure)
    st.caption(
        "Histogram bars show the fraction of simulated sessions in each shared bin. "
        "The empirical CDF shows P(session result ≤ X), including results equal to X. "
        "At X = 0 it includes break-even sessions, not just losses. "
        "These are final session outcomes, not drawdown paths or ruin probabilities."
    )
    st.subheader("Session risk percentiles")
    st.dataframe([
        {
            "Strategy": label,
            "5th percentile (BI)": number(float(np.quantile(values, 0.05, method="inverted_cdf"))),
            "95th percentile (BI)": number(float(np.quantile(values, 0.95, method="inverted_cdf"))),
        }
        for label, values in zip(labels, samples)
    ], hide_index=True, width="stretch")
    st.caption("Empirical discrete quantiles: the smallest observed result whose cumulative frequency reaches 5% or 95%. These describe session outcomes, not confidence intervals for EV.")


st.set_page_config(page_title="Poker Variance Simulator", layout="wide")
st.title("Poker Variance Simulator")
st.caption("Run it once vs multiple times · an independent-trials model")
st.write("Compare expected net profit and session risk when the same pot is split across multiple independent runs.")

with st.expander("Model and assumptions", expanded=False):
    st.markdown("""
    - Each session contains **N identical, independent all-ins**. T is the number of simulated sessions.
    - Heads-up with equal total contributions: the gross pot is P and your contribution is P/2 per hand.
    - Each run is a win/loss trial with fixed probability p; **no ties**. Poker equity with split pots can have the same mean but a different variance.
    - The model deducts rake qP **once per gross pot**, then splits P(1−q) equally among r runs. No cap, rounding, minimum fee or extra multi-run fee; this is not any site's rake schedule.
    - Net profit = payouts after rake minus your entire contribution. This is whole-hand profit, **not the incremental EV of a call after sunk bets**. No dead money, side pots or unequal investments.
    - Runs and hands are independent. Real boards drawn from one deck can be dependent; the 1/r variance law need not hold there. No cards are dealt here.
    """)
    st.latex(r"W_r\sim\mathrm{Binomial}(rN,p),\quad X_r=\frac{2(1-q)}{r}W_r-N\quad\text{BI}")
    st.latex(r"E[X_r]=N[2(1-q)p-1],\quad\mathrm{Var}(X_r)=\frac{4N(1-q)^2p(1-p)}{r}")
    st.caption("For one all-in set N=1. For a session, EV and variance multiply by N, while SD multiplies by sqrt(N).")

with st.form("scenario"):
    col1, col2 = st.columns(2)
    with col1:
        trials = st.number_input("Simulated sessions (T)", 1000, 500000, 50000, step=1000, key="trials")
        pot = st.number_input("Gross pot per all-in", 0.01, 1e12, 100.0, key="pot")
        equity = st.slider("Win probability (no ties)", 0.0, 1.0, 0.50, step=0.01, key="equity")
    with col2:
        allins = st.number_input("All-ins per session (N)", 1, 50000, 200, key="allins")
        run_times = st.selectbox("Runs per all-in", [1, 2, 3, 4], index=1, key="run_times")
        rake_pct = st.slider("Rake on gross pot (%)", 0.0, 20.0, 0.0, step=0.25, key="rake_pct")
    seed = st.number_input("Random seed", 0, 2**32 - 1, 42, step=1, key="seed")
    st.caption("The same inputs and seed reproduce the same simulation in the same NumPy environment. Change the seed for a new sample. Submit to update the results.")
    submitted = st.form_submit_button("Run Simulation", type="primary")

if submitted:
    try:
        config = SimulationConfig(trials, allins, pot, run_times, equity, rake_pct, seed)
        st.session_state["result"] = simulate(config)
    except ValueError as error:
        st.error(str(error))
        st.stop()

if "result" in st.session_state:
    show_results(st.session_state["result"])
else:
    st.info("Choose a scenario and select Run Simulation to compare EV and risk.")
