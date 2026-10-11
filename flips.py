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


def show_figure(figure, description):
    try:
        st.pyplot(figure, width="stretch", alt=description)
    finally:
        plt.close(figure)


def show_results(result):
    config = result.config
    once = theory(config, 1)
    multiple = theory(config, config.run_times)
    labels = ["Run once", f"{config.run_times} runs" if config.run_times > 1 else "1 run (same)"]
    samples = [result.once, result.multiple]
    counts = [result.once_outcomes, result.multiple_outcomes]
    st.subheader("Session results")
    st.caption(
        f"Submitted scenario: {config.trials:,} simulated sessions, "
        f"{config.allins:,} all-ins per session, {config.run_times} run(s), "
        f"win probability {config.equity:.0%}, rake {config.rake_pct:g}%, "
        f"gross pot {config.pot:,.2f}, seed {config.seed}. "
        "Submit again to update these results."
    )
    columns = st.columns(3)
    columns[0].metric("EV / session (BI)", number(once.ev_total), border=True)
    columns[1].metric(
        "Variance reduction",
        "N/A" if multiple.variance_ratio is None else f"{1 - multiple.variance_ratio:.1%}",
        border=True,
    )
    columns[2].metric(
        "SD reduction",
        "N/A" if multiple.sd_ratio is None else f"{1 - multiple.sd_ratio:.1%}",
        border=True,
    )
    st.caption("Theoretical values. Both strategies have the same EV. Reductions compare multiple runs with run once.")
    if once.variance_total == 0:
        st.info("Deterministic outcome: both variances are zero, so variance and SD ratios and percentage reductions are undefined (N/A).")
    if config.run_times == 1:
        st.info("One run selected: both series are the same sample and represent the same strategy.")
    st.caption(
        f"1 BI = your contribution per all-in = pot / 2 = {config.stake:,.3f} monetary units. "
        "This is whole-hand net profit, not incremental call EV or necessarily a table buy-in."
    )

    st.subheader("Profit distribution")
    show_figure(
        distribution_figure(*samples, labels, ev_total=once.ev_total),
        f"Overlaid distributions of final session profit for {labels[0]} and {labels[1]}, "
        f"with common bins, break-even at zero and shared theoretical EV of {number(once.ev_total)} BI.",
    )
    st.caption("Shared bins and axes. Blue: run once. Orange: selected runs. The dashed EV line is shared by both strategies.")

    st.subheader("How sessions finish")
    st.caption(
        "Simulated frequencies of final net loss, break-even or profit. "
        "A final loss is not ruin or drawdown during the session."
    )
    for column, label, outcomes in zip(st.columns(2), labels, counts):
        with column:
            st.markdown(f"**{label}**")
            st.table([
                {"Result": "Loss (< 0)", "Probability": f"{outcomes.loss / config.trials:.2%}"},
                {"Result": "Break-even (= 0)", "Probability": f"{outcomes.break_even / config.trials:.2%}"},
                {"Result": "Profit (> 0)", "Probability": f"{outcomes.profit / config.trials:.2%}"},
            ], hide_index=True, alt=f"Simulated final-outcome probabilities for {label}.")
    st.caption("Rounded percentages may not add to exactly 100%. Break-even is classified from integer wins and decimal rake, before rounding displayed profit.")

    with st.expander("Statistical details", expanded=False):
        st.caption(
            f"Theoretical EV per all-in: {number(once.ev_per_hand, 5)} BI. "
            "For a session, EV and variance scale with N; SD scales with sqrt(N). "
            "Theoretical variance ratio: " + number(multiple.variance_ratio)
            + " · SD ratio: " + number(multiple.sd_ratio) + "."
        )
        for column, label, values, exact in zip(st.columns(2), labels, samples, [once, multiple]):
            with column:
                st.markdown(f"**{label}**")
                mean = float(values[0]) if exact.variance_total == 0 else float(np.mean(values))
                st.table([
                    {"Measure": "Theoretical EV", "BI": number(exact.ev_total)},
                    {"Measure": "MC mean", "BI": number(mean)},
                    {"Measure": "MC minus EV", "BI": number(mean - exact.ev_total, 5)},
                    {"Measure": "MC standard error", "BI": number(exact.mc_se, 5)},
                    {"Measure": "Theoretical SD", "BI": number(exact.sd_total)},
                    {"Measure": "Sample SD", "BI": number(np.sqrt(sample_variance(values)))},
                    {"Measure": "5th percentile", "BI": number(float(np.quantile(values, 0.05, method="inverted_cdf")))},
                    {"Measure": "95th percentile", "BI": number(float(np.quantile(values, 0.95, method="inverted_cdf")))},
                ], hide_index=True, alt=f"Session statistics in buy-ins for {label}.")
        st.caption(
            "MC = Monte Carlo. Standard error = theoretical session SD / sqrt(T); "
            "it measures simulation error in the mean, not poker risk. "
            "EV ±1.96 standard errors is an approximate 95% sampling band for the MC mean, "
            "which can be inaccurate for rare outcomes. Sample SD uses denominator T−1."
        )
        variance_once = sample_variance(result.once)
        variance_multiple = sample_variance(result.multiple)
        observed_ratio = variance_multiple / variance_once if variance_once > 0 else None
        st.caption(
            f"Observed variance ratio: {number(observed_ratio)} · "
            f"Observed SD ratio: {number(np.sqrt(observed_ratio) if observed_ratio is not None else None)}. "
            "Observed ratios are N/A if the run-once sample has zero variance."
        )
        st.caption("The 5th and 95th percentiles are observed session outcomes from the inverse empirical CDF, not confidence intervals for EV.")
        st.subheader("Cumulative probability")
        show_figure(
            ecdf_figure(*samples, labels),
            "Empirical cumulative probability of final session profit, including outcomes equal to the threshold.",
        )
        st.caption("P(profit ≤ X) includes break-even sessions at X = 0; it is not the probability of a strict loss.")


st.set_page_config(page_title="Poker Variance Simulator", layout="centered")
st.title("Poker Variance")
st.caption("Run once vs multiple runs")
st.write("Compare the same expected profit with different session risk.")
st.caption("Simplified independent win/loss trials with fixed probability, no ties and proportional rake. Real boards from one deck can be dependent.")

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
    # Preserve a submitted scenario from an already-open session after upgrade.
    if not hasattr(st.session_state["result"], "once_outcomes"):
        previous_config = st.session_state["result"].config
        st.session_state["result"] = simulate(SimulationConfig(**vars(previous_config)))
    show_results(st.session_state["result"])
else:
    st.info("Choose a scenario and select Run Simulation to compare EV and risk.")
