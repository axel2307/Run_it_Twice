import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Poker Variance Simulator", layout="wide")

# ---------------------------------------------------
# TITLE
# ---------------------------------------------------

st.title("Poker Variance Simulator")
st.caption("Run it Once vs Run it Multiple Times")

# ---------------------------------------------------
# INPUTS
# ---------------------------------------------------

col1, col2, col3, col4 = st.columns(4)

with col1:
    trials = st.number_input("Simulations", 1000, 500000, 50000, step=1000)

with col2:
    allins = st.number_input("Number of All-ins", 1, 50000, 200)

with col3:
    pot = st.number_input("Total Pot Size", 0.01, value=100.0)

with col4:
    run_times = st.selectbox("Run it Times", [1,2,3,4], index=1)

equity = st.slider("Equity", 0.0, 1.0, 0.50, step=0.01)
rake_pct = st.slider("Rake (%)", 0.0, 20.0, 0.0, step=0.25)

simulate = st.button("Run Simulation")

# ---------------------------------------------------
# SIMULATION
# ---------------------------------------------------

if simulate:

    T = int(trials)
    N = int(allins)
    P = float(pot)
    r = int(run_times)
    p = float(equity)

    rake = rake_pct / 100
    P_net = P * (1 - rake)

    buyin = P / 2

    # run once
    wins_once = np.random.binomial(N, p, T)
    profit_once = wins_once * P_net - N * buyin

    # run r times
    wins_r = np.random.binomial(r * N, p, T)
    profit_r = wins_r * (P_net / r) - N * buyin

    # convert to buy-ins
    profit_once_bi = profit_once / buyin
    profit_r_bi = profit_r / buyin

    # ---------------------------------------------------
    # METRICS
    # ---------------------------------------------------

    ev_once = np.mean(profit_once_bi)
    ev_r = np.mean(profit_r_bi)

    sd_once = np.std(profit_once_bi)
    sd_r = np.std(profit_r_bi)

    var_reduction = sd_r / sd_once

    m1, m2, m3, m4 = st.columns(4)

    m1.metric("EV (Buy-ins)", round(ev_once,2))
    m2.metric("Std Dev Run Once", round(sd_once,2))
    m3.metric("Std Dev Run Multiple", round(sd_r,2))
    m4.metric("Variance Ratio", round(var_reduction,3))

    # ---------------------------------------------------
    # DISTRIBUTION PLOT
    # ---------------------------------------------------

    fig, ax = plt.subplots(figsize=(10,6))

    bins = 80

    ax.hist(
        profit_once_bi,
        bins=bins,
        density=True,
        alpha=0.5,
        label="Run Once"
    )

    ax.hist(
        profit_r_bi,
        bins=bins,
        density=True,
        alpha=0.5,
        label=f"Run {r} Times"
    )

    ax.axvline(0, linestyle="--")

    ax.set_xlabel("Result (Buy-ins)")
    ax.set_ylabel("Density")
    ax.set_title("Distribution of Results")

    ax.legend()

    st.pyplot(fig)

    # ---------------------------------------------------
    # CDF PLOT
    # ---------------------------------------------------

    fig2, ax2 = plt.subplots(figsize=(10,6))

    x1 = np.sort(profit_once_bi)
    y1 = np.arange(1, T+1) / T

    x2 = np.sort(profit_r_bi)
    y2 = np.arange(1, T+1) / T

    ax2.plot(x1, y1, label="Run Once")
    ax2.plot(x2, y2, label=f"Run {r} Times")

    ax2.set_xlabel("Result (Buy-ins)")
    ax2.set_ylabel("Probability Result ≤ X")

    ax2.set_title("Downside Risk Curve")

    ax2.legend()

    st.pyplot(fig2)

    # ---------------------------------------------------
    # PERCENTILES
    # ---------------------------------------------------

    p5_once = np.percentile(profit_once_bi,5)
    p5_r = np.percentile(profit_r_bi,5)

    p95_once = np.percentile(profit_once_bi,95)
    p95_r = np.percentile(profit_r_bi,95)

    st.subheader("Risk Percentiles")

    c1, c2 = st.columns(2)

    with c1:
        st.write("Run Once")
        st.write("5%:", round(p5_once,2), "buy-ins")
        st.write("95%:", round(p95_once,2), "buy-ins")

    with c2:
        st.write(f"Run {r} Times")
        st.write("5%:", round(p5_r,2), "buy-ins")
        st.write("95%:", round(p95_r,2), "buy-ins")