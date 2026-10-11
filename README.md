# Poker Variance Simulator

Small Streamlit portfolio project comparing expected net profit and risk when an all-in pot is split across independent runs. It illustrates analytical probability, Monte Carlo simulation, reproducibility and automated testing.

The simulator compares:

- Run it once
- Run it multiple times (2x, 3x, 4x)

Under the assumptions below, running more times preserves **expected profit**, while variance falls by **1/r** and standard deviation by **1/sqrt(r)**. These are different ratios.

## Features

- Exact theoretical EV shared by both strategies, plus variance and SD reduction cards
- Overlaid profit distributions with common bins, stable colors, EV and break-even markers
- Simulated probabilities of finishing with a loss, break-even or profit
- Seeded Monte Carlo means, deviations from EV and standard errors
- Comparable histograms, discrete empirical CDFs and risk percentiles
- Explicit session/hand units and deterministic edge cases
- Submitted scenarios remain visible until another simulation is submitted
- Compact two-column tables; Monte Carlo diagnostics and percentiles in Statistical details

## Demo

[Public app](https://runittwice.streamlit.app/) (may differ from a local branch).

## Model

This is a fixed-probability binomial model, not a card simulator. Each of T simulated sessions contains N identical, independent heads-up all-ins. Both players contribute P/2 to the gross pot P. Every run is won with probability p; there are no ties, side pots, dead money, unequal investments or changing equity. Poker equity that includes ties is not enough to determine this model's variance.

Rake is q times the **gross pot**, charged once before splitting it into r equal payouts. It has no cap, minimum, currency rounding or extra multi-run fee, and does not represent any poker site's rules. A winning run pays P(1−q)/r. Net profit subtracts the player's full N·P/2 contribution: it describes whole-hand profit, not the incremental EV of calling after earlier bets have become sunk costs.

One "buy-in" (BI) here means P/2, the player's contribution per hand; it is not necessarily a table buy-in. All output is in BI, so changing P only changes the equivalent monetary scale.

For W_r ~ Binomial(rN, p), total session net profit is:

```text
X_r = 2(1−q) W_r / r − N               [BI]
E[X_r] = N [2(1−q)p − 1]               [BI]
Var(X_r) = 4N(1−q)²p(1−p) / r          [BI²]
SD(X_r) = sqrt(Var(X_r))                [BI]
SE(Monte Carlo mean) = SD(X_r)/sqrt(T)  [BI]
```

Set N=1 for per-hand values. Increasing N multiplies EV and variance by N and SD by sqrt(N). At p=0 or p=1 both variances are zero: their ratios are undefined, shown as N/A. With r=1 both displayed samples are identical. For r>1 the two Monte Carlo samples are independent, so their sample means can differ even though their theoretical EVs agree.

Real boards dealt without replacement from one deck can be dependent. The independent 1/r variance reduction is therefore an illustration, not a claim about every real poker hand. Adding shared-card dependence, ties or a site's rake schedule would require an explicit model change.

## Reading results

- T means simulated **sessions**, not individual all-ins.
- Monte Carlo standard error measures error in estimating the mean; it is not the spread of poker results. Roughly EV ±1.96 SE is a normal-approximation 95% sampling band for the simulated mean; rare outcomes can make this approximation poor.
- Histograms use common bin edges and display probability mass per bin, not density or exact outcome probabilities.
- The main cards show reductions: `1 − 1/r` for variance and `1 − 1/sqrt(r)` for SD. At zero variance both reductions are undefined (N/A). One run has 0% reduction when variance is positive.
- The histogram overlays both samples in blue/orange. EV is shared by both strategies. If EV is exactly zero, one reference line is labeled "EV = break-even"; otherwise both reference lines are shown.
- Final loss, break-even and profit probabilities are Monte Carlo frequencies and may not sum to exactly 100% after display rounding. They describe final profit, not ruin or drawdown during the session.
- Outcome classification compares the **integer number of wins** to the exact break-even threshold implied by the input's decimal rake percentage. It does not use a tolerance around floating-point profit. Thus a tiny mathematical gain or loss is not classified as break-even, even if the displayed profit rounds to zero. Exactly break-even outcomes are set to zero in the plotted samples too, removing floating-point residuals without changing the model or sampling.
- The step CDF is P(session result ≤ X). At zero it includes break-even sessions as well as losses. Duplicate outcomes are aggregated and the tails run from 0 to 1.
- Percentiles use the inverse empirical CDF, so reported values are observed outcomes. They are not confidence intervals for EV.
- These are final session profits, not within-session drawdowns or risk-of-ruin paths.

Example: N=200, p=0.5, r=2 and no rake gives EV=0 BI, run-once SD≈14.142 BI and run-twice SD=10 BI. Variance ratio=0.5; SD ratio≈0.707. A 5% rake changes EV to −10 BI and the SDs to ≈13.435 and 9.5 BI.

## Installation

Use Python 3.12 (the version tested locally and configured in CI):

```bash
git clone https://github.com/axel2307/Run_it_Twice.git
cd Run_it_Twice
python -m venv .venv
# Activate .venv using the command for your shell, then:
python -m pip install -r requirements.txt
python -m streamlit run flips.py
```

On PowerShell, activate with `.\.venv\Scripts\Activate.ps1`; on macOS/Linux use `source .venv/bin/activate`. To use an assigned local port, append `--server.port <port> --server.address 127.0.0.1`.

## Tests and reproducibility

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

The numerical core is in `simulation.py`, charts in `charts.py`, and Streamlit presentation in `flips.py`. Tests independently enumerate small monetary outcomes, check analytical moments, seeded Monte Carlo tolerances, validation boundaries, ECDF semantics, shared histogram bins, and Streamlit reruns.

The seed is local to each run and does not alter NumPy's global RNG. The same inputs, seed and NumPy environment reproduce samples; identical streams across future NumPy versions are not promised. Direct dependency versions are pinned; this is not a full transitive lockfile.

Inputs are bounded to T=1,000–500,000, N=1–50,000, r=1–4, p=0–1, rake=0–20%, P=0.01–10¹² and seed=0–2³²−1. The pot ceiling is a numerical/UI limit, not a poker rule. Simulation samples aggregate binomial counts in O(T) memory rather than allocating T×N×r paths. CDF plotting aggregates repeated outcomes and figures are closed after rendering.
