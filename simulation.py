"""Independent, fixed-probability all-ins with equal heads-up contributions.

The input pot includes both contributions. Rake is a proportional deduction
from that pot, before it is divided among runs. Results are net profits over
the whole horizon, in units of the player's contribution (half the gross pot).
This model has no ties, card removal, rake cap, or currency rounding.
"""

from dataclasses import dataclass
from fractions import Fraction
from math import isfinite, sqrt
from numbers import Integral, Real

import numpy as np


def _integer_in_range(name: str, value: int, minimum: int, maximum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer between {minimum} and {maximum}.")
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}.")


def _finite_in_range(name: str, value: float, minimum: float, maximum: float) -> None:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number between {minimum} and {maximum}.")
    # Check bounds before converting through isfinite, including huge ints.
    if not minimum <= value <= maximum or not isfinite(value):
        raise ValueError(f"{name} must be a finite number between {minimum} and {maximum}.")


@dataclass(frozen=True)
class SimulationConfig:
    trials: int = 50_000
    allins: int = 200
    pot: float = 100.0
    run_times: int = 2
    equity: float = 0.5
    rake_pct: float = 0.0
    seed: int = 42

    def __post_init__(self) -> None:
        _integer_in_range("trials", self.trials, 1_000, 500_000)
        _integer_in_range("allins", self.allins, 1, 50_000)
        _integer_in_range("run_times", self.run_times, 1, 4)
        _integer_in_range("seed", self.seed, 0, 2**32 - 1)
        _finite_in_range("pot", self.pot, 0.01, 1e12)
        _finite_in_range("equity", self.equity, 0.0, 1.0)
        _finite_in_range("rake_pct", self.rake_pct, 0.0, 20.0)

    @property
    def stake(self) -> float:
        return self.pot / 2

    @property
    def rake_fraction(self) -> float:
        return self.rake_pct / 100

    @property
    def net_pot(self) -> float:
        return self.pot * (1 - self.rake_fraction)


@dataclass(frozen=True)
class Theory:
    """Exact moments; MC SE is for a mean across independent horizon samples.

    Variances are in squared contribution units, all other dimensional values
    are in contribution units. Ratios compare with running each all-in once.
    """

    ev_per_hand: float
    ev_total: float
    variance_per_hand: float
    variance_total: float
    sd_total: float
    mc_se: float
    variance_ratio: float | None
    sd_ratio: float | None


def theory(config: SimulationConfig, runs: int) -> Theory:
    """Return analytic moments for N independent hands, each with r runs."""
    _integer_in_range("runs", runs, 1, 4)
    p = config.equity
    payout = 2 * (1 - config.rake_fraction)
    ev_per_hand = payout * p - 1
    variance_once = payout**2 * p * (1 - p)
    variance_per_hand = variance_once / runs
    variance_total = config.allins * variance_per_hand
    return Theory(
        ev_per_hand=ev_per_hand,
        ev_total=config.allins * ev_per_hand,
        variance_per_hand=variance_per_hand,
        variance_total=variance_total,
        sd_total=sqrt(variance_total),
        mc_se=sqrt(variance_total / config.trials),
        variance_ratio=1 / runs if variance_once > 0 else None,
        sd_ratio=1 / sqrt(runs) if variance_once > 0 else None,
    )


@dataclass(frozen=True)
class OutcomeCounts:
    """Sample counts by the exact sign of the model's net horizon profit."""

    loss: int
    break_even: int
    profit: int

    @property
    def total(self) -> int:
        return self.loss + self.break_even + self.profit


def _break_even_threshold(config: SimulationConfig, runs: int) -> Fraction:
    """Exact winning-run count required to recover the initial contribution."""
    rake = Fraction(str(config.rake_pct)) / 100
    return Fraction(int(config.allins) * int(runs), 2) / (1 - rake)


def _outcome_counts(wins: np.ndarray, config: SimulationConfig, runs: int) -> OutcomeCounts:
    """Classify integer wins using the decimal rake input, without float signs.

    Net profit is zero at K = N*r / (2*(1-rake)). Only an integer K can
    break even. Rational arithmetic is confined to this scalar threshold;
    array comparisons use its small integer floor, avoiding object arrays
    and integer cross-products that could overflow.
    """
    threshold = _break_even_threshold(config, runs)
    floor = threshold.numerator // threshold.denominator
    if threshold.denominator == 1:
        loss = int(np.count_nonzero(wins < floor))
        break_even = int(np.count_nonzero(wins == floor))
    else:
        loss = int(np.count_nonzero(wins <= floor))
        break_even = 0
    return OutcomeCounts(loss, break_even, int(wins.size) - loss - break_even)


@dataclass(frozen=True)
class SimulationResult:
    config: SimulationConfig
    once: np.ndarray
    multiple: np.ndarray
    once_outcomes: OutcomeCounts
    multiple_outcomes: OutcomeCounts


def simulate(config: SimulationConfig) -> SimulationResult:
    """Sample net horizon results in O(trials) memory, independently for r > 1.

    A local RNG makes each configuration reproducible without changing NumPy's
    global random state. Normalizing before sampling keeps the pot scale out
    of intermediate arithmetic. Identical r=1 scenarios share sample values.
    Exact mathematical break-even outcomes are set to zero so roundoff cannot
    move their probability mass to either side of zero in the charts.
    """
    rng = np.random.default_rng(config.seed)
    payout = 2 * (1 - config.rake_fraction)
    wins_once = rng.binomial(config.allins, config.equity, config.trials)
    once = payout * wins_once - config.allins
    once_threshold = _break_even_threshold(config, 1)
    if once_threshold.denominator == 1:
        once[wins_once == once_threshold.numerator] = 0.0
    once_outcomes = _outcome_counts(wins_once, config, 1)
    if config.run_times == 1:
        multiple = once.copy()
        multiple_outcomes = once_outcomes
    else:
        wins_multiple = rng.binomial(
            config.run_times * config.allins, config.equity, config.trials
        )
        multiple = payout * (wins_multiple / config.run_times) - config.allins
        multiple_threshold = _break_even_threshold(config, config.run_times)
        if multiple_threshold.denominator == 1:
            multiple[wins_multiple == multiple_threshold.numerator] = 0.0
        multiple_outcomes = _outcome_counts(wins_multiple, config, config.run_times)
    return SimulationResult(
        config=config, once=once, multiple=multiple,
        once_outcomes=once_outcomes, multiple_outcomes=multiple_outcomes,
    )
