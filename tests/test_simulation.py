"""Analytic enumeration and deterministic Monte Carlo checks for the model."""

from dataclasses import replace
from fractions import Fraction
from itertools import product
from math import sqrt
import unittest
from unittest.mock import patch

import numpy as np

from simulation import OutcomeCounts, SimulationConfig, simulate, theory


class TheoryTests(unittest.TestCase):
    def test_exact_outcome_enumeration_in_currency(self):
        # Build every possible outcome, settle each gross pot after rake,
        # subtract both initial contributions, then convert to units. This
        # reference deliberately does not use the binomial moment formulas.
        pot = Fraction(120)
        contribution = pot / 2
        win_probability = Fraction(1, 4)
        rake = Fraction(1, 8)
        hands = 2
        for runs in range(1, 5):
            with self.subTest(runs=runs):
                outcomes = []
                for wins in product((0, 1), repeat=hands * runs):
                    probability = Fraction(1)
                    gross_receipts = Fraction(0)
                    for win in wins:
                        probability *= win_probability if win else 1 - win_probability
                        if win:
                            gross_receipts += pot * (1 - rake) / runs
                    net = (gross_receipts - hands * contribution) / contribution
                    outcomes.append((probability, net))
                self.assertEqual(sum(probability for probability, _ in outcomes), 1)
                mean = sum(probability * net for probability, net in outcomes)
                variance = sum(probability * (net - mean) ** 2 for probability, net in outcomes)
                config = SimulationConfig(
                    allins=hands, pot=float(pot), equity=float(win_probability),
                    rake_pct=float(rake * 100), run_times=runs,
                )
                result = theory(config, runs)
                self.assertAlmostEqual(result.ev_total, float(mean))
                self.assertAlmostEqual(result.variance_total, float(variance))
                self.assertAlmostEqual(result.ev_per_hand, float(mean / hands))
                self.assertAlmostEqual(result.variance_per_hand, float(variance / hands))
                self.assertAlmostEqual(result.sd_total, sqrt(float(variance)))
                self.assertAlmostEqual(result.mc_se, sqrt(float(variance) / config.trials))

    def test_ratios_distinguish_variance_and_standard_deviation(self):
        config = SimulationConfig(equity=0.63, rake_pct=7.25)
        for runs in range(1, 5):
            with self.subTest(runs=runs):
                result = theory(config, runs)
                self.assertAlmostEqual(result.variance_ratio, 1 / runs)
                self.assertAlmostEqual(result.sd_ratio, 1 / sqrt(runs))

    def test_zero_variance_ratios_are_undefined(self):
        for equity in (0.0, 1.0):
            for runs in range(1, 5):
                with self.subTest(equity=equity, runs=runs):
                    result = theory(SimulationConfig(equity=equity), runs)
                    self.assertIsNone(result.variance_ratio)
                    self.assertIsNone(result.sd_ratio)
                    self.assertEqual(result.variance_total, 0)
                    self.assertEqual(result.sd_total, 0)
                    self.assertEqual(result.mc_se, 0)

    def test_expected_rake_cost_and_horizon_scaling(self):
        config = SimulationConfig(allins=1, equity=0.5, rake_pct=5.0)
        single = theory(config, 1)
        many = theory(replace(config, allins=400), 1)
        self.assertAlmostEqual(single.ev_per_hand, -0.05)
        self.assertAlmostEqual(many.ev_total, -20)
        self.assertAlmostEqual(many.variance_total, 400 * single.variance_total)
        self.assertAlmostEqual(many.sd_total, 20 * single.sd_total)


class SimulationTests(unittest.TestCase):
    def test_seed_reproduces_samples_without_changing_global_rng(self):
        config = SimulationConfig(trials=1000, seed=1234)
        before = np.random.get_state()
        first = simulate(config)
        second = simulate(config)
        after = np.random.get_state()
        np.testing.assert_array_equal(first.once, second.once)
        np.testing.assert_array_equal(first.multiple, second.multiple)
        self.assertEqual(first.once_outcomes, second.once_outcomes)
        self.assertEqual(first.multiple_outcomes, second.multiple_outcomes)
        self.assertEqual(before[0], after[0])
        np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])
        different = simulate(replace(config, seed=1235))
        self.assertFalse(np.array_equal(first.once, different.once))
        self.assertFalse(np.array_equal(first.multiple, different.multiple))

    def test_run_once_comparison_is_identical(self):
        result = simulate(SimulationConfig(trials=1000, run_times=1))
        np.testing.assert_array_equal(result.once, result.multiple)
        self.assertEqual(result.once_outcomes, result.multiple_outcomes)

    def test_pot_scale_cancels_in_contribution_units(self):
        smallest = SimulationConfig(trials=1000, pot=0.01, rake_pct=12.25)
        largest = replace(smallest, pot=1e12)
        result_small = simulate(smallest)
        result_large = simulate(largest)
        np.testing.assert_array_equal(result_small.once, result_large.once)
        np.testing.assert_array_equal(result_small.multiple, result_large.multiple)
        self.assertEqual(result_small.once_outcomes, result_large.once_outcomes)
        self.assertEqual(result_small.multiple_outcomes, result_large.multiple_outcomes)
        self.assertEqual(smallest.stake, 0.005)
        self.assertAlmostEqual(largest.net_pot, 877_500_000_000)

    def test_certain_outcomes_and_boundaries(self):
        for equity, allins, runs, rake in product((0.0, 1.0), (1, 50_000), range(1, 5), (0.0, 20.0)):
            with self.subTest(equity=equity, allins=allins, runs=runs, rake=rake):
                config = SimulationConfig(
                    trials=1000, allins=allins, run_times=runs, equity=equity,
                    rake_pct=rake, seed=2**32 - 1,
                )
                result = simulate(config)
                expected = -allins if equity == 0 else allins * (1 - 2 * rake / 100)
                np.testing.assert_allclose(result.once, expected, rtol=0, atol=1e-10)
                np.testing.assert_array_equal(result.once, result.multiple)
                self.assertTrue(np.isfinite(result.once).all())
                expected_counts = OutcomeCounts(1000, 0, 0) if equity == 0 else OutcomeCounts(0, 0, 1000)
                self.assertEqual(result.once_outcomes, expected_counts)
                self.assertEqual(result.multiple_outcomes, expected_counts)

    def test_small_horizon_support_includes_split_payouts(self):
        result = simulate(SimulationConfig(trials=50_000, allins=1, run_times=4))
        np.testing.assert_array_equal(np.unique(result.once), [-1.0, 1.0])
        np.testing.assert_array_equal(np.unique(result.multiple), [-1.0, -0.5, 0.0, 0.5, 1.0])

    def test_monte_carlo_mean_and_variance_against_analytic_uncertainty(self):
        config = SimulationConfig(trials=100_000, allins=17, equity=0.37, rake_pct=7.25, seed=2026)
        result = simulate(config)
        for runs, sample in ((1, result.once), (config.run_times, result.multiple)):
            with self.subTest(runs=runs):
                reference = theory(config, runs)
                self.assertLess(abs(float(np.mean(sample)) - reference.ev_total), 6 * reference.mc_se)
                # Exact fourth central moment of the scaled binomial supplies
                # an uncertainty-aware tolerance for unbiased sample variance.
                binomial_variance = config.allins * runs * config.equity * (1 - config.equity)
                scale = 2 * (1 - config.rake_fraction) / runs
                fourth_moment = scale**4 * (
                    3 * binomial_variance**2
                    + binomial_variance * (1 - 6 * config.equity * (1 - config.equity))
                )
                variance_se = sqrt((
                    fourth_moment - (config.trials - 3) / (config.trials - 1)
                    * reference.variance_total**2
                ) / config.trials)
                self.assertLess(abs(float(np.var(sample, ddof=1)) - reference.variance_total), 7 * variance_se)

    def test_maximum_workload_is_finite_and_has_one_value_per_trial(self):
        config = SimulationConfig(trials=500_000, allins=50_000, run_times=4, pot=1e12, rake_pct=20, seed=0)
        result = simulate(config)
        for sample in (result.once, result.multiple):
            self.assertEqual(sample.shape, (500_000,))
            self.assertLessEqual(sample.nbytes, 4_000_000)
            self.assertTrue(np.isfinite(sample).all())
            self.assertGreaterEqual(float(sample.min()), -50_000)
            self.assertLessEqual(float(sample.max()), 30_000)
        for counts in (result.once_outcomes, result.multiple_outcomes):
            self.assertEqual(counts.total, config.trials)
            self.assertGreaterEqual(min(counts.loss, counts.break_even, counts.profit), 0)


class OutcomeCountTests(unittest.TestCase):
    def test_two_runs_have_loss_break_even_profit_probabilities(self):
        config = SimulationConfig(trials=100_000, allins=1, run_times=2, equity=0.5, rake_pct=0)
        result = simulate(config)
        np.testing.assert_array_equal(np.unique(result.multiple), [-1.0, 0.0, 1.0])
        self.assertEqual(result.once_outcomes.break_even, 0)
        for counts, probabilities in (
            (result.once_outcomes, (0.5, 0.0, 0.5)),
            (result.multiple_outcomes, (0.25, 0.5, 0.25)),
        ):
            self.assertEqual(counts.total, config.trials)
            for count, probability in zip((counts.loss, counts.break_even, counts.profit), probabilities):
                if probability == 0:
                    self.assertEqual(count, 0)
                else:
                    se = sqrt(probability * (1 - probability) / config.trials)
                    self.assertLess(abs(count / config.trials - probability), 6 * se)
        for sample, counts in ((result.once, result.once_outcomes), (result.multiple, result.multiple_outcomes)):
            self.assertEqual(counts.loss, int(np.count_nonzero(sample < 0)))
            self.assertEqual(counts.break_even, int(np.count_nonzero(sample == 0)))
            self.assertEqual(counts.profit, int(np.count_nonzero(sample > 0)))

    def test_exact_break_even_and_adjacent_rake_values(self):
        # Nine $50 contributions cost $450. Five wins in $100 gross pots
        # return $450 at 10% rake, with opposite signs immediately either side.
        for rake, expected in (
            (10.0, OutcomeCounts(0, 1000, 0)),
            (10.0 + 1e-12, OutcomeCounts(1000, 0, 0)),
            (10.0 - 1e-12, OutcomeCounts(0, 0, 1000)),
        ):
            with self.subTest(rake=rake), patch("simulation.np.random.default_rng") as rng:
                rng.return_value.binomial.return_value = np.full(1000, 5, dtype=np.int64)
                result = simulate(SimulationConfig(trials=1000, allins=9, run_times=1, rake_pct=rake))
                self.assertEqual(result.once_outcomes, expected)
                self.assertEqual(result.multiple_outcomes, expected)
                for sample in (result.once, result.multiple):
                    self.assertEqual(int(np.count_nonzero(sample < 0)), expected.loss)
                    self.assertEqual(int(np.count_nonzero(sample == 0)), expected.break_even)
                    self.assertEqual(int(np.count_nonzero(sample > 0)), expected.profit)

    def test_exact_break_even_roundoff_does_not_shift_cdf_at_zero(self):
        # In both fixtures K=200 is exact break-even; direct floating-point
        # settlement otherwise leaves a positive residual of about 1e-14.
        for hands, runs in ((133, 3), (399, 1)):
            with self.subTest(hands=hands, runs=runs):
                result = simulate(SimulationConfig(
                    trials=50_000, allins=hands, run_times=runs,
                    equity=0.5, rake_pct=0.25, seed=42,
                ))
                self.assertGreater(result.multiple_outcomes.break_even, 0)
                for sample, counts in (
                    (result.once, result.once_outcomes),
                    (result.multiple, result.multiple_outcomes),
                ):
                    self.assertEqual(int(np.count_nonzero(sample == 0)), counts.break_even)
                    self.assertEqual(int(np.count_nonzero(sample <= 0)), counts.loss + counts.break_even)

    def test_tiny_rake_loss_is_preserved_when_float_profit_rounds_to_zero(self):
        for rake in (1e-20, float(np.nextafter(0.0, 1.0))):
            with self.subTest(rake=rake), patch("simulation.np.random.default_rng") as rng:
                rng.return_value.binomial.side_effect = [
                    np.zeros(1000, dtype=np.int64), np.ones(1000, dtype=np.int64)
                ]
                result = simulate(SimulationConfig(trials=1000, allins=1, run_times=2, rake_pct=rake))
                np.testing.assert_array_equal(result.multiple, np.zeros(1000))
                self.assertEqual(result.multiple_outcomes, OutcomeCounts(1000, 0, 0))

    def test_counts_match_exact_currency_settlement_for_every_win_count(self):
        # The oracle settles each distinct possible K using scalar rational
        # dollars. It does not solve a threshold or inspect float profit arrays.
        for hands, runs, rake in product((1, 9), range(1, 5), (0.0, 10.0, 10.0 - 1e-12, 10.0 + 1e-12, 20.0, 1e-20)):
            with self.subTest(hands=hands, runs=runs, rake=rake):
                config = SimulationConfig(trials=1000, allins=hands, run_times=runs, rake_pct=rake)
                once_wins = np.resize(np.arange(hands + 1, dtype=np.int64), config.trials)
                multiple_wins = np.resize(np.arange(hands * runs + 1, dtype=np.int64), config.trials)
                with patch("simulation.np.random.default_rng") as rng:
                    rng.return_value.binomial.side_effect = [once_wins, multiple_wins]
                    result = simulate(config)
                if runs == 1:
                    multiple_wins = once_wins
                for count_runs, wins, counts in (
                    (1, once_wins, result.once_outcomes),
                    (runs, multiple_wins, result.multiple_outcomes),
                ):
                    expected = {-1: 0, 0: 0, 1: 0}
                    payout = Fraction(100) * (100 - Fraction(str(rake))) / (100 * count_runs)
                    values, frequencies = np.unique(wins, return_counts=True)
                    for wins_count, frequency in zip(values, frequencies):
                        settled_profit = payout * int(wins_count) - hands * 50
                        sign = (settled_profit > 0) - (settled_profit < 0)
                        expected[sign] += int(frequency)
                    self.assertEqual(counts, OutcomeCounts(expected[-1], expected[0], expected[1]))


class ValidationTests(unittest.TestCase):
    def test_invalid_inputs_are_rejected(self):
        invalid = {
            "trials": (0, 999, 500_001, 1000.5, True, "1000", None),
            "allins": (0, 50_001, 1.5, False),
            "run_times": (0, 5, 2.5, True),
            "seed": (-1, 2**32, 1.5, False),
            "pot": (0, 0.009, 1e12 + 1, 10**1000, float("inf"), float("nan"), True, "100"),
            "equity": (-0.01, 1.01, float("inf"), float("nan"), False),
            "rake_pct": (-0.25, 20.25, float("inf"), float("nan"), True),
        }
        for field, values in invalid.items():
            for value in values:
                with self.subTest(field=field, value=value), self.assertRaisesRegex(ValueError, field):
                    SimulationConfig(**{field: value})

    def test_theory_validates_run_count(self):
        for runs in (0, 5, 2.5, True):
            with self.subTest(runs=runs), self.assertRaisesRegex(ValueError, "runs"):
                theory(SimulationConfig(), runs)


if __name__ == "__main__":
    unittest.main()
