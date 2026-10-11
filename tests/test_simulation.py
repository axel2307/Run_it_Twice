"""Analytic enumeration and deterministic Monte Carlo checks for the model."""

from dataclasses import replace
from fractions import Fraction
from itertools import product
from math import sqrt
import unittest

import numpy as np

from simulation import SimulationConfig, simulate, theory


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
        self.assertEqual(before[0], after[0])
        np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])
        different = simulate(replace(config, seed=1235))
        self.assertFalse(np.array_equal(first.once, different.once))
        self.assertFalse(np.array_equal(first.multiple, different.multiple))

    def test_run_once_comparison_is_identical(self):
        result = simulate(SimulationConfig(trials=1000, run_times=1))
        np.testing.assert_array_equal(result.once, result.multiple)

    def test_pot_scale_cancels_in_contribution_units(self):
        smallest = SimulationConfig(trials=1000, pot=0.01, rake_pct=12.25)
        largest = replace(smallest, pot=1e12)
        result_small = simulate(smallest)
        result_large = simulate(largest)
        np.testing.assert_array_equal(result_small.once, result_large.once)
        np.testing.assert_array_equal(result_small.multiple, result_large.multiple)
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
