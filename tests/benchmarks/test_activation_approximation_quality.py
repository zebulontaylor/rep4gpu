"""Check CPU activation evaluation schedules and their declared range contract."""
import unittest

import numpy as np
import torch

from experiments.benchmarks.activations.activation_approximation_quality import (
    SCALE, evaluate, evaluate_quadratic, fit,
)
from rep4gpu.arithmetic.activations import quadratic_plan


class ActivationApproximationQualityTests(unittest.TestCase):
    def test_fixed_horner_matches_scalar_integer_oracle(self):
        x = np.array([-4., -1.2345, -.001, 0., .001, 1.2345, 4.])
        for kind in ('gelu', 'silu'):
            for profile in ('uniform', 'mixed', 'gaussian_std_0_5'):
                for degree in (4, 6, 8, 10):
                    coefficients = fit(kind, degree, profile)
                    actual, bounds = evaluate(x, coefficients, True)
                    f = bounds['internal_fractional_bits']
                    scale = 1 << f
                    c = [round(float(v) * scale) for v in coefficients]
                    expected = []
                    for v in x:
                        encoded = round(float(v) * SCALE)
                        t = encoded // (4 * (SCALE // scale))
                        z = t * t // scale
                        terms = c[2::2]
                        acc = terms[-1]
                        for term in reversed(terms[:-1]):
                            acc = acc * z // scale + term
                        expected.append((encoded // 2 + (acc * z // scale) * (SCALE // scale)) / SCALE)
                    np.testing.assert_array_equal(actual, expected)
                    self.assertLess(bounds['max_abs_raw_product'], bounds['conservative_product_bound'])
                    self.assertLess(bounds['conservative_product_bound'], 2**30)
                    direct = x / 2 + sum(coefficients[p] * (x / 4)**p for p in range(2, degree + 1, 2))
                    np.testing.assert_allclose(evaluate(x, coefficients)[0], direct, atol=1e-13)

    def test_quadratic_uses_repository_plan(self):
        x = np.linspace(-4, 4, 65537)
        for kind in ('gelu', 'silu'):
            for profile in ('local', 'uniform'):
                plan = quadratic_plan(kind, 4, fractional_bits=14, bits=32, profile=profile)
                actual, bounds = evaluate_quadratic(x, kind, profile)
                normalized = np.rint(x * SCALE).astype(np.int64) >> plan.prescale
                expected = plan.encoded_numerator(torch.from_numpy(normalized)).numpy() >> plan.shift
                np.testing.assert_array_equal(actual * SCALE, expected)
                self.assertLess(bounds['max_abs_numerator'], 2**30)

    def test_range_contract_rejects_outliers(self):
        for invalid in (np.array([4.00001]), np.array([-4.001]), np.array([np.nan])):
            with self.assertRaises(ValueError):
                evaluate(invalid, fit('gelu', 4, 'mixed'), True)
            with self.assertRaises(ValueError):
                evaluate_quadratic(invalid, 'silu', 'local')


if __name__ == '__main__':
    unittest.main()
