"""Removing quantized zero Horner stages preserves all nonzero arithmetic."""
import pytest
import torch

from rep4gpu.arithmetic.numerical import NumericalFixedPoint, EXP_COEFFICIENTS
from rep4gpu.proofs.witness_prover import _PlainFloor


class CountingOracle(_PlainFloor):
    def __init__(self, rounding='floor'):
        self.rounding = rounding
        self.secret_products = 0
        self.public_products = 0
        self.truncations = 0
        self.generator = torch.Generator().manual_seed(197)

    def mul(self, x, y):
        self.secret_products += 1
        return _PlainFloor.mul(self, x, y)

    def mul_public(self, x, value):
        self.public_products += 1
        return _PlainFloor.mul(self, x, self.public(value, like=x))

    def trunc(self, x, bits):
        self.truncations += 1
        floor = x >> bits
        remainder = x & ((1 << bits)-1)
        if self.rounding == 'ceil':
            return floor + (remainder != 0)
        if self.rounding == 'random':
            # Couple only the nontrivial random choices. Truncating zero is
            # deterministic even if a real runtime consumes fresh mask bytes.
            nonzero = remainder != 0
            draw = torch.zeros_like(x)
            draw[nonzero] = torch.randint(1 << bits, (int(nonzero.sum()),),
                                         generator=self.generator)
            return floor + (nonzero & (draw < remainder))
        return floor


class Untrimmed(NumericalFixedPoint):
    def polynomial(self, x, coefficients):
        if len(coefficients) == 1:
            return self.constant(coefficients[0], x)
        result = self.add(self._public_mul(x, coefficients[-1]),
                          self.constant(coefficients[-2], x))
        for coefficient in reversed(coefficients[:-2]):
            result = self.add(self.mul(result, x), self.constant(coefficient, x))
        return result


@pytest.mark.parametrize('fractional_bits', range(20, 25))
@pytest.mark.parametrize('rounding', ['floor', 'ceil', 'random'])
def test_all_numerical_outputs_preserved(fractional_bits, rounding):
    for kind, lower, upper in [('exp', -9, 9), ('reciprocal', 1, 2),
                                ('erf', -3, 3), ('silu', -4, 4),
                                ('gelu', -4, 4), ('swiglu', -4, 4)]:
        old = Untrimmed(CountingOracle(rounding), fractional_bits)
        new = NumericalFixedPoint(CountingOracle(rounding), fractional_bits)
        encoded = old.encode(torch.linspace(lower, upper, 1025, dtype=torch.float64))
        args = (encoded, encoded.flip(0)) if kind == 'swiglu' else (encoded,)
        assert torch.equal(getattr(old, kind)(*args), getattr(new, kind)(*args))


@pytest.mark.parametrize('fractional_bits', range(20, 25))
def test_exponential_removes_zero_stages(fractional_bits):
    old_runtime, new_runtime = CountingOracle(), CountingOracle()
    old = Untrimmed(old_runtime, fractional_bits)
    new = NumericalFixedPoint(new_runtime, fractional_bits)
    x = old.encode(torch.tensor([-8., 0., 8.], dtype=torch.float64))
    assert torch.equal(old.exp(x), new.exp(x))
    encoded = [round(c*old.scale) for c in EXP_COEFFICIENTS]
    degree = max(i for i, coefficient in enumerate(encoded) if coefficient)
    removed = 12-degree
    assert removed in (2, 3)
    assert old_runtime.secret_products-new_runtime.secret_products == removed
    assert old_runtime.truncations-new_runtime.truncations == removed
    assert old_runtime.public_products == new_runtime.public_products


@pytest.mark.parametrize('coefficients', [(0., 0., 0.), (1., 0., 0.),
                                         (1., -1., 1e-12), (0., 1e-12, -1e-12)])
def test_zero_and_constant_polynomials(coefficients):
    for rounding in ('floor', 'ceil', 'random'):
        old = Untrimmed(CountingOracle(rounding))
        new = NumericalFixedPoint(CountingOracle(rounding))
        x = old.encode(torch.tensor([-1., 0., 1.], dtype=torch.float64))
        # Public constants may retain scalar shape; values broadcast identically.
        assert torch.equal(old.polynomial(x, coefficients),
                           new.polynomial(x, coefficients).expand_as(x))
