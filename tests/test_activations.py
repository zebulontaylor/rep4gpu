"""Fixed-point composition and approximation tests using an integer oracle."""

from dataclasses import dataclass

import pytest
import torch

from rep4gpu.activations import FixedPoint


@dataclass
class Value:
    value: torch.Tensor

    @property
    def device(self):
        return self.value.device


class IntegerOracle:
    """Plain integer oracle for checking the activation layer's composition."""

    def public(self, value, like):
        return Value(torch.as_tensor(value, dtype=torch.int64).expand_as(like.value))

    def add(self, x, y):
        return Value(x.value + y.value)

    def sub(self, x, y):
        return Value(x.value - y.value)

    def mul(self, x, y):
        return Value(x.value * y.value)

    def matmul(self, x, y):
        return Value(x.value @ y.value)

    def mul_public(self, x, value):
        return Value(x.value * value)

    def trunc_relu(self, x, bits):
        return Value(torch.clamp(x.value >> bits, min=0))

    def trunc(self, x, bits):
        return Value(x.value >> bits)

    def sign_bit(self, x):
        return Value((x.value < 0).to(torch.int64))

    def lt_public(self, x, y):
        return Value((x.value < y).to(torch.int64))


@pytest.mark.parametrize('fractional_bits', [12, 16, 20])
def test_polynomial_integer_rounding(fractional_bits):
    fp = FixedPoint(IntegerOracle(), fractional_bits)
    x = Value(fp.encode([-2.123, -0.123, 0., 1.999]))
    coefficients = [-0.123, 0.456, -0.789, 0.234]
    expected = torch.full_like(x.value, int(round(coefficients[-1] * fp.scale)))
    for coefficient in reversed(coefficients[:-1]):
        expected = ((expected * x.value) >> fractional_bits) + int(round(coefficient * fp.scale))
    assert torch.equal(fp.polynomial(x, coefficients).value, expected)


def test_relu_and_clamp_full_signed_range():
    fp = FixedPoint(IntegerOracle())
    x = Value(torch.tensor([-(2**63), -1, 0, 1, 2**63-1]))
    assert torch.equal(fp.relu(x).value, torch.maximum(x.value, torch.tensor(0)))
    assert torch.equal(fp.clamp(x, -1., 1.).value, torch.clamp(x.value, -fp.scale, fp.scale))


@pytest.mark.parametrize('name,tolerance', [('gelu', .007), ('silu', .074)])
@pytest.mark.parametrize('fractional_bits', [12, 16, 20])
def test_activation_float_error(name, tolerance, fractional_bits):
    fp = FixedPoint(IntegerOracle(), fractional_bits)
    inputs = torch.linspace(-8., 8., 16001, dtype=torch.float64)
    x = Value(fp.encode(inputs))
    actual = fp.decode(getattr(fp, name)(x).value)
    reference = torch.nn.functional.gelu(inputs) if name == 'gelu' else torch.nn.functional.silu(inputs)
    # Quantized Horner evaluation and normalization add rounding error at f=12.
    assert torch.max(torch.abs(actual-reference)) < tolerance + 32 / fp.scale


def test_encoding_and_validation():
    fp = FixedPoint(IntegerOracle(), 2)
    assert torch.equal(fp.encode([.125, .375, -.125, -.375]), torch.tensor([0, 2, 0, -2]))
    for value in [float('inf'), float('nan'), 2.**63]:
        with pytest.raises(ValueError):
            fp.encode(value)
    with pytest.raises(TypeError):
        fp.decode(Value(torch.tensor(1)))
    with pytest.raises(ValueError):
        fp.polynomial(Value(torch.tensor(1)), [])
    with pytest.raises(ValueError):
        fp.clamp(Value(torch.tensor(1)), 2., 1.)


def test_matmul_truncates_after_accumulation():
    fp = FixedPoint(IntegerOracle(), 2)
    x = Value(torch.tensor([[1, 1]], dtype=torch.int64))
    y = Value(torch.tensor([[3], [3]], dtype=torch.int64))
    assert fp.matmul(x, y).value.item() == 1
    assert ((x.value * y.value.T) >> 2).sum().item() == 0


@pytest.mark.parametrize('bias_values', [None, [1, -2]])
def test_matmul_relu_fuses_bias_before_floor(bias_values):
    fp = FixedPoint(IntegerOracle(), 2)
    x = Value(torch.tensor([[1, -2], [2, 1]], dtype=torch.int64))
    y = Value(torch.tensor([[3, 1], [2, -3]], dtype=torch.int64))
    bias = None if bias_values is None else Value(torch.tensor(bias_values))
    expected = x.value @ y.value
    if bias is not None:
        expected += bias.value * fp.scale
    expected = torch.clamp(expected >> fp.fractional_bits, min=0)
    assert torch.equal(fp.matmul_relu(x, y, bias).value, expected)


def test_smooth_reuses_comparisons_and_even_polynomial():
    class CountingOracle(IntegerOracle):
        multiplications = 0
        comparisons = 0

        def mul(self, x, y):
            self.multiplications += 1
            return super().mul(x, y)

        def lt_public(self, x, y):
            self.comparisons += 1
            return super().lt_public(x, y)

    runtime = CountingOracle()
    fp = FixedPoint(runtime)
    fp.gelu(Value(fp.encode([-8., -1., 0., 1., 8.])))
    assert runtime.comparisons == 2
    assert runtime.multiplications == 9
