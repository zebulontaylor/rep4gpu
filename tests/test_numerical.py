"""Integer-oracle accuracy and range checks for numerical evaluation."""

from dataclasses import dataclass

import pytest
import torch

from rep4gpu.numerical import NumericalFixedPoint


@dataclass
class Value:
    value: torch.Tensor
    binary: bool = False

    @property
    def device(self):
        return self.value.device


class Oracle:
    bits = 64

    def __init__(self, rounding='floor'):
        self.rounding = rounding
        self.max_product = 0
        self.generator = torch.Generator().manual_seed(71)

    def public(self, x, like):
        return Value(torch.as_tensor(x, dtype=torch.int64).expand_as(like.value))

    def add(self, x, y):
        return Value(x.value + y.value)

    def sub(self, x, y):
        return Value(x.value - y.value)

    def mul(self, x, y):
        return self._product(x.value, y.value)

    def mul_public(self, x, y):
        return self._product(x.value, y)

    def _product(self, x, y):
        # Python integers detect overflow before torch's ring64 wraparound.
        maximum = max((abs(int(a)*int(b)) for a, b in zip(
            x.flatten().tolist(), torch.as_tensor(y).expand_as(x).flatten().tolist())), default=0)
        self.max_product = max(self.max_product, maximum)
        assert maximum < 2**62
        return Value(x*y)

    def trunc(self, x, shift):
        base = x.value >> shift
        if self.rounding == 'ceil':
            base = base + ((x.value & ((1 << shift)-1)) != 0)
        elif self.rounding == 'random':
            remainder = x.value & ((1 << shift)-1)
            draw = torch.randint(1 << shift, x.value.shape, generator=self.generator)
            base = base + (draw < remainder)
        return Value(base)

    def sign_bit(self, x):
        return Value((x.value < 0).to(torch.int64))


@pytest.mark.parametrize('fractional_bits', [20, 24])
@pytest.mark.parametrize('rounding', ['floor', 'ceil', 'random'])
def test_numerical_accuracy_and_quarter_ring(fractional_bits, rounding):
    rt = Oracle(rounding)
    fp = NumericalFixedPoint(rt, fractional_bits)
    for name, bound, reference, tolerance in [
        ('exp', 9., torch.exp, .5 if fractional_bits == 20 else .04),
        ('erf', 3., torch.erf, 1e-4),
        ('silu', 4., torch.nn.functional.silu, 1e-4),
        ('gelu', 4., torch.nn.functional.gelu, 1e-4),
    ]:
        inputs = torch.linspace(-bound, bound, 1001, dtype=torch.float64)
        actual = fp.decode(getattr(fp, name)(Value(fp.encode(inputs))).value)
        error = torch.max(torch.abs(actual-reference(inputs))).item()
        assert error < tolerance, (name, fractional_bits, rounding, error)
    inputs = torch.linspace(1., 2., 1001, dtype=torch.float64)
    actual = fp.decode(fp.reciprocal(Value(fp.encode(inputs))).value)
    assert torch.max(torch.abs(actual-1/inputs)) < 5/fp.scale
    gate = torch.linspace(-4., 4., 1001, dtype=torch.float64)
    value = torch.flip(gate, (0,))
    actual = fp.decode(fp.swiglu(Value(fp.encode(gate)), Value(fp.encode(value))).value)
    assert torch.max(torch.abs(actual-torch.nn.functional.silu(gate)*value)) < 4e-4
    assert rt.max_product < 2**62


def test_public_configuration_and_share_validation():
    for bits in [32, None]:
        rt = Oracle()
        rt.bits = bits
        with pytest.raises(ValueError):
            NumericalFixedPoint(rt)
    for fractional_bits in [True, 19, 25, 24.]:
        with pytest.raises(ValueError):
            NumericalFixedPoint(Oracle(), fractional_bits)
    fp = NumericalFixedPoint(Oracle())
    for name in ['exp', 'reciprocal', 'erf', 'silu', 'gelu']:
        with pytest.raises(ValueError):
            getattr(fp, name)(Value(torch.tensor([0]), binary=True))


@pytest.mark.parametrize('fractional_bits', [20, 24])
@pytest.mark.parametrize('rounding', ['floor', 'ceil', 'random'])
def test_nonpositive_exp_keeps_reciprocal_domain(fractional_bits, rounding):
    fp = NumericalFixedPoint(Oracle(rounding), fractional_bits)
    # Include single encoded ulps near zero, where reduction can round to zero.
    encoded = torch.cat((fp.encode(torch.linspace(-9., 0., 1001)),
                         torch.arange(-32, 1, dtype=torch.int64)))
    output = fp.exp(Value(encoded)).value
    assert torch.all((output >= 0) & (output <= fp.scale))


def test_swiglu_rejects_binary_inputs():
    fp = NumericalFixedPoint(Oracle())
    arithmetic = Value(torch.tensor([fp.scale]))
    binary = Value(torch.tensor([1]), binary=True)
    for gate, value in [(binary, arithmetic), (arithmetic, binary)]:
        with pytest.raises(ValueError):
            fp.swiglu(gate, value)
