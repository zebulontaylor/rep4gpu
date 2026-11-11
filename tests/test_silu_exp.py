"""Check the bounded exponential's integer contract and headroom."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from rep4gpu.silu_exp import encoded_coefficients, silu_exp_negative
from rep4gpu.witness_prover import _PlainFloor


class CountingFloor(_PlainFloor):
    def __init__(self):
        self.products = 0
        self.floors = []
        self.maximum_numerator = 0

    def mul(self, x, y):
        self.products += 1
        return super().mul(x, y)

    def trunc(self, x, bits):
        self.floors.append(bits)
        self.maximum_numerator = max(self.maximum_numerator, int(x.abs().max()))
        return super().trunc(x, bits)


@pytest.mark.parametrize('f', range(20, 25))
@pytest.mark.parametrize('bounded_floor', [False, True])
def test_remez_integer_contract(f, bounded_floor):
    scale = 1 << f
    coefficients = encoded_coefficients(f)
    assert coefficients[0] == scale
    assert all(a >= b >= 0 for a, b in zip(coefficients, coefficients[1:]))
    rng = np.random.default_rng(191)
    encoded = np.concatenate((np.linspace(-16*scale, 0, 65537).astype(np.int64),
                              rng.integers(-16*scale, 1, size=65536)))
    reduced = encoded // 16
    c0, c1, c2, c3, c4, c5, c6 = coefficients
    square = reduced*reduced//scale
    q2 = c4+(c5*reduced+c6*square)//scale
    q1 = c2+(c3*reduced+square*q2)//scale
    expected = c0+(c1*reduced+square*q1)//scale
    assert c4 >= c5 and c2 >= c3
    assert c2+c4+c6 <= c1
    assert np.all(square <= -reduced)
    assert np.all((q1 >= 0) & (q1 <= c1))
    for _ in range(4):
        expected = (expected * expected + scale//2) // scale
    runtime = CountingFloor()
    fp = SimpleNamespace(runtime=runtime, fractional_bits=f)
    actual = silu_exp_negative(fp, torch.from_numpy(encoded), bounded_floor=bounded_floor)
    np.testing.assert_array_equal(actual.numpy(), expected)
    assert int(actual.min()) >= 0 and int(actual.max()) <= scale
    assert runtime.maximum_numerator <= scale*scale+scale//2 < 2**62
    # Four public coefficient products plus seven secret products.
    assert runtime.products == 11
    assert runtime.floors == [4]+[f]*8


def test_remez_rejects_nonfloor_rounding():
    runtime = _PlainFloor()
    runtime.rounding = 'probabilistic'
    fp = SimpleNamespace(runtime=runtime, fractional_bits=24)
    with pytest.raises(ValueError, match='exact-floor'):
        silu_exp_negative(fp, torch.tensor([0], dtype=torch.int64))


@pytest.mark.parametrize('f', [19, 25, 24.0, True])
def test_remez_rejects_unsupported_precision(f):
    with pytest.raises(ValueError, match='fractional_bits'):
        encoded_coefficients(f)


def test_quadratic_q24_rounding_alignment_accuracy():
    # Consecutive encodings cover the argument-reduction boundary that made
    # nearest-encoded Remez coefficients exceed the SiLU error target.
    scale = 1 << 24
    centers = (36575297, round(2.1233864426612854*scale))
    magnitudes = np.concatenate([np.arange(x-32, x+33, dtype=np.int64)
                                 for x in centers])
    encoded = np.concatenate((-magnitudes, magnitudes))
    fp = SimpleNamespace(runtime=_PlainFloor(), fractional_bits=24)
    exponential = silu_exp_negative(fp, torch.from_numpy(-np.abs(encoded))).numpy()
    numerator = np.where(encoded < 0, encoded*exponential, encoded*scale)
    output = np.rint(numerator/(scale+exponential))/scale
    real = encoded/scale
    truth = real/(1+np.exp(-real))
    assert np.max(np.abs(output-truth))+1/scale <= 6.41e-7
