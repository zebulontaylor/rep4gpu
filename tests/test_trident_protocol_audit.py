"""Independent integer oracles and regression counterexamples for the audit."""
import numpy as np
import pytest

from benchmarks.trident_protocol_audit import (
    prefix_subtract, published_sign, truncation_check_residual, masked_signed_truncation,
)


@pytest.mark.parametrize('bits', [4, 8])
def test_prefix_subtract_exhaustive(bits):
    values = np.arange(1 << bits, dtype=np.uint64)
    a, b = np.meshgrid(values, values)
    assert np.array_equal(prefix_subtract(a, b, bits), (a - b) & ((1 << bits) - 1))


def test_prefix_subtract_ring32_boundaries_and_random():
    rng = np.random.default_rng(42)
    a = np.concatenate((np.array([0, 1, 0x7fffffff, 0x80000000, 0xffffffff], dtype=np.uint64),
                        rng.integers(0, 1 << 32, 10000, dtype=np.uint64)))
    b = rng.integers(0, 1 << 32, a.shape, dtype=np.uint64)
    oracle = np.array([(int(x) - int(y)) % (1 << 32) for x, y in zip(a, b)], dtype=np.uint64)
    assert np.array_equal(prefix_subtract(a, b), oracle)


def test_ndss_sign_counterexamples():
    assert int(published_sign(2, 1 << 30)) == 1
    assert int(published_sign(0, 1 << 31)) == 1
    assert int(published_sign(-1, 1)) == 1


def test_literal_truncation_check_rejects_an_honest_low_bit_carry():
    assert truncation_check_residual((16383, 1, 0), 14) == (1 << 32) - 16384
    assert truncation_check_residual((1, 2, 3), 14) == 0


def test_uncorrected_truncation_wrap_is_not_one_unit_rounding_error():
    assert masked_signed_truncation(1 << 29, -(1 << 31), 14) == -229376
    assert ((1 << 29) >> 14) == 32768
