"""Reference arithmetic rejects overflow before timing a fixture."""
import pytest
import torch

from benchmarks.inference import checked_add, checked_matmul, fixture, maximum_abs


def test_magnitude_handles_signed_minimum():
    assert maximum_abs(torch.tensor([-2**63])) == 2**63


def test_reference_rejects_accumulator_overflow():
    with pytest.raises(ValueError, match='matrix-product bound'):
        checked_matmul(torch.tensor([[2**62, 2**62]]), torch.tensor([[2], [2]]))
    with pytest.raises(ValueError, match='addition bound'):
        checked_add(torch.tensor([2**63-1]), torch.tensor([1]))


def test_fixture_rejects_invalid_encoding_scale():
    with pytest.raises(ValueError, match='fractional_bits'):
        fixture(1, 8, 8, 4, 63)
    with pytest.raises(ValueError, match='reference matrix-product bound'):
        fixture(1, 8, 8, 4, 40)
