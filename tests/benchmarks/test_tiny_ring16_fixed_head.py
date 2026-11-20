import itertools

import numpy as np
import pytest
import torch

from experiments.examples.fixed_head_text_classifier import integer_forward
from experiments.examples.tiny_ring16_fixed_head import TinyIntegerFixedHead, binary_bounds


def test_binary_bounds_cover_every_input_in_small_budget():
    w = np.array([[1, 0, -1, 1], [-1, 1, 0, 1], [0, -1, 1, 0],
                  [1, 1, 1, -1], [0, 0, 1, 1], [-1, 1, 0, 0]])
    b = np.array([1, -2, 0, 2])
    c = np.array([-3])
    signs = np.array([1, 1, -1, -1])
    x = np.array([row for row in itertools.product([0, 1], repeat=6) if sum(row) <= 3])
    bounds = binary_bounds(w, b, c, signs, 3)['absolute_bounds']
    hidden = x @ w + b
    assert np.abs(hidden).max() <= bounds['encoded']
    assert (hidden * hidden).max() <= bounds['squared']
    assert np.abs(integer_forward(x, w, b, c, signs)).max() <= bounds['output']


def test_quantized_training_circuit_matches_exported_integer_scores():
    torch.manual_seed(2026)
    model = TinyIntegerFixedHead(8, 4, 3)
    x = torch.tensor(list(itertools.product([0., 1.], repeat=8)))
    with torch.no_grad():
        actual = model(x).numpy() * 16
        params = [p.numpy().round().astype(np.int64) for p in model.quantized()]
    expected = integer_forward(x.numpy().astype(np.int64), *params, model.signs.numpy().astype(np.int64))
    np.testing.assert_array_equal(np.rint(actual), expected)
    assert all(np.count_nonzero(column) == 3 for column in params[0].T)


def test_large_bias_cannot_overflow_bound_into_a_valid_ring():
    result = binary_bounds(np.ones((2, 2), dtype=np.int64), np.array([2**40, 0]),
                           np.array([0]), np.array([1, -1]), 2)
    assert result['absolute_bounds']['output'] > 2**80
    assert not result['fits_signed_ring16']
    with pytest.raises(ValueError, match='ternary'):
        binary_bounds(np.full((2, 2), 2), np.zeros(2), np.zeros(1), np.array([1, -1]), 2)
