from itertools import product

import numpy as np
import torch

from experiments.examples.ring16_text_classifier import SparseIntegerSquare, integer_forward, range_bounds


def test_binary_budget_bounds_cover_every_small_input():
    w1 = np.array([[1, -1], [-1, 0], [1, 1]])
    b1 = np.array([1, -1])
    w2 = np.array([[3], [-2]])
    b2 = np.array([-7])
    result = range_bounds(w1, b1, w2, b2, max_tokens=2)
    for bits in product((0, 1), repeat=3):
        if sum(bits) > 2:
            continue
        x = np.array([bits])
        first = x @ w1
        h = first + b1
        square = h * h
        last = square @ w2
        score = integer_forward(x, w1, b1, w2, b2)
        for name, value in [('first_product', first), ('encoded', h), ('squared', square),
                            ('last_product', last), ('output', score)]:
            assert abs(value).max() <= result['absolute_bounds'][name]
    assert result['fits_signed_ring16']


def test_sparse_quantized_forward_matches_integer_circuit():
    torch.manual_seed(17)
    model = SparseIntegerSquare(16, 7, 4, 3)
    x = torch.randint(0, 2, (11, 16)).float()
    values = [p.detach().numpy().astype(np.int64) for p in model.quantized()]
    w1, b1, w2, _ = values
    assert ((w1 == -1) | (w1 == 0) | (w1 == 1)).all()
    assert (np.count_nonzero(w1, axis=0) <= 4).all()
    assert abs(b1).max() <= 1 and abs(w2).max() <= 3
    expected = integer_forward(x.numpy().astype(np.int64), *values)
    np.testing.assert_array_equal(model(x).detach().numpy() * 16, expected)


def test_zero_head_pruning_preserves_every_score():
    from experiments.examples.ring16_text_classifier import prune_zero_head
    w1 = np.array([[1, 1, -1], [0, -1, 1]])
    b1 = np.array([0, 1, -1])
    w2 = np.array([[2], [0], [-1]])
    b2 = np.array([3])
    pruned = prune_zero_head(w1, b1, w2, b2)
    x = np.array(list(product((0, 1), repeat=2)))
    np.testing.assert_array_equal(integer_forward(x, w1, b1, w2, b2), integer_forward(x, **pruned))
    assert pruned['w1'].shape == (2, 2)
