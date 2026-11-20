import itertools

import numpy as np
import pytest

from experiments.examples.fixed_head_text_classifier import encode, integer_forward, range_bounds


def test_integer_encoding_equals_exact_float_polynomial():
    params = dict(w1=np.array([[.25, -.125], [-.125, .25]]),
                  b1=np.array([.125, -.25]), b2=np.array([.5]))
    signs = np.array([1, -1])
    x = np.array([[0, 0], [3, 2], [1, 6]], dtype=np.int64)
    encoded, bounds, scale = encode(params, signs, 8, 8)
    hidden = x / 8 @ params['w1'] + params['b1']
    expected = (hidden**2 * signs).sum(1, keepdims=True) + params['b2']
    np.testing.assert_array_equal(integer_forward(x, **encoded, signs=signs), expected * scale)
    assert bounds['fits_signed_ring32']


def test_bounds_cover_exhaustive_token_budget():
    w = np.array([[2, -1], [-3, 2], [0, 1]])
    b = np.array([-2, 1])
    c = np.array([3])
    signs = np.array([1, -1])
    values = np.array([v for v in itertools.product(range(5), repeat=3) if sum(v) <= 4])
    bound = range_bounds(w, b, c, signs, 4)['absolute_bounds']
    h = values @ w + b
    assert abs(h).max() <= bound['encoded']
    assert (h * h).max() <= bound['squared']
    assert abs(integer_forward(values, w, b, c, signs)).max() <= bound['output']


def test_bad_scale_cannot_wrap_parameters():
    with pytest.raises(ValueError, match='int32'):
        encode(dict(w1=np.ones((2, 2)), b1=np.ones(2), b2=np.ones(1)), np.array([1, -1]), 2**20, 128)
