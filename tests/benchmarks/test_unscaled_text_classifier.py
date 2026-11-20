import numpy as np

from experiments.examples.unscaled_text_classifier import encode_polynomial, polynomial_bounds, polynomial_forward


def test_unscaled_polynomial_matches_float_for_exact_parameters():
    params = {"w1": np.array([[0.25, -0.125], [-0.5, 0.25]]), "b1": np.array([0.5, -0.25]),
              "w2": np.array([[0.5], [-0.25]]), "b2": np.array([-0.125])}
    encoded, bounds, scale = encode_polynomial(params, 16, 8, 128)
    x = np.array([[0, 0], [3, 7], [128, 0]])
    expected = ((x / 8 @ params["w1"] + params["b1"]) ** 2) @ params["w2"] + params["b2"]
    np.testing.assert_array_equal(polynomial_forward(x, **encoded) / scale, expected)
    assert bounds["fits_signed_ring32"]


def test_global_polynomial_bound_covers_all_small_budget_inputs():
    w1 = np.array([[-7, 21, -31], [15, -13, 5]])
    b1, w2, b2 = np.array([-3, 4, 6]), np.array([[-11], [9], [-17]]), np.array([5])
    x = np.array([(a, b) for a in range(6) for b in range(6 - a)])
    bounds = polynomial_bounds(w1, b1, w2, b2, 5)["absolute_bounds"]
    first = x @ w1
    hidden = first + b1
    square = hidden * hidden
    last = square @ w2
    for key, value in [("first_product", first), ("encoded", hidden), ("squared", square), ("last_product", last), ("output", last + b2)]:
        assert np.abs(value).max() <= bounds[key]
