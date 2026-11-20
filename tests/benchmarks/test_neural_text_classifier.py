import numpy as np

from experiments.examples.neural_text_classifier import integer_forward, range_bounds, select_threshold


def test_signed_floor_and_square_order():
    x = np.array([[8, 0], [0, 8]])
    w1 = np.array([[-9], [9]])
    b1, w2, b2 = np.array([1]), np.array([[-11]]), np.array([2])
    output, trace = integer_forward(x, w1, b1, w2, b2, 6)
    # -72 // 64 + 1 = -1, while truncation toward zero would produce zero.
    assert trace["encoded"].tolist() == [[-1], [2]]
    assert trace["hidden"].tolist() == [[0], [0]]
    assert output.tolist() == [[2], [2]]


def test_bounds_cover_all_small_budget_inputs():
    w1 = np.array([[-7, 21, -31], [15, -13, 5]])
    b1 = np.array([-3, 4, 6])
    w2 = np.array([[-11], [9], [-17]])
    b2 = np.array([5])
    counts = np.array([(a, b) for a in range(6) for b in range(6 - a)])
    bounds = range_bounds(w1, b1, w2, b2, 6, 5)
    output, trace = integer_forward(counts * 8, w1, b1, w2, b2, 6)
    trace["output"] = output
    for name, values in trace.items():
        assert np.abs(values).max() <= bounds["absolute_bounds"][name]
    assert bounds["fits_signed_ring32"]


def test_validation_threshold_can_change_decisions():
    labels = np.array([0, 0, 1, 1])
    logits = np.array([-3, -2, -1, 1])
    threshold, result, f1 = select_threshold(labels, logits)
    assert threshold == -1
    assert result["accuracy"] == f1 == 1.0
