import numpy as np
import pytest

from experiments.examples.text_classifier import featurize, load_data, quantize_weights, split_indices


def test_token_budget_bias_and_stable_hash():
    x = featurize(["Hello hello WORLD ignored", "", "HELLO"], features=16, max_tokens=3)
    assert x.dtype == np.int32
    assert x[:, -1].tolist() == [1, 1, 1]
    assert x[:, :-1].sum(axis=1).tolist() == [3, 0, 1]
    assert (x[0, :-1] >= 2 * x[2, :-1]).all()
    np.testing.assert_array_equal(x, featurize(["Hello hello WORLD ignored", "", "HELLO"], 16, 3))


def test_deduplicate_before_stratified_split(tmp_path):
    path = tmp_path / "sms"
    path.write_text("ham\tHello!\nham\thello\nham\tBye\nspam\tWin now\nspam\tWin later\n")
    texts, labels, digest = load_data(path)
    assert len(texts) == 4 and len(digest) == 64
    train, test = split_indices(labels, 42)
    assert not set(train) & set(test)
    assert set(labels[test]) == {0, 1}
    np.testing.assert_array_equal(test, split_indices(labels, 42)[1])


def test_quantized_bound_covers_extreme_inputs():
    w, bound = quantize_weights(np.array([[1.5], [-2.0], [-0.25]]), 256, 128)
    assert bound == 128 * 512 + 64
    extremes = np.array([[128, 0, 1], [0, 128, 1]], dtype=np.int64) @ w.astype(np.int64)
    assert np.abs(extremes).max() <= bound
    with pytest.raises(ValueError, match="does not fit"):
        quantize_weights(np.array([[2**30], [0.0]]), 256, 128)
