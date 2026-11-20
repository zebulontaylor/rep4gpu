"""Mixed-ring entry requires a certificate over the full binary input domain."""
from copy import deepcopy
from itertools import product

import numpy as np
import pytest

from experiments.benchmarks.inference.text_classifier import run
from experiments.benchmarks.inference.text_classifier_model import encoder_ring_bounds


def fixture():
    weights = dict(w1=np.array([[4, -7], [-6, 3], [2, 1]]),
                   b1=np.array([1, -2]), b2=np.array([1]))
    return dict(architecture='fixed_square_head', fractional_bits=0, input_divisor=1,
                x=np.zeros((1, 3), dtype=np.int64), weights=weights,
                float_weights={name: value.astype(float) for name, value in weights.items()})


def test_certificate_covers_all_binary_inputs_including_unseen_rows():
    model = fixture()
    certificate = encoder_ring_bounds(model, 8)
    x = np.array(list(product((0, 1), repeat=3)))
    values = x @ model['weights']['w1'] + model['weights']['b1']
    assert certificate == dict(input_domain='binary', minimum=int(values.min()), maximum=int(values.max()))


def test_certificate_rejects_overflow_not_present_in_evaluation_fixture():
    model = fixture()
    model['weights']['w1'][0, 0] = 127
    model['float_weights']['w1'] = model['weights']['w1'].astype(float)
    # The sole evaluation row is zero; the all-binary domain contains overflow.
    with pytest.raises(ValueError, match='exceeds signed ring8'):
        encoder_ring_bounds(model, 8)


@pytest.mark.parametrize('change', ['nonbinary', 'divisor', 'fractional', 'float_weights', 'architecture'])
def test_certificate_rejects_different_model_semantics(change):
    model = deepcopy(fixture())
    if change == 'nonbinary':
        model['x'][0, 0] = 2
    elif change == 'divisor':
        model['input_divisor'] = 2
    elif change == 'fractional':
        model['fractional_bits'] = 1
    elif change == 'float_weights':
        model['float_weights']['w1'][0, 0] += .5
    else:
        model['architecture'] = 'linear'
    with pytest.raises(ValueError):
        encoder_ring_bounds(model, 8)


def test_mixed_runtime_guard_precedes_checkpoint_access(tmp_path):
    with pytest.raises(ValueError, match='mixed encoder requires CUDA packed ring16'):
        run(tmp_path/'missing.npz', tmp_path/'output.json', encoder_ring_bits=8)
