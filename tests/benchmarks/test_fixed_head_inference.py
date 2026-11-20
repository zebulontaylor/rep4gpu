import numpy as np
import pytest

from experiments.benchmarks.inference.text_classifier import run
from experiments.benchmarks.inference.text_classifier_model import load_model


def checkpoint(path, signs=(1, -1)):
    params = dict(w1=np.eye(2, dtype=np.int32), b1=np.array([-1, 1], dtype=np.int32),
                  b2=np.zeros(1, dtype=np.int32))
    np.savez(path, architecture='fixed_square_head', input_divisor=1,
             x_test=np.array([[0, 0], [1, 0], [3, 0], [0, 2]], dtype=np.int32),
             signs=np.array(signs, dtype=np.int32), **params,
             **{'float_'+k: v.astype(np.float32) for k, v in params.items()})


def test_private_fixed_head_ring16_over_tls(tmp_path):
    path = tmp_path/'model.npz'
    checkpoint(path)
    model = load_model(path, 4, bits=16)
    assert model['expected'] == [1, 0, 1, 0]
    report = run(path, tmp_path/'report.json', device='cpu', bits=16, batch=4,
                 repeats=1, warmup=0)
    assert report['passed'] and report['exact_integer_match']
    assert report['return_codes'] == [0, 0, 0, 0]


def test_learned_sign_pattern_cannot_be_passed_as_fixed_architecture(tmp_path):
    path = tmp_path/'model.npz'
    checkpoint(path, (-1, 1))
    with pytest.raises(ValueError, match='canonical public'):
        load_model(path, 4)


def test_binary_fixed_head_uses_exact_sparse_native_weights(tmp_path):
    path = tmp_path / 'binary.npz'
    checkpoint(path)
    with np.load(path) as saved:
        values = dict(saved)
    values['x_test'] = np.minimum(values['x_test'], 1)
    np.savez(path, **values)
    model = load_model(path, 4, bits=16)
    assert model['sparse_native_eligible']
    weights = model['sparse_native_weights']
    assert np.array_equal(weights['w2'], [[1], [-1]])
    hidden = model['x'] @ weights['w1'] + weights['b1']
    labels = (hidden**2 @ weights['w2'] + weights['b2'] >= 0).ravel()
    assert labels.tolist() == model['expected']
    assert 'w2' not in model['weights']


def test_precomputed_sparse_baseline_is_not_remeasured(tmp_path, monkeypatch):
    from experiments.benchmarks.inference import sparse_classifier_native
    path = tmp_path / 'binary.npz'
    checkpoint(path)
    with np.load(path) as saved:
        values = dict(saved)
    values['x_test'] = np.minimum(values['x_test'], 1)
    np.savez(path, **values)

    def unexpected(*args, **kwargs):
        raise AssertionError('concurrent launch must use its premeasured baseline')

    monkeypatch.setattr(sparse_classifier_native, 'native_sparse_baselines', unexpected)
    baseline = {'float32_eager': 0.2, 'cpp_sparse_int32_threads1': 0.1}
    result = run(path, tmp_path / 'report.json', device='cpu', bits=16,
                 batch=4, repeats=1, warmup=0, native_baseline=baseline)
    assert result['passed']
    assert result['native_sparse_ms'] == {'cpp_sparse_int32_threads1': 0.1}
    assert result['native_dense_ms'] == {'float32_eager': 0.2}
