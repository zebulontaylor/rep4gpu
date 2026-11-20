"""Check the TLS classifier launcher and signed accumulator validation."""
import numpy as np
import pytest

from experiments.benchmarks.inference.text_classifier import run


def checkpoint(path, large=False):
    x = np.array([[0, 1], [1, 1], [2, 1]], dtype=np.int32)
    if large:
        x[2, 0] = 65535
    w = np.array([[65535 if large else 2], [-2]], dtype=np.int32)
    np.savez(path, x_test=x, weights=w, float_weights=w.astype(np.float32))


def test_signed_boundary_labels_over_tls(tmp_path):
    model = tmp_path/'model.npz'
    checkpoint(model)
    report = run(model, tmp_path/'result.json', batch=3, repeats=1, warmup=0)
    assert report['passed']
    assert report['return_codes'] == [0, 0, 0, 0]
    assert all(p['predictions'] == [0, 1, 1] for p in report['parties'])
    assert all(p['authenticated_tls'] for p in report['parties'])
    assert report['online_ms'] >= report['checked_inference_ms'] > 0


def test_reject_accumulator_overflow(tmp_path):
    model = tmp_path/'model.npz'
    checkpoint(model, large=True)
    with pytest.raises(ValueError, match='accumulator range'):
        run(model, tmp_path/'result.json', batch=3)


def test_private_square_encoder_over_tls(tmp_path):
    model = tmp_path/'square.npz'
    parameters = dict(w1=np.array([[4, 0], [0, 4]], dtype=np.int32),
                      b1=np.zeros(2, dtype=np.int32),
                      w2=np.array([[4], [-4]], dtype=np.int32),
                      b2=np.array([-1], dtype=np.int32))
    np.savez(model, x_test=np.array([[0, 0], [4, 0], [0, 4]], dtype=np.int32),
             fractional_bits=np.array(2), **parameters,
             **{'float_'+key: value.astype(np.float32)/4 for key, value in parameters.items()})
    report = run(model, tmp_path/'result.json', batch=3, repeats=1, warmup=0)
    assert report['passed'] and report['architecture'] == 'square'
    assert report['fractional_bits'] == 2
    assert all(p['predictions'] == [0, 1, 0] for p in report['parties'])
