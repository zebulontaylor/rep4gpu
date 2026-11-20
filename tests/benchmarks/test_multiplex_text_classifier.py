import numpy as np

from experiments.benchmarks.inference.multiplex_text_classifier import run_multiplex


def test_four_processes_two_independent_tls_sessions(tmp_path):
    model = tmp_path/'model.npz'
    np.savez(model, x_test=np.array([[0, 1], [1, 1], [2, 1]], dtype=np.int32),
             weights=np.array([[2], [-2]], dtype=np.int32),
             float_weights=np.array([[2], [-2]], dtype=np.float32))
    output = tmp_path/'result.json'
    report = run_multiplex(model, output, sessions=2, batch=3, repeats=8,
                           warmup=1, device='cpu', bits=16, timeout=60)
    assert report['passed']
    assert report['return_codes'] == [0, 0, 0, 0]
    assert report['party_processes'] == 4
    assert all(count > 0 for count in report['complete_batches_per_session'])
    assert report['texts_per_second'] > 0
    assert not list(tmp_path.rglob('*.key'))
