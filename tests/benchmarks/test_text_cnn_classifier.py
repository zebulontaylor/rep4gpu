"""Validation tests for the certified 100%-dense Text CNN and live 4-process CUDA IPC benchmark."""
from pathlib import Path
import shutil
import sys

import numpy as np
import pytest
import torch

from experiments.benchmarks.inference import text_cnn_classifier as bench
from experiments.examples.text_cnn_classifier import global_bounds, integer_forward


CHECKPOINT = Path('experiments/results/examples/text_cnn_classifier/model.npz')


def test_text_cnn_checkpoint_bounds_and_density():
    assert CHECKPOINT.exists(), f'missing Text CNN checkpoint {CHECKPOINT}'
    model = bench.load_cnn_model(CHECKPOINT, batch=64)
    assert model['architecture'] == 'text_cnn_square'
    assert model['primes'] == (251, 241, 239, 233)
    x = model['x']
    assert x.shape == (64, 48, 256)
    assert np.all((x == -1) | (x == 1))
    bounds = model['bounds']
    crt_limit = int(np.prod(model['primes'])) // 2
    assert bounds['score_max_abs'] < crt_limit
    scores = integer_forward(x, model['weights']['w1'], model['weights']['b1'],
                             model['weights']['w2'], model['weights']['b2'])
    assert int(np.max(np.abs(scores))) <= bounds['score_max_abs']
    assert (scores >= 0).reshape(-1).astype(int).tolist() == model['expected']


@pytest.mark.skipif(sys.platform != 'linux' or not torch.cuda.is_available()
                    or shutil.which('openssl') is None, reason='Linux, CUDA and OpenSSL required')
def test_text_cnn_live_four_process_cuda_ipc_exact(tmp_path):
    output = tmp_path / 'text_cnn_test.json'
    report = bench.run(
        CHECKPOINT, output, batch=64, repeats=2, warmup=1,
        timeout=120, phase_graphs=True, device_transport='cuda-ipc',
        resident_inputs=False, include_isolated_estimate=True,
    )
    assert report['passed'] is True
    assert report['exact_integer_match'] is True
    assert report['return_codes'] == [0, 0, 0, 0]
    assert all(p['control_authenticated_tls'] is True for p in report['parties'])
    assert report['isolated_party_gpu_estimate']['bottleneck_party_ms'] > 0
