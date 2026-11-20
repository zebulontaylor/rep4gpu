"""Tests for the AG News Topic MLP ReLU classifier export and Rep4 inference."""
from pathlib import Path
import numpy as np
import pytest
import torch

from experiments.examples.topic_mlp_classifier import (
    homogeneous_rescale,
    propagate_bounds,
)


def test_homogeneous_rescale_preserves_sign_and_bounds():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(64, 32)).astype(np.float64)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    params = dict(
        w1=rng.normal(scale=0.5, size=(32, 16)),
        b1=rng.normal(scale=0.5, size=(16,)),
        w2=rng.normal(scale=0.5, size=(16, 8)),
        b2=rng.normal(scale=0.5, size=(8,)),
        w3=rng.normal(scale=0.5, size=(8, 1)),
        b3=rng.normal(scale=0.5, size=(1,)),
    )

    def eval_mlp(p):
        z1 = x @ p['w1'] + p['b1']
        z2 = np.maximum(z1, 0.0) @ p['w2'] + p['b2']
        z3 = np.maximum(z2, 0.0) @ p['w3'] + p['b3']
        return z1, z2, z3

    _, _, z3_before = eval_mlp(params)
    scaled = homogeneous_rescale(params, x, target_max=1.5)
    z1_s, z2_s, z3_s = eval_mlp(scaled)

    assert np.allclose(np.max(np.abs(z1_s)), 1.5, rtol=1e-6)
    assert np.allclose(np.max(np.abs(z2_s)), 1.5, rtol=1e-6)
    assert np.allclose(np.max(np.abs(z3_s)), 1.5, rtol=1e-6)
    assert np.array_equal(z3_before >= 0, z3_s >= 0)


def test_propagate_bounds_enforces_quarter_ring_contract():
    scale = 1 << 14
    x = np.full((2, 4), scale, dtype=np.int64)
    # Accumulator = 4 * scale * scale = 2^30 -> violates strict quarter-ring (< 2^30) for probabilistic.
    params = dict(
        w1=np.full((4, 4), scale, dtype=np.int64),
        b1=np.zeros(4, dtype=np.int64),
        w2=np.zeros((4, 4), dtype=np.int64),
        b2=np.zeros(4, dtype=np.int64),
        w3=np.zeros((4, 1), dtype=np.int64),
        b3=np.zeros(1, dtype=np.int64),
    )
    with pytest.raises(ValueError, match='probabilistic limit'):
        propagate_bounds(x, params, 14, bits=32, rounding='probabilistic')
    # Floor allows up to 2^31 - 1, so 2^30 passes floor.
    bounds, _, _, _ = propagate_bounds(x, params, 14, bits=32, rounding='floor')
    assert bounds['raw1_max_abs'] == 1 << 30


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_topic_mlp_live_floor_and_probabilistic(tmp_path):
    ckpt = Path('experiments/results/examples/topic_mlp_classifier/model.npz')
    if not ckpt.exists():
        pytest.skip('trained topic MLP checkpoint not present')
    from experiments.benchmarks.inference.topic_mlp_classifier import run

    rep_floor = run(
        ckpt, tmp_path / 'floor.json', batch=16, repeats=1, warmup=1,
        rounding='floor', weight_storage='words32', device_transport='cuda-ipc',
        resident_inputs=True,
    )
    assert rep_floor['passed']
    assert rep_floor['exact_floor_match']

    rep_prob = run(
        ckpt, tmp_path / 'prob.json', batch=16, repeats=1, warmup=1,
        rounding='probabilistic', weight_storage='strassen32', device_transport='cuda-ipc',
        resident_inputs=False,
    )
    assert rep_prob['passed']
    assert rep_prob['interval_valid']
