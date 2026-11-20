"""Validate GPU-crypto inference on a trained two-layer ReLU MLP."""
import pytest
import torch

from examples.mlp_demo import run_demo


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('bits', [32, 64])
@pytest.mark.parametrize('rounding', ['floor', 'probabilistic'])
def test_trained_mlp_demo_all_contexts(bits, rounding):
    result = run_demo(steps=600, bits=bits, fractional_bits=12, rounding=rounding, capture_graphs=True)
    assert result['float_accuracy'] == 1.0
    assert result['private_accuracy'] == 1.0
    if rounding == 'floor':
        assert result['exact_integer_match']
    assert result['max_logit_error'] < 0.05
