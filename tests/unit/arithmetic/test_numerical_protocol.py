"""Private ring64 numerical outputs agree with an integer floor oracle."""
import pytest
import torch

from experiments.benchmarks.activations.numerical_primitives import run_case


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('kind',['exp','reciprocal','erf','silu','gelu','swiglu'])
def test_private_numerical_floor(kind):
    torch.set_num_threads(1)
    result=run_case(kind,elements=9,repeats=1,rounding='floor',replay=False)
    assert result['integer_oracle_passed']
    assert result['maximum_rounding_interval_width']==0
