"""Check signed forward bounds against small independent integer computations."""
import pytest
import torch

from experiments.benchmarks.inference.numerical_inference_common import ForwardOracle, fixture, oracle, reference
from experiments.benchmarks.activations.numerical_primitives import evaluate


def test_signed_interval_matmul():
    runtime = ForwardOracle(True)
    a = runtime.public(torch.tensor([[-3, 2], [4, -1]]))
    a.hi = a.lo+2
    weights = torch.tensor([[2, -5], [-3, 7]])
    result = runtime.matmul(a, runtime.public(weights))
    for first in range(3):
        for second in range(3):
            actual = (a.lo+torch.tensor([first, second])) @ weights
            assert bool(((actual >= result.lo) & (actual <= result.hi)).all())
    assert torch.equal(result.lo, torch.tensor([[-18, 19], [5, -37]]))
    assert torch.equal(result.hi, torch.tensor([[-8, 43], [15, -13]]))


@pytest.mark.parametrize('kind', ['silu', 'gelu', 'swiglu'])
def test_forward_floor_matches_independent_plain_steps(kind):
    floats, integers = fixture(kind, 2, 3, 4, 2)
    interval, proof = oracle(kind, integers)
    x, gate, bias, down, output_bias, *value = integers
    hidden = (x @ gate >> 24)+bias
    runtime = ForwardOracle(False)
    value_projection = runtime.public((x @ value[0] >> 24)+value[1]) if value else None
    activation = evaluate(runtime, kind, runtime.public(hidden), value_projection)
    expected = (activation.lo @ down >> 24)+output_bias
    assert torch.equal(interval.lo, expected)
    assert torch.equal(interval.hi, expected)
    assert proof['maximum_pretruncation_abs_bound'] < 2**62
    probabilistic, _ = oracle(kind, integers, probabilistic=True)
    assert bool(((expected >= probabilistic.lo) & (expected <= probabilistic.hi)).all())
    assert reference(floats, kind).shape == (2, 2)


def test_reject_overflow_before_dot():
    runtime = ForwardOracle()
    with pytest.raises(ValueError, match='2\\*\\*62'):
        runtime.matmul(runtime.public(torch.tensor([[2**40]])), runtime.public(torch.tensor([[2**30]])))


@pytest.mark.parametrize('kind', ['silu', 'gelu', 'swiglu'])
def test_activation_bypass_preserves_affine_and_gate_steps(kind):
    floats, integers = fixture(kind, 2, 3, 4, 2)
    x, gate, bias, down, output_bias, *value = integers
    hidden = (x @ gate >> 24)+bias
    if kind == 'swiglu':
        hidden = hidden*((x @ value[0] >> 24)+value[1]) >> 24
    expected = (hidden @ down >> 24)+output_bias
    interval, _ = oracle(kind, integers, activation_enabled=False)
    assert torch.equal(interval.lo, expected)
    assert torch.equal(interval.hi, expected)
    probabilistic, _ = oracle(kind, integers, probabilistic=True, activation_enabled=False)
    assert bool(((expected >= probabilistic.lo) & (expected <= probabilistic.hi)).all())

    x, gate, bias, down, output_bias, *value = floats
    hidden = x @ gate+bias
    if kind == 'swiglu':
        hidden = hidden*(x @ value[0]+value[1])
    expected_float = hidden @ down+output_bias
    assert torch.equal(reference(floats, kind, activation_enabled=False), expected_float)
    assert not torch.equal(reference(floats, kind), expected_float)


@pytest.mark.parametrize('kind,index', [('silu', 2), ('swiglu', 6)])
def test_reject_out_of_domain_projection(kind, index):
    _, values = fixture(kind, 1, 2, 3, 2)
    values[index].fill_(5 << 24)
    with pytest.raises(ValueError, match='activation input'):
        oracle(kind, values, probabilistic=True)


def test_fixture_original_stream():
    from experiments.benchmarks.inference.inference import fixture as original
    _, actual = fixture('silu', 2, 3, 4, 2)
    _, expected, _ = original(2, 3, 4, 2, 24)
    assert all(torch.equal(a, b) for a, b in zip(actual, expected))
