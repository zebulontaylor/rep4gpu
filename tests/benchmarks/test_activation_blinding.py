import math

import numpy as np
import pytest
import torch

from experiments.benchmarks.activations.activation_blinding import ReplicaSimulation, leakage_witnesses, wire_model


def test_blinding_leakage_and_epsilon_counterexample():
    witness = leakage_witnesses()
    assert witness['softmax']['sorted_pairwise_gaps_preserved']
    assert witness['softmax']['probability_multiset_preserved']
    assert witness['softmax']['fp16_large_shift_logits'] == [8192.,8192.,8192.]
    assert witness['softmax']['fp16_large_shift_max_probability_error'] > 0.2
    assert witness['rms']['zero_epsilon_normalized_multiset_preserved']
    assert witness['rms']['absolute_pairwise_ratio_preserved']
    assert witness['rms']['fixed_epsilon_max_error'] > 0.01
    assert witness['rms']['scaled_epsilon_max_error'] < 1e-12
    assert witness['rms']['scale_recovered_from_public_epsilons'] == 7
    assert witness['gelu']['all_original_values_still_visible']


@pytest.mark.parametrize('activation', ['softmax', 'rms', 'gelu'])
def test_replicas_agree_and_detect_independent_corruption(activation):
    simulator = ReplicaSimulation()
    fixture = torch.tensor([[-2., 0., 1., 3.]], dtype=torch.float64)
    first = simulator.run(fixture, activation, 0)
    second = simulator.run(fixture, activation, 1)
    assert first['accepted'] and second['accepted']
    values = fixture.numpy()
    if activation == 'softmax':
        exponents = np.exp(values-values.max(axis=-1, keepdims=True))
        expected = exponents/exponents.sum(axis=-1, keepdims=True)
    elif activation == 'rms':
        expected = values/np.sqrt(np.mean(values*values, axis=-1, keepdims=True)+1e-6)
    else:
        expected = 0.5*values*(1+np.vectorize(math.erf)(values/math.sqrt(2)))
    encoded = np.rint(expected*2**16).astype(np.int64)&0xffffffff
    assert np.array_equal(first['quantized_output'], encoded)
    assert np.array_equal(first['quantized_output'], second['quantized_output'])
    assert not np.array_equal(first['components'], second['components'])
    assert not simulator.run(fixture, activation, 2, corrupt_second=True)['accepted']
    with pytest.raises(ValueError, match='fresh'):
        simulator.run(fixture, activation, 2)


@pytest.mark.parametrize('bits', [32,64])
def test_wire_model_counts_all_six_rep4_joints_and_tags(bits):
    count = 8192
    model = wire_model(count, bits)
    assert model['rep4_generic_product']['application_bytes'] == 6*count*(bits//8)+192
    assert model['rep4_balanced_cross_product']['application_bytes'] == 4*count*(bits//8)+128
    simulated = model['simulated_two_helper_relu']
    assert simulated['input']['application_bytes'] == 2*count*(bits//8)+64
    assert simulated['output']['application_bytes'] == 6*count*(bits//8)+64
    assert simulated['application_bytes'] == 8*count*(bits//8)+128
    assert simulated['application_bytes'] == simulated['input']['application_bytes']+simulated['output']['application_bytes']
    assert simulated['checked_rounds'] == 2
    assert simulated['supported_by_benchmark'] == (bits==32)
    assert model['hypothetical_two_helper_activation']['application_bytes'] == 11*count*(bits//8)+96
    assert model['tee_latency_ms'] is None
