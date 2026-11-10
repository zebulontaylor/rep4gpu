"""Checkpoint fidelity, full-domain field bounds, and compact clear baselines."""
import copy
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from examples import spdz_compact_toy as compact
from examples.spdz_trained_toy import Square
from examples.toy_next_token import dataset
from rep4gpu.spdz import PRIME


CHECKPOINT = Path(__file__).resolve().parents[1] / 'results/spdz_compact_toy/model.pt'


@pytest.fixture(scope='module')
def model():
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    result = nn.Sequential(nn.Linear(16, 8), Square(), nn.Linear(8, 8)).eval()
    state = torch.load(CHECKPOINT, map_location='cpu', weights_only=True)
    result.load_state_dict(state, strict=True)
    yield result
    torch.set_num_threads(previous_threads)


def _python_reference(encoded):
    x, w1, b1, w2, b2 = (value.tolist() for value in encoded)
    first, outputs, bounds, field_outputs = [], [], [], []
    for row in x:
        affine = [sum(int(row[k])*int(w1[k][h]) for k in range(16)) + int(b1[0][h]) for h in range(8)]
        hidden = [value*value for value in affine]
        logits = [sum(hidden[h]*int(w2[h][j]) for h in range(8)) + int(b2[0][j]) for j in range(8)]
        absolute = [sum(hidden[h]*abs(int(w2[h][j])) for h in range(8)) + abs(int(b2[0][j])) for j in range(8)]
        field_first = [(sum((int(row[k]) % PRIME)*(int(w1[k][h]) % PRIME) for k in range(16))
                        + int(b1[0][h])) % PRIME for h in range(8)]
        field = [(sum((field_first[h]*field_first[h] % PRIME)*(int(w2[h][j]) % PRIME)
                      for h in range(8)) + int(b2[0][j])) % PRIME for j in range(8)]
        signed = [value - PRIME if value > PRIME//2 else value for value in field]
        first.append(affine)
        outputs.append(logits)
        bounds.append(absolute)
        field_outputs.append(signed)
    return tuple(torch.tensor(value, dtype=torch.int64) for value in (first, outputs, bounds, field_outputs))


def test_checkpoint_predicts_all_64_contexts_in_float_and_integer(model):
    contexts, inputs, targets = dataset()
    fixture = compact.quantized_fixture(model)
    report = compact.numerical_report(model, fixture)
    assert contexts.shape == (64, 2) and torch.unique(contexts, dim=0).shape[0] == 64
    assert torch.equal(targets, contexts.sum(1).remainder(8))
    assert torch.equal(fixture['encoded'][0], inputs.long())
    assert torch.equal(fixture['targets'], targets)
    assert torch.equal(inputs[:, :8].sum(1), torch.ones(64))
    assert torch.equal(inputs[:, 8:].sum(1), torch.ones(64))
    with torch.inference_mode():
        floating = model(inputs)
    assert torch.equal(floating.argmax(1), targets)
    assert torch.equal(fixture['expected'].argmax(1), targets)
    assert report['float_accuracy'] == report['integer_accuracy'] == 1.
    assert report['architecture'] == [16, 8, 8]
    assert report['evaluation_rows'] == report['unique_contexts'] == 64
    assert 'training contexts' in report['evaluation']
    assert report['max_logit_error'] < .05 and report['max_logprob_error'] < .07
    assert model[0].in_features == 16 and model[0].out_features == 8
    assert model[2].in_features == model[2].out_features == 8


def test_python_integer_field_and_absolute_sum_references_match(model):
    fixture = compact.quantized_fixture(model)
    first, expected, absolute, field_expected = _python_reference(fixture['encoded'])
    proof = fixture['proof']
    assert torch.equal(expected, fixture['expected'])
    assert torch.equal(field_expected, expected)
    assert bool((expected.abs() <= absolute).all())
    assert proof['first_affine_abs_bound'] == int(first.abs().max())
    assert proof['square_abs_bound'] == int(first.abs().max())**2 < PRIME//2
    assert proof['output_abs_bound'] == int(absolute.max()) < PRIME//2
    assert proof['actual_output_max_abs'] == int(expected.abs().max())
    assert proof['field_prime'] == PRIME and proof['signed_limit'] == PRIME//2
    assert proof['no_field_wrap'] and proof['no_private_truncation']
    assert bool((first.amax(0) != first.amin(0)).all())


def test_scale_contract_matches_quantized_float_function_without_truncation(model):
    fixture = compact.quantized_fixture(model)
    x, w1, b1, w2, b2 = fixture['encoded']
    proof, scale = fixture['proof'], 256
    assert [proof[name] for name in ('input_scale', 'first_affine_scale', 'square_scale', 'output_scale')] == [1, scale, scale**2, scale**3]
    assert torch.equal(w1, (model[0].weight.detach().T.double()*scale).round().long())
    assert torch.equal(b1, (model[0].bias.detach()[None, :].double()*scale).round().long())
    assert torch.equal(w2, (model[2].weight.detach().T.double()*scale).round().long())
    assert torch.equal(b2, (model[2].bias.detach()[None, :].double()*scale**3).round().long())
    decoded = (x.double() @ (w1.double()/scale) + b1.double()/scale).square()
    decoded = decoded @ (w2.double()/scale) + b2.double()/scale**3
    assert torch.equal(decoded, fixture['expected'].double()/scale**3)


@pytest.mark.parametrize('batch', [1, 65])
def test_manifest_and_repeated_contexts_use_compact_dimensions(model, batch):
    fixture = compact.quantized_fixture(model, batch=batch)
    complete = compact.quantized_fixture(model)
    indices = torch.arange(batch).remainder(64)
    assert torch.equal(fixture['expected'], complete['expected'][indices])
    assert torch.equal(fixture['targets'], complete['targets'][indices])
    assert fixture['proof']['output_abs_bound'] == complete['proof']['output_abs_bound']
    manifest = compact.preprocessing_manifest(batch)
    assert manifest['matrices'] == [dict(name='first', m=batch, k=16, n=8),
                                    dict(name='second', m=batch, k=8, n=8)]
    assert manifest['squares'] == [dict(name='hidden', shape=[batch, 8])]
    assert [(entry['name'], entry['owner'], entry['shape']) for entry in manifest['inputs']] == [
        ('x', 0, [batch, 16]), ('w1', 1, [16, 8]), ('b1', 1, [1, 8]),
        ('w2', 1, [8, 8]), ('b2', 1, [1, 8])]
    assert [list(value.shape) for value in fixture['encoded']] == [entry['shape'] for entry in manifest['inputs']]


def test_range_guard_covers_unselected_contexts_even_for_decode_batch(model):
    unsafe = copy.deepcopy(model)
    with torch.no_grad():
        for parameter in unsafe.parameters():
            parameter.zero_()
        # Context (0,0) does not select feature7; contexts starting with token7
        # would make this square exceed the signed field range at scale256.
        unsafe[0].weight[0, 7] = 128.
    with pytest.raises(ValueError, match='square exceeds'):
        compact.quantized_fixture(unsafe, batch=1)


def test_range_guard_uses_absolute_contributions_despite_output_cancellation(model):
    unsafe = copy.deepcopy(model)
    with torch.no_grad():
        for parameter in unsafe.parameters():
            parameter.zero_()
        unsafe[0].bias.fill_(1.)
        unsafe[2].weight[:, ::2] = float(PRIME//8)
        unsafe[2].weight[:, 1::2] = -float(PRIME//8)
    # At scale1 the eight hidden values are1 and the signed weights cancel.
    # Their absolute sum still exceeds p/2, so the fixture must reject them.
    assert int(unsafe[2].weight[0].double().round().long().sum()) == 0
    with pytest.raises(ValueError, match='output exceeds'):
        compact.quantized_fixture(unsafe, fractional_bits=0)


def test_nonfinite_parameters_are_rejected_before_integer_conversion(model):
    unsafe = copy.deepcopy(model)
    with torch.no_grad():
        unsafe[0].weight[0, 0] = float('inf')
    with pytest.raises(ValueError, match='parameter exceeds'):
        compact.quantized_fixture(unsafe)


def test_native_clear_baseline_executes_the_same_compact_trained_model(model, monkeypatch):
    from benchmarks import spdz_native_replay
    calls = []
    def measure(compute, device, repeats, warmups):
        assert torch.device(device).type == 'cpu'
        value = compute()
        calls.append(torch.as_tensor(np.asarray(value)).clone() if not isinstance(value, torch.Tensor) else value.clone())
        return dict(median_ms=1., minimum_ms=1., maximum_ms=1., samples_ms=[1.])
    monkeypatch.setattr(spdz_native_replay, 'wall_measurement', measure)
    fixture = compact.quantized_fixture(model)
    report = spdz_native_replay.trained_baselines(model, fixture, 'cpu', repeats=1)
    with torch.inference_mode():
        expected = model(fixture['encoded'][0].float())
    assert len(calls) == 2 and all(value.shape == (64, 8) for value in calls)
    assert torch.equal(calls[0], expected)
    torch.testing.assert_close(calls[1], expected, rtol=1e-5, atol=1e-5)
    assert report['accuracy']['fp32']['max_logit_error'] == 0.
    assert 'Linear(16,8), square, Linear(8,8)' in report['model']
