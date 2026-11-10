"""Dense trained model fidelity, active features, and full-domain field bounds."""
import copy

import pytest
import torch

from examples.spdz_trained_dense_toy import (dataset, numerical_report,
                                             preprocessing_manifest, quantized_fixture, train)
from rep4gpu.spdz import PRIME


@pytest.fixture(scope='module')
def model():
    return train()


def test_all_features_vary_and_all_quantized_hidden_units_are_active(model):
    _, inputs, _ = dataset(model.embedding_table)
    assert inputs.shape == (64, 512)
    assert bool((inputs.abs() == 1).all())
    assert not bool((inputs == inputs[0]).all(dim=0).any())
    fixture = quantized_fixture(model)
    report = numerical_report(model, fixture)
    assert report['float_accuracy'] == report['integer_accuracy'] == 1.
    assert report['active_hidden_units'] == 2048
    assert all(value > .8 for value in report['quantized_weight_nonzero_fraction'])
    assert report['max_logit_error'] < .25
    assert report['max_logprob_error'] < .35


def test_full_domain_absolute_sum_bound_and_quantized_float_equivalence(model):
    fixture = quantized_fixture(model)
    x, w1, b1, w2, b2 = fixture['encoded']
    first = x @ w1+b1
    row_bounds = first.square() @ w2.abs()+b2.abs()
    assert int(row_bounds.max()) == fixture['proof']['output_abs_bound'] < PRIME//2
    assert bool((fixture['expected'].abs() <= row_bounds).all())
    scale = 256
    with torch.no_grad():
        quantized = (x.double() @ (w1.double()/scale)+b1.double()/scale).square()
        quantized = quantized @ (w2.double()/scale)+b2.double()/scale**3
    assert torch.equal(quantized, fixture['expected'].double()/scale**3)


def test_no_constant_embedding_padding_and_no_unsafe_model_accepted(model):
    bad = model.embedding_table.clone()
    bad[:, 0] = 1
    with pytest.raises(ValueError, match='constant feature'):
        dataset(bad)
    unsafe = copy.deepcopy(model)
    with torch.no_grad():
        unsafe[0].weight.fill_(10000.)
    with pytest.raises(ValueError, match='square activation'):
        quantized_fixture(unsafe)


def test_native_decode_manifest_and_repeated_context_semantics(model):
    manifest = preprocessing_manifest(model, batch=1)
    assert manifest['matrices'] == [dict(name='first', m=1, k=512, n=2048),
                                    dict(name='second', m=1, k=2048, n=8)]
    assert manifest['inputs'][1]['shape'] == [512, 2048]
    complete = quantized_fixture(model)
    repeated = quantized_fixture(model, batch=65)
    assert torch.equal(repeated['expected'][:64], complete['expected'])
    assert torch.equal(repeated['expected'][64], complete['expected'][0])
