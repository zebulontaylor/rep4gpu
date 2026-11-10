"""Numerical bounds and authenticated composition for the trained square model."""
import copy
import os

import pytest
import torch

from examples.spdz_trained_toy import (evaluate_authenticated, numerical_report,
                                      quantized_fixture, train)
from examples.toy_next_token import dataset
from rep4gpu.spdz import PRIME, TrustedDealer
from rep4gpu.spdz_distributed import DistributedParty, split_triple
from rep4gpu.spdz_inputs import make_input_masks
from rep4gpu.spdz_polynomial import make_square_masks
from rep4gpu.transport import IdealNetwork


@pytest.fixture(scope='module')
def trained():
    return train()


@pytest.mark.parametrize('fractional_bits', [4, 6, 8])
def test_trained_integer_model_matches_float_predictions_with_proven_bounds(trained, fractional_bits):
    fixture = quantized_fixture(trained, fractional_bits)
    report = numerical_report(trained, fixture)
    assert report['float_accuracy'] == report['integer_accuracy'] == 1.
    assert report['range_proof']['output_abs_bound'] < PRIME//2
    x, w1, b1, w2, b2 = fixture['encoded']
    expected = (x @ w1 + b1).square() @ w2 + b2
    assert torch.equal(fixture['expected'], expected)
    assert int(expected.abs().max()) <= report['range_proof']['output_abs_bound']
    if fractional_bits == 8:
        assert report['max_logit_error'] < .03


def test_model_with_unsafe_square_or_output_bound_is_rejected(trained):
    unsafe = copy.deepcopy(trained)
    with torch.no_grad():
        unsafe[0].weight.fill_(10000.)
    with pytest.raises(ValueError, match='square bound'):
        quantized_fixture(unsafe)
    unsafe = copy.deepcopy(trained)
    with torch.no_grad():
        unsafe[2].weight.fill_(10000.)
    with pytest.raises(ValueError, match='second affine bound'):
        quantized_fixture(unsafe)


def test_q8_matches_float_model_with_identically_quantized_parameters(trained):
    fixture = quantized_fixture(trained)
    x, w1, b1, w2, b2 = fixture['encoded']
    scale = 256
    # Float64 is exact for these bounded integer products and powers of two.
    quantized_logits = ((x.double() @ (w1.double()/scale) + b1.double()/scale).square()
                        @ (w2.double()/scale) + b2.double()/scale**3)
    assert torch.equal(quantized_logits, fixture['expected'].double()/scale**3)


def test_authenticated_model_matches_integer_oracle_with_test_only_correlations(trained):
    # This fixture tests online algebra. Deployment preprocessing is supplied
    # externally; no trusted dealer is constructed by the example itself.
    fixture = quantized_fixture(trained)
    dealer = TrustedDealer(mac_keys=10)
    encoded = fixture['encoded']
    batch = encoded[0].shape[0]
    material = dict(
        input_masks=[make_input_masks(dealer, owner, tuple(value.shape))
                     for owner, value in zip([0, 1, 1, 1, 1], encoded)],
        matrix_triples=[split_triple(dealer.triple(batch, 16, 64)),
                        split_triple(dealer.triple(batch, 64, 8))],
        square_masks=make_square_masks(dealer, (batch, 64)))
    network = IdealNetwork(timeout=30)
    session = os.urandom(32)
    parties = [DistributedParty(p, network.endpoint(p), dealer.alpha_shares[:, p].clone(),
                                session=session, _allow_insecure_test_transport=True)
               for p in range(4)]
    result = evaluate_authenticated(fixture, parties, material)
    assert result['exact_integer_match'] and result['integer_accuracy'] == 1.
    assert all(mask.used for masks in material['input_masks'] for mask in masks)
    assert all(triple.used for triples in material['matrix_triples'] for triple in triples)
    assert all(mask.used for mask in material['square_masks'])


def test_small_native_batch_preserves_all_context_bound_and_manifest(trained):
    from examples.spdz_trained_toy import preprocessing_manifest
    complete = quantized_fixture(trained)
    small = quantized_fixture(trained, batch=4)
    assert torch.equal(small['expected'], complete['expected'][:4])
    assert small['proof']['output_abs_bound'] == complete['proof']['output_abs_bound']
    manifest = preprocessing_manifest(4)
    assert [entry['shape'] for entry in manifest['inputs']] == [list(x.shape) for x in small['encoded']]
    assert manifest['matrices'] == [dict(name='first', m=4, k=16, n=64),
                                    dict(name='second', m=4, k=64, n=8)]


@pytest.mark.parametrize('batch', [128, 129])
def test_larger_batches_repeat_existing_contexts_without_changing_bounds(trained, batch):
    from examples.spdz_trained_toy import preprocessing_manifest
    complete = quantized_fixture(trained)
    repeated = quantized_fixture(trained, batch=batch)
    rows = torch.arange(batch) % 64
    assert torch.equal(repeated['expected'], complete['expected'][rows])
    assert torch.equal(repeated['targets'], complete['targets'][rows])
    assert repeated['proof']['output_abs_bound'] == complete['proof']['output_abs_bound']
    report = numerical_report(trained, repeated)
    assert report['unique_evaluation_contexts'] == 64
    assert report['integer_accuracy'] == 1.
    assert 'cycle' in report['evaluation']
    assert preprocessing_manifest(batch)['matrices'][0]['m'] == batch
