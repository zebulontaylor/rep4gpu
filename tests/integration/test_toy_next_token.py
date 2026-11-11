"""Check trained synthetic predictions and exact four-party evaluation."""
import pytest
import torch
from experiments.examples.toy_next_token import dataset,evaluate_private,quantized_fixture,train


@pytest.fixture(scope='module')
def model():
    return train()


def test_all_context_training_accuracy(model):
    contexts,inputs,targets=dataset()
    assert contexts.shape==(64,2)
    assert torch.equal(targets,contexts.sum(-1)%8)
    assert (model(inputs).argmax(-1)==targets).all()


def test_private_fixed_point_logits(model):
    report=evaluate_private(model)
    assert report['exact_integer_match']
    assert report['float_next_token_accuracy']==1
    assert report['fixed_next_token_accuracy']==1
    assert report['range_proof']['all_fixture_values_fit_signed_ring']
    assert report['max_logprob_error'] < .1


def test_oversized_quantization_rejected(model):
    with pytest.raises(ValueError,match='exceeds signed ring32'):
        quantized_fixture(model,fractional_bits=20,bits=32)
