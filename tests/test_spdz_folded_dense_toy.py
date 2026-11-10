"""Public embedding compilation must preserve the trained integer function."""
import copy

import pytest
import torch

from examples import spdz_folded_dense_toy as folded
from examples import spdz_trained_dense_toy as dense
from rep4gpu.spdz import PRIME


@pytest.fixture(scope='module')
def model():
    return folded.train()


def test_all_context_integer_logits_and_first_affine_are_identical(model):
    original = dense.quantized_fixture(model)
    result = folded.quantized_fixture(model)
    assert torch.equal(result['expected'], original['expected'])
    x, w1, b1, _, _ = original['encoded']
    fx, fw1, fb1, _, _ = result['encoded']
    assert fx.shape == (64, 16) and fw1.shape == (16, 2048)
    assert torch.equal(x @ w1+b1, fx @ fw1+fb1)
    assert result['proof']['output_abs_bound'] == original['proof']['output_abs_bound'] < PRIME//2
    assert result['proof']['exact_integer_embedding_fold']


def test_float_fold_matches_original_function_and_is_the_clear_baseline(model):
    source = copy.deepcopy(model).double()
    baseline = folded.baseline_model(source)
    contexts, dense_inputs, _ = dense.dataset(source.embedding_table)
    one_hot = torch.nn.functional.one_hot(contexts, num_classes=8).reshape(64, 16).double()
    with torch.no_grad():
        expected = source(dense_inputs.double())
        actual = baseline(one_hot)
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)
    assert baseline[0].in_features == 16 and baseline[0].out_features == 2048
    assert baseline[2].in_features == 2048 and baseline[2].out_features == 8


def test_quantization_precedes_folding_and_source_checkpoint_is_unchanged(model):
    before = {key:value.clone() for key,value in model.state_dict().items()}
    result = folded.quantized_fixture(model)
    dense_weights = dense.quantized_fixture(model)['encoded'][1]
    table = model.embedding_table
    expected = torch.cat((table @ dense_weights[:256], table @ dense_weights[256:]))
    assert torch.equal(result['encoded'][1], expected)
    rounded_after_fold = (folded.baseline_model(model)[0].weight.detach().T*256).round().long()
    assert not torch.equal(expected, rounded_after_fold)
    assert all(torch.equal(value, before[key]) for key,value in model.state_dict().items())


def test_native_manifest_and_full_domain_accuracy(model):
    manifest = folded.preprocessing_manifest(model, batch=1)
    assert manifest['matrices'] == [dict(name='first', m=1, k=16, n=2048),
                                    dict(name='second', m=1, k=2048, n=8)]
    assert sum(x['m']*x['k']*x['n'] for x in manifest['matrices']) == 49152
    result = folded.numerical_report(model, folded.quantized_fixture(model))
    assert result['float_accuracy'] == result['integer_accuracy'] == 1.
    assert result['float_fold_max_logit_error'] < 1e-4
    assert result['active_hidden_units'] == 2048
