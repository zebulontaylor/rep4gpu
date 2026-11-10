"""Compile the public embedding into the trained dense model's first layer.

The private input becomes two one-hot token vectors instead of 512 embedding
features. Fold the already quantized integer weights to preserve every integer
logit exactly. Clear timings use the corresponding folded floating model.
The original trained dense checkpoint is loaded without modification.
"""
import argparse
import copy
import json
from pathlib import Path

import torch
from torch import nn

from examples import spdz_trained_dense_toy as dense
from examples.spdz_trained_toy import Square, infer_authenticated
from rep4gpu.spdz import PRIME


train = dense.train


def _one_hot_inputs(table, batch):
    contexts, _, targets = dense.dataset(table, batch)
    inputs = nn.functional.one_hot(contexts, num_classes=8).reshape(batch, 16).long()
    return inputs, targets


def quantized_fixture(model, fractional_bits=8, batch=64):
    """Return the exact dense-model integer function in a smaller first layer."""
    original = dense.quantized_fixture(model, fractional_bits, batch)
    table = model.embedding_table.cpu()
    dimension = table.shape[1]
    _, w1, b1, w2, b2 = original['encoded']
    folded = torch.cat((table @ w1[:dimension], table @ w1[dimension:]), dim=0)
    if int(folded.abs().max()) >= PRIME//2:
        raise ValueError('folded first-layer weight exceeds the signed field range')
    inputs, targets = _one_hot_inputs(table, batch)
    expected = (inputs @ folded+b1).square() @ w2+b2
    if not torch.equal(expected, original['expected']):
        raise AssertionError('public embedding fold changed the integer function')
    proof = dict(original['proof'])
    proof['contract'] = 'Two one-hot 8-token inputs; exact public-embedding compilation; no private domain check.'
    proof['exact_integer_embedding_fold'] = True
    proof['folded_weight_max_abs'] = int(folded.abs().max())
    return dict(encoded=[inputs, folded, b1, w2, b2], expected=expected,
                targets=targets, proof=proof,
                active_hidden_units=original['active_hidden_units'])


def baseline_model(model):
    """Compile the same public embedding into the clear floating first layer."""
    device, dtype = model[0].weight.device, model[0].weight.dtype
    table = model.embedding_table.to(device=device, dtype=dtype)
    dimension = table.shape[1]
    hidden = model[0].out_features
    result = nn.Sequential(nn.Linear(16, hidden, device=device, dtype=dtype),
                           Square(), copy.deepcopy(model[2]))
    with torch.no_grad():
        weight = model[0].weight.T
        folded = torch.cat((table @ weight[:dimension], table @ weight[dimension:]), dim=0)
        result[0].weight.copy_(folded.T)
        result[0].bias.copy_(model[0].bias)
    return result.eval()


def numerical_report(model, fixture):
    inputs = fixture['encoded'][0]
    baseline = baseline_model(model)
    with torch.no_grad():
        floating = baseline(inputs.to(device=baseline[0].weight.device,
                                      dtype=baseline[0].weight.dtype)).cpu().double()
        all_contexts, dense_inputs, _ = dense.dataset(model.embedding_table.cpu())
        one_hot = nn.functional.one_hot(all_contexts, num_classes=8).reshape(64, 16)
        original_float = model(dense_inputs.to(device=model[0].weight.device,
                                               dtype=model[0].weight.dtype)).cpu().double()
        folded_float = baseline(one_hot.to(device=baseline[0].weight.device,
                                           dtype=baseline[0].weight.dtype)).cpu().double()
    decoded = fixture['expected'].double()/fixture['proof']['output_scale']
    return dict(task='synthetic two-token sum modulo 8',
                logical_dense_architecture=[model[0].in_features, model[0].out_features, model[2].out_features],
                evaluated_architecture=[16, model[0].out_features, model[2].out_features],
                optimization='Public embedding multiplied into the first weight matrix at model preparation.',
                comparator='Same folded Linear(16,H), square, Linear(H,8) floating model; no dense-model denominator.',
                integer_equivalence='Folded already-quantized weights preserve every dense integer logit exactly.',
                evaluation_rows=inputs.shape[0], unique_contexts=min(inputs.shape[0], 64),
                evaluation='All 64 contexts are training contexts; no held-out accuracy claim.',
                float_accuracy=float((floating.argmax(-1)==fixture['targets']).float().mean()),
                integer_accuracy=float((fixture['expected'].argmax(-1)==fixture['targets']).float().mean()),
                max_logit_error=float((decoded-floating).abs().max()),
                max_logprob_error=float((decoded.log_softmax(-1)-floating.log_softmax(-1)).abs().max()),
                float_fold_max_logit_error=float((folded_float-original_float).abs().max()),
                active_hidden_units=fixture['active_hidden_units'], range_proof=fixture['proof'])


def preprocessing_manifest(model, batch=1):
    result = dense.preprocessing_manifest(model, batch)
    result['matrices'][0]['k'] = 16
    result['inputs'][0]['shape'][1] = 16
    result['inputs'][1]['shape'][0] = 16
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=Path('results/spdz_trained_dense_toy/model.pt'))
    parser.add_argument('--output', type=Path, default=Path('results/spdz_folded_dense_toy'))
    parser.add_argument('--batch', type=int, default=1)
    args = parser.parse_args()
    model = train()
    model.load_state_dict(torch.load(args.checkpoint, map_location='cpu', weights_only=True))
    fixture = quantized_fixture(model, batch=64)
    report = numerical_report(model, fixture)
    report['private_batch'] = args.batch
    report['original_checkpoint'] = str(args.checkpoint)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'metadata.json').write_text(json.dumps(report, indent=2)+'\n')
    (args.output/'preprocessing.json').write_text(json.dumps(preprocessing_manifest(model, args.batch), indent=2)+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
