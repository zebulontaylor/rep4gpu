"""Train a dense-embedding square MLP on the same 64 two-token contexts.

A fixed public signed embedding maps each token to dense features. The input
owner performs this lookup before either clear or private MLP evaluation. Every
embedding feature varies across tokens; input dimensions are not zero padding.
All contexts are used for training and evaluation, with no held-out claim.
"""
import argparse
import json
from pathlib import Path

import torch
from torch import nn

from examples.spdz_trained_toy import Square, infer_authenticated
from rep4gpu.spdz import PRIME


def embedding_table(dimension=256):
    if not isinstance(dimension, int) or dimension < 1:
        raise ValueError('embedding dimension must be positive')
    generator = torch.Generator().manual_seed(20261111)
    table = torch.randint(0, 2, (8, dimension), generator=generator)*2-1
    constant = (table == table[0]).all(dim=0)
    while bool(constant.any()):
        table[:, constant] = torch.randint(0, 2, (8, int(constant.sum())), generator=generator)*2-1
        constant = (table == table[0]).all(dim=0)
    return table


def dataset(table, batch=64):
    if not isinstance(batch, int) or batch < 1:
        raise ValueError('batch must be positive')
    if table.dtype != torch.int64 or table.ndim != 2 or table.shape[0] != 8 or not bool((table.abs()==1).all()):
        raise ValueError('public embedding must be an int64 matrix of eight signed rows')
    if bool((table == table[0]).all(dim=0).any()):
        raise ValueError('public embedding must not contain constant feature columns')
    contexts = torch.cartesian_prod(torch.arange(8), torch.arange(8))
    contexts = contexts[torch.arange(batch) % 64]
    inputs = table[contexts].reshape(batch, -1)
    return contexts, inputs, contexts.sum(dim=1) % 8


def train(steps=300, *, hidden=2048, dimension=256, learning_rate=.001, weight_decay=0., dropout=.1, target_loss=.02):
    """Train dense weights from scratch on one CPU thread."""
    if not isinstance(hidden, int) or hidden < 1 or not isinstance(steps, int) or steps < 1:
        raise ValueError('hidden width and training steps must be positive')
    torch.set_num_threads(1)
    torch.manual_seed(20261112)
    table = embedding_table(dimension)
    _, inputs, targets = dataset(table)
    model = nn.Sequential(nn.Linear(2*dimension, hidden), Square(), nn.Linear(hidden, 8))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    completed_steps = 0
    for step in range(steps):
        optimizer.zero_grad()
        hidden_values = model[1](model[0](inputs.float()))
        logits = model[2](nn.functional.dropout(hidden_values, p=dropout, training=True))
        loss = nn.functional.cross_entropy(logits, targets)
        loss.backward()
        optimizer.step()
        completed_steps = step+1
        if completed_steps % 5 == 0:
            with torch.no_grad():
                evaluation_loss = nn.functional.cross_entropy(model(inputs.float()), targets)
            if float(evaluation_loss) < target_loss:
                break
    model.training_steps = completed_steps
    model.register_buffer('embedding_table', table)
    return model.eval()


def quantized_fixture(model, fractional_bits=8, batch=64):
    """Bound the full 64-context domain while deferring public scale changes."""
    if not isinstance(fractional_bits, int) or not 0 <= fractional_bits <= 8:
        raise ValueError('fractional_bits must be in [0,8]')
    table = model.embedding_table.cpu()
    _, all_inputs, _ = dataset(table)
    _, inputs, targets = dataset(table, batch)
    scale = 1 << fractional_bits
    values = [model[0].weight.detach().T, model[0].bias.detach()[None, :],
              model[2].weight.detach().T, model[2].bias.detach()[None, :]]
    encoded = []
    for value, factor in zip(values, [scale, scale, scale, scale**3]):
        rounded = torch.round(value.double()*factor)
        if not bool((torch.isfinite(rounded) & (rounded.abs() < PRIME//2)).all()):
            raise ValueError('encoded parameter exceeds the signed field range')
        encoded.append(rounded.long())
    w1, b1, w2, b2 = encoded
    # Inputs are signs and parameters are field-bounded, so this first affine
    # accumulation fits int64 for the supported input dimensions.
    if all_inputs.shape[1]*int(w1.abs().max()) + int(b1.abs().max()) >= 2**63:
        raise ValueError('first affine reference can overflow int64')
    all_first = all_inputs @ w1 + b1
    first_bound = all_first.abs().amax(dim=0)
    maximum_first = int(first_bound.max())
    if maximum_first**2 >= PRIME//2:
        raise ValueError('square activation can exceed the signed field range')
    output_bounds = [sum(int(first_bound[h])**2*abs(int(w2[h, j]))
                         for h in range(w2.shape[0])) + abs(int(b2[0, j]))
                     for j in range(w2.shape[1])]
    if max(output_bounds) >= 2**63:
        raise ValueError('output absolute-sum reference can overflow int64')
    # Maxima for different hidden units need not occur in the same context.
    # Bound each valid context's sum of absolute terms before taking its max.
    context_bounds = all_first.square() @ w2.abs() + b2.abs()
    maximum_output = int(context_bounds.max())
    if maximum_output >= PRIME//2:
        raise ValueError('output absolute bound exceeds the signed field range')
    first = inputs @ w1 + b1
    expected = first.square() @ w2 + b2
    active = (w2 != 0).any(dim=1) & (all_first.square().amax(dim=0) != all_first.square().amin(dim=0))
    proof = dict(field_prime=PRIME, signed_limit=PRIME//2,
                 weight_fractional_bits=fractional_bits, input_scale=1,
                 first_affine_scale=scale, square_scale=scale**2, output_scale=scale**3,
                 first_affine_abs_bound=maximum_first, square_abs_bound=maximum_first**2,
                 output_abs_bound=maximum_output, actual_output_max_abs=int(expected.abs().max()),
                 output_bound_method='Maximum sum of absolute output terms across the 64 valid contexts',
                 no_field_wrap=True, no_private_truncation=True,
                 contract='Public fixed embedding table and all 64 valid contexts; no private input-domain check.')
    return dict(encoded=[inputs, w1, b1, w2, b2], expected=expected,
                targets=targets, proof=proof, active_hidden_units=int(active.sum()))


def numerical_report(model, fixture):
    inputs = fixture['encoded'][0]
    with torch.no_grad():
        floating = model(inputs.float()).double()
    decoded = fixture['expected'].double()/fixture['proof']['output_scale']
    w1, w2 = fixture['encoded'][1], fixture['encoded'][3]
    return dict(task='synthetic two-token sum modulo 8',
                architecture=[inputs.shape[1], w1.shape[1], w2.shape[1]],
                embedding='Fixed public dense signed code; owner-local lookup outside both MLP timers.',
                evaluation_rows=inputs.shape[0], unique_contexts=min(inputs.shape[0], 64),
                evaluation='All 64 contexts are also used for training; no held-out accuracy claim.',
                float_accuracy=float((floating.argmax(-1)==fixture['targets']).float().mean()),
                integer_accuracy=float((fixture['expected'].argmax(-1)==fixture['targets']).float().mean()),
                max_logit_error=float((decoded-floating).abs().max()),
                max_logprob_error=float((decoded.log_softmax(-1)-floating.log_softmax(-1)).abs().max()),
                input_nonzero_fraction=float((inputs != 0).float().mean()),
                active_hidden_units=fixture['active_hidden_units'],
                quantized_weight_nonzero_fraction=[float((w != 0).float().mean()) for w in (w1, w2)],
                range_proof=fixture['proof'])


def preprocessing_manifest(model, batch=1):
    inputs, hidden, outputs = model[0].in_features, model[0].out_features, model[2].out_features
    return dict(matrices=[dict(name='first', m=batch, k=inputs, n=hidden),
                          dict(name='second', m=batch, k=hidden, n=outputs)],
                squares=[dict(name='hidden', shape=[batch, hidden])],
                inputs=[dict(name=name, owner=owner, shape=list(shape))
                        for name, owner, shape in [('x', 0, (batch, inputs)),
                                                  ('w1', 1, (inputs, hidden)), ('b1', 1, (1, hidden)),
                                                  ('w2', 1, (hidden, outputs)), ('b2', 1, (1, outputs))]])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hidden', type=int, default=2048)
    parser.add_argument('--dimension', type=int, default=256)
    parser.add_argument('--steps', type=int, default=300)
    parser.add_argument('--fractional-bits', type=int, default=8)
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--output', type=Path, default=Path('results/spdz_trained_dense_toy'))
    args = parser.parse_args()
    model = train(args.steps, hidden=args.hidden, dimension=args.dimension)
    fixture = quantized_fixture(model, args.fractional_bits, args.batch)
    report = numerical_report(model, quantized_fixture(model, args.fractional_bits, 64))
    report['private_batch'] = args.batch
    report['training'] = dict(embedding_seed=20261111, model_seed=20261112, max_steps=args.steps,
                              completed_steps=model.training_steps, optimizer='Adam', learning_rate=.001,
                              weight_decay=0., dropout=.1, early_stop_loss=.02, check_every_steps=5,
                              cpu_threads=1)
    args.output.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.output/'model.pt')
    (args.output/'metadata.json').write_text(json.dumps(report, indent=2)+'\n')
    (args.output/'preprocessing.json').write_text(json.dumps(preprocessing_manifest(model, args.batch), indent=2)+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
