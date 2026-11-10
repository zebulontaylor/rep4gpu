"""Train a square-activation next-token toy without private rescaling.

The dataset is the same 64 two-token contexts as ``examples.toy_next_token``.
Training and evaluation use all contexts; accuracy is not held-out accuracy.
Run ``python -m examples.spdz_trained_toy`` to train and check the integer
fixture. Authenticated execution accepts externally generated preprocessing;
this module does not generate trusted-dealer correlations.
"""
from concurrent.futures import ThreadPoolExecutor
import argparse
import json
from pathlib import Path

import torch
from torch import nn

from examples.toy_next_token import dataset
from rep4gpu.spdz import PRIME
from rep4gpu.spdz_distributed import AuthShare
from rep4gpu.spdz_inputs import private_input
from rep4gpu.spdz_polynomial import square


class Square(nn.Module):
    def forward(self, value):
        return value.square()


def train(steps=800):
    """Train the floating square MLP deterministically on one CPU thread."""
    if not isinstance(steps, int) or steps < 1:
        raise ValueError('training steps must be a positive integer')
    torch.set_num_threads(1)
    torch.manual_seed(20261110)
    model = nn.Sequential(nn.Linear(16, 64), Square(), nn.Linear(64, 8))
    _, inputs, targets = dataset()
    optimizer = torch.optim.Adam(model.parameters(), lr=.02, weight_decay=.001)
    for _ in range(steps):
        optimizer.zero_grad()
        loss = nn.functional.cross_entropy(model(inputs), targets)
        loss.backward()
        optimizer.step()
    return model.eval()


def quantized_fixture(model, fractional_bits=8, batch=64):
    """Encode weights while deferring every scale change to public decoding.

    One-hot inputs have scale one. The first affine output has scale S, its
    square S**2, and final logits S**3, where S=2**fractional_bits. Batches
    larger than 64 repeat the training contexts; they add no new examples.
    """
    if not isinstance(fractional_bits, int) or not 0 <= fractional_bits <= 8:
        raise ValueError('fractional_bits must be an integer in [0,8]')
    if not isinstance(batch, int) or batch < 1:
        raise ValueError('batch must be a positive integer')
    _, inputs, targets = dataset()
    rows = torch.arange(batch) % inputs.shape[0]
    inputs, targets = inputs[rows], targets[rows]
    scale = 1 << fractional_bits
    tensors = [inputs, model[0].weight.detach().T, model[0].bias.detach()[None, :],
               model[2].weight.detach().T, model[2].bias.detach()[None, :]]
    scales = [1, scale, scale, scale, scale**3]
    encoded = []
    for value, factor in zip(tensors, scales):
        rounded = torch.round(value.double()*factor)
        if not bool((torch.isfinite(rounded) & (rounded.abs() < PRIME//2)).all()):
            raise ValueError('encoded model value exceeds the signed prime-field range')
        encoded.append(rounded.long())
    x, w1, b1, w2, b2 = encoded
    # Exactly one token is selected from each eight-feature group. This bound
    # covers every possible context, rather than only observed predictions.
    first_bound = w1[:8].abs().amax(0) + w1[8:].abs().amax(0) + b1[0].abs()
    max_first = int(first_bound.max())
    if max_first**2 >= PRIME//2:
        raise ValueError('first affine or square bound exceeds the signed field range')
    hidden_bound = first_bound.square()
    output_bounds = [sum(int(hidden_bound[h])*abs(int(w2[h, j]))
                         for h in range(w2.shape[0])) + abs(int(b2[0, j]))
                     for j in range(w2.shape[1])]
    max_output = max(output_bounds)
    if max_output >= PRIME//2:
        raise ValueError('second affine bound exceeds the signed field range')
    affine = x @ w1 + b1
    hidden = affine.square()
    expected = hidden @ w2 + b2
    proof = dict(field_prime=PRIME, signed_limit=PRIME//2,
                 weight_fractional_bits=fractional_bits, input_scale=1,
                 first_affine_scale=scale, square_scale=scale**2, output_scale=scale**3,
                 first_affine_abs_bound=max_first, square_abs_bound=max_first**2,
                 output_abs_bound=max_output, actual_output_max_abs=int(expected.abs().max()),
                 no_field_wrap=True, no_private_truncation=True,
                 contract='Public model bound over all two-one-hot inputs; not a private range check.')
    return dict(encoded=encoded, expected=expected, targets=targets, proof=proof)


def numerical_report(model, fixture):
    inputs, targets = fixture['encoded'][0].float(), fixture['targets']
    with torch.no_grad():
        floating = model(inputs).double()
    integer = fixture['expected']
    decoded = integer.double()/fixture['proof']['output_scale']
    return dict(task='synthetic target=(previous_token_1+previous_token_2)%8',
                model='Linear(16,64), square, Linear(64,8); not a transformer',
                evaluation=f'{len(targets)} input rows cycle through the 64 training contexts; no held-out claim.',
                unique_evaluation_contexts=min(len(targets), 64),
                float_accuracy=float((floating.argmax(-1)==targets).float().mean()),
                integer_accuracy=float((integer.argmax(-1)==targets).float().mean()),
                max_logit_error=float((decoded-floating).abs().max()),
                max_logprob_error=float((decoded.log_softmax(-1)-floating.log_softmax(-1)).abs().max()),
                range_proof=fixture['proof'])


def _broadcast_rows(share, batch):
    if share.values.ndim != 2 or share.values.shape[0] != 1:
        raise ValueError('bias shares must have one row')
    return AuthShare(share.values.expand(batch, -1), share.macs.expand(-1, batch, -1))


def infer_authenticated(party, values, first_triple, second_triple, square_mask):
    """Evaluate authenticated inputs without opening an intermediate or rescaling."""
    x, w1, b1, w2, b2 = values
    first = party.add(party.matmul(x, w1, first_triple), _broadcast_rows(b1, x.values.shape[0]))
    hidden = square(party, first, square_mask)
    return party.add(party.matmul(hidden, w2, second_triple), _broadcast_rows(b2, x.values.shape[0]))


def evaluate_authenticated(fixture, parties, preprocessing):
    """Consume fresh externally prepared inputs, matrix triples, and square masks.

    ``preprocessing`` contains ``input_masks`` (five lists of role-local masks),
    ``matrix_triples`` (two lists), and ``square_masks`` (one list). The caller
    establishes their security/provenance and provisions each party only its
    own MAC-key shares. Every correlation is consumed once by the runtime.
    """
    owners = [0, 1, 1, 1, 1]
    def role(party):
        try:
            p = party.party
            values = [private_input(party, masks[p], value.to(party.device) if p == owner else None)
                      for masks, value, owner in zip(preprocessing['input_masks'], fixture['encoded'], owners)]
            output = infer_authenticated(party, values, preprocessing['matrix_triples'][0][p],
                                         preprocessing['matrix_triples'][1][p], preprocessing['square_masks'][p])
            opened = party.open(output)
            return torch.where(opened > PRIME//2, opened-PRIME, opened).cpu()
        except BaseException as error:
            party.abort(error)
            raise
    with ThreadPoolExecutor(len(parties)) as pool:
        outputs = list(pool.map(role, parties))
    if any(not torch.equal(output, fixture['expected']) for output in outputs):
        raise AssertionError('authenticated output differs from the integer model')
    return dict(exact_integer_match=True, integer_accuracy=float(
        (outputs[0].argmax(-1)==fixture['targets']).float().mean()),
        global_payload_bytes=sum(p.transport.bytes_sent for p in parties))


def preprocessing_manifest(batch=64):
    """Describe fresh dealer-free material for one authenticated evaluation."""
    if not isinstance(batch, int) or batch < 1:
        raise ValueError('batch must be a positive integer')
    return dict(matrices=[dict(name='first', m=batch, k=16, n=64),
                          dict(name='second', m=batch, k=64, n=8)],
                squares=[dict(name='hidden', shape=[batch, 64])],
                inputs=[dict(name=name, owner=owner, shape=list(shape))
                        for name, owner, shape in [('x', 0, (batch, 16)),
                                                  ('w1', 1, (16, 64)), ('b1', 1, (1, 64)),
                                                  ('w2', 1, (64, 8)), ('b2', 1, (1, 8))]])


def material_from_bundles(bundles):
    """Arrange role-local native preprocessing dictionaries for evaluation."""
    return dict(input_masks=[[bundle['input_masks'][name] for bundle in bundles]
                             for name in ('x', 'w1', 'b1', 'w2', 'b2')],
                matrix_triples=[[bundle['matrix_triples'][name] for bundle in bundles]
                                for name in ('first', 'second')],
                square_masks=[bundle['square_masks']['hidden'] for bundle in bundles])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=800)
    parser.add_argument('--fractional-bits', type=int, default=8)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--output', type=Path, default=Path('results/spdz_trained_toy'))
    parser.add_argument('--write-manifest', type=Path, help='Write required preprocessing shapes and exit.')
    args = parser.parse_args()
    if args.write_manifest is not None:
        args.write_manifest.write_text(json.dumps(preprocessing_manifest(args.batch), indent=2)+'\n')
        return
    model = train(args.steps)
    fixture = quantized_fixture(model, args.fractional_bits, args.batch)
    report = numerical_report(model, fixture)
    report['execution'] = 'CPU floating and integer reference; authenticated execution requires external preprocessing.'
    report['training'] = dict(seed=20261110, steps=args.steps, optimizer='Adam',
                              learning_rate=.02, weight_decay=.001, cpu_threads=1)
    args.output.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.output/'model.pt')
    (args.output/'metadata.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
