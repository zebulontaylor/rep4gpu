"""Train compact ring16 classifiers with a fixed signed sum-of-squares head.

Binary hashed feature inputs feed a learned sparse ternary encoder. Head signs
are public architecture constants; all encoder weights, sparsity locations,
biases and the final bias remain private. No inference rescaling is required.

From the repository root, run:
``python -m experiments.examples.tiny_ring16_fixed_head``

The default output directory is ``experiments/results/examples/tiny_ring16_fixed_head``.
``model.npz`` is selected using validation F1, precision, and hidden width;
``metadata.json`` records its held-out metrics and certified binary-input bounds.
Candidate checkpoints retain ``w1``, ``b1``, ``b2``, public ``signs``, raw binary
``x_test``, and corresponding ``float_*`` arrays for native comparisons. The
architecture tag is ``fixed_square_head`` and ``input_divisor`` is one.
Use ``--evaluate-checkpoint PATH`` to evaluate a fixed candidate without tuning.
A native benchmark can use the learned sparse indices; MPC must keep them private.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from experiments.examples.text_classifier import DATA_CITATION, DATA_URL, featurize, load_data, metrics, split_indices
from experiments.examples.neural_text_classifier import select_threshold
from experiments.examples.fixed_head_text_classifier import integer_forward


class TinyIntegerFixedHead(torch.nn.Module):
    def __init__(self, features, hidden, nonzeros):
        super().__init__()
        if hidden < 2 or hidden % 2 or not 0 < nonzeros <= features:
            raise ValueError('require even hidden width and valid nonzero count')
        self.nonzeros = nonzeros
        self.encoder = torch.nn.Parameter(torch.randn(features, hidden) * .1)
        self.encoder_bias = torch.nn.Parameter(torch.full((hidden,), .6))
        self.output_bias = torch.nn.Parameter(torch.tensor([-8.]))
        self.register_buffer('signs', torch.cat((torch.ones(hidden // 2), -torch.ones(hidden // 2))))

    def quantized(self):
        mask = torch.zeros_like(self.encoder).scatter_(0, self.encoder.abs().topk(self.nonzeros, dim=0).indices, 1)
        targets = (mask * self.encoder.sign(), self.encoder_bias.round().clamp(-2, 2), self.output_bias.round())
        return tuple(p + (target - p).detach() for p, target in zip((self.encoder, self.encoder_bias, self.output_bias), targets))

    def forward(self, x):
        w, b, c = self.quantized()
        return ((x @ w + b).square() * self.signs).sum(1, keepdim=True) / 16 + c / 16

    def constrain(self):
        with torch.no_grad():
            self.encoder.clamp_(-1, 1)
            self.encoder_bias.clamp_(-2, 2)
            self.output_bias.clamp_(-256, 256)


def binary_bounds(w1, b1, b2, signs, max_tokens=128):
    """Bound every binary feature input whose L1 norm is at most max_tokens."""
    w1, b1, b2, signs = [np.asarray(v, dtype=np.int64) for v in (w1, b1, b2, signs)]
    if w1.ndim != 2 or b1.shape != (w1.shape[1],) or signs.shape != b1.shape or b2.shape != (1,):
        raise ValueError('invalid fixed-head shapes')
    if max_tokens < 1 or not np.isin(w1, [-1, 0, 1]).all() or not np.isin(signs, [-1, 1]).all():
        raise ValueError('require positive token bound, ternary weights and unit head signs')
    lo = [-min(int(np.count_nonzero(v < 0)), max_tokens) for v in w1.T]
    hi = [min(int(np.count_nonzero(v > 0)), max_tokens) for v in w1.T]
    lower = [v + int(b) for v, b in zip(lo, b1)]
    upper = [v + int(b) for v, b in zip(hi, b1)]
    squares = [max(abs(a), abs(b)) ** 2 for a, b in zip(lower, upper)]
    bound = dict(first_product=max(map(abs, lo + hi)), encoded=max(map(abs, lower + upper)),
                 squared=max(squares), sum_absolute_squares=sum(squares), output=sum(squares) + abs(int(b2[0])),
                 parameters=max(max(abs(int(v)) for v in b1), abs(int(b2[0])), 1))
    maximum = max(bound.values())
    return dict(absolute_bounds=bound, fits_signed_ring16=maximum < 32768,
                minimum_signed_ring_bits=max(2, maximum.bit_length() + 1))


def evaluate(checkpoint, output):
    with np.load(checkpoint, allow_pickle=False) as model:
        x, y = model['x_test'], model['y_test']
        params = {k: model[k] for k in ('w1', 'b1', 'b2', 'signs')}
        max_tokens = int(model['max_tokens'])
    logits = integer_forward(x, **params)
    result = dict(checkpoint=str(checkpoint), checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                  selection='Evaluation of a fixed checkpoint; no parameter or threshold selection.',
                  hidden=params['w1'].shape[1], **binary_bounds(**params, max_tokens=max_tokens),
                  integer=metrics(y, logits), observed_max_absolute_logit=int(np.abs(logits).max()))
    output.write_text(json.dumps(result, indent=2) + '\n')
    return result


def train(args):
    torch.set_num_threads(args.threads)
    texts, labels, digest = load_data(args.data)
    binary = (featurize(texts, args.features, args.max_tokens)[:, :-1] > 0).astype(np.int32)
    development, test = split_indices(labels, args.seed)
    train_local, validation_local = split_indices(labels[development], args.seed + 1)
    training, validation = development[train_local], development[validation_local]
    x = torch.from_numpy(binary.astype(np.float32))
    y = torch.from_numpy(labels.astype(np.float32)).reshape(-1, 1)
    configurations = [(4, 32), (4, 64), (8, 32), (8, 48), (16, 16), (16, 24), (16, 32)]
    args.output.mkdir(parents=True, exist_ok=True)
    candidates = []
    for hidden, nonzeros in configurations:
        torch.manual_seed(args.seed)
        model = TinyIntegerFixedHead(args.features, hidden, nonzeros)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        best = None
        for epoch in range(1, args.epochs + 1):
            ids = np.random.default_rng(args.seed + epoch).permutation(training)
            for start in range(0, len(ids), args.batch):
                batch = ids[start:start + args.batch]
                optimizer.zero_grad()
                loss = torch.nn.functional.binary_cross_entropy_with_logits(model(x[batch]), y[batch], pos_weight=torch.tensor(2.))
                loss.backward()
                optimizer.step()
                model.constrain()
            with torch.no_grad():
                w, b, c = [p.detach().numpy().round().astype(np.int32) for p in model.quantized()]
                signs = model.signs.numpy().astype(np.int32)
            validation_logits = integer_forward(binary[validation], w, b, c, signs)
            threshold, validation_metrics, f1 = select_threshold(labels[validation], validation_logits)
            c = c - int(threshold)
            bounds = binary_bounds(w, b, c, signs, args.max_tokens)
            rank = (f1, validation_metrics['spam_precision'], -hidden)
            if bounds['fits_signed_ring16'] and (best is None or rank > best[0]):
                best = (rank, dict(w1=w, b1=b, b2=c, signs=signs),
                        dict(hidden=hidden, encoder_nonzeros=nonzeros, epoch=epoch, validation=validation_metrics,
                             validation_f1=f1, threshold=int(threshold), **bounds))
            if best is not None and epoch - best[2]['epoch'] >= args.patience:
                break
        if best is None:
            raise ValueError('no ring16-safe model')
        params, info = best[1:]
        path = args.output / f'model_h{hidden}_k{nonzeros}.npz'
        np.savez(path, architecture=np.array('fixed_square_head'), x_test=binary[test], y_test=labels[test],
                 **params, **{f'float_{k}': v.astype(np.float32) for k, v in params.items() if k != 'signs'},
                 input_divisor=np.array(1), max_tokens=np.array(args.max_tokens), ring_bits=np.array(16),
                 encoder_scale=np.array(1), output_scale=np.array(1),
                 train_indices=training, validation_indices=validation, test_indices=test)
        candidates.append((best[0], path, info))
        print(json.dumps(info), flush=True)
    # No candidate's test predictions are consulted until selection is frozen.
    _, path, selected = max(candidates, key=lambda c: c[0])
    chosen_path = args.output / 'model.npz'
    chosen_path.write_bytes(path.read_bytes())
    result = evaluate(chosen_path, args.output / 'model.metrics.json')
    metadata = dict(dataset=dict(url=DATA_URL, citation=DATA_CITATION, sha256=digest,
                                train_rows=len(training), validation_rows=len(validation), test_rows=len(test)),
                    architecture='binary token presence -> learned sparse ternary encoder -> fixed signed sum of squares + learned bias',
                    privacy='All learned weights, sparsity locations and biases remain private; head signs are architecture constants.',
                    input_contract=f'Binary features with at most {args.max_tokens} active buckets.',
                    features=args.features, max_tokens=args.max_tokens,
                    selection='Maximum validation F1, then precision, then smaller width; test labels only evaluate the selected model.',
                    seed=args.seed, learning_rate=args.learning_rate, epochs=args.epochs, patience=args.patience,
                    selected=selected, selected_checkpoint=str(path), candidates=[c[2] for c in candidates], test=result['integer'])
    (args.output / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('.local/datasets/SMSSpamCollection'))
    parser.add_argument('--output', type=Path, default=Path('experiments/results/examples/tiny_ring16_fixed_head'))
    parser.add_argument('--evaluate-checkpoint', type=Path)
    parser.add_argument('--features', type=int, default=512)
    parser.add_argument('--max-tokens', type=int, default=128)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=25)
    parser.add_argument('--batch', type=int, default=256)
    parser.add_argument('--learning-rate', type=float, default=.02)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if args.evaluate_checkpoint is not None:
        print(json.dumps(evaluate(args.evaluate_checkpoint, args.evaluate_checkpoint.with_suffix('.metrics.json')), indent=2))
        return
    if args.features < 64 or min(args.max_tokens, args.epochs, args.patience, args.batch, args.threads) < 1:
        parser.error('at least 64 features and positive counts required')
    print(json.dumps(train(args), indent=2))


if __name__ == '__main__':
    main()
