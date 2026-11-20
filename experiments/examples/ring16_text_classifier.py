"""Train a sparse integer square classifier whose entire circuit fits ring16.

Binary hashed token-presence features feed a learned sparse ternary encoder and
an integer head. Straight-through quantization trains the deployed integer
circuit. Validation selects architecture, epoch, and threshold; test labels are
used only after selection. Every learned parameter remains private at inference.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiments.examples.text_classifier import DATA_CITATION, DATA_URL, featurize, load_data, metrics, split_indices
from experiments.examples.neural_text_classifier import select_threshold


class SparseIntegerSquare(torch.nn.Module):
    def __init__(self, features, hidden, nonzeros, head_limit):
        super().__init__()
        self.nonzeros, self.head_limit = nonzeros, head_limit
        self.encoder = torch.nn.Parameter(torch.randn(features, hidden) * 0.1)
        self.encoder_bias = torch.nn.Parameter(torch.full((hidden,), 0.6))
        self.head = torch.nn.Parameter(torch.randn(hidden, 1))
        self.head_bias = torch.nn.Parameter(torch.tensor([-24.]))

    def quantized(self):
        mask = torch.zeros_like(self.encoder).scatter_(0, self.encoder.abs().topk(self.nonzeros, dim=0).indices, 1)
        targets = (mask * self.encoder.sign(), self.encoder_bias.round().clamp(-1, 1),
                   self.head.round().clamp(-self.head_limit, self.head_limit), self.head_bias.round())
        parameters = (self.encoder, self.encoder_bias, self.head, self.head_bias)
        return tuple(p + (target - p).detach() for p, target in zip(parameters, targets))

    def forward(self, x):
        w1, b1, w2, b2 = self.quantized()
        return ((x @ w1 + b1).square() @ w2 + b2) / 16

    def constrain(self):
        with torch.no_grad():
            self.encoder.clamp_(-1, 1)
            self.encoder_bias.clamp_(-1, 1)
            self.head.clamp_(-self.head_limit, self.head_limit)
            self.head_bias.clamp_(-128, 128)


def integer_forward(x, w1, b1, w2, b2):
    hidden = np.asarray(x, dtype=np.int64) @ np.asarray(w1, dtype=np.int64) + b1
    return (hidden * hidden) @ w2 + b2


def prune_zero_head(w1, b1, w2, b2):
    """Remove hidden coordinates that contribute exactly zero to every score."""
    keep = np.asarray(w2).reshape(-1) != 0
    if not keep.any():
        raise ValueError('Cannot export a classifier with no active hidden coordinates')
    return dict(w1=w1[:, keep], b1=b1[keep], w2=w2[keep], b2=b2)


def range_bounds(w1, b1, w2, b2, max_tokens=128):
    """Bound all wires for binary feature vectors with at most max_tokens ones."""
    w1, b1, w2, b2 = [np.asarray(v, dtype=np.int64) for v in (w1, b1, w2, b2)]
    positive = np.sort(np.maximum(w1, 0), axis=0)[-max_tokens:].sum(axis=0)
    negative = np.sort(np.minimum(w1, 0), axis=0)[:max_tokens].sum(axis=0)
    low, high = negative + b1, positive + b1
    square_high = np.maximum(np.abs(low), np.abs(high)) ** 2
    output_low = square_high @ np.minimum(w2, 0)
    output_high = square_high @ np.maximum(w2, 0)
    bounds = dict(first_product=int(max(abs(negative).max(), abs(positive).max())),
                  encoded=int(max(abs(low).max(), abs(high).max())),
                  squared=int(square_high.max()),
                  last_product=int(max(abs(output_low).max(), abs(output_high).max())),
                  output=int(max(abs(output_low + b2).max(), abs(output_high + b2).max())),
                  parameters=max(int(np.abs(v).max()) for v in (w1, b1, w2, b2)))
    maximum = max(bounds.values())
    return dict(absolute_bounds=bounds, fits_signed_ring16=maximum < 32768,
                minimum_signed_ring_bits=max(2, maximum.bit_length() + 1))


def train(args):
    torch.set_num_threads(args.threads)
    texts, labels, digest = load_data(args.data)
    binary = (featurize(texts, args.features, args.max_tokens)[:, :-1] > 0).astype(np.int32)
    development, test_ids = split_indices(labels, args.seed)
    local_train, local_validation = split_indices(labels[development], args.seed + 1)
    train_ids, validation_ids = development[local_train], development[local_validation]
    x = torch.from_numpy(binary.astype(np.float32))
    y = torch.from_numpy(labels.astype(np.float32)).reshape(-1, 1)
    configurations = [(128, 8, 3), (256, 4, 3), (128, 4, 3), (256, 8, 1), (512, 4, 2)]
    candidates, selected = [], None
    for hidden, nonzeros, head_limit in configurations:
        torch.manual_seed(args.seed)
        model = SparseIntegerSquare(args.features, hidden, nonzeros, head_limit)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        best = None
        for epoch in range(1, args.epochs + 1):
            order = np.random.default_rng(epoch).permutation(train_ids)
            for start in range(0, len(order), args.batch):
                ids = order[start:start + args.batch]
                optimizer.zero_grad()
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    model(x[ids]), y[ids], pos_weight=torch.tensor(2.))
                loss.backward()
                optimizer.step()
                model.constrain()
            with torch.no_grad():
                validation_logits = model(x[validation_ids]).numpy() * 16
                parameters = [p.detach().numpy().round().astype(np.int64) for p in model.quantized()]
            threshold, validation_metrics, f1 = select_threshold(labels[validation_ids], validation_logits)
            parameters[-1] -= int(threshold)
            bounds = range_bounds(*parameters, max_tokens=args.max_tokens)
            rank = (f1, validation_metrics['spam_precision'])
            if bounds['fits_signed_ring16'] and (best is None or rank > best['rank']):
                best = dict(rank=rank, epoch=epoch, threshold=int(threshold), parameters=parameters,
                            validation=validation_metrics, **bounds)
            if best is not None and epoch - best['epoch'] >= args.patience:
                break
        if best is None:
            raise ValueError('No ring16-safe candidate')
        row = {k: v for k, v in best.items() if k not in ('parameters', 'rank')}
        row.update(hidden=hidden, encoder_nonzeros_per_hidden=nonzeros, head_limit=head_limit)
        candidates.append(row)
        if selected is None or best['rank'] > selected['rank']:
            selected = {**best, 'configuration': row}
        print(json.dumps(row), flush=True)
    # Test labels are accessed only after architecture, epoch, and threshold selection.
    params = dict(zip(('w1', 'b1', 'w2', 'b2'), selected['parameters']))
    logits = integer_forward(binary[test_ids], **params)
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez(args.output / 'model.npz', architecture=np.array('square_unscaled'),
             x_test=binary[test_ids], y_test=labels[test_ids], **params,
             **{f'float_{name}': value.astype(np.float32) for name, value in params.items()},
             input_divisor=np.array(1), max_tokens=np.array(args.max_tokens), ring_bits=np.array(16),
             train_indices=train_ids, validation_indices=validation_ids, test_indices=test_ids)
    pruned = prune_zero_head(**params)
    if not np.array_equal(integer_forward(binary[test_ids], **pruned), logits):
        raise AssertionError('Zero-head pruning changed predictions')
    np.savez(args.output / 'model_pruned.npz', architecture=np.array('square_unscaled'),
             x_test=binary[test_ids], y_test=labels[test_ids], **pruned,
             **{f'float_{name}': value.astype(np.float32) for name, value in pruned.items()},
             input_divisor=np.array(1), max_tokens=np.array(args.max_tokens), ring_bits=np.array(16),
             train_indices=train_ids, validation_indices=validation_ids, test_indices=test_ids)
    report = dict(dataset=dict(url=DATA_URL, citation=DATA_CITATION, sha256=digest,
                               train_rows=len(train_ids), validation_rows=len(validation_ids), test_rows=len(test_ids)),
                  architecture='binary hashed token presence -> sparse integer encoder -> square -> integer head',
                  privacy='All learned encoder/head weights, sparsity locations, and biases stay private.',
                  input_contract=f'Binary feature values and at most {args.max_tokens} nonzero features; hash first {args.max_tokens} tokens then clamp counts to one.',
                  ring_bits=16, seed=args.seed, learning_rate=args.learning_rate, max_epochs=args.epochs,
                  patience=args.patience, training_logit_divisor=16,
                  selection='Maximum validation F1, then precision; all architecture/epoch/threshold choices precede test evaluation.',
                  selected=selected['configuration'], pruned_hidden=pruned['w1'].shape[1],
                  pruned_bounds=range_bounds(**pruned, max_tokens=args.max_tokens), candidates=candidates,
                  test=metrics(labels[test_ids], logits), observed_max_absolute_score=int(np.abs(logits).max()),
                  inference='H = x @ w1 + b1; score = (H*H) @ w2 + b2; label = score >= 0; no rescaling.',
                  runtime_scope='This checkpoint is ring16-safe; a ring16 MPC runtime must be selected separately.')
    (args.output / 'metadata.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('.local/datasets/SMSSpamCollection'))
    parser.add_argument('--output', type=Path, default=Path('experiments/results/examples/ring16_text_classifier'))
    parser.add_argument('--features', type=int, default=512)
    parser.add_argument('--max-tokens', type=int, default=128)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--batch', type=int, default=256)
    parser.add_argument('--learning-rate', type=float, default=0.02)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if args.features < 8 or min(args.max_tokens, args.epochs, args.patience, args.batch, args.threads) < 1:
        parser.error('features must be at least8; dimensions and counts must be positive')
    print(json.dumps(train(args), indent=2))


if __name__ == '__main__':
    main()
