"""Train a private encoder with a fixed public signed sum-of-squares head.

Only the encoder, its bias, and the final bias are learned. The public signs
are architecture constants: the first half are +1 and the rest are -1.
Integer inference needs one matrix product and one reduced square product.
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


class FixedHeadClassifier(torch.nn.Module):
    def __init__(self, features, hidden):
        super().__init__()
        if hidden < 2 or hidden % 2:
            raise ValueError('hidden width must be positive and even')
        self.encoder = torch.nn.Linear(features, hidden)
        self.bias = torch.nn.Parameter(torch.tensor([-1.0]))
        self.register_buffer('signs', torch.cat((torch.ones(hidden // 2), -torch.ones(hidden // 2))))
        torch.nn.init.normal_(self.encoder.weight, std=.04)
        torch.nn.init.uniform_(self.encoder.bias, -.25, .25)

    def forward(self, x):
        return (self.encoder(x).square() * self.signs).sum(-1, keepdim=True) + self.bias


def integer_forward(x, w1, b1, b2, signs):
    x, w1, b1, b2, signs = [np.asarray(v, dtype=np.int64) for v in (x, w1, b1, b2, signs)]
    h = x @ w1 + b1
    return (h * h * signs).sum(axis=1, keepdims=True) + b2


def range_bounds(w1, b1, b2, signs, max_tokens):
    """Bound all valid nonnegative count vectors, including empty messages."""
    w1, b1, b2, signs = [np.asarray(v, dtype=np.int64) for v in (w1, b1, b2, signs)]
    if w1.ndim != 2 or b1.shape != (w1.shape[1],) or signs.shape != b1.shape or b2.shape != (1,):
        raise ValueError('invalid fixed-head parameter shapes')
    if not np.isin(signs, [-1, 1]).all() or max_tokens < 1:
        raise ValueError('require signed unit head and positive token bound')
    lo = [min(int(v), 0) * max_tokens for v in w1.min(0)]
    hi = [max(int(v), 0) * max_tokens for v in w1.max(0)]
    a = [v + int(bias) for v, bias in zip(lo, b1)]
    b = [v + int(bias) for v, bias in zip(hi, b1)]
    square_max = [max(abs(left), abs(right)) ** 2 for left, right in zip(a, b)]
    bound_sum = sum(square_max)
    bounds = dict(first_product=max(map(abs, lo + hi)), encoded=max(map(abs, a + b)),
                  squared=max(square_max), sum_absolute_squares=bound_sum,
                  output=bound_sum + int(np.abs(b2).max()),
                  parameters=max(int(np.abs(v).max()) for v in (w1, b1, b2)))
    maximum = max(bounds.values())
    return dict(absolute_bounds=bounds, minimum_signed_ring_bits=max(2, maximum.bit_length() + 1),
                fits_signed_ring16=maximum < 2**15, fits_signed_ring32=maximum < 2**31)


def encode(params, signs, scale, max_tokens):
    hidden_scale = 8 * scale
    if scale < 1:
        raise ValueError('scale must be positive')
    values = {}
    for name, multiplier in [('w1', scale), ('b1', hidden_scale), ('b2', hidden_scale**2)]:
        rounded = np.rint(np.asarray(params[name], dtype=np.float64) * multiplier)
        if not np.isfinite(rounded).all() or np.any(np.abs(rounded) >= 2**31):
            raise ValueError('encoded parameters exceed signed int32')
        values[name] = rounded.astype(np.int32)
    return values, range_bounds(**values, signs=signs, max_tokens=max_tokens), hidden_scale**2


def train(args):
    torch.set_num_threads(args.threads)
    texts, labels, digest = load_data(args.data)
    counts = featurize(texts, args.features, args.max_tokens)[:, :-1]
    development, test_ids = split_indices(labels, args.seed)
    local_train, local_validation = split_indices(labels[development], args.seed + 1)
    train_ids, validation_ids = development[local_train], development[local_validation]
    x = torch.from_numpy(counts.astype(np.float32) / 8)
    y = torch.from_numpy(labels.astype(np.float32)).reshape(-1, 1)
    candidates = []
    args.output.mkdir(parents=True, exist_ok=True)
    for hidden in args.hidden:
        torch.manual_seed(args.seed)
        model = FixedHeadClassifier(args.features, hidden)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.l2)
        generator = torch.Generator().manual_seed(args.seed)
        best = None
        for epoch in range(1, args.epochs + 1):
            order = torch.randperm(len(train_ids), generator=generator).numpy()
            for start in range(0, len(order), args.batch):
                ids = train_ids[order[start:start + args.batch]]
                optimizer.zero_grad()
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    model(x[ids]), y[ids], pos_weight=torch.tensor(2.0))
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    model.encoder.weight.clamp_(-.25, .25)
                    model.encoder.bias.clamp_(-1, 1)
                    model.bias.clamp_(-8, 8)
            with torch.no_grad():
                logits = model(x[validation_ids]).numpy()
            threshold, validation_metrics, f1 = select_threshold(labels[validation_ids], logits)
            rank = (f1, validation_metrics['spam_precision'])
            if best is None or rank > best[0]:
                best = (rank, epoch, threshold, {k: v.detach().clone() for k, v in model.state_dict().items()}, validation_metrics)
        model.load_state_dict(best[3])
        with torch.no_grad():
            model.bias.sub_(best[2])
        params = dict(w1=model.encoder.weight.detach().numpy().T.copy(), b1=model.encoder.bias.detach().numpy().copy(),
                      b2=model.bias.detach().numpy().copy())
        signs = model.signs.numpy().astype(np.int32)
        variants = [(scale, *encode(params, signs, scale, args.max_tokens)) for scale in args.scales]
        safe = [row for row in variants if row[2]['fits_signed_ring32']]
        if not safe:
            raise ValueError('no ring32-safe scale')
        scale, encoded, bounds, output_scale = max(safe, key=lambda row: row[0])
        validation_integer = integer_forward(counts[validation_ids], **encoded, signs=signs)
        validation_result = metrics(labels[validation_ids], validation_integer)
        precision, recall = validation_result['spam_precision'], validation_result['spam_recall']
        rank = (2 * precision * recall / max(1e-12, precision + recall), precision, -hidden)
        info = dict(hidden=hidden, selected_epoch=best[1], validation_float=best[4], validation_integer=validation_result,
                    encoder_scale=scale, output_scale=output_scale, **bounds,
                    scales=[dict(scale=s, **b) for s, _, b, _ in variants])
        candidates.append((rank, info, params, encoded, signs))
        np.savez(args.output / f'model_h{hidden}.npz', architecture=np.array('fixed_square_head'),
                 x_test=counts[test_ids].astype(np.int32), y_test=labels[test_ids], **encoded,
                 **{f'float_{k}': v for k, v in params.items()}, signs=signs, input_divisor=np.array(8),
                 encoder_scale=np.array(scale), output_scale=np.array(output_scale), max_tokens=np.array(args.max_tokens),
                 train_indices=train_ids, validation_indices=validation_ids, test_indices=test_ids)
    # Architecture and encoding choices are fixed before test metrics are read.
    _, info, params, encoded, signs = max(candidates, key=lambda row: row[0])
    hidden = info['hidden']
    (args.output / 'model.npz').write_bytes((args.output / f'model_h{hidden}.npz').read_bytes())
    integer_logits = integer_forward(counts[test_ids], **encoded, signs=signs)
    h = counts[test_ids] / 8 @ params['w1'] + params['b1']
    float_logits = (h * h * signs).sum(axis=1, keepdims=True) + params['b2']
    report = dict(dataset=dict(url=DATA_URL, citation=DATA_CITATION, sha256=digest,
                              train_rows=len(train_ids), validation_rows=len(validation_ids), test_rows=len(test_ids)),
                  architecture='counts/8 -> learned linear encoder -> fixed signed sum of squares + learned bias',
                  privacy='All learned encoder weights, encoder biases and output bias remain private. Head signs are fixed architecture constants.',
                  selection='Maximum integer validation F1, then precision, then smaller width. Encoder scales selected by ring32 bounds alone.',
                  seed=args.seed, epochs=args.epochs, features=args.features, max_tokens=args.max_tokens,
                  selected=info, candidates=[row[1] for row in candidates],
                  float=metrics(labels[test_ids], float_logits), integer=metrics(labels[test_ids], integer_logits),
                  prediction_agreement=float(np.mean((integer_logits >= 0) == (float_logits >= 0))))
    (args.output / 'metadata.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def evaluate_checkpoint(checkpoint, output):
    """Evaluate a preselected fixed checkpoint without changing its parameters."""
    with np.load(checkpoint, allow_pickle=False) as model:
        x, y = model['x_test'], model['y_test']
        params = {name: model[name] for name in ('w1', 'b1', 'b2')}
        floats = {name: model['float_' + name] for name in params}
        signs = model['signs']
        divisor = int(model['input_divisor'])
        max_tokens = int(model['max_tokens'])
        encoder_scale, output_scale = int(model['encoder_scale']), int(model['output_scale'])
    logits = integer_forward(x, **params, signs=signs)
    h = x / divisor @ floats['w1'] + floats['b1']
    float_logits = (h * h * signs).sum(axis=1, keepdims=True) + floats['b2']
    report = dict(checkpoint=str(checkpoint), checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                  selection='Preselected checkpoint evaluation; no training, scale, or threshold selection.',
                  hidden=len(signs), encoder_scale=encoder_scale, output_scale=output_scale,
                  **range_bounds(**params, signs=signs, max_tokens=max_tokens),
                  float=metrics(y, float_logits), integer=metrics(y, logits),
                  prediction_agreement=float(np.mean((logits >= 0) == (float_logits >= 0))),
                  observed_max_absolute_logit=int(np.abs(logits).max()))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluate-checkpoint', type=Path)
    parser.add_argument('--evaluation-output', type=Path)
    parser.add_argument('--data', type=Path, default=Path('.local/datasets/SMSSpamCollection'))
    parser.add_argument('--output', type=Path, default=Path('experiments/results/examples/fixed_head_text_classifier'))
    parser.add_argument('--features', type=int, default=512)
    parser.add_argument('--hidden', type=int, nargs='+', default=[32, 64, 128])
    parser.add_argument('--scales', type=int, nargs='+', default=[1, 2, 4, 8, 16, 32, 64, 128])
    parser.add_argument('--max-tokens', type=int, default=128)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--batch', type=int, default=256)
    parser.add_argument('--learning-rate', type=float, default=.005)
    parser.add_argument('--l2', type=float, default=.0001)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if args.evaluate_checkpoint is not None:
        output = args.evaluation_output or args.evaluate_checkpoint.with_suffix('.metrics.json')
        print(json.dumps(evaluate_checkpoint(args.evaluate_checkpoint, output), indent=2))
        return
    if min(args.features, args.max_tokens, args.epochs, args.batch, args.threads, *args.scales) < 1 or any(h < 2 or h % 2 for h in args.hidden):
        parser.error('positive dimensions/scales and even hidden widths required')
    result = train(args)
    print(json.dumps({k: v for k, v in result.items() if k != 'candidates'}, indent=2))


if __name__ == '__main__':
    main()
