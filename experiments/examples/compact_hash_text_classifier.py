"""Train private ring16 encoders with fewer public hash buckets.

Run from the repository root:
``python -m experiments.examples.compact_hash_text_classifier``

Each feature dimension gets a complete train/validation run. The family and its
configuration are selected by validation F1, then precision, then fewer input
buckets and hidden coordinates. Held-out metrics are reported after that choice.
The input owner still performs only public hashing and binary clipping; every
learned encoder parameter stays private.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.examples.tiny_ring16_fixed_head import train


def choose_family(reports):
    """Select by validation only; held-out fields never enter the rank."""
    def rank(report):
        selected = report['selected']
        return (selected['validation_f1'], selected['validation']['spam_precision'],
                -report['features'], -selected['hidden'])
    return max(reports, key=rank)


def run(args):
    reports = []
    args.output.mkdir(parents=True, exist_ok=True)
    for features in args.features:
        configuration = argparse.Namespace(**vars(args))
        configuration.features = features
        configuration.output = args.output / f'features{features}'
        reports.append(train(configuration))
    chosen = choose_family(reports)
    dimension = chosen['features']
    source = args.output / f'features{dimension}' / 'model.npz'
    (args.output / 'model.npz').write_bytes(source.read_bytes())
    # Family selection is frozen before assembling any held-out comparison.
    comparisons = [dict(features=report['features'], hidden=report['selected']['hidden'],
                        encoder_nonzeros=report['selected']['encoder_nonzeros'],
                        validation=report['selected']['validation'],
                        validation_f1=report['selected']['validation_f1'],
                        test=report['test'],
                        absolute_bounds=report['selected']['absolute_bounds'],
                        checkpoint=f"features{report['features']}/model.npz",
                        two_recipient_input_payload_bytes_per_row=4*report['features'])
                   for report in reports]
    result = dict(dataset=chosen['dataset'],
                  architecture='binary public hash buckets -> private ternary encoder -> fixed signed sum of squares + private bias',
                  selection='Maximum validation F1, then precision, then fewer input buckets and hidden coordinates; held-out metrics do not select the family.',
                  input_contract='Hash at most max_tokens public word tokens, then clip each bucket to binary presence.',
                  max_tokens=args.max_tokens, ring_bits=16, selected_features=dimension,
                  selected_checkpoint=str(source), candidates=comparisons,
                  privacy='Hashing has no learned parameters. Encoder weights, learned sparsity locations, encoder biases and output bias remain private; signs are fixed architecture constants.',
                  wire_accounting='Payload estimate is two corrections per input, two bytes per feature; fixed check tags and TLS framing are additional.')
    (args.output / 'metadata.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('.local/datasets/SMSSpamCollection'))
    parser.add_argument('--output', type=Path, default=Path('experiments/results/examples/compact_hash_text_classifier'))
    parser.add_argument('--features', type=int, nargs='+', default=[64, 128])
    parser.add_argument('--max-tokens', type=int, default=128)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=25)
    parser.add_argument('--batch', type=int, default=256)
    parser.add_argument('--learning-rate', type=float, default=.02)
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    if not args.features or min(args.features) < 64 or min(args.max_tokens, args.epochs, args.patience, args.batch, args.threads) < 1:
        parser.error('at least 64 hash buckets and positive counts required')
    print(json.dumps(run(args), indent=2))


if __name__ == '__main__':
    main()
