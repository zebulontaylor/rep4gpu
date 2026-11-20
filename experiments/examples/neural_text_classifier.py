"""Train a private-encoder square-activation SMS classifier on CPU.

The input owner applies public token hashing and divides counts by eight. Both
linear layers are learned and must be secret-shared for private inference.
Training selects an epoch and decision threshold using validation messages;
the held-out test split is the same as the linear classifier's split.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiments.examples.text_classifier import DATA_CITATION, DATA_URL, featurize, load_data, metrics, split_indices


class SquareClassifier(torch.nn.Module):
    def __init__(self, features: int, hidden: int):
        super().__init__()
        self.encoder = torch.nn.Linear(features, hidden)
        self.head = torch.nn.Linear(hidden, 1)
        torch.nn.init.normal_(self.encoder.weight, std=0.04)
        torch.nn.init.uniform_(self.encoder.bias, -0.25, 0.25)

    def forward(self, x):
        return self.head(self.encoder(x).square())


def integer_forward(x, w1, b1, w2, b2, fractional_bits):
    """Evaluate signed int64 references with exact floor rescaling after products."""
    scale = 1 << fractional_bits
    x, w1, b1, w2, b2 = [np.asarray(a, dtype=np.int64) for a in (x, w1, b1, w2, b2)]
    first_product = x @ w1
    encoded = first_product // scale + b1
    squared = encoded * encoded
    hidden = squared // scale
    last_product = hidden @ w2
    output = last_product // scale + b2
    return output, {"first_product": first_product, "encoded": encoded,
                    "squared": squared, "hidden": hidden, "last_product": last_product}


def range_bounds(w1, b1, w2, b2, fractional_bits, max_tokens):
    """Bound every accumulator for any nonnegative count input within the budget.

    Counts/8 are encoded exactly for Q3 and above. For each hidden coordinate,
    the first dot product lies between the most negative and most positive
    weight times the input L1 budget. These intervals include the empty message.
    Squaring and the output dot product then use interval arithmetic.
    """
    if fractional_bits < 3 or max_tokens < 1:
        raise ValueError("Require at least three fractional bits and a positive token budget")
    scale = 1 << fractional_bits
    w1, b1, w2, b2 = [np.asarray(a, dtype=np.int64) for a in (w1, b1, w2, b2)]
    budget = max_tokens * (scale // 8)
    low_product = np.minimum(w1.min(axis=0), 0) * budget
    high_product = np.maximum(w1.max(axis=0), 0) * budget
    low_encoded = low_product // scale + b1
    high_encoded = high_product // scale + b1
    squared_high = np.maximum(np.abs(low_encoded), np.abs(high_encoded)) ** 2
    hidden_high = squared_high // scale
    output_low = hidden_high @ np.minimum(w2, 0)
    output_high = hidden_high @ np.maximum(w2, 0)
    bounds = {
        "first_product": int(max(np.abs(low_product).max(), np.abs(high_product).max())),
        "encoded": int(max(np.abs(low_encoded).max(), np.abs(high_encoded).max())),
        "squared": int(squared_high.max()), "hidden": int(hidden_high.max()),
        "last_product": int(max(np.abs(output_low).max(), np.abs(output_high).max())),
        "output": int(max(np.abs(output_low // scale + b2).max(), np.abs(output_high // scale + b2).max())),
    }
    maximum = max(bounds.values())
    return {"absolute_bounds": bounds, "fits_signed_ring32": maximum < 2**31,
            "minimum_signed_ring_bits": max(2, int(maximum).bit_length() + 1)}


def select_threshold(labels, logits):
    """Select maximum validation F1, breaking ties toward higher precision."""
    logits = np.asarray(logits).reshape(-1)
    candidates = np.unique(np.r_[0.0, logits])
    best = None
    for threshold in candidates:
        result = metrics(labels, logits - threshold)
        precision, recall = result["spam_precision"], result["spam_recall"]
        f1 = 2 * precision * recall / max(1e-12, precision + recall)
        rank = (f1, precision, -abs(float(threshold)))
        if best is None or rank > best[0]:
            best = (rank, float(threshold), result)
    return best[1], best[2], best[0][0]


def train(args):
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    texts, labels, digest = load_data(args.data)
    counts = featurize(texts, args.features, args.max_tokens)[:, :-1]
    development, test_ids = split_indices(labels, args.seed)
    local_train, local_validation = split_indices(labels[development], args.seed + 1)
    train_ids, validation_ids = development[local_train], development[local_validation]
    x = torch.from_numpy(counts.astype(np.float32) / 8)
    y = torch.from_numpy(labels.astype(np.float32)).reshape(-1, 1)
    model = SquareClassifier(args.features, args.hidden)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.l2)
    generator = torch.Generator().manual_seed(args.seed)
    best = None
    for epoch in range(1, args.epochs + 1):
        order = torch.randperm(len(train_ids), generator=generator).numpy()
        model.train()
        for start in range(0, len(order), args.batch):
            ids = train_ids[order[start:start + args.batch]]
            optimizer.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                model(x[ids]), y[ids], pos_weight=torch.tensor(args.positive_weight))
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                model.encoder.weight.clamp_(-0.25, 0.25)
                model.encoder.bias.clamp_(-1, 1)
                model.head.weight.clamp_(-0.5, 0.5)
                model.head.bias.clamp_(-8, 8)
        with torch.no_grad():
            validation_logits = model(x[validation_ids]).numpy()
        threshold, validation_metrics, f1 = select_threshold(labels[validation_ids], validation_logits)
        rank = (f1, validation_metrics["spam_precision"])
        if best is None or rank > best[0]:
            best = (rank, epoch, threshold, {k: v.detach().clone() for k, v in model.state_dict().items()}, validation_metrics)
    model.load_state_dict(best[3])
    with torch.no_grad():
        model.head.bias.sub_(best[2])
        float_logits = model(x[test_ids]).numpy()
    params = {"w1": model.encoder.weight.detach().numpy().T.copy(),
              "b1": model.encoder.bias.detach().numpy().copy(),
              "w2": model.head.weight.detach().numpy().T.copy(),
              "b2": model.head.bias.detach().numpy().copy()}
    args.output.mkdir(parents=True, exist_ok=True)
    variants = []
    for bits in args.fractional_bits:
        scale = 1 << bits
        quantized = {k: np.rint(v.astype(np.float64) * scale).astype(np.int32) for k, v in params.items()}
        encoded_x = counts[test_ids].astype(np.int32) * (scale // 8)
        bounds = range_bounds(**quantized, fractional_bits=bits, max_tokens=args.max_tokens)
        if not bounds["fits_signed_ring32"]:
            raise ValueError(f"Q{bits} violates the ring32 accumulator bound")
        logits, intermediates = integer_forward(encoded_x, **quantized, fractional_bits=bits)
        observed = {k: int(np.abs(v).max()) for k, v in intermediates.items()}
        observed["output"] = int(np.abs(logits).max())
        variants.append({"fractional_bits": bits, **bounds, "observed_absolute_maxima": observed,
                         "metrics": metrics(labels[test_ids], logits),
                         "prediction_agreement": float(np.mean((logits >= 0) == (float_logits >= 0))),
                         "maximum_absolute_logit_error": float(np.abs(logits / scale - float_logits).max())})
        np.savez(args.output / f"model_q{bits}.npz", x_test=encoded_x, y_test=labels[test_ids],
                 **quantized, **{f"float_{k}": v for k, v in params.items()},
                 float_x_test=counts[test_ids].astype(np.float32) / 8,
                 train_indices=train_ids, validation_indices=validation_ids, test_indices=test_ids,
                 fractional_bits=np.array(bits), max_tokens=np.array(args.max_tokens))
    report = {"dataset": {"url": DATA_URL, "citation": DATA_CITATION, "license": "CC BY 4.0", "sha256": digest,
                          "train_rows": len(train_ids), "validation_rows": len(validation_ids), "test_rows": len(test_ids)},
              "architecture": f"counts/8 -> Linear({args.features},{args.hidden}) -> square -> Linear({args.hidden},1)",
              "privacy": "All learned weights and biases require secret sharing; token hashing is public and runs on the input owner.",
              "features": args.features, "hidden": args.hidden, "max_tokens": args.max_tokens, "ring_bits": 32,
              "seed": args.seed, "epochs": args.epochs, "selected_epoch": best[1],
              "selection": "Maximum validation F1, then precision; epoch and threshold selected without test labels.",
              "selected_float_threshold_folded_into_bias": best[2], "validation": best[4],
              "learning_rate": args.learning_rate, "l2": args.l2, "positive_weight": args.positive_weight,
              "training_batch": args.batch, "float": metrics(labels[test_ids], float_logits),
              "integer_variants": variants,
              "rounding": "Nearest-even parameter encoding; exact signed floor after each matrix product and square.",
              "range_contract": "Nonnegative integer counts, total <= max_tokens; x=count*2**fractional_bits/8. Bounds cover every valid input, not only the test set."}
    (args.output / "metadata.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(".local/datasets/SMSSpamCollection"))
    parser.add_argument("--output", type=Path, default=Path("experiments/results/examples/neural_text_classifier"))
    parser.add_argument("--features", type=int, default=512)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--fractional-bits", type=int, nargs="+", default=[6, 8])
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.005)
    parser.add_argument("--l2", type=float, default=0.0001)
    parser.add_argument("--positive-weight", type=float, default=2.0)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if min(args.features, args.hidden, args.max_tokens, args.epochs, args.batch, args.threads) < 1 or not all(3 <= b <= 12 for b in args.fractional_bits):
        parser.error("Dimensions and counts must be positive; fractional bits must be between 3 and 12")
    print(json.dumps(train(args), indent=2))


if __name__ == "__main__":
    main()
