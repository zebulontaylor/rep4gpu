"""Encode the trained square classifier as a polynomial with no rescaling.

Only the output sign is needed. Keeping each product's scale avoids all three
intermediate fixed-point truncations. The scale choice uses global accumulator
bounds alone, prioritizing encoder precision and then output-weight precision.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.examples.text_classifier import metrics


def encode_polynomial(params, encoder_scale, head_scale, max_tokens):
    if encoder_scale < 1 or head_scale < 1 or max_tokens < 1:
        raise ValueError("Scales and token budget must be positive")
    hidden_scale = 8 * encoder_scale
    output_scale = hidden_scale**2 * head_scale
    encoded = {
        "w1": np.rint(params["w1"] * encoder_scale).astype(np.int64),
        "b1": np.rint(params["b1"] * hidden_scale).astype(np.int64),
        "w2": np.rint(params["w2"] * head_scale).astype(np.int64),
        "b2": np.rint(params["b2"] * output_scale).astype(np.int64),
    }
    bounds = polynomial_bounds(**encoded, max_tokens=max_tokens)
    return encoded, bounds, output_scale


def polynomial_bounds(w1, b1, w2, b2, max_tokens):
    """Certify signed bounds for every nonnegative count vector in the budget."""
    w1, b1, w2, b2 = [np.asarray(v, dtype=np.int64) for v in (w1, b1, w2, b2)]
    product_low = np.minimum(w1.min(axis=0), 0) * max_tokens
    product_high = np.maximum(w1.max(axis=0), 0) * max_tokens
    low, high = product_low + b1, product_high + b1
    square_high = np.maximum(np.abs(low), np.abs(high)) ** 2
    final_low = square_high @ np.minimum(w2, 0)
    final_high = square_high @ np.maximum(w2, 0)
    absolute = {
        "first_product": int(max(np.abs(product_low).max(), np.abs(product_high).max())),
        "encoded": int(max(np.abs(low).max(), np.abs(high).max())),
        "squared": int(square_high.max()),
        "last_product": int(max(np.abs(final_low).max(), np.abs(final_high).max())),
        "output": int(max(np.abs(final_low + b2).max(), np.abs(final_high + b2).max())),
        "parameters": max(int(np.abs(v).max()) for v in (w1, b1, w2, b2)),
    }
    maximum = max(absolute.values())
    return {"absolute_bounds": absolute, "fits_signed_ring32": maximum < 2**31,
            "minimum_signed_ring_bits": max(2, maximum.bit_length() + 1)}


def polynomial_forward(x, w1, b1, w2, b2):
    x, w1, b1, w2, b2 = [np.asarray(v, dtype=np.int64) for v in (x, w1, b1, w2, b2)]
    hidden = x @ w1 + b1
    return (hidden * hidden) @ w2 + b2


def convert(model_path, output, encoder_scales, head_scales):
    with np.load(model_path, allow_pickle=False) as source:
        fractional_bits = int(source["fractional_bits"])
        unit = 2**fractional_bits // 8
        if np.any(source["x_test"] % unit):
            raise ValueError("Source inputs are not exactly encoded raw token counts")
        x = source["x_test"].astype(np.int64) // unit
        y = source["y_test"].copy()
        max_tokens = int(source["max_tokens"])
        params = {key: source[f"float_{key}"].astype(np.float64) for key in ("w1", "b1", "w2", "b2")}
        indices = {key: source[key].copy() for key in ("train_indices", "validation_indices", "test_indices")}
    if (x < 0).any() or (x.sum(axis=1) > max_tokens).any():
        raise ValueError("Source inputs violate the token-count budget")
    candidates = []
    for encoder_scale in sorted(set(encoder_scales)):
        for head_scale in sorted(set(head_scales)):
            encoded, bounds, output_scale = encode_polynomial(params, encoder_scale, head_scale, max_tokens)
            candidates.append((encoder_scale, head_scale, encoded, bounds, output_scale))
    feasible = [row for row in candidates if row[3]["fits_signed_ring32"]]
    if not feasible:
        raise ValueError("No candidate scale pair fits ring32")
    selected = max(feasible, key=lambda row: (row[0], row[1]))
    # The selection is frozen before reading any prediction metrics.
    float_logits = ((x / 8 @ params["w1"] + params["b1"]) ** 2) @ params["w2"] + params["b2"]
    rows = []
    for encoder_scale, head_scale, encoded, bounds, output_scale in candidates:
        logits = polynomial_forward(x, **encoded)
        rows.append({"encoder_scale": encoder_scale, "head_scale": head_scale, "output_scale": output_scale,
                     **bounds, "metrics": metrics(y, logits),
                     "prediction_agreement": float(np.mean((logits >= 0) == (float_logits >= 0))),
                     "observed_max_absolute_logit": int(np.abs(logits).max())})
    encoder_scale, head_scale, encoded, bounds, output_scale = selected
    selected_metrics = next(row for row in rows if (row["encoder_scale"], row["head_scale"]) == (encoder_scale, head_scale))
    output.mkdir(parents=True, exist_ok=True)
    np.savez(output / "model.npz", architecture=np.array("square_unscaled"),
             x_test=x.astype(np.int32), y_test=y,
             **{key: value.astype(np.int32) for key, value in encoded.items()},
             **{f"float_{key}": value.astype(np.float32) for key, value in params.items()},
             float_x_test=(x / 8).astype(np.float32), **indices,
             max_tokens=np.array(max_tokens), encoder_scale=np.array(encoder_scale),
             head_scale=np.array(head_scale), output_scale=np.array(output_scale), input_divisor=np.array(8))
    report = {"architecture": "square_unscaled", "source_checkpoint": str(model_path),
              "features": x.shape[1], "hidden": params["w1"].shape[1], "max_tokens": max_tokens,
              "ring_bits": 32, "test_rows": len(y),
              "selection": "Largest encoder scale, then largest head scale, among globally ring32-safe candidates. Metrics do not influence selection.",
              "forward": "H = raw_counts @ w1 + b1; logits = (H * H) @ w2 + b2; classify logits >= 0.",
              "privacy": "Both learned layers and both biases remain secret-shared. No intermediate truncation or opening is needed.",
              "float": metrics(y, float_logits), "selected": selected_metrics, "candidates": rows}
    (output / "metadata.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("experiments/results/examples/neural_text_classifier/model_q6.npz"))
    parser.add_argument("--output", type=Path, default=Path("experiments/results/examples/unscaled_text_classifier"))
    parser.add_argument("--encoder-scales", type=int, nargs="+", default=[16, 32, 64])
    parser.add_argument("--head-scales", type=int, nargs="+", default=[4, 8, 16, 32])
    args = parser.parse_args()
    report = convert(args.model, args.output, args.encoder_scales, args.head_scales)
    print(json.dumps({key: value for key, value in report.items() if key != "candidates"}, indent=2))


if __name__ == "__main__":
    main()
