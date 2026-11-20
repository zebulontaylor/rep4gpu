"""Measure weight quantization and certified ring bounds for the SMS classifier.

Run from the repository root after ``experiments.examples.text_classifier``.
The signed-ring requirement is an arithmetic bound; the MPC runtime still uses
32-bit shares, so this sweep does not measure a 16-bit protocol speedup.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.examples.text_classifier import metrics, quantize_weights


def sweep(model_path: Path, metadata_path: Path, scales: list[int]) -> dict:
    metadata = json.loads(metadata_path.read_text())
    max_tokens = int(metadata["max_tokens"])
    with np.load(model_path, allow_pickle=False) as model:
        x = model["x_test"].astype(np.int64)
        labels = model["y_test"]
        float_weights = model["float_weights"]
    if x.ndim != 2 or float_weights.shape != (x.shape[1], 1):
        raise ValueError("Expected a matrix of features and one linear output")
    if (x[:, :-1] < 0).any() or (x[:, :-1].sum(axis=1) > max_tokens).any() or not (x[:, -1] == 1).all():
        raise ValueError("Test inputs violate the bounded-count and bias contract")
    float_logits = x.astype(np.float32) @ float_weights
    rows = []
    for scale in scales:
        weights, bound = quantize_weights(float_weights, scale, max_tokens)
        logits = x @ weights.astype(np.int64)
        rows.append({
            "weight_scale": scale,
            "metrics": metrics(labels, logits),
            "prediction_agreement": float(np.mean((float_logits >= 0) == (logits >= 0))),
            "signed_accumulator_absolute_bound": bound,
            "minimum_signed_ring_bits": max(2, bound.bit_length() + 1),
            "fits_signed_ring16": bound < 2**15,
            "fits_signed_ring32": bound < 2**31,
            "observed_max_absolute_logit": int(np.max(np.abs(logits))),
        })
    return {
        "model": str(model_path), "dataset_sha256": metadata["dataset"]["sha256"],
        "test_rows": len(labels), "max_tokens": max_tokens,
        "bound_formula": "max_tokens * max(abs(nonbias_weights)) + abs(bias)",
        "runtime_ring_bits": 32,
        "interpretation": "Minimum ring width is an arithmetic requirement, not a benchmark of a narrower MPC runtime.",
        "float": metrics(labels, float_logits), "scales": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("experiments/results/examples/text_classifier/model.npz"))
    parser.add_argument("--metadata", type=Path, default=Path("experiments/results/examples/text_classifier/metadata.json"))
    parser.add_argument("--scales", type=int, nargs="+", default=[256, 128, 64, 32, 16, 8, 4, 2, 1])
    parser.add_argument("--output", type=Path, default=Path("experiments/results/inference/text_classifier_precision.json"))
    args = parser.parse_args()
    result = sweep(args.model, args.metadata, args.scales)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
