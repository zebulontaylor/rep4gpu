"""Train a bounded, integer linear classifier on the UCI SMS Spam Collection.

Run from the repository root with ``python -m experiments.examples.text_classifier``.
The input owner hashes tokens locally. Inference needs one secret matrix product;
its integer logit has the same sign convention as the floating-point model.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import urllib.request
import zipfile

import numpy as np
import torch

DATA_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
DATA_CITATION = "Almeida, T. & Hidalgo, J. (2011). SMS Spam Collection. https://doi.org/10.24432/C5CC84"
TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def featurize(texts: list[str], features: int = 512, max_tokens: int = 128) -> np.ndarray:
    """Return nonnegative token counts and a final constant-one bias column.

    SHA256 fixes bucket assignments across Python processes. Only the first
    ``max_tokens`` lowercased word tokens contribute to each row.
    """
    if features < 1 or not 0 < max_tokens < 2**31:
        raise ValueError("features must be positive and max_tokens must fit int32")
    x = np.zeros((len(texts), features + 1), dtype=np.int32)
    x[:, -1] = 1
    buckets = {}
    for row, text in enumerate(texts):
        indices = []
        for token in TOKEN_RE.findall(text.lower())[:max_tokens]:
            if token not in buckets:
                buckets[token] = int.from_bytes(hashlib.sha256(token.encode()).digest()[:8], "little") % features
            indices.append(buckets[token])
        # Repeated words share a digest within this input-owner batch. Bincount
        # preserves collisions and repeated-token counts without scalar writes.
        x[row, :-1] = np.bincount(indices, minlength=features)
    return x


def load_data(path: Path) -> tuple[list[str], np.ndarray, str]:
    """Download the public corpus if absent, then remove duplicate text rows."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(DATA_URL, timeout=60) as response:
            payload = response.read()
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            path.write_bytes(archive.read("SMSSpamCollection"))
    payload = path.read_bytes()
    rows: dict[str, int] = {}
    for line in payload.decode("utf-8").splitlines():
        if not line.strip():
            continue
        label, text = line.split("\t", 1)
        if label not in {"ham", "spam"}:
            raise ValueError(f"Unexpected label {label!r}")
        # Match the tokenizer's case folding before splitting to prevent exact
        # normalized duplicate messages from appearing on both sides.
        text = " ".join(TOKEN_RE.findall(text.lower()))
        value = int(label == "spam")
        if text in rows and rows[text] != value:
            raise ValueError("Conflicting labels for duplicate text")
        rows[text] = value
    return list(rows), np.asarray(list(rows.values()), dtype=np.int64), hashlib.sha256(payload).hexdigest()


def split_indices(labels: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Use a deterministic stratified 80/20 split after text deduplication."""
    rng = np.random.default_rng(seed)
    train, test = [], []
    for label in (0, 1):
        indices = rng.permutation(np.flatnonzero(labels == label))
        if len(indices) < 2:
            raise ValueError("Each class needs at least two distinct messages")
        count = max(1, int(len(indices) * 0.2))
        test.extend(indices[:count])
        train.extend(indices[count:])
    return rng.permutation(train), rng.permutation(test)


def quantize_weights(weights: np.ndarray, scale: int, max_tokens: int) -> tuple[np.ndarray, int]:
    """Round weights and prove a signed int32 bound for every allowed input."""
    if scale < 1 or max_tokens < 1 or not np.isfinite(weights).all():
        raise ValueError("Require finite weights and positive scale and max_tokens")
    rounded = np.rint(np.asarray(weights, dtype=np.float64) * scale)
    bound = max_tokens * float(np.max(np.abs(rounded[:-1]))) + float(np.abs(rounded[-1]).max())
    if bound >= 2**31:
        raise ValueError(f"Accumulator bound {bound:g} does not fit signed int32")
    return rounded.astype(np.int32), int(bound)


def metrics(labels: np.ndarray, logits: np.ndarray) -> dict[str, float | int]:
    predicted = logits.reshape(-1) >= 0
    positive = labels == 1
    tp = int(np.count_nonzero(predicted & positive))
    fp = int(np.count_nonzero(predicted & ~positive))
    fn = int(np.count_nonzero(~predicted & positive))
    return {"accuracy": float(np.mean(predicted == positive)), "spam_precision": tp / max(1, tp + fp),
            "spam_recall": tp / max(1, tp + fn), "false_positives": fp, "false_negatives": fn}


def train(args: argparse.Namespace) -> dict:
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    texts, labels, digest = load_data(args.data)
    x = featurize(texts, args.features, args.max_tokens)
    train_ids, test_ids = split_indices(labels, args.seed)
    xt = torch.from_numpy(x[train_ids].astype(np.float32))
    yt = torch.from_numpy(labels[train_ids].astype(np.float32)).reshape(-1, 1)
    weights = torch.zeros((x.shape[1], 1), requires_grad=True)
    optimizer = torch.optim.LBFGS([weights], max_iter=args.iterations, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(xt @ weights, yt)
        loss = loss + args.l2 * weights[:-1].square().sum() / 2
        loss.backward()
        return loss

    optimizer.step(closure)
    float_weights = weights.detach().numpy()
    quantized, bound = quantize_weights(float_weights, args.scale, args.max_tokens)
    test_x, test_y = x[test_ids], labels[test_ids]
    float_logits = test_x.astype(np.float32) @ float_weights
    integer_logits = test_x.astype(np.int64) @ quantized.astype(np.int64)
    result = {
        "dataset": {"url": DATA_URL, "citation": DATA_CITATION, "license": "CC BY 4.0", "sha256": digest,
                    "deduplication": "lowercased word-token sequence", "train_rows": len(train_ids), "test_rows": len(test_ids)},
        "features": args.features, "max_tokens": args.max_tokens, "weight_scale": args.scale,
        "seed": args.seed, "l2": args.l2, "iterations": args.iterations,
        "architecture": "hashed token counts plus bias, single linear logit", "ring_bits": 32,
        "signed_accumulator_absolute_bound": bound,
        "float": metrics(test_y, float_logits), "integer": metrics(test_y, integer_logits),
        "prediction_agreement": float(np.mean((float_logits >= 0) == (integer_logits >= 0))),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez(args.output / "model.npz", x_test=test_x, y_test=test_y, weights=quantized,
             float_weights=float_weights, train_indices=train_ids, test_indices=test_ids)
    (args.output / "metadata.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(".local/datasets/SMSSpamCollection"))
    parser.add_argument("--output", type=Path, default=Path("experiments/results/examples/text_classifier"))
    parser.add_argument("--features", type=int, default=512)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--scale", type=int, default=256)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--l2", type=float, default=0.001)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.iterations < 1 or args.l2 < 0 or args.threads < 1:
        parser.error("iterations and threads must be positive; l2 must be nonnegative")
    print(json.dumps(train(args), indent=2))


if __name__ == "__main__":
    main()
