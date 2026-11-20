"""Train and export a certified integer dense-input Text CNN on SMS Spam.

Run from the repository root:
    python -m experiments.examples.text_cnn_classifier

Architecture (Kim-style 1D Text CNN with energy pooling):
    X in {-1, +1}^{B x L x D}  (L=48 tokens, D=256 dense subword/char-ngram Rademacher features)
    H = Conv1d(X, W1) + b1     (k=5, C=128 -> B x 44 x 128)
    P = sum_t H[:, t, :]^2     (B x 128)
    score = P @ w2 + b2        (B x 1)
    label = (score >= 0)

Every input entry is +/-1 (100% dense, including the PAD token), and every
learned parameter (W1, b1, w2, b2) is secret-shared during MPC inference.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.examples.neural_text_classifier import select_threshold
from experiments.examples.text_classifier import DATA_CITATION, DATA_URL, load_data, metrics, split_indices
from rep4gpu.gpu.matmul.crt_conv import DEFAULT_CRT_PRIMES

TOKEN_RE = re.compile(r'[a-z0-9]+|[^\s\w]', re.UNICODE)


def _token_vector(token: str, dim: int) -> np.ndarray:
    """Map one token to a deterministic dense {-1, +1}^dim Rademacher vector."""
    if dim < 8 or dim % 8 or dim > 512:
        raise ValueError('embedding dim must be a multiple of 8 in [8, 512]')
    padded = f'<{token}>'
    grams = [f'W:{token}']
    if token != '<PAD>':
        if any(c.isdigit() for c in token):
            grams.append('F:HAS_DIGIT')
        if token.isdigit():
            grams.append('F:ALL_DIGIT')
        if len(token) >= 5 and token.isdigit():
            grams.append('F:LONG_DIGIT')
        for n in (2, 3, 4):
            for start in range(max(0, len(padded) - n + 1)):
                grams.append(f'{n}:{padded[start:start + n]}')
    acc = np.zeros(dim, dtype=np.int32)
    nbytes = dim // 8
    for gram in grams:
        digest = hashlib.sha256(gram.encode('utf-8')).digest()
        raw = digest + hashlib.sha256(digest).digest()
        bits = np.unpackbits(np.frombuffer(raw[:nbytes], dtype=np.uint8))[:dim]
        acc += bits.astype(np.int32) * 2 - 1
    return np.where(acc >= 0, 1, -1).astype(np.int8)


def featurize_sequences(texts: list[str], seq_len: int = 48, dim: int = 256) -> np.ndarray:
    """Return dense int8 tensor of shape (N, seq_len, dim) with entries in {-1, +1}."""
    if seq_len < 1 or dim < 8:
        raise ValueError('seq_len must be positive and dim >= 8')
    pad_vec = _token_vector('<PAD>', dim)
    cache: dict[str, np.ndarray] = {}
    out = np.broadcast_to(pad_vec, (len(texts), seq_len, dim)).copy()
    for row, text in enumerate(texts):
        for pos, token in enumerate(TOKEN_RE.findall(text.lower())[:seq_len]):
            vec = cache.get(token)
            if vec is None:
                vec = _token_vector(token, dim)
                cache[token] = vec
            out[row, pos] = vec
    return out


class TextCNNSquare(nn.Module):
    def __init__(self, dim: int = 256, channels: int = 128, kernel_size: int = 5):
        super().__init__()
        self.conv = nn.Conv1d(dim, channels, kernel_size)
        self.head = nn.Linear(channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)
        p = (h * h).mean(dim=2)
        return self.head(p)


def global_bounds(w1: np.ndarray, b1: np.ndarray, w2: np.ndarray, b2: np.ndarray, seq_out: int) -> dict[str, int]:
    """Certify worst-case absolute bounds over all inputs x in {-1, +1}^{seq_len x dim}."""
    # w1 shape: (k, D, C), b1: (C,), w2: (C, 1), b2: (1,)
    h_max = np.abs(w1.astype(np.int64)).sum(axis=(0, 1)) + np.abs(b1.astype(np.int64))
    p_max = (h_max ** 2) * seq_out
    score_max = int((p_max * np.abs(w2[:, 0].astype(np.int64))).sum() + abs(int(b2[0])))
    return dict(
        conv_max_abs=int(h_max.max()),
        pool_max_abs=int(p_max.max()),
        score_max_abs=score_max,
    )


def integer_forward(x: np.ndarray, w1: np.ndarray, b1: np.ndarray, w2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """Exact int64 reference forward pass; x: (B, L, D), w1: (k, D, C)."""
    # Use float64 conv1d only when exact (all products bounded << 2^53), verified explicitly.
    h_bound = int(np.abs(w1.astype(np.int64)).sum(axis=(0, 1)).max() + np.abs(b1.astype(np.int64)).max())
    if h_bound >= 2**50:
        raise ValueError('conv accumulator exceeds exact float64 integer range')
    xt = torch.from_numpy(x.astype(np.float64)).permute(0, 2, 1)
    wt = torch.from_numpy(w1.astype(np.float64)).permute(2, 1, 0)
    bt = torch.from_numpy(b1.astype(np.float64))
    h = F.conv1d(xt, wt, bt).round().to(torch.int64).permute(0, 2, 1).numpy()
    p = (h * h).sum(axis=1)
    return p @ w2.astype(np.int64) + b2.astype(np.int64)


def train_and_export(args):
    torch.set_num_threads(args.threads)
    texts, labels, data_sha = load_data(args.data)
    dev_ids, test_ids = split_indices(labels, args.seed)
    tr_loc, val_loc = split_indices(labels[dev_ids], args.seed + 1)
    train_ids, val_ids = dev_ids[tr_loc], dev_ids[val_loc]

    x_all = featurize_sequences(texts, seq_len=args.seq_len, dim=args.dim)
    seq_out = args.seq_len - args.kernel_size + 1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    xt = torch.from_numpy(x_all).to(device=device, dtype=torch.float32).permute(0, 2, 1)
    yt = torch.from_numpy(labels.astype(np.float32)).to(device).unsqueeze(1)

    torch.manual_seed(args.seed)
    model = TextCNNSquare(args.dim, args.channels, args.kernel_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_val_f1, best_thr, best_state = -1.0, 0.0, None
    for _ in range(args.epochs):
        model.train()
        perm = train_ids[torch.randperm(len(train_ids)).numpy()]
        for start in range(0, len(perm), 128):
            idx = perm[start:start + 128]
            optimizer.zero_grad()
            loss = (F.binary_cross_entropy_with_logits(model(xt[idx]), yt[idx])
                    + args.l1 * model.conv.weight.abs().mean())
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_scores = model(xt[val_ids]).squeeze(1).cpu().numpy()
            thr, _, f1 = select_threshold(labels[val_ids], val_scores)
            if f1 > best_val_f1:
                best_val_f1, best_thr = f1, thr
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()
    with torch.no_grad():
        val_float = model(xt[val_ids]).squeeze(1).cpu().numpy() - best_thr
        test_float = model(xt[test_ids]).squeeze(1).cpu().numpy() - best_thr
    float_test_metrics = metrics(labels[test_ids], test_float)

    # Export canonical (k, D, C) layout for CRT Conv1d and (C, 1) for head.
    w1_f = best_state['conv.weight'].permute(2, 1, 0).numpy().astype(np.float64)
    b1_f = best_state['conv.bias'].numpy().astype(np.float64)
    w2_f = best_state['head.weight'].numpy().astype(np.float64).T
    b2_f = np.array([(float(best_state['head.bias'].item()) - float(best_thr)) * seq_out], dtype=np.float64)

    primes = tuple(args.primes)
    limit = math.prod(primes) // 2
    best_q = None
    for s1 in (8, 12, 16, 20, 24, 32, 48, 64):
        w1_q = np.rint(w1_f * s1).astype(np.int64)
        b1_q = np.rint(b1_f * s1).astype(np.int64)
        h_max = np.abs(w1_q).sum(axis=(0, 1)) + np.abs(b1_q)
        if int(h_max.max()) ** 2 * seq_out >= limit:
            continue
        xv = torch.from_numpy(x_all[val_ids].astype(np.float64)).permute(0, 2, 1)
        hv = F.conv1d(xv, torch.from_numpy(w1_q.astype(np.float64)).permute(2, 1, 0),
                      torch.from_numpy(b1_q.astype(np.float64))).round().to(torch.int64).permute(0, 2, 1).numpy()
        pv = (hv * hv).sum(axis=1)
        for s2 in (8, 16, 32, 64, 128, 256):
            w2_q = np.rint(w2_f * s2).astype(np.int64)
            raw_val = (pv @ w2_q).reshape(-1)
            thr_q, vm, f1_q = select_threshold(labels[val_ids], raw_val)
            b2_q = np.array([-int(math.floor(thr_q))], dtype=np.int64)
            bounds = global_bounds(w1_q, b1_q, w2_q, b2_q, seq_out)
            if bounds['score_max_abs'] >= limit:
                continue
            agree_val = float((((raw_val + b2_q[0]) >= 0) == (val_float >= 0)).mean())
            rank = (f1_q, agree_val, vm['accuracy'])
            if best_q is None or rank > best_q[0]:
                best_q = (rank, s1, s2, w1_q, b1_q, w2_q, b2_q, bounds)

    if best_q is None:
        raise RuntimeError('no integer quantization fits the certified CRT modulus bound')
    _, s1, s2, w1_q, b1_q, w2_q, b2_q, bounds = best_q
    test_int_scores = integer_forward(x_all[test_ids], w1_q, b1_q, w2_q, b2_q).reshape(-1)
    int_test_metrics = metrics(labels[test_ids], test_int_scores)
    agreement = float(((test_int_scores >= 0) == (test_float >= 0)).mean())

    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output / 'model.npz'
    np.savez(
        checkpoint,
        architecture=np.array('text_cnn_square'),
        primes=np.asarray(primes, dtype=np.int64),
        seq_len=np.int64(args.seq_len),
        dim=np.int64(args.dim),
        channels=np.int64(args.channels),
        kernel_size=np.int64(args.kernel_size),
        x_test=x_all[test_ids].astype(np.int8),
        y_test=labels[test_ids].astype(np.int64),
        w1=w1_q.astype(np.int32),
        b1=b1_q.astype(np.int32),
        w2=w2_q.astype(np.int32),
        b2=b2_q.astype(np.int64),
        float_w1=w1_f.astype(np.float32),
        float_b1=b1_f.astype(np.float32),
        float_w2=w2_f.astype(np.float32),
        float_b2=b2_f.astype(np.float32),
    )
    meta = dict(
        architecture='text_cnn_square',
        dataset_url=DATA_URL,
        dataset_citation=DATA_CITATION,
        dataset_sha256=data_sha,
        train_examples=int(len(train_ids)),
        validation_examples=int(len(val_ids)),
        test_examples=int(len(test_ids)),
        seq_len=args.seq_len,
        dim=args.dim,
        channels=args.channels,
        kernel_size=args.kernel_size,
        input_nonzero_fraction=float((x_all[test_ids] != 0).mean()),
        primes=list(primes),
        crt_signed_limit=limit,
        scales=dict(s1=s1, s2=s2),
        global_certified_bounds=bounds,
        float_test_metrics=float_test_metrics,
        integer_test_metrics=int_test_metrics,
        float_integer_agreement=agreement,
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    )
    (args.output / 'metadata.json').write_text(json.dumps(meta, indent=2) + '\n')
    print(json.dumps(meta, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('.local/datasets/SMSSpamCollection'))
    parser.add_argument('--output', type=Path, default=Path('experiments/results/examples/text_cnn_classifier'))
    parser.add_argument('--seq-len', type=int, default=48)
    parser.add_argument('--dim', type=int, default=256)
    parser.add_argument('--channels', type=int, default=128)
    parser.add_argument('--kernel-size', type=int, default=5)
    parser.add_argument('--primes', type=int, nargs='+', default=list(DEFAULT_CRT_PRIMES))
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=2e-2)
    parser.add_argument('--l1', type=float, default=2e-4)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--threads', type=int, default=4)
    train_and_export(parser.parse_args())


if __name__ == '__main__':
    main()
