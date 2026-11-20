"""Train and export a Q14 fixed-point multi-layer ReLU topic classifier on AG News.

Task: binary classification of AG News topic "Sci/Tech" (label == 3) vs other
topics (World, Sports, Business). All learned parameters (4,720,641 weights and
biases across three Linear layers) are exported for secret-shared Ring32 Rep4
inference. Input featurization is deterministic signed n-gram hashing with zero
learned parameters.

Run from the repository root:
    python -m experiments.examples.topic_mlp_classifier
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy import sparse
from sklearn.feature_extraction.text import HashingVectorizer
import torch
import torch.nn as nn
import torch.nn.functional as F


TOPIC_NAMES = ('World', 'Sports', 'Business', 'Sci/Tech')
TARGET_LABEL = 3  # Sci/Tech


def load_ag_news(train_path: Path, test_path: Path) -> tuple[list[str], np.ndarray, list[str], np.ndarray, dict[str, str]]:
    """Load local AG News parquet files and return binary Sci/Tech labels."""
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(f'AG News parquet files not found at {train_path} and {test_path}')
    tr = pq.read_table(train_path)
    te = pq.read_table(test_path)
    tr_texts = tr['text'].to_pylist()
    te_texts = te['text'].to_pylist()
    tr_y = (tr['label'].to_numpy() == TARGET_LABEL).astype(np.int64)
    te_y = (te['label'].to_numpy() == TARGET_LABEL).astype(np.int64)
    hashes = dict(
        train_parquet_sha256=hashlib.sha256(train_path.read_bytes()).hexdigest(),
        test_parquet_sha256=hashlib.sha256(test_path.read_bytes()).hexdigest(),
    )
    return tr_texts, tr_y, te_texts, te_y, hashes


def featurize_hashed_ngrams(texts: list[str], features: int = 4096) -> sparse.csr_matrix:
    """Deterministic signed feature hashing of word (1,2)-grams + char_wb 4-grams, L2-normalized."""
    if features < 64 or features % 2 != 0:
        raise ValueError('features must be an even integer >= 64')
    word_vec = HashingVectorizer(
        n_features=features, analyzer='word', ngram_range=(1, 2),
        lowercase=True, alternate_sign=True, norm='l2', dtype=np.float32,
    )
    char_vec = HashingVectorizer(
        n_features=features, analyzer='char_wb', ngram_range=(4, 4),
        lowercase=True, alternate_sign=True, norm='l2', dtype=np.float32,
    )
    x = (word_vec.transform(texts) + 0.5 * char_vec.transform(texts)).tocsr()
    row_norms = np.sqrt(np.asarray(x.multiply(x).sum(axis=1)).reshape(-1))
    inv_norms = np.where(row_norms > 0, 1.0 / np.maximum(row_norms, 1e-12), 0.0).astype(np.float32)
    return sparse.diags(inv_norms) @ x


def stratified_split(labels: np.ndarray, val_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_ids, val_ids = [], []
    for cls in (0, 1):
        idx = rng.permutation(np.flatnonzero(labels == cls))
        n_val = max(1, int(round(len(idx) * val_fraction)))
        val_ids.append(idx[:n_val])
        train_ids.append(idx[n_val:])
    return rng.permutation(np.concatenate(train_ids)), rng.permutation(np.concatenate(val_ids))


class TopicMLP(nn.Module):
    """3-layer ReLU MLP: D -> H1 -> H2 -> 1."""

    def __init__(self, features: int = 4096, h1: int = 1024, h2: int = 512, dropout: float = 0.35):
        super().__init__()
        self.fc1 = nn.Linear(features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, 1)
        self.dropout = nn.Dropout(dropout)

    def forward_with_preacts(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z1 = self.fc1(x)
        a1 = self.dropout(F.relu(z1))
        z2 = self.fc2(a1)
        a2 = self.dropout(F.relu(z2))
        z3 = self.fc3(a2)
        return z1, z2, z3

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_with_preacts(x)[2]


def binary_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float | int]:
    pred = scores.reshape(-1) >= 0
    pos = labels == 1
    tp = int(np.count_nonzero(pred & pos))
    fp = int(np.count_nonzero(pred & ~pos))
    fn = int(np.count_nonzero(~pred & pos))
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-12, prec + rec)
    return dict(
        accuracy=float(np.mean(pred == pos)),
        precision=float(prec),
        recall=float(rec),
        f1=float(f1),
        false_positives=fp,
        false_negatives=fn,
    )


def select_threshold(labels: np.ndarray, logits: np.ndarray) -> tuple[float, dict]:
    """Select decision threshold maximizing validation F1 on a fine quantile grid."""
    logits = logits.reshape(-1)
    candidates = np.unique(np.concatenate(([0.0], np.quantile(logits, np.linspace(0.05, 0.95, 361)))))
    best = None
    for thr in candidates:
        m = binary_metrics(labels, logits - thr)
        key = (m['f1'], m['accuracy'], -abs(float(thr)))
        if best is None or key > best[0]:
            best = (key, float(thr), m)
    return best[1], best[2]


def homogeneous_rescale(params: dict[str, np.ndarray], x_dense: np.ndarray, target_max: float = 1.6) -> dict[str, np.ndarray]:
    """Exactly rescale ReLU layers using positive homogeneity so pre-activations fit Q14 quarter-ring.

    ReLU(c * z) = c * ReLU(z) for c > 0. Scaling (w1, b1) by c1 scales h1 by c1
    and requires scaling b2 by c1; scaling (w2, b2) by c2 scales h2 by c2 and
    requires scaling b3 by c2; scaling (w3, b3) by c3 preserves the sign of score.
    """
    w1, b1 = params['w1'].astype(np.float64).copy(), params['b1'].astype(np.float64).copy()
    w2, b2 = params['w2'].astype(np.float64).copy(), params['b2'].astype(np.float64).copy()
    w3, b3 = params['w3'].astype(np.float64).copy(), params['b3'].astype(np.float64).copy()

    z1 = x_dense.astype(np.float64) @ w1 + b1
    c1 = target_max / max(float(np.max(np.abs(z1))), 1e-12)
    w1 *= c1
    b1 *= c1
    b2 *= c1
    a1 = np.maximum(z1 * c1, 0.0)

    z2 = a1 @ w2 + b2
    c2 = target_max / max(float(np.max(np.abs(z2))), 1e-12)
    w2 *= c2
    b2 *= c2
    b3 *= c2
    a2 = np.maximum(z2 * c2, 0.0)

    z3 = a2 @ w3 + b3
    c3 = target_max / max(float(np.max(np.abs(z3))), 1e-12)
    w3 *= c3
    b3 *= c3

    return dict(w1=w1, b1=b1, w2=w2, b2=b2, w3=w3, b3=b3)


def checked_int_matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    max_a = int(np.max(np.abs(a), initial=0))
    max_b = int(np.max(np.abs(b), initial=0))
    if max_a * max_b * a.shape[1] >= 2**62:
        raise ValueError('integer matmul can overflow int64')
    return a.astype(np.int64) @ b.astype(np.int64)


def propagate_bounds(x_q: np.ndarray, q_params: dict[str, np.ndarray],
                     fractional_bits: int, bits: int = 32, rounding: str = 'probabilistic') -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    """Validate exact floor outputs and propagated rounding intervals in ring32.

    Matches the Rep4 fixed-point schedule:
      raw1 = x @ w1 + b1 * scale;  h1 = ReLU(trunc(raw1, f))
      raw2 = h1 @ w2 + b2 * scale; h2 = ReLU(trunc(raw2, f))
      raw3 = h2 @ w3 + b3 * scale; label = (raw3 >= 0)  [no truncation needed for sign]
    """
    if rounding not in ('floor', 'probabilistic'):
        raise ValueError('rounding must be floor or probabilistic')
    scale = 1 << fractional_bits
    half_ring = 1 << (bits - 1)
    quarter_ring = 1 << (bits - 2)
    w1, b1 = q_params['w1'].astype(np.int64), q_params['b1'].astype(np.int64)
    w2, b2 = q_params['w2'].astype(np.int64), q_params['b2'].astype(np.int64)
    w3, b3 = q_params['w3'].astype(np.int64), q_params['b3'].astype(np.int64)

    # Exact floor reference
    raw1 = checked_int_matmul(x_q, w1) + b1 * scale
    h1_floor = np.maximum(raw1 >> fractional_bits, 0)
    raw2_floor = checked_int_matmul(h1_floor, w2) + b2 * scale
    h2_floor = np.maximum(raw2_floor >> fractional_bits, 0)
    raw3_floor = checked_int_matmul(h2_floor, w3) + b3 * scale

    if rounding == 'floor':
        raw2_lo = raw2_hi = raw2_floor
        raw3_lo = raw3_hi = raw3_floor
        h1_hi, h2_hi = h1_floor, h2_floor
    else:
        h1_lo = h1_floor
        h1_hi = np.maximum((raw1 + (scale - 1)) >> fractional_bits, 0)
        w2_pos, w2_neg = np.maximum(w2, 0), np.minimum(w2, 0)
        raw2_lo = checked_int_matmul(h1_lo, w2_pos) + checked_int_matmul(h1_hi, w2_neg) + b2 * scale
        raw2_hi = checked_int_matmul(h1_hi, w2_pos) + checked_int_matmul(h1_lo, w2_neg) + b2 * scale
        h2_lo = np.maximum(raw2_lo >> fractional_bits, 0)
        h2_hi = np.maximum((raw2_hi + (scale - 1)) >> fractional_bits, 0)
        w3_pos, w3_neg = np.maximum(w3, 0), np.minimum(w3, 0)
        raw3_lo = checked_int_matmul(h2_lo, w3_pos) + checked_int_matmul(h2_hi, w3_neg) + b3 * scale
        raw3_hi = checked_int_matmul(h2_hi, w3_pos) + checked_int_matmul(h2_lo, w3_neg) + b3 * scale

    max_raw1 = int(np.max(np.abs(raw1)))
    max_raw2 = int(max(np.max(np.abs(raw2_lo)), np.max(np.abs(raw2_hi))))
    max_raw3 = int(max(np.max(np.abs(raw3_lo)), np.max(np.abs(raw3_hi))))
    max_h1 = int(np.max(np.abs(h1_hi)))
    max_h2 = int(np.max(np.abs(h2_hi)))

    # Truncated layers (raw1, raw2) require quarter-ring for probabilistic rounding;
    # raw3 is not truncated (only sign-checked) so requires signed half-ring.
    trunc_limit = quarter_ring if rounding == 'probabilistic' else half_ring
    if max(max_raw1, max_raw2) >= trunc_limit:
        raise ValueError(
            f'truncation accumulator max({max_raw1}, {max_raw2}) violates {rounding} limit {trunc_limit}'
        )
    if max_raw3 >= half_ring:
        raise ValueError(f'output score accumulator {max_raw3} violates signed ring{bits} limit {half_ring}')
    for name, arr in [('x', x_q), ('w1', w1), ('b1', b1), ('w2', w2), ('b2', b2), ('w3', w3), ('b3', b3)]:
        if int(arr.min()) < -half_ring or int(arr.max()) >= half_ring:
            raise ValueError(f'{name} exceeds signed ring{bits}')

    ambiguous = int(np.count_nonzero((raw3_lo < 0) & (raw3_hi >= 0)))
    bounds = dict(
        ring_bits=bits,
        fractional_bits=fractional_bits,
        rounding=rounding,
        truncation_limit=trunc_limit,
        signed_ring_limit=half_ring,
        raw1_max_abs=max_raw1,
        h1_max_abs=max_h1,
        raw2_max_abs=max_raw2,
        h2_max_abs=max_h2,
        raw3_max_abs=max_raw3,
        max_score_interval_width_q28=int(np.max(raw3_hi - raw3_lo)),
        max_score_interval_width_float=float(np.max(raw3_hi - raw3_lo) / (scale * scale)),
        ambiguous_sign_examples=ambiguous,
    )
    return bounds, raw3_floor, raw3_lo, raw3_hi


def train_and_export(args: argparse.Namespace) -> dict:
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    tr_texts, tr_y, te_texts, te_y, data_hashes = load_ag_news(args.train_parquet, args.test_parquet)
    train_ids, val_ids = stratified_split(tr_y, val_fraction=0.1, seed=args.seed)

    x_tr_all = featurize_hashed_ngrams(tr_texts, features=args.features)
    x_te_csr = featurize_hashed_ngrams(te_texts, features=args.features)
    x_train_csr = x_tr_all[train_ids]
    y_train = tr_y[train_ids]
    x_val_dense = x_tr_all[val_ids].toarray()
    y_val = tr_y[val_ids]
    x_test_dense = x_te_csr.toarray()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TopicMLP(args.features, args.h1, args.h2, dropout=args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    pos_weight = torch.tensor([args.pos_weight], device=device)

    x_val_t = torch.from_numpy(x_val_dense).to(device)
    best = None
    rng = np.random.default_rng(args.seed)

    for epoch in range(1, args.epochs + 1):
        model.train()
        order = rng.permutation(len(y_train))
        for start in range(0, len(order), args.batch_size):
            idx = order[start:start + args.batch_size]
            xb = torch.from_numpy(x_train_csr[idx].toarray()).to(device)
            yb = torch.from_numpy(y_train[idx].astype(np.float32)).to(device).unsqueeze(1)
            optimizer.zero_grad(set_to_none=True)
            z1, z2, z3 = model.forward_with_preacts(xb)
            bce = F.binary_cross_entropy_with_logits(z3, yb, pos_weight=pos_weight)
            # Soft range regularizer keeps pre-activations comfortably inside Q14 quarter-ring.
            reg = 0.01 * (
                F.relu(z1.abs() - 1.5).square().mean()
                + F.relu(z2.abs() - 1.5).square().mean()
                + F.relu(z3.abs() - 1.5).square().mean()
            )
            (bce + reg).backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_logits = []
            for start in range(0, len(y_val), 1024):
                val_logits.append(model(x_val_t[start:start + 1024]).squeeze(1).cpu().numpy())
            val_scores = np.concatenate(val_logits)
        thr, val_m = select_threshold(y_val, val_scores)
        key = (val_m['f1'], val_m['accuracy'])
        if best is None or key > best[0]:
            best = (key, epoch, thr, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, val_m)

    _, best_epoch, best_thr, best_state, best_val_metrics = best
    raw_params = dict(
        w1=best_state['fc1.weight'].numpy().T.astype(np.float64),
        b1=best_state['fc1.bias'].numpy().astype(np.float64),
        w2=best_state['fc2.weight'].numpy().T.astype(np.float64),
        b2=best_state['fc2.bias'].numpy().astype(np.float64),
        w3=best_state['fc3.weight'].numpy().T.astype(np.float64),
        b3=np.array([float(best_state['fc3.bias'].item()) - best_thr], dtype=np.float64),
    )

    # Exact homogeneous rescaling calibrated on validation set (never touching test set).
    scaled_params = homogeneous_rescale(raw_params, x_val_dense, target_max=args.target_max_preact)

    # Evaluate float model on held-out test set.
    h1_f = np.maximum(x_test_dense.astype(np.float64) @ scaled_params['w1'] + scaled_params['b1'], 0.0)
    h2_f = np.maximum(h1_f @ scaled_params['w2'] + scaled_params['b2'], 0.0)
    test_float_scores = (h2_f @ scaled_params['w3'] + scaled_params['b3']).reshape(-1)
    float_test_metrics = binary_metrics(te_y, test_float_scores)

    # Quantize to Q_f integer fixed-point.
    f_bits = args.fractional_bits
    scale = 1 << f_bits
    x_test_q = np.rint(x_test_dense.astype(np.float64) * scale).astype(np.int32)
    q_params = {k: np.rint(v * scale).astype(np.int32) for k, v in scaled_params.items()}

    # Validate bounds for BOTH exact floor and probabilistic Q14 on the full 7,600-example test set.
    floor_bounds, raw3_floor, _, _ = propagate_bounds(x_test_q, q_params, f_bits, bits=32, rounding='floor')
    prob_bounds, _, raw3_lo, raw3_hi = propagate_bounds(x_test_q, q_params, f_bits, bits=32, rounding='probabilistic')

    # Also check Layer 1 global Cauchy-Schwarz worst-case bound over ALL unit-L2 inputs.
    max_x_l2 = float(np.sqrt((x_test_q.astype(np.float64) ** 2).sum(axis=1)).max())
    max_w1_col_l2 = float(np.sqrt((q_params['w1'].astype(np.float64) ** 2).sum(axis=0)).max())
    l1_global_cs_bound = int(np.ceil(max_x_l2 * max_w1_col_l2 + np.max(np.abs(q_params['b1'])) * scale))

    int_test_metrics = binary_metrics(te_y, raw3_floor.reshape(-1))
    float_int_agreement = float(np.mean((raw3_floor.reshape(-1) >= 0) == (test_float_scores >= 0)))
    max_logit_err = float(np.max(np.abs(raw3_floor.reshape(-1) / (scale * scale) - test_float_scores)))

    total_params = sum(int(v.size) for v in q_params.values())
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output / 'model.npz'
    np.savez(
        checkpoint,
        architecture=np.array('topic_mlp_relu'),
        fractional_bits=np.int64(f_bits),
        ring_bits=np.int64(32),
        x_test=x_test_q,
        y_test=te_y,
        **q_params,
        **{f'float_{k}': v.astype(np.float32) for k, v in scaled_params.items()},
    )
    meta = dict(
        architecture=f'Linear({args.features},{args.h1}) -> ReLU -> Linear({args.h1},{args.h2}) -> ReLU -> Linear({args.h2},1)',
        dataset='AG News (fancyzhx/ag_news)',
        task='Binary topic classification: Sci/Tech (label==3) vs World/Sports/Business',
        dataset_hashes=data_hashes,
        train_examples=int(len(train_ids)),
        validation_examples=int(len(val_ids)),
        test_examples=int(len(te_y)),
        total_parameters=total_params,
        input_nonzero_fraction=float((x_test_q != 0).mean()),
        fractional_bits=f_bits,
        ring_bits=32,
        selected_epoch=best_epoch,
        validation_metrics=best_val_metrics,
        float_test_metrics=float_test_metrics,
        integer_floor_test_metrics=int_test_metrics,
        float_integer_prediction_agreement=float_int_agreement,
        max_abs_logit_error=max_logit_err,
        layer1_global_cauchy_schwarz_bound=l1_global_cs_bound,
        layer1_global_fits_quarter_ring=bool(l1_global_cs_bound < (1 << 30)),
        test_floor_bounds=floor_bounds,
        test_probabilistic_bounds=prob_bounds,
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    )
    (args.output / 'metadata.json').write_text(json.dumps(meta, indent=2) + '\n')
    print(json.dumps(meta, indent=2))
    return meta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-parquet', type=Path, default=Path('.local/datasets/ag_news_train.parquet'))
    parser.add_argument('--test-parquet', type=Path, default=Path('.local/datasets/ag_news_test.parquet'))
    parser.add_argument('--output', type=Path, default=Path('experiments/results/examples/topic_mlp_classifier'))
    parser.add_argument('--features', type=int, default=4096)
    parser.add_argument('--h1', type=int, default=1024)
    parser.add_argument('--h2', type=int, default=512)
    parser.add_argument('--fractional-bits', type=int, default=14)
    parser.add_argument('--target-max-preact', type=float, default=1.6)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=1.5e-3)
    parser.add_argument('--weight-decay', type=float, default=2e-2)
    parser.add_argument('--dropout', type=float, default=0.35)
    parser.add_argument('--pos-weight', type=float, default=1.5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--threads', type=int, default=4)
    train_and_export(parser.parse_args())


if __name__ == '__main__':
    main()
