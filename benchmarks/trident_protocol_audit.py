"""Executable arithmetic checks of Trident's published nonlinear steps.

Run ``python -m benchmarks.trident_protocol_audit --output results/trident_audit.json``.
This uses public fixtures, not a private inference implementation. Failed paper
identities are retained as counterexamples; they are never installed as runtime
operations. A zero exit status means the audit ran, not that Trident passed.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from benchmarks.inference import fixture


def published_sign(value, random_mask, bits=32):
    """Evaluate NDSS Figure 19 literally, including modular multiplication."""
    mask = (1 << bits) - 1
    v = np.asarray(value, dtype=np.int64).astype(np.uint64) & np.uint64(mask)
    r = np.asarray(random_mask, dtype=np.uint64) & np.uint64(mask)
    return ((r >> (bits - 1)) ^ (((v * r) & np.uint64(mask)) >> (bits - 1))).astype(np.int64)


def prefix_subtract(a, b, bits=32):
    """Public Boolean-circuit oracle for a-b; not a shared protocol."""
    mask = np.uint64((1 << bits) - 1)
    a = np.asarray(a, dtype=np.uint64) & mask
    b = (~np.asarray(b, dtype=np.uint64)) & mask
    propagate = a ^ b
    initial = propagate.copy()
    generate = (a & b) | (propagate & np.uint64(1))
    distance = 1
    while distance < bits:
        generate |= propagate & ((generate << distance) & mask)
        propagate &= ((propagate << distance) | np.uint64((1 << distance) - 1)) & mask
        distance *= 2
    return (initial ^ ((generate << 1) | np.uint64(1))) & mask


def truncation_check_residual(parts, shift, bits=32):
    """Residual of Figure 18's honest-mask check, with c cancelled."""
    modulus = 1 << bits
    r = sum(parts) % modulus
    truncated = r >> shift
    low_sum = sum(p & ((1 << shift) - 1) for p in parts)
    return (r - (truncated << shift) - low_sum) % modulus


def masked_signed_truncation(value, mask, shift, bits=32):
    """Evaluate the uncorrected masked truncation identity on signed words."""
    def signed(x):
        return ((x + (1 << (bits - 1))) % (1 << bits)) - (1 << (bits - 1))
    return signed((signed(value - mask) >> shift) + (signed(mask) >> shift))


def run_audit(width=2048, hidden=8192, out=2048, seed=20261110):
    torch.set_num_threads(1)
    rng = np.random.default_rng(seed)
    _, tensors, _ = fixture(1, width, hidden, out, 14)
    x, w1, b1, w2, b2 = [t.numpy() for t in tensors]
    preactivation = ((x @ w1) + (b1 << 14)) >> 14
    masks = rng.integers(0, 1 << 32, preactivation.shape, dtype=np.uint64)
    observed = published_sign(preactivation, masks)
    expected = (preactivation < 0).astype(np.int64)
    candidate_h = preactivation * (1 - observed)
    expected_h = np.maximum(preactivation, 0)
    candidate_logits = ((candidate_h @ w2) >> 14) + b2
    expected_logits = ((expected_h @ w2) >> 14) + b2

    # The revised split requires subtraction: v=(m-lambda3)-(lambda1+lambda2).
    split_mask = rng.integers(0, 1 << 32, preactivation.shape, dtype=np.uint64)
    left = (preactivation.astype(np.uint64) + split_mask) & np.uint64(0xffffffff)
    corrected = (prefix_subtract(left, split_mask) >> 31).astype(np.int64)
    triples = rng.integers(0, 1 << 32, (10000, 3), dtype=np.uint64)
    rejects = sum(truncation_check_residual(tuple(map(int, p)), 14) != 0 for p in triples)
    paths = [Path(__file__), Path('benchmarks/inference.py')]
    paths += [p for p in (Path('work/trident-audit/trident.pdf'),
                           Path('work/trident-audit/trident-ndss.pdf')) if p.exists()]
    return dict(
        methodology='Public arithmetic audit; no private nonlinear execution or network timing.',
        seed=seed, shape=[1, width, hidden, out], ring_bits=32, fractional_bits=14,
        sources=dict(ndss='https://www.ndss-symposium.org/wp-content/uploads/2020/02/23005-paper.pdf',
                     revised='https://arxiv.org/abs/1912.02631v2'),
        source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        original_sign=dict(counterexample=dict(value=2, mask=1 << 30, expected=0,
                                               actual=int(published_sign(2, 1 << 30))),
                           hidden_elements=int(preactivation.size),
                           wrong_signs=int(np.count_nonzero(observed != expected)),
                           wrong_logits=int(np.count_nonzero(candidate_logits != expected_logits)),
                           max_logit_error=float(np.max(np.abs(candidate_logits - expected_logits))) / (1 << 14),
                           privacy_counterexample='Opened r*v is always zero for v=0 and always even for even v.'),
        revised_sign_oracle=dict(wrong_signs=int(np.count_nonzero(corrected != expected)),
                                 private_protocol_implemented=False,
                                 convention='Subtract the positive mask sum; Boolean prefix circuit evaluated on public operands.'),
        literal_truncation_check=dict(parts=[16383, 1, 0], shift=14,
                                      honest_residual=truncation_check_residual((16383, 1, 0), 14),
                                      expected_residual=0, random_honest_cases=10000,
                                      random_honest_rejections=rejects),
        uncorrected_ring32_truncation=dict(value=1 << 29, mask=-(1 << 31), shift=14,
                                           expected=(1 << 29) >> 14,
                                           actual=masked_signed_truncation(1 << 29, -(1 << 31), 14),
                                           interpretation='Wrap correction or a stronger range/slack contract is required.'),
        full_private_ffn_validated=False, sub500kb_validated=False,
        earlier_453kb_projection_valid=False,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('results/trident_audit.json'))
    args = parser.parse_args()
    result = run_audit()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
