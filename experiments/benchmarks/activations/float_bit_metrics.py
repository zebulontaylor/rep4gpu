"""Compare output encodings after casting to floating-point storage formats.

These measurements cast completed float64 outputs. They do not simulate native
low-precision matmul or activation evaluation. ULP distances count representable
encoding steps between finite values, with positive and negative zero merged.
A Hamming distance counts differing bits, not lost numerical precision.
"""
from __future__ import annotations

import numpy as np
import torch


FORMATS = {
    'fp32': (torch.float32, 32),
    'fp16': (torch.float16, 16),
    'bf16': (torch.bfloat16, 16),
    'fp8_e4m3fn': (torch.float8_e4m3fn, 8),
    'fp8_e5m2': (torch.float8_e5m2, 8),
}


def _words(tensor, bits):
    integer = {8: torch.uint8, 16: torch.int16, 32: torch.int32}[bits]
    # Mask after conversion so signed integer views retain their raw bit pattern.
    return tensor.contiguous().reshape(-1).view(integer).numpy().astype(np.int64) & ((1 << bits) - 1)


def _ordered(words, bits):
    """Map finite signed-magnitude float encodings to monotone integer ranks."""
    sign = 1 << (bits - 1)
    magnitude = words & (sign - 1)
    return np.where((words & sign) != 0, -magnitude, magnitude)


def _nonfinite_counts(values):
    return {
        'finite': int(np.isfinite(values).sum()),
        'nan': int(np.isnan(values).sum()),
        'positive_infinity': int(np.isposinf(values).sum()),
        'negative_infinity': int(np.isneginf(values).sum()),
    }


def compare_formats(actual: torch.Tensor, reference: torch.Tensor):
    """Return per-format bit, numerical, and finite-pair ULP mismatch metrics.

    Both arguments must be nonempty CPU float64 tensors with the same shape.
    Bit metrics include every encoding, including NaNs and infinities. Numerical
    equality treats signed zeros as equal and NaNs as unequal. ULP and absolute
    error summaries include only pairs whose cast values are both finite; their
    denominator and excluded-pair count are explicit. Input nonfinite counts
    remain visible even if a format's conversion saturates them to finite values.
    """
    if (not isinstance(actual, torch.Tensor) or not isinstance(reference, torch.Tensor)
            or actual.dtype != torch.float64 or reference.dtype != torch.float64
            or actual.device.type != 'cpu' or reference.device.type != 'cpu'
            or actual.shape != reference.shape or actual.numel() == 0):
        raise ValueError('comparison requires nonempty matching CPU float64 tensors')
    actual = actual.detach()
    reference = reference.detach()
    count = actual.numel()
    source_actual = _nonfinite_counts(actual.numpy())
    source_reference = _nonfinite_counts(reference.numpy())
    results = {}
    for name, (dtype, bits) in FORMATS.items():
        cast_actual, cast_reference = actual.to(dtype), reference.to(dtype)
        a, r = _words(cast_actual, bits), _words(cast_reference, bits)
        av = cast_actual.double().reshape(-1).numpy()
        rv = cast_reference.double().reshape(-1).numpy()
        xor = a ^ r
        per_bit = ((xor[:, None] >> np.arange(bits)) & 1)
        hamming = per_bit.sum(axis=1)
        finite = np.isfinite(av) & np.isfinite(rv)
        distances = np.abs(_ordered(a[finite], bits) - _ordered(r[finite], bits))
        errors = np.abs(av[finite] - rv[finite])
        matches = int((xor == 0).sum())
        results[name] = {
            'format_bits': bits,
            'elements': count,
            'exact_match_count': matches,
            'exact_match_fraction': matches / count,
            'encoded_mismatch_count': count - matches,
            'numerical_mismatch_count': int((av != rv).sum()),
            'signed_zero_only_mismatch_count': int(((av == 0) & (rv == 0) & (xor != 0)).sum()),
            'hamming_total': int(hamming.sum()),
            'hamming_mean': float(hamming.mean()),
            'hamming_max': int(hamming.max()),
            'hamming_histogram': {str(i): int((hamming == i).sum()) for i in range(bits + 1)},
            'per_bit_mismatch_counts_lsb_first': per_bit.sum(axis=0).tolist(),
            'source_actual_counts': source_actual,
            'source_reference_counts': source_reference,
            'cast_actual_counts': _nonfinite_counts(av),
            'cast_reference_counts': _nonfinite_counts(rv),
            'finite_pair_count': int(finite.sum()),
            'nonfinite_pair_count': int((~finite).sum()),
            'ulp_distance_max': int(distances.max()) if distances.size else None,
            'ulp_distance_mean': float(distances.mean()) if distances.size else None,
            'ulp_distance_p99': int(np.quantile(distances, .99, method='nearest')) if distances.size else None,
            'cast_max_abs_error_finite_pairs': float(errors.max()) if errors.size else None,
        }
    return results
