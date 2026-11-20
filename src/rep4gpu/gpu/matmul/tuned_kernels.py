"""Validated tile selection for exact GPU Rep4 local products."""
from functools import lru_cache

import torch
import triton

from .kernels import CompactWeight, _tensor_mm, local_products, ring_matmul


@lru_cache(maxsize=None)
def _supports_tuned(device):
    # These integer MMA tiles and shared-memory budgets are tested on SM 8.9.
    return torch.cuda.get_device_capability(device) == (8, 9)


def local_products_tuned(x0, x1, x2, y0, y1, y2, bits=64):
    """Use exact ring32 modular-add reconstruction for SM 8.9 prefill batches.

    Compact ring32 weights with at least 16 rows select the validated 32/128/32
    modular-add tile on SM 8.9. Other devices, rings, formats, and smaller
    batches retain the baseline kernel.
    """
    xs = (x0, x1, x2)
    ys = (y0, y1, y2)
    if (bits != 32 or not all(isinstance(y, CompactWeight) for y in ys)
            or x0.ndim != 2 or x0.shape[0] < 16 or not _supports_tuned(x0.device)):
        return local_products(*xs, *ys, bits)
    from .ring32_opaque import local_products_opaque
    return local_products_opaque(*xs, *ys, tile=(32, 128, 32))


def ring_matmul_tuned(a, b, bits=64):
    """Use the validated prefill tile for SM 8.9 compact ring32 matrix products."""
    if bits != 32 or not isinstance(b, CompactWeight) or a.ndim != 2 or a.shape[0] < 32 or not _supports_tuned(a.device):
        return ring_matmul(a, b, bits)
    if a.dtype != torch.int64 or not a.is_cuda or a.device != b.device or a.shape[1] != b.shape[0]:
        raise ValueError('expected a compatible int64 CUDA input matrix')
    m, k = a.shape
    n = b.shape[1]
    out = torch.empty((m, n), device=a.device, dtype=torch.int64)
    if m and n and k:
        with torch.cuda.device(a.device):
            _tensor_mm[(triton.cdiv(m, 32), triton.cdiv(n, 128))](
                a.contiguous(), b.words, out, m, n, k, 32, 32, 128, 64, False, num_warps=4, num_stages=1)
    elif k == 0:
        out.zero_()
    return out
