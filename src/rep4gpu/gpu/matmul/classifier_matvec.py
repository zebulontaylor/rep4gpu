"""Exact ring32 local products for a classifier with one output logit."""

import torch
import triton

from .kernels import _local_decode


def local_products_classifier(x0, x1, x2, y0, y1, y2, bits=32):
    """Return the three Rep4 contributions for ``[batch, width] @ [width, 1]``.

    All six inputs must be ordinary int64 CUDA tensors on the same device.
    Their full bit patterns are accepted; arithmetic uses the low 32 bits.
    The output components have shape ``[batch, 1]`` and contain canonical
    unsigned ring32 values in int64 storage. Noncontiguous inputs are copied.

    Each row uses direct modular integer products, avoiding tensor-core tiles
    with unused output columns. This changes only local arithmetic; callers
    still perform the protocol's masking, resharing, and consistency checks.
    """
    if bits != 32:
        raise ValueError('classifier matvec requires ring32')
    xs, ys = (x0, x1, x2), (y0, y1, y2)
    if any(not isinstance(value, torch.Tensor) or value.ndim != 2 or
           value.dtype != torch.int64 or not value.is_cuda for value in (*xs, *ys)):
        raise ValueError('shares must be two-dimensional int64 CUDA tensors')
    if any(value.device != x0.device for value in (*xs, *ys)):
        raise ValueError('shares must use the same CUDA device')
    if any(value.shape != x0.shape for value in xs):
        raise ValueError('input share shapes must match')
    batch, width = x0.shape
    if any(value.shape != (width, 1) for value in ys):
        raise ValueError('weight shares must have shape [input width, 1]')
    out = torch.empty((3, batch, 1), dtype=torch.int64, device=x0.device)
    if width == 0:
        out.zero_()
    elif batch:
        # Capping the reduction tile keeps long feature vectors bounded in registers.
        block = min(1024, max(32, triton.next_power_of_2(width)))
        with torch.cuda.device(x0.device):
            _local_decode[(batch, 1)](
                *(value.contiguous() for value in (*xs, *ys)), out,
                batch, 1, width, 32, 1, block, num_warps=4)
    return out.unbind(0)
