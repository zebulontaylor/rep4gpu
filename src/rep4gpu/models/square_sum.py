"""Checked contraction of squared hidden states into one shared score."""
from dataclasses import dataclass

import torch

from ..protocol.runtime import Share


@dataclass(frozen=True)
class PublicSquareSigns:
    values: tuple
    tensor: torch.Tensor


def prepare_square_signs(rt, signs):
    """Validate and cache public +/-1 coefficients outside CUDA graph capture.

    The ordinary phase-graph warmup populates this cache before recording.
    Call explicitly before a manual CUDA capture that has no warmup.
    """
    values = tuple(signs)
    if any(type(value) is not int or value not in (-1, 1) for value in values):
        raise ValueError('square sum signs must be public Python integers equal to -1 or +1')
    cache = rt.__dict__.setdefault('_square_sum_sign_cache', {})
    if values not in cache:
        cache[values] = PublicSquareSigns(values, torch.tensor(values, dtype=torch.int64, device=rt.device))
    return cache[values]


def signed_square_sum(rt, x, signs):
    """Return a shared ``sum(signs[j] * x[:, j]**2)`` of shape ``[batch, 1]``.

    Signs are public architecture constants. Inputs and the result remain
    shared. Each role sums its three local product contributions before the
    ordinary checked Rep4 resharing, so payload size is independent of hidden
    width. The ring wraps normally; callers establish signed score bounds.
    """
    if not isinstance(x, Share) or x.binary or len(x.shape) != 2:
        raise ValueError('square sum requires an arithmetic matrix share')
    rt._validate_share_storage(x)
    if x.device != rt.device:
        raise ValueError('square sum input must use the runtime device')
    public = signs if isinstance(signs, PublicSquareSigns) else prepare_square_signs(rt, signs)
    if len(public.values) != x.shape[1] or public.tensor.device != x.device:
        raise ValueError('public square signs must match the hidden width and device')
    if not x.shape[0] or not x.shape[1]:
        return rt.public(torch.zeros((x.shape[0], 1), dtype=torch.int64, device=x.device))
    if x.device.type == 'cuda':
        from ..gpu.matmul.square_sum import local_square_sum
        local = local_square_sum(x.parts, public.tensor, rt.bits)
    else:
        a, b, c = x.parts
        local = tuple(rt._normalize((part * public.tensor).sum(dim=1, keepdim=True))
                      for part in (a*a + 2*a*b, b*b + 2*b*c, 2*a*c))
    contributions = [None] * 5
    contributions[(rt.party - 1) % 4], contributions[rt.party], contributions[4] = local
    return rt._reshare_product(contributions, False, x.device)
