"""Pipelined ring32 local products for the 512-by-128 classifier encoder."""
from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from .kernels import CompactWeight, _pack_weights


@triton.jit
def _digit(word, index: tl.constexpr):
    return ((word >> (8 * index)).to(tl.uint8).to(tl.int32) - 128).to(tl.int8)


@triton.jit
def _encoder(X0, X1, X2, Y0, Y1, Y2, C, M: tl.constexpr,
             BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, STAGES: tl.constexpr,
             PREPACKED: tl.constexpr = False):
    component = tl.program_id(2)
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    ks = tl.arange(0, BK)
    d0 = tl.full((BM, BN), 0, tl.int32)
    d1 = tl.full((BM, BN), 0, tl.int32)
    d2 = tl.full((BM, BN), 0, tl.int32)
    d3 = tl.full((BM, BN), 0, tl.int32)
    for term in tl.static_range(2):
        if PREPACKED:
            # Fold additions into the static weights of the first two contributions.
            if component == 0:
                A = X0 if term == 0 else X1
                matrix = 3 if term == 0 else 0
            elif component == 1:
                A = X1 if term == 0 else X2
                matrix = 4 if term == 0 else 1
            else:
                A = X0 if term == 0 else X2
                matrix = 2 if term == 0 else 0
            B = Y0 + matrix * 4 * 512 * 128
        else:
            if component == 0:
                A, A2 = X0, X1
                B = Y0 if term == 0 else Y1
            elif component == 1:
                A, A2 = X1, X2
                B = Y1 if term == 0 else Y2
            else:
                A = X0 if term == 0 else X2
                A2 = X0
                B = Y2 if term == 0 else Y0
        for block in tl.range(512 // BK, num_stages=STAGES):
            k = block * BK + ks
            offsets = rows[:, None] * 512 + k[None, :]
            a = tl.load(A + offsets, rows[:, None] < M, 0).to(tl.uint32)
            if not PREPACKED and term == 0:
                a += tl.load(A2 + offsets, (rows[:, None] < M) & (component < 2), 0).to(tl.uint32)
            a += 0x80808080
            a0, a1 = _digit(a, 0), _digit(a, 1)
            a2, a3 = _digit(a, 2), _digit(a, 3)
            w = B + k[:, None] * 128 + cols[None, :]
            if PREPACKED:
                b0 = tl.load(w)
                b1 = tl.load(w + 512 * 128)
                b2 = tl.load(w + 2 * 512 * 128)
                b3 = tl.load(w + 3 * 512 * 128)
            else:
                b = tl.load(w).to(tl.uint32) + 0x80808080
                b0, b1 = _digit(b, 0), _digit(b, 1)
                b2, b3 = _digit(b, 2), _digit(b, 3)
            d0 = tl.dot(a0, b0, d0, out_dtype=tl.int32)
            d1 = tl.dot(a0, b1, d1, out_dtype=tl.int32)
            d1 = tl.dot(a1, b0, d1, out_dtype=tl.int32)
            d2 = tl.dot(a0, b2, d2, out_dtype=tl.int32)
            d2 = tl.dot(a1, b1, d2, out_dtype=tl.int32)
            d2 = tl.dot(a2, b0, d2, out_dtype=tl.int32)
            d3 = tl.dot(a0, b3, d3, out_dtype=tl.int32)
            d3 = tl.dot(a1, b2, d3, out_dtype=tl.int32)
            d3 = tl.dot(a2, b1, d3, out_dtype=tl.int32)
            d3 = tl.dot(a3, b0, d3, out_dtype=tl.int32)

    # Even the longest diagonal fits int32: 2 terms * 512 * 4 * 128**2.
    result = (d0.to(tl.uint32) + (d1.to(tl.uint32) << 8)
              + (d2.to(tl.uint32) << 16) + (d3.to(tl.uint32) << 24))
    dst = (component * M + rows[:, None]) * 128 + cols[None, :]
    tl.store(C + dst, result.to(tl.int64), rows[:, None] < M)


def local_products_pipelined(x0, x1, x2, y0, y1, y2, *, stages=3, tile=(32, 128, 32)):
    """Compute three full-width ring32 contributions with staged reduction tiles.

    Accepts int64 CUDA activation shares and compact weights of shape (512, 128).
    Independent diagonal accumulators let each tile feed all ten int8 tensor-core
    products before the next tile. The output uses canonical ring32 values in
    int64 storage, as required by the resharing step.
    """
    xs, ys = (x0, x1, x2), (y0, y1, y2)
    if any(not isinstance(y, CompactWeight) or y.shape != (512, 128) for y in ys):
        raise ValueError("pipelined encoder requires compact ring32 weights of shape (512, 128)")
    if any(y.device != x0.device for y in ys):
        raise ValueError("weights must share the input device")
    return _launch(xs, tuple(y.words for y in ys), stages, tile, prepacked=False)


@dataclass(frozen=True)
class PreparedEncoderWeights:
    """Balanced int8 planes for y0, y1, y2, y0+y1, and y1+y2, in that order."""
    limbs: torch.Tensor

    def __post_init__(self):
        if (self.limbs.shape != (5, 4, 512, 128) or self.limbs.dtype != torch.int8
                or not self.limbs.is_cuda or not self.limbs.is_contiguous()):
            raise ValueError("encoder weights require five contiguous sets of four CUDA int8 planes")

    @property
    def shape(self):
        return self.limbs.shape[2:]

    @property
    def device(self):
        return self.limbs.device


def prepare_pipelined_encoder(y0, y1, y2):
    """Pack static weight combinations once per model into five sets of byte planes."""
    ys = y0, y1, y2
    if any(not isinstance(y, CompactWeight) or y.shape != (512, 128) or y.device != y0.device
           for y in ys):
        raise ValueError("expected three matching compact 512-by-128 ring32 weights")
    a, b, c = (y.words for y in ys)
    limbs = torch.empty((5, 4, 512, 128), dtype=torch.int8, device=a.device)
    with torch.cuda.device(a.device):
        for i, word in enumerate((a, b, c, a + b, b + c)):
            _pack_weights[(triton.cdiv(word.numel(), 256), 4)](word, limbs[i], word.numel(), 4, 256)
    return PreparedEncoderWeights(limbs)


def local_products_pipelined_prepared(x0, x1, x2, weights, *, stages=3, tile=(32, 128, 32),
                                      num_warps=4):
    """Stream static signed byte planes directly into the six local matrix products."""
    if not isinstance(weights, PreparedEncoderWeights) or weights.device != x0.device:
        raise ValueError("expected prepared encoder weights on the input device")
    return _launch((x0, x1, x2), (weights.limbs, None, None), stages, tile, prepacked=True,
                   num_warps=num_warps)


def _launch(xs, weight_ptrs, stages, tile, *, prepacked, num_warps=4):
    x0 = xs[0]
    if any(x.ndim != 2 or x.shape != x0.shape or x.shape[1] != 512 or
           x.dtype != torch.int64 or not x.is_cuda or x.device != x0.device for x in xs):
        raise ValueError("pipelined encoder requires matching int64 CUDA matrices of width 512")
    bm, bn, bk = tile
    if bm not in (16, 32, 64) or bn not in (32, 64, 128) or bk not in (32, 64):
        raise ValueError("unsupported encoder tile")
    max_stages = 5 if prepacked else 4
    if stages not in range(1, max_stages + 1):
        raise ValueError(f"encoder stages must be between one and {max_stages}")
    if num_warps not in (4, 8):
        raise ValueError("encoder requires four or eight warps per block")
    m = x0.shape[0]
    out = torch.empty((3, m, 128), dtype=torch.int64, device=x0.device)
    if m:
        with torch.cuda.device(x0.device):
            _encoder[(triton.cdiv(m, bm), 128 // bn, 3)](
                *(x.contiguous() for x in xs), *weight_ptrs, out,
                m, bm, bn, bk, stages, prepacked, num_warps=num_warps, num_stages=stages)
    return out.unbind(0)
