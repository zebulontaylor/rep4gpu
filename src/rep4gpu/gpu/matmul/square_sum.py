"""Exact local Rep4 contractions for a public signed sum of squares."""
import torch
import triton
import triton.language as tl


@triton.jit
def _local_square_sum(X0, X1, X2, SIGNS, OUT, BATCH: tl.constexpr,
                      WIDTH: tl.constexpr, BITS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    dtype = tl.uint64 if BITS == 64 else tl.uint32
    a = tl.full((), 0, dtype)
    b = tl.full((), 0, dtype)
    c = tl.full((), 0, dtype)
    for start in range(tl.cdiv(WIDTH, BLOCK)):
        col = start * BLOCK + tl.arange(0, BLOCK)
        x0 = tl.load(X0 + row * WIDTH + col, col < WIDTH, 0).to(dtype)
        x1 = tl.load(X1 + row * WIDTH + col, col < WIDTH, 0).to(dtype)
        x2 = tl.load(X2 + row * WIDTH + col, col < WIDTH, 0).to(dtype)
        positive = tl.load(SIGNS + col, col < WIDTH, 1) > 0
        t0 = x0 * x0 + 2 * x0 * x1
        t1 = x1 * x1 + 2 * x1 * x2
        t4 = 2 * x0 * x2
        a += tl.sum(tl.where(positive, t0, -t0), 0)
        b += tl.sum(tl.where(positive, t1, -t1), 0)
        c += tl.sum(tl.where(positive, t4, -t4), 0)
    if BITS < 64:
        a &= (1 << BITS) - 1
        b &= (1 << BITS) - 1
        c &= (1 << BITS) - 1
    tl.store(OUT + row, a.to(tl.int64))
    tl.store(OUT + BATCH + row, b.to(tl.int64))
    tl.store(OUT + 2 * BATCH + row, c.to(tl.int64))


def local_square_sum(parts, signs, bits):
    """Return the three local contributions as ``[batch, 1]`` tensors.

    ``signs`` is a previously validated public int64 CUDA vector. Arithmetic
    wraps modulo ``2**bits`` even when input components use full int64 words.
    """
    if bits not in (16, 32, 64):
        raise ValueError('square sum requires ring16, ring32, or ring64')
    allowed = (torch.int64, torch.uint16) if bits == 16 else (torch.int64,)
    if len(parts) != 3 or any(not isinstance(p, torch.Tensor) or p.ndim != 2
                              or p.dtype not in allowed or not p.is_cuda for p in parts):
        raise ValueError('square sum requires three int64 or ring16 uint16 CUDA matrices')
    first = parts[0]
    if any(p.shape != first.shape or p.device != first.device or p.dtype != first.dtype for p in parts):
        raise ValueError('square sum components must match in shape, device, and dtype')
    batch, width = first.shape
    if (signs.shape != (width,) or signs.dtype != torch.int64
            or signs.device != first.device or not signs.is_contiguous()):
        raise ValueError('public signs must be a contiguous int64 vector on the share device')
    output = torch.empty((3, batch, 1), dtype=first.dtype, device=first.device)
    if width == 0:
        output.zero_()
    elif batch:
        block = min(1024, max(32, triton.next_power_of_2(width)))
        with torch.cuda.device(first.device):
            _local_square_sum[(batch,)](*(p.contiguous() for p in parts), signs,
                                       output, batch, width, bits, block, num_warps=4)
    return tuple(output.unbind(0))
