"""ChaCha20 mask rows whose draw nonces are CUDA graph inputs."""
import triton
import triton.language as tl

from .gpu_prg_batch import _block_words


@triton.jit
def _rows(K0, K1, K2, K3, K4, K5, NONCES, OUT, COUNT: tl.constexpr,
          BITS: tl.constexpr, VALID: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(1)
    keys = (K0, K1, K2, K3, K4, K5)
    key = K0
    active = tl.full((), False, tl.int1)
    for i in tl.static_range(6):
        key = tl.where(row == i, keys[i], key)
        active |= (row == i) & VALID[i]
    nonce = tl.load(NONCES + row)
    block = tl.program_id(0).to(tl.uint64)*BLOCK + tl.arange(0, BLOCK).to(tl.uint64)
    if active:
        words = _block_words(key, nonce, block, BLOCK)
        for i in tl.static_range(16 if BITS == 32 else 8):
            idx = block*(16 if BITS == 32 else 8)+i
            if BITS == 32:
                value = words[i].to(tl.uint64)
            else:
                value = words[2*i].to(tl.uint64) | (words[2*i+1].to(tl.uint64) << 32)
            tl.store(OUT+row*COUNT+idx, value.to(tl.int64), idx < COUNT)
    else:
        for i in tl.static_range(16 if BITS == 32 else 8):
            idx = block*(16 if BITS == 32 else 8)+i
            tl.store(OUT+row*COUNT+idx, tl.full((BLOCK,), 0, tl.int64), idx < COUNT)


def generate_rows_into(keys, nonces, output, bits=64):
    """Fill six fixed mask rows from a persistent uint64 CUDA nonce vector.

    The caller reserves and uploads fresh nonces before every replay. This
    function performs no reservations and must never own a live draw counter.
    """
    import torch
    if len(keys) != 6 or bits not in (32, 64):
        raise ValueError('six keys and a 32-bit or 64-bit ring are required')
    if output.ndim != 2 or output.shape[0] != 6 or output.dtype != torch.int64 or not output.is_cuda or not output.is_contiguous():
        raise ValueError('output must be contiguous CUDA int64 with six rows')
    if nonces.shape != (6,) or nonces.dtype != torch.uint64 or nonces.device != output.device or not nonces.is_contiguous():
        raise ValueError('nonces must be six contiguous CUDA uint64 values on the output device')
    active = [key for key in keys if key is not None]
    if not active:
        raise ValueError('at least one key is required')
    for key in active:
        if key.shape != (8,) or key.dtype != torch.uint32 or key.device != output.device or not key.is_contiguous():
            raise ValueError('keys must be contiguous eight-word CUDA uint32 tensors')
    count = output.shape[1]
    blocks = triton.cdiv(count, 16 if bits == 32 else 8)
    if blocks >= 2**32:
        raise ValueError('a draw must use fewer than 2**32 ChaCha blocks')
    if count:
        launch_keys = tuple(key if key is not None else active[0] for key in keys)
        _rows[(triton.cdiv(blocks, 128), 6)](*launch_keys, nonces, output, count, bits,
                                         tuple(key is not None for key in keys), 128, num_warps=4)
