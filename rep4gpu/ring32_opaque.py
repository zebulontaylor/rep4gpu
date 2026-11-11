"""Experimental ring32 reconstruction with an opaque modular add.

Each diagonal starts from zero and stays within a bounded int32 diagonal. The
opaque add separates modular reconstruction from saturating tensor-core
accumulators without retaining a uint64 output accumulator.
"""
import torch
import triton
import triton.language as tl
from .kernels import CompactWeight

@triton.jit
def _opaque_local(X0, X1, X2, Y0, Y1, Y2, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
               BITS: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
               BK: tl.constexpr, PREPARED: tl.constexpr = False, OUTPUT: tl.constexpr = -1):
    tl.static_assert(BITS == 32, 'opaque reconstruction requires ring32')
    output = tl.program_id(2) if OUTPUT < 0 else OUTPUT
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    kk = tl.arange(0, BK)
    out = tl.full((BM, BN), 0, tl.uint32)
    for term in tl.static_range(2):
        for start in range(tl.cdiv(K, BK)):
            if output == 0:
                A = X0
                A2 = X1
                if term == 0:
                    B = Y0
                else:
                    B = Y1
            elif output == 1:
                A = X1
                A2 = X2
                if term == 0:
                    B = Y1
                else:
                    B = Y2
            else:
                if term == 0:
                    A = X0
                    B = Y2
                else:
                    A = X2
                    B = Y0
                A2 = X0
            a = tl.load(A + rows[:, None] * K + start * BK + kk[None, :],
                        (rows[:, None] < M) & (start * BK + kk[None, :] < K), 0).to(tl.uint32)
            if term == 0:
                if output < 2:
                    a += tl.load(A2 + rows[:,None]*K + start*BK+kk[None,:],
                                 (rows[:,None]<M)&(start*BK+kk[None,:]<K),0).to(tl.uint32)
            if not PREPARED:
                b = tl.load(B + (start * BK + kk[:, None]) * N + cols[None, :],
                            (start * BK + kk[:, None] < K) & (cols[None, :] < N), 0).to(tl.uint32)
                b += 0x80808080
            # Adding this constant turns byte extraction into balanced signed digits.
            a += 0x80808080
            for diagonal in tl.static_range(BITS // 8):
                acc = tl.full((BM, BN), 0, tl.int32)
                for i in tl.static_range(diagonal + 1):
                    ai = (((a >> (8 * i)) & 255).to(tl.int32) - 128).to(tl.int8)
                    if PREPARED:
                        bi = tl.load(B + (diagonal-i)*K*N + (start*BK+kk[:,None])*N + cols[None,:],
                                     (start*BK+kk[:,None]<K)&(cols[None,:]<N), 0)
                    else:
                        bi = (((b >> (8 * (diagonal-i))) & 255).to(tl.int32) - 128).to(tl.int8)
                    acc = tl.dot(ai, bi, acc, out_dtype=tl.int32)
                out = tl.inline_asm_elementwise(
                    "add.u32 $0, $1, $2;", constraints="=r,r,r",
                    args=[out, acc.to(tl.uint32) << (8 * diagonal)],
                    dtype=tl.uint32, is_pure=True, pack=1)
    tl.store(C + output*M*N + rows[:, None] * N + cols[None, :], out.to(tl.int64),
             (rows[:, None] < M) & (cols[None, :] < N))


def local_products_opaque(x0,x1,x2,y0,y1,y2,*,tile=(32,128,64),warps=4,stages=1,separate=False):
    xs,ys=(x0,x1,x2),(y0,y1,y2)
    if any(not isinstance(y,CompactWeight) for y in ys):
        raise ValueError('opaque ring32 products require compact weights')
    if any(x.dtype!=torch.int64 or x.ndim!=2 or not x.is_cuda or x.shape!=x0.shape or x.device!=x0.device for x in xs):
        raise ValueError('inputs must be matching int64 CUDA matrices')
    if any(y.shape!=y0.shape or y.device!=x0.device for y in ys) or x0.shape[1]!=y0.shape[0]:
        raise ValueError('matrix dimensions and devices must match')
    m,k=x0.shape
    n=y0.shape[1]
    bm,bn,bk=tile
    if bk<32 or bk>=32768 or bk&(bk-1):
        raise ValueError('K tile must be a power of two in [32,16384]')
    out=torch.empty((3,m,n),dtype=torch.int64,device=x0.device)
    if m and n and k:
        with torch.cuda.device(x0.device):
            for output in range(3) if separate else (-1,):
                _opaque_local[(triton.cdiv(m,bm),triton.cdiv(n,bn),1 if separate else 3)](
                    *(x.contiguous() for x in xs),*(y.words for y in ys),out,m,n,k,32,bm,bn,bk,False,output,
                    num_warps=warps,num_stages=stages)
    elif k==0:
        out.zero_()
    return out.unbind(0)
