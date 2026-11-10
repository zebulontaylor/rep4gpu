"""Compact-input Mersenne-field GPU products with int64 reconstruction."""
import torch
import triton
import triton.language as tl

from .field_kernels import PRIME,_reduce31


@triton.jit
def _tensor(A,B,C,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,
            BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
    rows=tl.program_id(0)*BM+tl.arange(0,BM)
    cols=tl.program_id(1)*BN+tl.arange(0,BN)
    kk=tl.arange(0,BK)
    out=tl.full((BM,BN),0,tl.int64)
    for start in range(tl.cdiv(K,BK)):
        ki=start*BK+kk
        a=_reduce31(tl.load(A+rows[:,None]*K+ki[None,:],(rows[:,None]<M)&(ki[None,:]<K),0).to(tl.int64))
        b=_reduce31(tl.load(B+ki[:,None]*N+cols[None,:],(ki[:,None]<K)&(cols[None,:]<N),0).to(tl.int64))
        a=(tl.where(a>1073741823,a-2147483647,a)+2155905152).to(tl.uint32)
        b=(tl.where(b>1073741823,b-2147483647,b)+2155905152).to(tl.uint32)
        for diagonal in tl.static_range(7):
            acc=tl.full((BM,BN),0,tl.int32)
            for i in tl.static_range(4):
                if i<=diagonal and diagonal-i<4:
                    ai=(((a>>(8*i))&255).to(tl.int32)-128).to(tl.int8)
                    bi=(((b>>(8*(diagonal-i)))&255).to(tl.int32)-128).to(tl.int8)
                    acc=tl.dot(ai,bi,acc,out_dtype=tl.int32)
            out+=acc.to(tl.int64)<<(8*diagonal%31)
        # Each tile contributes less than 2**49 in absolute value. Folding
        # once here bounds the next iteration without a second int64 tile.
        out=_reduce31(out)
    tl.store(C+rows[:,None]*N+cols[None,:],out,(rows[:,None]<M)&(cols[None,:]<N))


@triton.jit
def _decode(A,B,C,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
    row=tl.program_id(0)
    cols=tl.program_id(1)*BN+tl.arange(0,BN)
    kk=tl.arange(0,BK)
    sums=tl.full((BK,BN),0,tl.int64)
    for start in range(tl.cdiv(K,BK)):
        ki=start*BK+kk
        a=_reduce31(tl.load(A+row*K+ki,ki<K,0).to(tl.int64))
        b=_reduce31(tl.load(B+ki[:,None]*N+cols[None,:],(ki[:,None]<K)&(cols[None,:]<N),0).to(tl.int64))
        sums=_reduce31(sums+_reduce31(a[:,None]*b))
    tl.store(C+row*N+cols,_reduce31(tl.sum(sums,0)),cols<N)


def field_matmul(a,b,*,packed=False,backend="auto",tile=(32,64,64),warps=4,prime=PRIME):
    """Exact prime31 product with int32/int64 CUDA inputs and optional int32 output.

    Field normalization and reconstruction use int64 arithmetic; signed-byte
    tensor products have bounded exact int32 accumulators.
    Packed outputs are canonical field values, which fit positive signed int32.
    The CPU path accepts the same dtypes and retains the reference algorithm.
    """
    if prime!=PRIME:
        raise ValueError("only the prime 2**31-1 is supported")
    if a.dtype not in (torch.int32,torch.int64) or b.dtype not in (torch.int32,torch.int64):
        raise TypeError("field inputs must use int32 or int64")
    if a.ndim!=2 or b.ndim!=2 or a.shape[1]!=b.shape[0] or a.device!=b.device:
        raise ValueError("expected compatible matrices on one device")
    if backend not in ("auto","tensor","decode"):
        raise ValueError("backend must be auto, tensor or decode")
    if not a.is_cuda:
        from .spdz import field_matmul as reference
        result=reference(a.to(torch.int64),b.to(torch.int64))
        return result.to(torch.int32) if packed else result
    a,b=a.contiguous(),b.contiguous()
    m,k=a.shape
    n=b.shape[1]
    out=torch.empty((m,n),device=a.device,dtype=torch.int32 if packed else torch.int64)
    if not m or not n:
        return out
    if not k:
        return out.zero_()
    if backend=="auto":
        backend="decode" if m<=4 else "tensor"
    with torch.cuda.device(a.device):
        if backend=="decode":
            _decode[(m,triton.cdiv(n,32))](a,b,out,m,n,k,32,32,num_warps=4)
        else:
            bm,bn,bk=tile
            if bk>128:
                raise ValueError("K tile must be at most 128")
            _tensor[(triton.cdiv(m,bm),triton.cdiv(n,bn))](a,b,out,m,n,k,bm,bn,bk,num_warps=warps,num_stages=1)
    return out
