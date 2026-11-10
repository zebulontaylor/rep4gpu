"""Exact CUDA matrix products modulo the Mersenne prime 2**31-1."""
import torch
import triton
import triton.language as tl

PRIME=2147483647


@triton.jit
def _reduce31(x):
    x=(x&2147483647)+(x>>31)
    x=(x&2147483647)+(x>>31)
    x=tl.where(x<0,x+2147483647,x)
    return tl.where(x>=2147483647,x-2147483647,x)


@triton.jit
def _field_tensor(A,B,C,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,
                  BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
    rows=tl.program_id(0)*BM+tl.arange(0,BM)
    cols=tl.program_id(1)*BN+tl.arange(0,BN)
    kk=tl.arange(0,BK)
    out=tl.full((BM,BN),0,tl.int64)
    for start in range(tl.cdiv(K,BK)):
        ki=start*BK+kk
        a=_reduce31(tl.load(A+rows[:,None]*K+ki[None,:],(rows[:,None]<M)&(ki[None,:]<K),0))
        b=_reduce31(tl.load(B+ki[:,None]*N+cols[None,:],(ki[:,None]<K)&(cols[None,:]<N),0))
        # Centering keeps the four balanced bytes equal to the integer value,
        # rather than merely equal modulo 2**32.
        a=tl.where(a>1073741823,a-2147483647,a)+2155905152
        b=tl.where(b>1073741823,b-2147483647,b)+2155905152
        tile=tl.full((BM,BN),0,tl.int64)
        for diagonal in tl.static_range(7):
            acc=tl.full((BM,BN),0,tl.int32)
            for i in tl.static_range(4):
                if i<=diagonal and diagonal-i<4:
                    ai=(((a>>(8*i))&255).to(tl.int32)-128).to(tl.int8)
                    bi=(((b>>(8*(diagonal-i)))&255).to(tl.int32)-128).to(tl.int8)
                    acc=tl.dot(ai,bi,acc,out_dtype=tl.int32)
            # 2**31 == 1 in the field. Reconstruction remains int64 so an
            # arbitrary field residue cannot fuse into a saturating int32 MMA.
            tile+=acc.to(tl.int64)<<(8*diagonal%31)
        out=_reduce31(out+_reduce31(tile))
    tl.store(C+rows[:,None]*N+cols[None,:],out,(rows[:,None]<M)&(cols[None,:]<N))


@triton.jit
def _field_decode(A,B,C,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
    row=tl.program_id(0)
    cols=tl.program_id(1)*BN+tl.arange(0,BN)
    kk=tl.arange(0,BK)
    sums=tl.full((BK,BN),0,tl.int64)
    for start in range(tl.cdiv(K,BK)):
        ki=start*BK+kk
        a=_reduce31(tl.load(A+row*K+ki,ki<K,0))
        b=_reduce31(tl.load(B+ki[:,None]*N+cols[None,:],(ki[:,None]<K)&(cols[None,:]<N),0))
        sums=_reduce31(sums+_reduce31(a[:,None]*b))
    out=_reduce31(tl.sum(sums,0))
    tl.store(C+row*N+cols,out,cols<N)


def field_matmul(a,b,*,backend="auto"):
    """Multiply int64 matrices exactly over F_(2**31-1).

    Inputs may contain any signed int64 value; outputs are canonical field
    elements. CUDA uses exact signed-byte tensor products or scalar decode.
    CPU inputs retain the existing implementation as a fallback.
    """
    if a.dtype!=torch.int64 or b.dtype!=torch.int64:
        raise TypeError("field matrices must have dtype int64")
    if a.ndim!=2 or b.ndim!=2 or a.shape[1]!=b.shape[0] or a.device!=b.device:
        raise ValueError("expected compatible matrices on one device")
    if backend not in ("auto","tensor","decode"):
        raise ValueError("backend must be auto, tensor or decode")
    if not a.is_cuda:
        from .spdz import field_matmul as fallback
        return fallback(a,b)
    a,b=a.contiguous(),b.contiguous()
    m,k=a.shape
    n=b.shape[1]
    out=torch.empty((m,n),device=a.device,dtype=torch.int64)
    if not m or not n:
        return out
    if not k:
        return out.zero_()
    if backend=="auto":
        backend="decode" if m<=4 else "tensor"
    with torch.cuda.device(a.device):
        if backend=="decode":
            _field_decode[(m,triton.cdiv(n,32))](a,b,out,m,n,k,32,32,num_warps=4)
        else:
            _field_tensor[(triton.cdiv(m,32),triton.cdiv(n,64))](a,b,out,m,n,k,32,64,64,num_warps=4,num_stages=1)
    return out
