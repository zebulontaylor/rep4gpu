"""Prepared-right decode for field tensors whose canonical ranges are established."""
import torch
import triton
import triton.language as tl

from .field_kernels import PRIME,_reduce31


@triton.jit
def _canonical_decode(D,WV,WM,FV,FM,OUT,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,
                      BN:tl.constexpr,BK:tl.constexpr):
    plane=tl.program_id(0)
    row=tl.program_id(1).to(tl.int64)
    cols=tl.program_id(2).to(tl.int64)*BN+tl.arange(0,BN)
    kk=tl.arange(0,BK)
    sums=tl.full((BK,BN),0,tl.int64)
    for start in range(tl.cdiv(K,BK)):
        ki=start*BK+kk
        d=tl.load(D+row*K+ki,ki<K,0).to(tl.int64)
        wp=ki[:,None].to(tl.int64)*N+cols[None,:]
        mask=(ki[:,None]<K)&(cols[None,:]<N)
        if plane==0:
            w=tl.load(WV+wp,mask,0).to(tl.int64)
        else:
            w=tl.load(WM+(plane.to(tl.int64)-1)*K*N+wp,mask,0).to(tl.int64)
        product=d[:,None]*w
        # Canonical operands imply 0 <= product < p**2. One fold is <2p.
        reduced=(product&2147483647)+(product>>31)
        reduced=tl.where(reduced>=2147483647,reduced-2147483647,reduced)
        # K <2**31 keeps the entire sum plus F below 2**62.
        sums+=reduced
    fp=row*N+cols
    if plane==0:
        base=tl.load(FV+fp,cols<N,0).to(tl.int64)
    else:
        base=tl.load(FM+(plane.to(tl.int64)-1)*M*N+fp,cols<N,0).to(tl.int64)
    out=_reduce31(base+tl.sum(sums,0))
    tl.store(OUT+plane.to(tl.int64)*M*N+fp,out,cols<N)


def prepared_right_online_canonical(d,w_values,w_macs,f_values,f_macs,*,packed=True,bn=32,bk=32):
    """Compute F+D@W after the protocol establishes every input is in [0,p).

    Shapes and output layout match ``prepared_right_online``. This specialized
    API does not scan tensor values. D must come from canonical reconstruction;
    W and F must be the validated, owned snapshots of the one-use prepared
    handle. Use the general helper for arbitrary signed representatives.

    Int32/int64 loads are widened before multiplication. Every product is
    reduced exactly, and K<2**31 bounds the final int64 sum below 2**62.
    """
    tensors=(d,w_values,w_macs,f_values,f_macs)
    if any(t.dtype not in (torch.int32,torch.int64) or t.device!=d.device for t in tensors):
        raise ValueError("canonical field sources must use int32/int64 on one device")
    if d.ndim!=2 or w_values.ndim!=2 or f_macs.ndim!=3:
        raise ValueError("invalid matrix or MAC-plane dimensions")
    m,k=d.shape
    n=w_values.shape[1]
    h=f_macs.shape[0]
    if h<1 or k>=2**31 or w_values.shape[0]!=k or w_macs.shape!=(h,k,n) or f_values.shape!=(m,n) or f_macs.shape!=(h,m,n):
        raise ValueError("prepared-right shapes or accumulation bound are invalid")
    if bn not in (16,32,64) or bk not in (16,32,64):
        raise ValueError("decode tiles must be 16, 32 or 64")
    if not d.is_cuda:
        from .field_prepared import prepared_right_online
        return prepared_right_online(*tensors,packed=packed)
    tensors=tuple(t.contiguous() for t in tensors)
    out=torch.empty((h+1,m,n),device=d.device,dtype=torch.int32 if packed else torch.int64)
    if m and n:
        with torch.cuda.device(d.device):
            _canonical_decode[(h+1,m,triton.cdiv(n,bn))](*tensors,out,m,n,k,bn,bk,num_warps=4)
    return out[0],out[1:]
