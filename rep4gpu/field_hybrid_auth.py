"""Hybrid tensor/SIMT authenticated prime31 decode with fresh input buffers."""
import torch
import triton
import triton.language as tl

from .field_fast import field_matmul
from .field_fusion import _integer_tensors
from .field_kernels import _reduce31


@triton.jit
def _finish(D,BV,BM,CV,CM,ALPHA,RIGHT,OUT,
             M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,PARTY:tl.constexpr,
             BN:tl.constexpr,BK:tl.constexpr):
    plane=tl.program_id(0)
    row=tl.program_id(1)
    cols=tl.program_id(2)*BN+tl.arange(0,BN)
    kk=tl.arange(0,BK)
    sums=tl.full((BK,BN),0,tl.int64)
    for start in range(tl.cdiv(K,BK)):
        ki=start*BK+kk
        d=_reduce31(tl.load(D+row*K+ki,ki<K,0).to(tl.int64))
        bp=ki[:,None]*N+cols[None,:]
        mask=(ki[:,None]<K)&(cols[None,:]<N)
        if plane==0:
            b=_reduce31(tl.load(BV+bp,mask,0).to(tl.int64))
        else:
            b=_reduce31(tl.load(BM+(plane-1)*K*N+bp,mask,0).to(tl.int64))
        sums=_reduce31(sums+_reduce31(d[:,None]*b))
    cp=row*N+cols
    if plane==0:
        out=_reduce31(tl.load(CV+cp,cols<N,0).to(tl.int64))
    else:
        out=_reduce31(tl.load(CM+(plane-1)*M*N+cp,cols<N,0).to(tl.int64))
    right=_reduce31(tl.load(RIGHT+(plane+1)*M*N+cp,cols<N,0).to(tl.int64))
    public=_reduce31(tl.load(RIGHT+cp,cols<N,0).to(tl.int64))
    out+=right
    if plane==0:
        if PARTY==0:
            out+=public
    else:
        alpha=_reduce31(tl.load(ALPHA+plane-1).to(tl.int64))
        out+=_reduce31(alpha*public)
    tl.store(OUT+plane*M*N+cp,_reduce31(out+tl.sum(sums,0)),cols<N)


def authenticated_local_hybrid(d,e,a_values,a_macs,b_values,b_macs,c_values,c_macs,alpha_share,party,
                               *,packed=True,right_tile=(16,64,64),bn=32,bk=32):
    """Complete a fresh Beaver product using a small row stack and two field kernels.

    D, A_value and A_macs are copied into one [(H+2)*M,K] input. One tensor
    product computes D@E and every A@E. A SIMT kernel adds D@B, C and the
    public/MAC corrections. B and its large MAC planes are never concatenated.
    This helper caches no triple, mask, opened value or intermediate product.
    """
    d,e,av,am,bv,bm,cv,cm,alpha=_integer_tensors((d,e,a_values,a_macs,b_values,b_macs,c_values,c_macs,alpha_share))
    if d.ndim!=2 or e.ndim!=2 or alpha.ndim!=1 or not alpha.numel() or party<0:
        raise ValueError("invalid authenticated matrix dimensions, key count or party")
    m,k=d.shape
    n=e.shape[1]
    h=alpha.numel()
    if e.shape[0]!=k or av.shape!=(m,k) or am.shape!=(h,m,k) or bv.shape!=(k,n) or bm.shape!=(h,k,n) or cv.shape!=(m,n) or cm.shape!=(h,m,n):
        raise ValueError("authenticated matrices and MAC planes have incompatible shapes")
    if bn not in (16,32,64) or bk not in (16,32,64):
        raise ValueError("SIMT tile dimensions must be 16, 32 or 64")
    rows=torch.cat((d,av,am.reshape(h*m,k)),dim=0)
    right=field_matmul(rows,e,packed=packed,backend="tensor",tile=right_tile)
    out=torch.empty((h+1,m,n),device=d.device,dtype=torch.int32 if packed else torch.int64)
    if m and n:
        with torch.cuda.device(d.device):
            _finish[(h+1,m,triton.cdiv(n,bn))](d,bv,bm,cv,cm,alpha,right,out,m,n,k,party,bn,bk,num_warps=4)
    return out[0],out[1:]
