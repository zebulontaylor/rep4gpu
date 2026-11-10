"""SIMT authenticated prime31 decode without padded tensor-core rows."""
import torch
import triton
import triton.language as tl

from .field_fast import field_matmul
from .field_fusion import _integer_tensors
from .field_kernels import _reduce31


@triton.jit
def _finish(D,E,AV,AM,BV,BM,CV,CM,ALPHA,PUBLIC,OUT,
             M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,PARTY:tl.constexpr,
             BN:tl.constexpr,BK:tl.constexpr):
    plane=tl.program_id(0)
    row=tl.program_id(1)
    cols=tl.program_id(2)*BN+tl.arange(0,BN)
    kk=tl.arange(0,BK)
    sums=tl.full((BK,BN),0,tl.int64)
    for start in range(tl.cdiv(K,BK)):
        ki=start*BK+kk
        xp=row*K+ki
        yp=ki[:,None]*N+cols[None,:]
        xm=ki<K
        ym=(ki[:,None]<K)&(cols[None,:]<N)
        d=_reduce31(tl.load(D+xp,xm,0).to(tl.int64))
        e=_reduce31(tl.load(E+yp,ym,0).to(tl.int64))
        if plane==0:
            a=_reduce31(tl.load(AV+xp,xm,0).to(tl.int64))
            b=_reduce31(tl.load(BV+yp,ym,0).to(tl.int64))
        else:
            a=_reduce31(tl.load(AM+(plane-1)*M*K+xp,xm,0).to(tl.int64))
            b=_reduce31(tl.load(BM+(plane-1)*K*N+yp,ym,0).to(tl.int64))
        left=_reduce31(d[:,None]*b)
        right=_reduce31(a[:,None]*e)
        sums=_reduce31(sums+left+right)
    cp=row*N+cols
    if plane==0:
        out=_reduce31(tl.load(CV+cp,cols<N,0).to(tl.int64))
    else:
        out=_reduce31(tl.load(CM+(plane-1)*M*N+cp,cols<N,0).to(tl.int64))
    public=_reduce31(tl.load(PUBLIC+cp,cols<N,0).to(tl.int64))
    if plane==0:
        if PARTY==0:
            out+=public
    else:
        alpha=_reduce31(tl.load(ALPHA+plane-1).to(tl.int64))
        out+=_reduce31(alpha*public)
    out=_reduce31(out+tl.sum(sums,0))
    tl.store(OUT+plane*M*N+cp,out,cols<N)


def authenticated_local_decode(d,e,a_values,a_macs,b_values,b_macs,c_values,c_macs,alpha_share,party,
                               *,packed=True,bn=32,bk=32):
    """Compute authenticated matrix Beaver outputs with scalar int64 products.

    The signature and canonical output layout match ``authenticated_local``.
    This path avoids the minimum 32-row tile for batches of one to four rows.
    Inputs and correlations belong to the caller's current one-use triple;
    no mask or triple data is cached by this helper.
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
        raise ValueError("decode tiles must be 16, 32 or 64")
    public=field_matmul(d,e,packed=packed,backend="decode")
    out=torch.empty((h+1,m,n),device=d.device,dtype=torch.int32 if packed else torch.int64)
    if m and n:
        with torch.cuda.device(d.device):
            _finish[(h+1,m,triton.cdiv(n,bn))](d,e,av,am,bv,bm,cv,cm,alpha,public,out,m,n,k,party,bn,bk,num_warps=4)
    return out[0],out[1:]
