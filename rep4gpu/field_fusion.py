"""GPU-local authenticated Beaver arithmetic and independent-key MAC reductions."""
import torch
import triton
import triton.language as tl

from .field_fast import field_matmul
from .field_kernels import _reduce31


@triton.jit
def _authenticated(D,E,AV,AM,BV,BM,CV,CM,ALPHA,PUBLIC,OUT,
                    M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,
                    PARTY:tl.constexpr,TM:tl.constexpr,TN:tl.constexpr,TK:tl.constexpr):
    plane=tl.program_id(0)
    rows=tl.program_id(1)*TM+tl.arange(0,TM)
    cols=tl.program_id(2)*TN+tl.arange(0,TN)
    kk=tl.arange(0,TK)
    cp=rows[:,None]*N+cols[None,:]
    cm=(rows[:,None]<M)&(cols[None,:]<N)
    if plane==0:
        out=_reduce31(tl.load(CV+cp,cm,0).to(tl.int64))
    else:
        out=_reduce31(tl.load(CM+(plane-1)*M*N+cp,cm,0).to(tl.int64))
    public=_reduce31(tl.load(PUBLIC+cp,cm,0).to(tl.int64))
    if plane==0:
        if PARTY==0:
            out=_reduce31(out+public)
    else:
        alpha=_reduce31(tl.load(ALPHA+plane-1).to(tl.int64))
        out=_reduce31(out+_reduce31(alpha*public))
    for term in tl.static_range(2):
        for start in range(tl.cdiv(K,TK)):
            ki=start*TK+kk
            ap=rows[:,None]*K+ki[None,:]
            bp=ki[:,None]*N+cols[None,:]
            am=(rows[:,None]<M)&(ki[None,:]<K)
            bm=(ki[:,None]<K)&(cols[None,:]<N)
            if term==0:
                a=tl.load(D+ap,am,0).to(tl.int64)
                if plane==0:
                    b=tl.load(BV+bp,bm,0).to(tl.int64)
                else:
                    b=tl.load(BM+(plane-1)*K*N+bp,bm,0).to(tl.int64)
            else:
                if plane==0:
                    a=tl.load(AV+ap,am,0).to(tl.int64)
                else:
                    a=tl.load(AM+(plane-1)*M*K+ap,am,0).to(tl.int64)
                b=tl.load(E+bp,bm,0).to(tl.int64)
            a=_reduce31(a)
            b=_reduce31(b)
            a=(tl.where(a>1073741823,a-2147483647,a)+2155905152).to(tl.uint32)
            b=(tl.where(b>1073741823,b-2147483647,b)+2155905152).to(tl.uint32)
            for diagonal in tl.static_range(7):
                acc=tl.full((TM,TN),0,tl.int32)
                for i in tl.static_range(4):
                    if i<=diagonal and diagonal-i<4:
                        ai=(((a>>(8*i))&255).to(tl.int32)-128).to(tl.int8)
                        bi=(((b>>(8*(diagonal-i)))&255).to(tl.int32)-128).to(tl.int8)
                        acc=tl.dot(ai,bi,acc,out_dtype=tl.int32)
                out+=acc.to(tl.int64)<<(8*diagonal%31)
            out=_reduce31(out)
    tl.store(OUT+plane*M*N+cp,out,cm)


def _integer_tensors(tensors):
    device=tensors[0].device
    if any(t.dtype not in (torch.int32,torch.int64) or not t.is_cuda or t.device!=device for t in tensors):
        raise ValueError("field tensors must use int32/int64 on one CUDA device")
    return tuple(t.contiguous() for t in tensors)


def authenticated_local(d,e,a_values,a_macs,b_values,b_macs,c_values,c_macs,alpha_share,party,
                        *,packed=True,tile=(32,64,64),warps=4):
    """Complete one party's authenticated matrix Beaver product on the GPU.

    For H independent MAC keys, inputs are D[M,K], E[K,N], A[M,K],
    A_macs[H,M,K], B[K,N], B_macs[H,K,N], C[M,N], C_macs[H,M,N]
    and alpha_share[H]. Returns value[M,N], macs[H,M,N]. Public D@E is
    computed once, followed by one batched launch for all value/MAC planes.
    Only party zero adds D@E to its value share; every MAC receives alpha_i D@E.
    """
    tensors=_integer_tensors((d,e,a_values,a_macs,b_values,b_macs,c_values,c_macs,alpha_share))
    d,e,av,am,bv,bm,cv,cm,alpha=tensors
    if d.ndim!=2 or e.ndim!=2 or alpha.ndim!=1 or not alpha.numel() or party<0:
        raise ValueError("invalid authenticated matrix dimensions, key count or party")
    m,k=d.shape
    n=e.shape[1]
    h=alpha.numel()
    if e.shape[0]!=k or av.shape!=(m,k) or am.shape!=(h,m,k) or bv.shape!=(k,n) or bm.shape!=(h,k,n) or cv.shape!=(m,n) or cm.shape!=(h,m,n):
        raise ValueError("authenticated matrices and MAC planes have incompatible shapes")
    public=field_matmul(d,e,packed=packed)
    out=torch.empty((h+1,m,n),device=d.device,dtype=torch.int32 if packed else torch.int64)
    if m and n:
        tm,tn,tk=tile
        if tk>128:
            raise ValueError("K tile must be at most 128")
        with torch.cuda.device(d.device):
            _authenticated[(h+1,triton.cdiv(m,tm),triton.cdiv(n,tn))](
                d,e,av,am,bv,bm,cv,cm,alpha,public,out,m,n,k,party,tm,tn,tk,
                num_warps=warps,num_stages=1)
    return out[0],out[1:]


@triton.jit
def _mac_partial(MACS,OPENED,ALPHA,CHI,PARTS,N:tl.constexpr,NP:tl.constexpr,BLOCK:tl.constexpr):
    key=tl.program_id(0)
    idx=tl.program_id(1)*BLOCK+tl.arange(0,BLOCK)
    m=_reduce31(tl.load(MACS+key*N+idx,idx<N,0).to(tl.int64))
    opened=_reduce31(tl.load(OPENED+idx,idx<N,0).to(tl.int64))
    alpha=_reduce31(tl.load(ALPHA+key).to(tl.int64))
    chi=_reduce31(tl.load(CHI+key*N+idx,idx<N,0).to(tl.int64))
    residual=_reduce31(m-alpha*opened)
    weighted=_reduce31(residual*chi)
    tl.store(PARTS+key*NP+tl.program_id(1),_reduce31(tl.sum(weighted,0)))


@triton.jit
def _mac_final(PARTS,OUT,NP:tl.constexpr,BLOCK:tl.constexpr):
    key=tl.program_id(0)
    idx=tl.arange(0,BLOCK)
    sums=tl.full((BLOCK,),0,tl.int64)
    for start in range(tl.cdiv(NP,BLOCK)):
        at=start*BLOCK+idx
        sums+=tl.load(PARTS+key*NP+at,at<NP,0).to(tl.int64)
    tl.store(OUT+key,_reduce31(tl.sum(sums,0)))


def field_mac_check(macs,opened,alpha_share,challenge,*,packed=True):
    """Reduce MAC residuals with an independent challenge row for every key.

    Shapes are MACs[H,N], opened[N], alpha_share[H], challenge[H,N]. A shared
    challenge[N] is rejected: independent key domains are part of the protocol.
    The H canonical output residues use two launches and no H-by-N residual
    allocation. Sampling and commitment/coin-toss ordering remain the caller's job.
    """
    macs,opened,alpha,chi=_integer_tensors((macs,opened,alpha_share,challenge))
    if macs.ndim!=2 or opened.ndim!=1 or alpha.ndim!=1 or chi.shape!=macs.shape:
        raise ValueError("MAC checks require independent challenge[H,N] rows")
    h,n=macs.shape
    if h<1 or opened.numel()!=n or alpha.numel()!=h or n>=2**31:
        raise ValueError("MAC-check dimensions or accumulation bound are invalid")
    out=torch.empty((h,),device=macs.device,dtype=torch.int32 if packed else torch.int64)
    if not n:
        return out.zero_()
    parts=triton.cdiv(n,256)
    partial=torch.empty((h,parts),device=macs.device,dtype=torch.int64)
    with torch.cuda.device(macs.device):
        _mac_partial[(h,parts)](macs,opened,alpha,chi,partial,n,parts,256,num_warps=4)
        _mac_final[(h,)](partial,out,parts,256,num_warps=4)
    return out
