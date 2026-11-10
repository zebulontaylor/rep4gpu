"""GPU field products for fresh model-dependent prepared-right correlations."""
import torch
import triton
import triton.language as tl

from .field_kernels import PRIME,_reduce31


@triton.jit
def _tensor(AV,AM,BV,BM,CV,CM,OUT,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,
            A_PLANES:tl.constexpr,B_PLANES:tl.constexpr,TM:tl.constexpr,TN:tl.constexpr,TK:tl.constexpr):
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
    for start in range(tl.cdiv(K,TK)):
        ki=start*TK+kk
        ap=rows[:,None]*K+ki[None,:]
        bp=ki[:,None]*N+cols[None,:]
        am=(rows[:,None]<M)&(ki[None,:]<K)
        bm=(ki[:,None]<K)&(cols[None,:]<N)
        if A_PLANES:
            if plane==0:
                a=tl.load(AV+ap,am,0).to(tl.int64)
            else:
                a=tl.load(AM+(plane-1)*M*K+ap,am,0).to(tl.int64)
        else:
            a=tl.load(AV+ap,am,0).to(tl.int64)
        if B_PLANES:
            if plane==0:
                b=tl.load(BV+bp,bm,0).to(tl.int64)
            else:
                b=tl.load(BM+(plane-1)*K*N+bp,bm,0).to(tl.int64)
        else:
            b=tl.load(BV+bp,bm,0).to(tl.int64)
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
            # Keep the arbitrary field residue int64: fusing it into an int32
            # saturating MMA would change modular arithmetic near INT32_MAX.
            out+=acc.to(tl.int64)<<(8*diagonal%31)
        out=_reduce31(out)
    tl.store(OUT+plane*M*N+cp,out,cm)


@triton.jit
def _decode(AV,AM,BV,BM,CV,CM,OUT,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,
            A_PLANES:tl.constexpr,B_PLANES:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
    plane=tl.program_id(0)
    row=tl.program_id(1)
    cols=tl.program_id(2)*BN+tl.arange(0,BN)
    kk=tl.arange(0,BK)
    sums=tl.full((BK,BN),0,tl.int64)
    for start in range(tl.cdiv(K,BK)):
        ki=start*BK+kk
        ap=row*K+ki
        bp=ki[:,None]*N+cols[None,:]
        am=ki<K
        bm=(ki[:,None]<K)&(cols[None,:]<N)
        if A_PLANES:
            if plane==0:
                a=tl.load(AV+ap,am,0).to(tl.int64)
            else:
                a=tl.load(AM+(plane-1)*M*K+ap,am,0).to(tl.int64)
        else:
            a=tl.load(AV+ap,am,0).to(tl.int64)
        if B_PLANES:
            if plane==0:
                b=tl.load(BV+bp,bm,0).to(tl.int64)
            else:
                b=tl.load(BM+(plane-1)*K*N+bp,bm,0).to(tl.int64)
        else:
            b=tl.load(BV+bp,bm,0).to(tl.int64)
        a=_reduce31(a)
        b=_reduce31(b)
        sums=_reduce31(sums+_reduce31(a[:,None]*b))
    cp=row*N+cols
    if plane==0:
        base=_reduce31(tl.load(CV+cp,cols<N,0).to(tl.int64))
    else:
        base=_reduce31(tl.load(CM+(plane-1)*M*N+cp,cols<N,0).to(tl.int64))
    tl.store(OUT+plane*M*N+cp,_reduce31(base+tl.sum(sums,0)),cols<N)


def _apply(av,am,bv,bm,cv,cm,*,a_planes,b_planes,packed,backend,tile,warps):
    tensors=(av,am,bv,bm,cv,cm)
    if any(t.dtype not in (torch.int32,torch.int64) or t.device!=av.device for t in tensors):
        raise ValueError("field sources must use int32/int64 on one device")
    if av.ndim!=2 or bv.ndim!=2 or cm.ndim!=3:
        raise ValueError("matrix and MAC-plane dimensions are invalid")
    m,k=av.shape
    n=bv.shape[1]
    h=cm.shape[0]
    if h<1 or bv.shape[0]!=k or cv.shape!=(m,n) or cm.shape!=(h,m,n) or (a_planes and am.shape!=(h,m,k)) or (b_planes and bm.shape!=(h,k,n)):
        raise ValueError("prepared-right source shapes are incompatible")
    if backend not in ("auto","tensor","decode"):
        raise ValueError("backend must be auto, tensor or decode")
    if not av.is_cuda:
        from .spdz import field_matmul
        outputs=[]
        for plane in range(h+1):
            a=am[plane-1] if a_planes and plane else av
            b=bm[plane-1] if b_planes and plane else bv
            c=cm[plane-1] if plane else cv
            product=field_matmul(a.to(torch.int64),b.to(torch.int64))
            outputs.append((c.to(torch.int64).remainder(PRIME)+product).remainder(PRIME))
        result=torch.stack(outputs)
        if packed:
            result=result.to(torch.int32)
        return result[0],result[1:]
    tensors=tuple(t.contiguous() for t in tensors)
    out=torch.empty((h+1,m,n),device=av.device,dtype=torch.int32 if packed else torch.int64)
    if m and n:
        if backend=="auto":
            backend="decode" if m<=4 else "tensor"
        with torch.cuda.device(av.device):
            if backend=="decode":
                _decode[(h+1,m,triton.cdiv(n,32))](*tensors,out,m,n,k,a_planes,b_planes,32,32,num_warps=4)
            else:
                tm,tn,tk=tile
                if tk>128:
                    raise ValueError("K tile must be at most 128")
                _tensor[(h+1,triton.cdiv(m,tm),triton.cdiv(n,tn))](
                    *tensors,out,m,n,k,a_planes,b_planes,tm,tn,tk,num_warps=warps,num_stages=1)
    return out[0],out[1:]


def prepared_right_factor(a_values,a_macs,c_values,c_macs,e,*,packed=True,backend="auto",tile=(32,64,64),warps=4):
    """Compute offline F=C+A@E for a fresh authenticated correlation.

    MAC planes use Gamma_F=Gamma_C+Gamma_A@E. The caller authenticates E and
    enforces one-use state; this helper stores no model, triple or randomness.
    """
    return _apply(a_values,a_macs,e,e,c_values,c_macs,a_planes=True,b_planes=False,
                  packed=packed,backend=backend,tile=tile,warps=warps)


def prepared_right_online(d,w_values,w_macs,f_values,f_macs,*,packed=True,backend="auto",tile=(32,64,64),warps=4):
    """Compute online Z=F+D@W and Gamma_Z=Gamma_F+D@Gamma_W.

    Only one field matrix product per value/MAC plane is required. The caller
    consumes the prepared handle once and enforces its immediate or deferred
    masked-opening checks before any unmasked output is sent.
    """
    return _apply(d,d,w_values,w_macs,f_values,f_macs,a_planes=False,b_planes=True,
                  packed=packed,backend=backend,tile=tile,warps=warps)
