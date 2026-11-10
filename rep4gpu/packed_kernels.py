"""Experimental exact tensor products with both operands packed into int8 limbs."""
import torch
import triton
import triton.language as tl

from .kernels import PreparedWeight


@triton.jit
def _pack_inputs(X0,X1,X2,P,SIZE:tl.constexpr,LIMBS:tl.constexpr,BLOCK:tl.constexpr):
    idx=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    limb=tl.program_id(1)
    mask=idx<SIZE
    x0=tl.load(X0+idx,mask,0).to(tl.uint64)
    x1=tl.load(X1+idx,mask,0).to(tl.uint64)
    x2=tl.load(X2+idx,mask,0).to(tl.uint64)
    for channel in tl.static_range(5):
        if channel==0:
            word=x0
        elif channel==1:
            word=x1
        elif channel==2:
            word=x2
        elif channel==3:
            word=x0+x1
        else:
            word=x1+x2
        word+=0x8080808080808080
        digit=(((word>>(8*limb))&255).to(tl.int32)-128).to(tl.int8)
        tl.store(P+(channel*LIMBS+limb)*SIZE+idx,digit,mask)


@triton.jit
def _packed_local(A,Y0,Y1,Y2,C,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,
                  BITS:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
    rows=tl.program_id(0)*BM+tl.arange(0,BM)
    cols=tl.program_id(1)*BN+tl.arange(0,BN)
    kk=tl.arange(0,BK)
    output=tl.program_id(2)
    out=tl.full((BM,BN),0,tl.uint64)
    for term in tl.static_range(2):
        if output==0:
            if term==0:
                ac=3
                B=Y0
            else:
                ac=0
                B=Y1
        elif output==1:
            if term==0:
                ac=4
                B=Y1
            else:
                ac=1
                B=Y2
        else:
            if term==0:
                ac=0
                B=Y2
            else:
                ac=2
                B=Y0
        for start in range(tl.cdiv(K,BK)):
            ap=rows[:,None]*K+start*BK+kk[None,:]
            bp=(start*BK+kk[:,None])*N+cols[None,:]
            am=(rows[:,None]<M)&(start*BK+kk[None,:]<K)
            bm=(start*BK+kk[:,None]<K)&(cols[None,:]<N)
            for diagonal in tl.static_range(BITS//8):
                acc=tl.full((BM,BN),0,tl.int32)
                for i in tl.static_range(diagonal+1):
                    ai=tl.load(A+(ac*(BITS//8)+i)*M*K+ap,am,0)
                    bi=tl.load(B+(diagonal-i)*K*N+bp,bm,0)
                    acc=tl.dot(ai,bi,acc,out_dtype=tl.int32)
                out+=acc.to(tl.int64).to(tl.uint64)<<(8*diagonal)
    if BITS==32:
        out&=0xffffffff
    tl.store(C+output*M*N+rows[:,None]*N+cols[None,:],out.to(tl.int64),
             (rows[:,None]<M)&(cols[None,:]<N))


def local_products_packed(x0,x1,x2,y0,y1,y2,bits=64,*,tile=(16,32,64),warps=4,accumulate=False):
    """Pack five input combinations, then compute three exact local outputs.

    The weight arguments must be prepared static weights from ``prepare_weight``.
    Packing is included in the call and therefore in benchmark timings.
    ``accumulate=True`` retains each diagonal in int32 until K is reduced;
    K <= 8191 keeps the sum of both products within the signed int32 range.
    """
    xs=(x0,x1,x2)
    ys=(y0,y1,y2)
    if bits not in (32,64) or any(not isinstance(y,PreparedWeight) or y.bits!=bits for y in ys):
        raise ValueError("all weights must be prepared in the requested ring")
    if any(x.ndim!=2 or x.dtype!=torch.int64 or not x.is_cuda or x.shape!=x0.shape or x.device!=x0.device for x in xs):
        raise ValueError("input shares must have matching int64 CUDA matrices")
    if any(y.device!=x0.device or y.shape!=y0.shape for y in ys):
        raise ValueError("weight share shapes and devices must match")
    m,k=x0.shape
    n=y0.shape[1]
    if y0.shape[0]!=k:
        raise ValueError("input and weight matrix dimensions do not match")
    xs=tuple(x.contiguous() for x in xs)
    packed=torch.empty((5,bits//8,m,k),device=x0.device,dtype=torch.int8)
    out=torch.empty((3,m,n),device=x0.device,dtype=torch.int64)
    bm,bn,bk=tile
    if m and k and n:
        with torch.cuda.device(x0.device):
            _pack_inputs[(triton.cdiv(m*k,256),bits//8)](*xs,packed,m*k,bits//8,256)
            if accumulate and k>8191:
                raise ValueError("full-K accumulation requires K <=8191")
            kernel=_packed_local_accum if accumulate else _packed_local
            kernel[(triton.cdiv(m,bm),triton.cdiv(n,bn),3)](
                packed,*(y.limbs for y in ys),out,m,n,k,bits,bm,bn,bk,num_warps=warps,num_stages=1)
    elif k==0:
        out.zero_()
    return out.unbind(0)

@triton.jit
def _packed_local_accum(A,Y0,Y1,Y2,C,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,
                  BITS:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
    rows=tl.program_id(0)*BM+tl.arange(0,BM)
    cols=tl.program_id(1)*BN+tl.arange(0,BN)
    kk=tl.arange(0,BK)
    output=tl.program_id(2)
    # Retain dot accumulators across K so each diagonal needs one ring conversion.
    acc0=tl.full((BM,BN),0,tl.int32)
    acc1=tl.full((BM,BN),0,tl.int32)
    acc2=tl.full((BM,BN),0,tl.int32)
    acc3=tl.full((BM,BN),0,tl.int32)
    acc4=tl.full((BM,BN),0,tl.int32)
    acc5=tl.full((BM,BN),0,tl.int32)
    acc6=tl.full((BM,BN),0,tl.int32)
    acc7=tl.full((BM,BN),0,tl.int32)
    for term in tl.static_range(2):
        if output==0:
            if term==0:
                ac=3
                B=Y0
            else:
                ac=0
                B=Y1
        elif output==1:
            if term==0:
                ac=4
                B=Y1
            else:
                ac=1
                B=Y2
        else:
            if term==0:
                ac=0
                B=Y2
            else:
                ac=2
                B=Y0
        for start in range(tl.cdiv(K,BK)):
            ap=rows[:,None]*K+start*BK+kk[None,:]
            bp=(start*BK+kk[:,None])*N+cols[None,:]
            am=(rows[:,None]<M)&(start*BK+kk[None,:]<K)
            bm=(start*BK+kk[:,None]<K)&(cols[None,:]<N)
            if BITS>0:
                for i in tl.static_range(1):
                    ai=tl.load(A+(ac*(BITS//8)+i)*M*K+ap,am,0)
                    bi=tl.load(B+(0-i)*K*N+bp,bm,0)
                    acc0=tl.dot(ai,bi,acc0,out_dtype=tl.int32)
            if BITS>8:
                for i in tl.static_range(2):
                    ai=tl.load(A+(ac*(BITS//8)+i)*M*K+ap,am,0)
                    bi=tl.load(B+(1-i)*K*N+bp,bm,0)
                    acc1=tl.dot(ai,bi,acc1,out_dtype=tl.int32)
            if BITS>16:
                for i in tl.static_range(3):
                    ai=tl.load(A+(ac*(BITS//8)+i)*M*K+ap,am,0)
                    bi=tl.load(B+(2-i)*K*N+bp,bm,0)
                    acc2=tl.dot(ai,bi,acc2,out_dtype=tl.int32)
            if BITS>24:
                for i in tl.static_range(4):
                    ai=tl.load(A+(ac*(BITS//8)+i)*M*K+ap,am,0)
                    bi=tl.load(B+(3-i)*K*N+bp,bm,0)
                    acc3=tl.dot(ai,bi,acc3,out_dtype=tl.int32)
            if BITS>32:
                for i in tl.static_range(5):
                    ai=tl.load(A+(ac*(BITS//8)+i)*M*K+ap,am,0)
                    bi=tl.load(B+(4-i)*K*N+bp,bm,0)
                    acc4=tl.dot(ai,bi,acc4,out_dtype=tl.int32)
            if BITS>40:
                for i in tl.static_range(6):
                    ai=tl.load(A+(ac*(BITS//8)+i)*M*K+ap,am,0)
                    bi=tl.load(B+(5-i)*K*N+bp,bm,0)
                    acc5=tl.dot(ai,bi,acc5,out_dtype=tl.int32)
            if BITS>48:
                for i in tl.static_range(7):
                    ai=tl.load(A+(ac*(BITS//8)+i)*M*K+ap,am,0)
                    bi=tl.load(B+(6-i)*K*N+bp,bm,0)
                    acc6=tl.dot(ai,bi,acc6,out_dtype=tl.int32)
            if BITS>56:
                for i in tl.static_range(8):
                    ai=tl.load(A+(ac*(BITS//8)+i)*M*K+ap,am,0)
                    bi=tl.load(B+(7-i)*K*N+bp,bm,0)
                    acc7=tl.dot(ai,bi,acc7,out_dtype=tl.int32)
    out=tl.full((BM,BN),0,tl.uint64)
    if BITS>0:
        out+=acc0.to(tl.int64).to(tl.uint64)<<0
    if BITS>8:
        out+=acc1.to(tl.int64).to(tl.uint64)<<8
    if BITS>16:
        out+=acc2.to(tl.int64).to(tl.uint64)<<16
    if BITS>24:
        out+=acc3.to(tl.int64).to(tl.uint64)<<24
    if BITS>32:
        out+=acc4.to(tl.int64).to(tl.uint64)<<32
    if BITS>40:
        out+=acc5.to(tl.int64).to(tl.uint64)<<40
    if BITS>48:
        out+=acc6.to(tl.int64).to(tl.uint64)<<48
    if BITS>56:
        out+=acc7.to(tl.int64).to(tl.uint64)<<56
    if BITS==32:
        out&=0xffffffff
    tl.store(C+output*M*N+rows[:,None]*N+cols[None,:],out.to(tl.int64),
             (rows[:,None]<M)&(cols[None,:]<N))

