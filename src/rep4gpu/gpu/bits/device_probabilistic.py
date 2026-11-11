"""Optional CUDA/BLAKE3 port of Rep4 small-gap probabilistic truncation.

The caller must establish abs(signed(x)) < 2**(bits-2). This is a public range
contract, not a private range check. Nonempty, nonidentity calls use three
checked device exchanges; input sharing and explicit output opening are extra.
"""
import math

import torch
import triton
import triton.language as tl

from ...protocol.runtime import Share


@triton.jit
def _prepare(X0,X1,X2,R2,R3,WIRE,SPLIT,N:tl.constexpr,BITS:tl.constexpr,
             SHIFT:tl.constexpr,PARTY:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    a=tl.load(X1+i,i<N,0).to(tl.uint64)
    if PARTY==0:
        b=tl.load(X2+i,i<N,0).to(tl.uint64)
    else:
        b=tl.load(X0+i,i<N,0).to(tl.uint64)
    r=tl.load(R2+i,i<N,0).to(tl.uint64)+tl.load(R3+i,i<N,0).to(tl.uint64)
    partial=a+b+r
    if BITS==32:
        r&=0xffffffff
        partial&=0xffffffff
    tl.store(WIRE+i,partial,i<N)
    tl.store(SPLIT+i,(r>>(BITS-1))&1,i<N)
    tl.store(SPLIT+N+i,(r&((1<<(BITS-1))-1))>>SHIFT,i<N)


@triton.jit
def _unmask(WIRE,X0,X1,X2,SPLIT,N:tl.constexpr,BITS:tl.constexpr,
            SHIFT:tl.constexpr,PARTY:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    a=tl.load(X1+i,i<N,0).to(tl.uint64)
    if PARTY==2:
        b=tl.load(X2+i,i<N,0).to(tl.uint64)
    else:
        b=tl.load(X0+i,i<N,0).to(tl.uint64)
    value=tl.load(WIRE+i,i<N,0).to(tl.uint64)+a+b+(1<<(BITS-2))
    if BITS==32:
        value&=0xffffffff
    tl.store(SPLIT+i,(value>>(BITS-1))&1,i<N)
    tl.store(SPLIT+N+i,(value&((1<<(BITS-1))-1))>>SHIFT,i<N)


@triton.jit
def _finish(R0,R1,R2,C0,C1,C2,B0,B1,B2,OUT,N:tl.constexpr,
            BITS:tl.constexpr,SHIFT:tl.constexpr,PARTY:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    DT:tl.constexpr=tl.uint32 if BITS==32 else tl.uint64
    for slot in tl.static_range(3):
        if slot==0:
            r,c,b=R0,C0,B0
        elif slot==1:
            r,c,b=R1,C1,B1
        else:
            r,c,b=R2,C2,B2
        # The joint-input routes put r only in components 2/3 and c in 0/1.
        # Their other components are public zeros, so each slot reads one pair.
        if (PARTY+slot+1)%4>=2:
            sign=tl.load(r+i,i<N,0).to(DT)
            upper=0-tl.load(r+N+i,i<N,0).to(DT)
        else:
            sign=tl.load(c+i,i<N,0).to(DT)
            upper=tl.load(c+N+i,i<N,0).to(DT)
        both=tl.load(b+i,i<N,0).to(DT)
        value=upper+((sign-2*both)<<(BITS-1-SHIFT))
        if (PARTY+slot+1)%4==0:
            value-=1<<(BITS-SHIFT-2)
        if BITS==32:
            value&=0xffffffff
        tl.store(OUT+slot*N+i,value.to(tl.int64),i<N)


def trunc_pr(rt,x,shift):
    """Stochastically round x/2**shift to its floor or ceil on CUDA.

    Requires arithmetic DeviceParty shares with the public range contract
    abs(signed(x)) < 2**(rt.bits-2). Rounding up has probability
    (x mod 2**shift)/2**shift; exact multiples are unchanged by rounding.
    BLAKE3 tags intentionally differ from the CPU BLAKE2b implementation.
    Empty tensors and shift zero return x without communication or mask draws.
    """
    if x.binary:
        raise ValueError('probabilistic truncation requires arithmetic shares')
    if not isinstance(shift,int) or not 0<=shift<rt.bits-1:
        raise ValueError('shift must be an integer in [0, ring_bits-2]')
    if x.device.type!='cuda' or x.device!=rt.device:
        raise ValueError('device probabilistic truncation requires shares on the party CUDA device')
    if not hasattr(rt,'_exchange_joint_wires'):
        raise TypeError('device probabilistic truncation requires DeviceParty')
    count=math.prod(x.shape)
    if shift==0 or count==0:
        return x
    # All three holders advance each mask stream, including holders that do
    # not need the sampled value in this local opening calculation.
    parts=tuple(part.contiguous() for part in x.parts)
    masks={e:rt._random(e,x.shape,x.device,packed=True) for e in (2,3) if rt.party!=e}
    split=torch.empty((2,*x.shape),dtype=torch.int64,device=x.device)
    wire=None
    if rt.party<2:
        dtype=torch.uint32 if rt.bits==32 else torch.int64
        wire=torch.empty(count,dtype=dtype,device=x.device)
        _prepare[(triton.cdiv(count,256),)](*parts,masks[2],masks[3],wire,split,
                                           count,rt.bits,shift,rt.party,256)
        wire=wire.view(torch.uint8)
    entries=[(0,1,2,wire if rt.party<2 else None),
             (1,0,3,wire if rt.party<2 else None)]
    received=rt._exchange_joint_wires(entries,x.shape,False)
    if rt.party>=2:
        dtype=torch.uint32 if rt.bits==32 else torch.int64
        checked=received[rt.party-2].view(dtype)
        _unmask[(triton.cdiv(count,256),)](checked,*parts,split,count,rt.bits,shift,rt.party,256)
    r,c=rt.joint_input_batch(
        [(0,1,3,2,split if rt.party<2 else None),
         (2,3,1,0,split if rt.party>=2 else None)],
        (2,*x.shape),False,x.device)
    if getattr(rt,'narrow_products',False) and shift<=16:
        from ..matmul.narrow_product_gpu import product16
        both=product16(rt,r[0],c[0])
    else:
        both=(rt._mul_sparse_pairs(r[0],c[0]) if getattr(rt,'sparse_products',False)
              else rt.mul(r[0],c[0]))
    output=torch.empty((3,*x.shape),dtype=torch.int64,device=x.device)
    _finish[(triton.cdiv(count,256),)](*r.parts,*c.parts,*both.parts,output,
                                      count,rt.bits,shift,rt.party,256)
    return Share(tuple(output.unbind(0)),False)
