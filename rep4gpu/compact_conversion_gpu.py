"""Fuse local preparation and weighted reduction for compact bit conversion."""

import math

import torch
import triton
import triton.language as tl

from .compact_conversion import weighted_b2a
from .runtime import Share


@triton.jit
def _pair_products(A, B, C, OUT, N: tl.constexpr, PARTY: tl.constexpr,
                   BLOCK: tl.constexpr):
    i = tl.program_id(0)*BLOCK + tl.arange(0, BLOCK)
    a = tl.load(A+i, i<N, 0) & 1
    b = tl.load(B+i, i<N, 0) & 1
    c = tl.load(C+i, i<N, 0) & 1
    if PARTY == 0 or PARTY == 2:
        value = b*c
    else:
        value = a*b
    tl.store(OUT+i, value, i<N)


@triton.jit
def _parity_slot(A, Q, R, offsets, mask, COMPONENT: tl.constexpr,
                 DT: tl.constexpr):
    bit = (tl.load(A+offsets, mask, 0) & 1).to(DT)
    q = tl.load(Q+offsets, mask, 0).to(DT)
    r = tl.load(R+offsets, mask, 0).to(DT)
    if COMPONENT < 2:
        return bit-2*q, 0-2*r
    else:
        return 0-2*q, bit-2*r


@triton.jit
def _reduce(A,B,C,Q0,Q1,Q2,R0,R1,R2,W,OUT,
            P:tl.constexpr,D:tl.constexpr,O:tl.constexpr,PARTY:tl.constexpr,
            BITS:tl.constexpr,BP:tl.constexpr,BD:tl.constexpr):
    item = tl.program_id(0)*BD+tl.arange(0,BD)
    plane = tl.arange(0,BP)
    offset = plane[:,None]*D+item[None,:]
    mask = (plane[:,None]<P)&(item[None,:]<D)
    DT:tl.constexpr = tl.uint32 if BITS==32 else tl.uint64
    x0,y0 = _parity_slot(A,Q0,R0,offset,mask,(PARTY+1)%4,DT)
    x1,y1 = _parity_slot(B,Q1,R1,offset,mask,(PARTY+2)%4,DT)
    x2,y2 = _parity_slot(C,Q2,R2,offset,mask,(PARTY+3)%4,DT)
    v0 = (x0+x1)*y0+x0*y1
    v1 = (x1+x2)*y1+x1*y2
    v2 = x0*y2+x2*y0
    for output in range(O):
        weight = tl.load(W+output*P+plane,plane<P,0).to(DT)[:,None]
        dest = output*D+item
        for slot in tl.static_range(6):
            if slot == 0:
                value = v0
            elif slot == 1:
                value = v1
            elif slot == 2:
                value = v2
            elif slot == 3:
                value = x0+y0
            elif slot == 4:
                value = x1+y1
            else:
                value = x2+y2
            total = tl.sum(value*weight,0).to(tl.uint64)
            if BITS == 32:
                total &= 0xffffffff
            tl.store(OUT+slot*O*D+dest,total.to(tl.int64),item<D)


@triton.jit
def _finish(L,A,B,C,OUT,N:tl.constexpr,BITS:tl.constexpr,BLOCK:tl.constexpr):
    item = tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    for slot in tl.static_range(3):
        linear = tl.load(L+slot*N+item,item<N,0).to(tl.uint64)
        if slot == 0:
            product = tl.load(A+item,item<N,0).to(tl.uint64)
        elif slot == 1:
            product = tl.load(B+item,item<N,0).to(tl.uint64)
        else:
            product = tl.load(C+item,item<N,0).to(tl.uint64)
        value = linear-2*product
        if BITS == 32:
            value &= 0xffffffff
        tl.store(OUT+slot*N+item,value.to(tl.int64),item<N)


def weighted_b2a_gpu(rt, planes, weights):
    """Convert weighted planes with the compact protocol's exact local shares.

    CUDA planes with up to 64 rows fuse parity preparation, local products,
    and weighted linear sums. Other shapes use the reference implementation.
    The joint inputs and product resharing retain their original draw order.
    """
    if not planes.binary or not len(planes.shape):
        raise ValueError('weighted_b2a requires binary bit planes with a leading plane axis')
    weights = torch.as_tensor(weights, dtype=torch.int64, device=planes.device)
    if weights.ndim not in (1,2) or weights.shape[-1] != planes.shape[0]:
        raise ValueError('weights must have shape (planes,) or (outputs,planes)')
    p = planes.shape[0]
    outputs = 1 if weights.ndim == 1 else weights.shape[0]
    d = math.prod(planes.shape[1:])
    if planes.device.type != 'cuda' or not 1 <= p <= 64 or not d or not outputs:
        return weighted_b2a(rt, planes, weights)
    parts = tuple(part.contiguous() for part in planes.parts)
    pair = torch.empty(planes.shape, dtype=torch.int64, device=planes.device)
    _pair_products[(triton.cdiv(p*d,256),)](*parts,pair,p*d,rt.party,256)
    q,r = rt.joint_input_batch(
        [(2,3,1,0,pair if rt.party in (2,3) else None),
         (0,1,3,2,pair if rt.party in (0,1) else None)],
        planes.shape,False,planes.device)
    shape = tuple(planes.shape[1:]) if weights.ndim == 1 else (outputs,*planes.shape[1:])
    result = torch.empty((6,*shape), dtype=torch.int64, device=planes.device)
    bd = 32 if p>16 else 64
    _reduce[(triton.cdiv(d,bd),)](*parts,*q.parts,*r.parts,weights.contiguous(),result,
                                 p,d,outputs,rt.party,rt.bits,triton.next_power_of_2(p),bd,
                                 num_warps=8 if p>16 else 4)
    values = result.unbind(0)
    contributions = [None]*5
    contributions[(rt.party-1)%4],contributions[rt.party],contributions[4] = values[:3]
    product = rt._reshare_product(contributions,False,planes.device)
    final = torch.empty((3,*shape),dtype=torch.int64,device=planes.device)
    count = outputs*d
    _finish[(triton.cdiv(count,256),)](result[3:],*product.parts,final,count,rt.bits,256)
    return Share(tuple(final.unbind(0)),False)
