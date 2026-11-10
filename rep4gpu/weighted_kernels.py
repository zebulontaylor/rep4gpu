"""Exact weighted local Rep4 products reduced before communication."""
from __future__ import annotations

import math
import operator

import torch
import triton
import triton.language as tl


@triton.jit
def _weighted_products(X0,X1,X2,Y0,Y1,Y2,W,C,L:tl.constexpr,D:tl.constexpr,
                       O:tl.constexpr,BITS:tl.constexpr,BL:tl.constexpr,BD:tl.constexpr):
    item=tl.program_id(0)*BD+tl.arange(0,BD)
    plane=tl.arange(0,BL)
    offsets=plane[:,None]*D+item[None,:]
    mask=(plane[:,None]<L)&(item[None,:]<D)
    DT:tl.constexpr=tl.uint32 if BITS==32 else tl.uint64
    x0=tl.load(X0+offsets,mask,0).to(DT)
    x1=tl.load(X1+offsets,mask,0).to(DT)
    x2=tl.load(X2+offsets,mask,0).to(DT)
    y0=tl.load(Y0+offsets,mask,0).to(DT)
    y1=tl.load(Y1+offsets,mask,0).to(DT)
    y2=tl.load(Y2+offsets,mask,0).to(DT)
    p0=(x0+x1)*y0+x0*y1
    p1=(x1+x2)*y1+x1*y2
    p4=x0*y2+x2*y0
    for output in range(O):
        weight=tl.load(W+output*L+plane,plane<L,0).to(DT)[:,None]
        v0=tl.sum(p0*weight,0).to(tl.uint64)
        v1=tl.sum(p1*weight,0).to(tl.uint64)
        v4=tl.sum(p4*weight,0).to(tl.uint64)
        if BITS==32:
            v0&=0xffffffff
            v1&=0xffffffff
            v4&=0xffffffff
        offset=output*D+item
        tl.store(C+offset,v0.to(tl.int64),item<D)
        tl.store(C+O*D+offset,v1.to(tl.int64),item<D)
        tl.store(C+2*O*D+offset,v4.to(tl.int64),item<D)


def _signed_weights(value):
    if hasattr(value,"tolist"):
        value=value.tolist()
    if isinstance(value,(tuple,list)):
        return [_signed_weights(item) for item in value]
    value=operator.index(value)&0xffffffffffffffff
    return value if value<0x8000000000000000 else value-0x10000000000000000


def weighted_local_products(xs,ys,weights,bits=64):
    """Return three local products after weighting and reducing their leading axis.

    ``xs`` and ``ys`` each contain three int64 CUDA tensors shaped ``(L,*S)``,
    with L <= 64. Weights shaped ``(L,)`` produce outputs shaped ``S``;
    weights shaped ``(O,L)`` produce outputs shaped ``(O,*S)``. Weights can be
    signed int64 CUDA tensors or Python integer sequences, interpreted modulo
    the ring. Keep weights on the GPU to avoid a host transfer on repeated calls.

    Each output equals the weighted sum of the corresponding ``v0``, ``v1`` or
    ``v4`` contribution from ``local_products``, using elementwise products.
    A 32-bit result is represented canonically in [0,2**32); a 64-bit result
    retains its signed int64 bit pattern. No operand is opened.
    """
    if bits not in (32,64):
        raise ValueError("bits must be 32 or 64")
    if len(xs)!=3 or len(ys)!=3:
        raise ValueError("provide three x shares and three y shares")
    tensors=tuple(xs)+tuple(ys)
    first=tensors[0]
    if first.ndim<1:
        raise ValueError("share tensors need a leading reduction axis")
    if any(t.dtype!=torch.int64 or not t.is_cuda or t.shape!=first.shape or t.device!=first.device for t in tensors):
        raise ValueError("share tensors must have matching int64 CUDA shapes and devices")
    l=first.shape[0]
    if l>64:
        raise ValueError("the leading reduction axis must have at most 64 planes")
    if not isinstance(weights,torch.Tensor):
        weights=torch.tensor(_signed_weights(weights),device=first.device,dtype=torch.int64)
    elif weights.dtype!=torch.int64:
        raise TypeError("weights must use int64")
    weights=weights.to(device=first.device).contiguous()
    if weights.ndim not in (1,2) or weights.shape[-1]!=l:
        raise ValueError("weights must have shape (L,) or (O,L)")
    outputs=1 if weights.ndim==1 else weights.shape[0]
    shape=tuple(first.shape[1:])
    output_shape=shape if weights.ndim==1 else (outputs,*shape)
    d=math.prod(shape)
    result=torch.empty((3,*output_shape),device=first.device,dtype=torch.int64)
    if not d or not outputs:
        return result.unbind(0)
    if l==0:
        return result.zero_().unbind(0)
    tensors=tuple(t.contiguous() for t in tensors)
    bd=32 if l>16 else 64
    warps=8 if l>16 else 4
    with torch.cuda.device(first.device):
        _weighted_products[(triton.cdiv(d,bd),)](
            *tensors,weights,result,l,d,outputs,bits,triton.next_power_of_2(l),bd,num_warps=warps)
    return result.unbind(0)
