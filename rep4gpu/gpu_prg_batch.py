"""Generate a batch of correlated ChaCha20 mask rows in one CUDA launch."""
from __future__ import annotations

import operator

import torch
import triton
import triton.language as tl

from .gpu_prg import _quarter


@triton.jit
def _block_words(KEY,nonce,counter,BLOCK:tl.constexpr):
    z=tl.full((BLOCK,),0,tl.uint32)
    s0=z+0x61707865
    s1=z+0x3320646e
    s2=z+0x79622d32
    s3=z+0x6b206574
    s4=z+tl.load(KEY+0)
    s5=z+tl.load(KEY+1)
    s6=z+tl.load(KEY+2)
    s7=z+tl.load(KEY+3)
    s8=z+tl.load(KEY+4)
    s9=z+tl.load(KEY+5)
    s10=z+tl.load(KEY+6)
    s11=z+tl.load(KEY+7)
    s12=counter.to(tl.uint32)
    s13=(counter>>32).to(tl.uint32)
    s14=z+nonce.to(tl.uint32)
    s15=z+(nonce>>32).to(tl.uint32)
    x0,x1,x2,x3=s0,s1,s2,s3
    x4,x5,x6,x7=s4,s5,s6,s7
    x8,x9,x10,x11=s8,s9,s10,s11
    x12,x13,x14,x15=s12,s13,s14,s15
    for _ in range(10):
        x0,x4,x8,x12=_quarter(x0,x4,x8,x12)
        x1,x5,x9,x13=_quarter(x1,x5,x9,x13)
        x2,x6,x10,x14=_quarter(x2,x6,x10,x14)
        x3,x7,x11,x15=_quarter(x3,x7,x11,x15)
        x0,x5,x10,x15=_quarter(x0,x5,x10,x15)
        x1,x6,x11,x12=_quarter(x1,x6,x11,x12)
        x2,x7,x8,x13=_quarter(x2,x7,x8,x13)
        x3,x4,x9,x14=_quarter(x3,x4,x9,x14)
    return (x0+s0,x1+s1,x2+s2,x3+s3,x4+s4,x5+s5,x6+s6,x7+s7,
            x8+s8,x9+s9,x10+s10,x11+s11,x12+s12,x13+s13,x14+s14,x15+s15)


@triton.jit(do_not_specialize=["N0","N1","N2","N3","N4","N5"])
def _rows(K0,K1,K2,K3,K4,K5,N0,N1,N2,N3,N4,N5,OUT,COUNT:tl.constexpr,
          BITS:tl.constexpr,VALID:tl.constexpr,BLOCK:tl.constexpr):
    row=tl.program_id(1)
    keys=(K0,K1,K2,K3,K4,K5)
    nonces=(N0.to(tl.uint64),N1.to(tl.uint64),N2.to(tl.uint64),
            N3.to(tl.uint64),N4.to(tl.uint64),N5.to(tl.uint64))
    key=K0
    nonce=N0.to(tl.uint64)
    active=tl.full((),False,tl.int1)
    for i in tl.static_range(6):
        key=tl.where(row==i,keys[i],key)
        nonce=tl.where(row==i,nonces[i],nonce)
        active|=(row==i)&VALID[i]
    block=tl.program_id(0).to(tl.uint64)*BLOCK+tl.arange(0,BLOCK).to(tl.uint64)
    if active:
        words=_block_words(key,nonce,block,BLOCK)
        if BITS==32:
            for i in tl.static_range(16):
                idx=block*16+i
                tl.store(OUT+row*COUNT+idx,words[i].to(tl.uint64).to(tl.int64),idx<COUNT)
        else:
            for i in tl.static_range(8):
                idx=block*8+i
                value=words[2*i].to(tl.uint64)|(words[2*i+1].to(tl.uint64)<<32)
                tl.store(OUT+row*COUNT+idx,value.to(tl.int64),idx<COUNT)
    else:
        for i in tl.static_range(16 if BITS==32 else 8):
            idx=block*(16 if BITS==32 else 8)+i
            tl.store(OUT+row*COUNT+idx,tl.full((BLOCK,),0,tl.int64),idx<COUNT)


def generate_rows(keys,nonces,count,bits=64,*,packed=False):
    """Return 1–6 mask rows, generating all active draws in one launch.

    A ``None`` key and nonce produce a zero row for a joint input's receiver.
    Other rows use cached eight-word uint32 CUDA keys and fresh unsigned 64-bit
    nonces. Row order is unchanged; repeated keys still require distinct nonces.
    The caller reserves draw counters in its protocol's public schedule order.
    ``packed=True`` stores ring32 rows as uint32, halving their memory traffic.
    Other cases retain int64 storage; the generated bits are unchanged.
    """
    keys=tuple(keys)
    nonces=tuple(nonces)
    count=operator.index(count)
    if not 1<=len(keys)<=6 or len(keys)!=len(nonces):
        raise ValueError("provide matching key and nonce rows, with 1–6 rows")
    if bits not in (32,64):
        raise ValueError("bits must be 32 or 64")
    if count<0:
        raise ValueError("element count must be nonnegative")
    blocks=triton.cdiv(count,16 if bits==32 else 8)
    if blocks>=2**32:
        raise ValueError("a draw must use fewer than 2**32 ChaCha blocks")
    active=[key for key in keys if key is not None]
    if not active:
        raise ValueError("at least one active key is required")
    device=active[0].device
    valid=tuple(key is not None for key in keys)
    checked=[]
    for key,nonce in zip(keys,nonces):
        if key is None:
            if nonce is not None:
                raise ValueError("zero rows require a None nonce")
            checked.append(0)
            continue
        if key.dtype!=torch.uint32 or key.shape!=(8,) or not key.is_cuda or key.device!=device or not key.is_contiguous():
            raise ValueError("keys must be matching contiguous eight-word uint32 CUDA tensors")
        nonce=operator.index(nonce)
        if not 0<=nonce<2**64:
            raise ValueError("draw nonces must fit in 64 bits")
        checked.append(nonce)
    dtype=torch.uint32 if packed and bits==32 else torch.int64
    result=torch.empty((len(keys),count),device=device,dtype=dtype)
    if count:
        launch_keys=tuple(key if key is not None else active[0] for key in keys)
        launch_keys=launch_keys+(active[0],)*(6-len(keys))
        launch_nonces=tuple(checked)+(0,)*(6-len(keys))
        flags=valid+(False,)*(6-len(keys))
        with torch.cuda.device(device):
            _rows[(triton.cdiv(blocks,128),len(keys))](
                *launch_keys,*launch_nonces,result,count,bits,flags,128,num_warps=4)
    return result
