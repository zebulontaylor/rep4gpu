"""Packed low-prefix carries for exact truncation with quarter-ring headroom."""
import math

import torch
import triton
import triton.language as tl

from .runtime import Share


@triton.jit
def _pack_pair(PAIR, OUT, N: tl.constexpr, SHIFT: tl.constexpr,
               BITS: tl.constexpr, WIDTH: tl.constexpr, WORDS: tl.constexpr,
               BLOCK: tl.constexpr):
    word = tl.program_id(0)*BLOCK+tl.arange(0, BLOCK)
    bit = tl.arange(0, WIDTH)
    stream = word[:, None]*WIDTH+bit[None, :]
    item, position = stream//(SHIFT+1), stream % (SHIFT+1)
    source = tl.load(PAIR+item, item<N, 0).to(tl.uint64)
    position = tl.where(position == SHIFT, BITS-1, position)
    value = (source >> position) & 1
    value = tl.where(item<N, value, 0)
    packed = tl.sum(value << bit[None, :], 1)
    tl.store(OUT+word, packed.to(tl.int64), word<WORDS)


@triton.jit
def _unpack_initial(G0,G1,G2,P0,P1,P2,STATE,TOP,
                    N:tl.constexpr,SHIFT:tl.constexpr,WIDTH:tl.constexpr,
                    BLOCK:tl.constexpr):
    component = tl.program_id(1)
    item = tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    if component == 0:
        gp,pp=G0,P0
    elif component == 1:
        gp,pp=G1,P1
    else:
        gp,pp=G2,P2
    valid=item<N*SHIFT
    coordinate,position=item//SHIFT,item%SHIFT
    stream=coordinate*(SHIFT+1)+position
    g=(tl.load(gp+stream//WIDTH,valid,0).to(tl.uint64) >> (stream%WIDTH))&1
    p=(tl.load(pp+stream//WIDTH,valid,0).to(tl.uint64) >> (stream%WIDTH))&1
    tl.store(STATE+component*N*SHIFT+item,g.to(tl.int64),valid)
    tl.store(STATE+3*N*SHIFT+component*N*SHIFT+item,p.to(tl.int64),valid)
    stream=item*(SHIFT+1)+SHIFT
    top=(tl.load(gp+stream//WIDTH,item<N,0).to(tl.uint64) >> (stream%WIDTH))&1
    tl.store(TOP+component*N+item,top.to(tl.int64),item<N)


@triton.jit
def _pack_level(STATE,OUT,N:tl.constexpr,SIZE:tl.constexpr,
                WIDTH:tl.constexpr,WORDS:tl.constexpr,FINAL:tl.constexpr,
                BLOCK:tl.constexpr):
    component=tl.program_id(1)
    word=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    bit=tl.arange(0,WIDTH)
    gate=word[:,None]*WIDTH+bit[None,:]
    FACTOR:tl.constexpr=1 if FINAL else 2
    PAIRS:tl.constexpr=SIZE//2
    coordinate=gate//(PAIRS*FACTOR)
    slot=gate%(PAIRS*FACTOR)
    pair=slot//FACTOR
    propagate=False if FINAL else (slot%2==1)
    low=coordinate*SIZE+pair*2
    high=low+1
    valid=coordinate<N
    left=tl.load(STATE+3*N*SIZE+component*N*SIZE+high,valid,0).to(tl.uint64)&1
    right_offset=tl.where(propagate,3*N*SIZE,0)
    right=tl.load(STATE+right_offset+component*N*SIZE+low,valid,0).to(tl.uint64)&1
    left=tl.sum(tl.where(valid,left,0)<<bit[None,:],1)
    right=tl.sum(tl.where(valid,right,0)<<bit[None,:],1)
    tl.store(OUT+component*WORDS+word,left.to(tl.int64),word<WORDS)
    tl.store(OUT+3*WORDS+component*WORDS+word,right.to(tl.int64),word<WORDS)


@triton.jit
def _next_level(STATE,Q0,Q1,Q2,OUT,N:tl.constexpr,SIZE:tl.constexpr,
                NEXT:tl.constexpr,WIDTH:tl.constexpr,FINAL:tl.constexpr,
                BLOCK:tl.constexpr):
    component=tl.program_id(1)
    item=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    coordinate,node=item//NEXT,item%NEXT
    valid=coordinate<N
    paired=node<SIZE//2
    source=coordinate*SIZE+node*2+tl.where(paired,1,0)
    old_g=tl.load(STATE+component*N*SIZE+source,valid,0).to(tl.uint64)&1
    old_p=tl.load(STATE+3*N*SIZE+component*N*SIZE+source,valid,0).to(tl.uint64)&1
    if component == 0: qp=Q0
    elif component == 1: qp=Q1
    else: qp=Q2
    FACTOR:tl.constexpr=1 if FINAL else 2
    gate=(coordinate*(SIZE//2)+node)*FACTOR
    product=(tl.load(qp+gate//WIDTH,valid&paired,0).to(tl.uint64)>>(gate%WIDTH))&1
    if FINAL:
        propagation=tl.full((BLOCK,),0,tl.uint64)
    else:
        gate+=1
        propagation=(tl.load(qp+gate//WIDTH,valid&paired,0).to(tl.uint64)>>(gate%WIDTH))&1
    g=tl.where(paired,old_g^product,old_g)
    p=tl.where(paired,propagation,old_p)
    tl.store(OUT+component*N*NEXT+item,g.to(tl.int64),valid)
    tl.store(OUT+3*N*NEXT+component*N*NEXT+item,p.to(tl.int64),valid)


def low_carry_and_top_product(rt, pair, shape, shift):
    """Jointly share only low ``shift`` bits and the sign bit of each pair."""
    count,width=math.prod(shape),rt.binary_bits
    words=(count*(shift+1)+width-1)//width
    packed=torch.empty(words,dtype=torch.int64,device=pair.device)
    _pack_pair[(triton.cdiv(words,4),)](pair.contiguous().reshape(-1),packed,
        count,shift,rt.bits,width,words,4)
    a,b=rt.joint_input_batch(
        [(0,1,3,2,packed if rt.party<2 else None),
         (2,3,1,0,packed if rt.party>=2 else None)],
        (words,),True,pair.device)
    generate=(rt._mul_sparse_pairs(a,b) if getattr(rt,'sparse_products',False)
              else rt.mul(a,b))
    propagate=rt.add(a,b)
    state=torch.empty((2,3,count*shift),dtype=torch.int64,device=pair.device)
    top=torch.empty((3,*shape),dtype=torch.int64,device=pair.device)
    _unpack_initial[(triton.cdiv(count*shift,256),3)](*generate.parts,*propagate.parts,
        state,top,count,shift,width,256)
    size=shift
    while size>1:
        final=size==2
        gates_per_item=(size//2)*(1 if final else 2)
        gate_words=(count*gates_per_item+width-1)//width
        gates=torch.empty((2,3,gate_words),dtype=torch.int64,device=pair.device)
        _pack_level[(triton.cdiv(gate_words,4),3)](state,gates,count,size,width,gate_words,final,4)
        product=rt.mul(Share(tuple(gates[0].unbind(0)),True),
                       Share(tuple(gates[1].unbind(0)),True))
        next_size=(size+1)//2
        next_state=torch.empty((2,3,count*next_size),dtype=torch.int64,device=pair.device)
        _next_level[(triton.cdiv(count*next_size,256),3)](state,*product.parts,next_state,
            count,size,next_size,width,final,256)
        state,size=next_state,next_size
    low=Share(tuple(part.reshape(shape) for part in state[0].unbind(0)),True)
    return low,Share(tuple(top.unbind(0)),True)
