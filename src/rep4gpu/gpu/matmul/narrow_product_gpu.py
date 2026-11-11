"""Checked 16-bit products for coefficient-annihilated truncation corrections."""
import math

import torch
import triton
import triton.language as tl

from ...protocol.device_runtime import DeviceParty, _SPARSE_SCHEDULE, mask_domain
from ..crypto.gpu_prg_batch import generate_rows
from ...protocol.runtime import Share


_DOMAIN = b'rep4/trunc-pr/product16/v1\0'


def product16_domain(bits, binary_bits, excluded):
    """Separate packed 16-bit product masks from ordinary arithmetic masks."""
    return _DOMAIN+mask_domain(bits,binary_bits,False,excluded)


class _Wire16:
    """Reuse checked device framing without changing the party's ring width."""
    def __init__(self,rt):
        self.rt=rt
    def __getattr__(self,name):
        return getattr(self.rt,name)
    def _wire_bits(self,binary):
        if binary:
            raise ValueError('narrow products require arithmetic wires')
        return 16


@triton.jit
def _masked16(X0,X1,X2,Y0,Y1,Y2,MASKS,OUT,N:tl.constexpr,
              PADDED:tl.constexpr,JOINTS:tl.constexpr,COLUMNS:tl.constexpr,
              BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    x0=tl.load(X0+i,i<N,0).to(tl.uint32)
    x1=tl.load(X1+i,i<N,0).to(tl.uint32)
    x2=tl.load(X2+i,i<N,0).to(tl.uint32)
    y0=tl.load(Y0+i,i<N,0).to(tl.uint32)
    y1=tl.load(Y1+i,i<N,0).to(tl.uint32)
    y2=tl.load(Y2+i,i<N,0).to(tl.uint32)
    a0=(x0+x1)*y0+x0*y1
    a1=(x1+x2)*y1+x1*y2
    a4=x0*y2+x2*y0
    for j in tl.static_range(2):
        if COLUMNS[j]==0:
            value=a0
        elif COLUMNS[j]==1:
            value=a1
        else:
            value=a4
        mask=tl.load(MASKS+JOINTS[j]*PADDED+i,i<N,0).to(tl.uint32)
        tl.store(OUT+j*N+i,(value-mask)&0xffff,i<N)


@triton.jit
def _assemble16(MASKS,LOCAL,RECEIVED,OUT,N:tl.constexpr,PADDED:tl.constexpr,
                MASK_SLOTS:tl.constexpr,OUTSIDE_SLOTS:tl.constexpr,
                LOCAL_IDS:tl.constexpr,RECEIVE_ID:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    for slot in tl.static_range(3):
        value=tl.full((BLOCK,),0,tl.uint32)
        for k in tl.static_range(4):
            if MASK_SLOTS[k]==slot:
                value+=tl.load(MASKS+k*PADDED+i,i<N,0).to(tl.uint32)
            if OUTSIDE_SLOTS[k]==slot:
                if LOCAL_IDS[k]>=0:
                    value+=tl.load(LOCAL+LOCAL_IDS[k]*N+i,i<N,0).to(tl.uint32)
                elif RECEIVE_ID==k:
                    value+=tl.load(RECEIVED+i,i<N,0).to(tl.uint32)
        tl.store(OUT+slot*N+i,(value&0xffff).to(tl.int64),i<N)


def product16(rt,x,y):
    """Multiply complementary-support arithmetic shares modulo 2**16.

    The caller establishes global supports {0,1} and {2,3}, in either order.
    The returned int64 components represent a 16-bit product, not a lifted
    full-ring product. They may only enter a full-ring linear expression whose
    coefficient annihilates 2**16, such as -2**(bits-shift) for shift<=16.
    All retained terms use fresh independent masks and checked balanced joints.
    """
    if x.binary or y.binary or x.shape!=y.shape or x.device!=y.device or x.device!=rt.device:
        raise ValueError('narrow products require matching arithmetic shares on the party device')
    if x.device.type!='cuda' or rt.bits not in (32,64):
        raise ValueError('narrow products require a 32-bit or 64-bit CUDA party')
    count=math.prod(x.shape)
    if not count:
        return rt.public(0,like=x)
    words=triton.cdiv(count,2)
    excluded=tuple(None if rt.party==r else r for _,_,r,_ in _SPARSE_SCHEDULE)
    keys=tuple(None if e is None else rt._device_keys[e].key(
        product16_domain(rt.bits,rt.binary_bits,e)) for e in excluded)
    builder=getattr(rt,'_phase_builder',None)
    if builder is None:
        reservations=iter(rt._reserve_nonces(e for e in excluded if e is not None))
        nonces=tuple(None if e is None else next(reservations) for e in excluded)
        masks=generate_rows(keys,nonces,words,bits=32,packed=True)
    else:
        masks=builder.masks(keys,excluded,(words,),32,packed=True)
    # Each row has its own padded final halfword for odd element counts.
    masks=masks.view(torch.uint16)
    local_joints=tuple(k for k,(s,b,_,_) in enumerate(_SPARSE_SCHEDULE) if rt.party in (s,b))
    joint_terms=(1,3,4,5)
    columns=tuple(0 if joint_terms[k]==(rt.party-1)%4 else
                  1 if joint_terms[k]==rt.party else 2 for k in local_joints)
    masked=torch.empty((2,count),dtype=torch.uint16,device=x.device)
    _masked16[(triton.cdiv(count,256),)](*(p.contiguous() for p in x.parts),
        *(p.contiguous() for p in y.parts),masks,masked,count,2*words,local_joints,columns,256)
    wires={k:masked[j].view(torch.uint8) for j,k in enumerate(local_joints)}
    entries=[(s,b,r,wires.get(k)) for k,(s,b,r,_) in enumerate(_SPARSE_SCHEDULE)]
    received=DeviceParty._exchange_joint_wires(_Wire16(rt),entries,x.shape,False)
    receive_id=next(k for k,(_,_,r,_) in enumerate(_SPARSE_SCHEDULE) if rt.party==r)
    incoming=received[receive_id].view(torch.uint16)
    result=torch.empty((3,*x.shape),dtype=torch.int64,device=x.device)
    mask_slots=tuple(-1 if rt.party==r else (r-rt.party-1)%4 for _,_,r,_ in _SPARSE_SCHEDULE)
    outside_slots=tuple(-1 if rt.party==o else (o-rt.party-1)%4 for _,_,_,o in _SPARSE_SCHEDULE)
    local_ids=tuple(local_joints.index(k) if k in local_joints else -1 for k in range(4))
    _assemble16[(triton.cdiv(count,256),)](masks,masked,incoming,result,count,2*words,
        mask_slots,outside_slots,local_ids,receive_id,256)
    rt.joint_inputs+=4
    rt.multiplications+=1
    return Share(tuple(result.unbind(0)),False)
