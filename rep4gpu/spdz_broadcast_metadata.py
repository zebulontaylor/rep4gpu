"""Allocate both checked-broadcast stages before the value exchange."""
import struct
import torch
import triton
import triton.language as tl

from .spdz_metadata import _header_byte


@triton.jit
def _prepare(SESSION,PAYLOAD,VHEAD,VFRAME,EHEAD,EFRAME,H0,H1,H2,H3,ROUND,PARTY,
             N:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    if tl.program_id(0)<2:
        vh=_header_byte(SESSION,i//93,i%93,H0,H1,H2,H3,ROUND,N,86)
        eh=_header_byte(SESSION,i//93,i%93,H0,H1,H2,H3,ROUND,128,69)
        tl.store(VHEAD+i,vh,i<372)
        tl.store(EHEAD+i,eh,i<372)
    if tl.program_id(0)==0:
        vh=_header_byte(SESSION,PARTY,i,H0,H1,H2,H3,ROUND,N,86)
        eh=_header_byte(SESSION,PARTY,i,H0,H1,H2,H3,ROUND,128,69)
        tl.store(VFRAME+i,vh,i<93)
        tl.store(EFRAME+i,eh,i<93)
    value=tl.load(PAYLOAD+i-93,(i>=93)&(i<N+93),0)
    tl.store(VFRAME+i,value,(i>=93)&(i<N+93))


def prepare_broadcast(session_hash,label_hash,round_id,party,payload):
    """Return owned V headers/frame and E headers/frame in one launch.

    The final 128 bytes of the echo frame are an output buffer for the frame
    digest operation. The caller must fill them before sending the echo frame.
    No returned frame aliases the caller's payload or a later call's storage.
    """
    if session_hash.dtype!=torch.uint8 or session_hash.shape!=(32,) or not session_hash.is_cuda:
        raise ValueError('session hash must contain 32 CUDA bytes')
    if payload.dtype!=torch.uint8 or payload.ndim!=1 or payload.device!=session_hash.device or not payload.is_contiguous():
        raise ValueError('payload must be contiguous CUDA bytes on the session device')
    if not isinstance(label_hash,bytes) or len(label_hash)!=32 or party not in range(4):
        raise ValueError('invalid label hash or party')
    if not 0<=round_id<1<<64:raise ValueError('round counter must fit uint64')
    size=payload.numel()
    storage=torch.empty(2*372+93+size+221,dtype=torch.uint8,device=payload.device)
    vheaders=storage[:372].reshape(4,93)
    eheaders=storage[372:744].reshape(4,93)
    own=storage[744:837+size]
    echo=storage[837+size:]
    _prepare[(triton.cdiv(max(372,93+size),256),)](session_hash,payload,vheaders,own,eheaders,echo,
        *struct.unpack('<4q',label_hash),round_id,party,size,256,num_warps=4)
    return vheaders,own,eheaders,echo
