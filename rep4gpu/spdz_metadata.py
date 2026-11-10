"""Small protocol metadata transfers and device-generated broadcast headers."""
import struct
import torch
import triton
import triton.language as tl


def upload_bytes(data,device):
    """Copy at most 4 KiB of metadata without synchronizing the CUDA stream.

    Pinned allocator events retain the host storage until the asynchronous copy
    completes. Bulk values and MAC shares must stay on the device instead.
    """
    if not isinstance(data,bytes) or len(data)>4096:
        raise ValueError('metadata must be bytes with length at most 4096')
    device=torch.device(device)
    if device.type!='cuda':raise ValueError('metadata upload requires CUDA')
    if not data:return torch.empty(0,dtype=torch.uint8,device=device)
    host=torch.empty(len(data),dtype=torch.uint8,pin_memory=True)
    host.copy_(torch.frombuffer(bytearray(data),dtype=torch.uint8))
    return host.to(device,non_blocking=True)


@triton.jit
def _header_byte(SESSION,ROW,COLUMN,H0,H1,H2,H3,ROUND,SIZE,STAGE:tl.constexpr):
    value=(tl.cast(3550860927832445011,tl.uint64)>>(8*COLUMN)).to(tl.uint8)
    session=tl.load(SESSION+COLUMN-8,(COLUMN>=8)&(COLUMN<40),0)
    value=tl.where((COLUMN>=8)&(COLUMN<40),session,value)
    value=tl.where((COLUMN>=40)&(COLUMN<48),(tl.cast(ROUND,tl.uint64)>>(8*(COLUMN-40))).to(tl.uint8),value)
    value=tl.where((COLUMN>=48)&(COLUMN<52),(ROW>>(8*(COLUMN-48))).to(tl.uint8),value)
    value=tl.where(COLUMN==52,STAGE,value)
    which=(COLUMN-53)//8
    word=tl.where(which==0,H0,tl.where(which==1,H1,tl.where(which==2,H2,H3))).to(tl.uint64)
    value=tl.where((COLUMN>=53)&(COLUMN<85),(word>>(8*((COLUMN-53)%8))).to(tl.uint8),value)
    return tl.where(COLUMN>=85,(tl.cast(SIZE,tl.uint64)>>(8*(COLUMN-85))).to(tl.uint8),value)


@triton.jit
def _headers(SESSION,OUT,H0,H1,H2,H3,ROUND,SIZE,STAGE:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.arange(0,BLOCK)
    value=_header_byte(SESSION,i//93,i%93,H0,H1,H2,H3,ROUND,SIZE,STAGE)
    tl.store(OUT+i,value,i<372)


@triton.jit
def _header_frame(SESSION,PAYLOAD,HEADERS,FRAME,H0,H1,H2,H3,ROUND,PARTY,
                   N:tl.constexpr,STAGE:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    if tl.program_id(0)<2:
        expected=_header_byte(SESSION,i//93,i%93,H0,H1,H2,H3,ROUND,N,STAGE)
        tl.store(HEADERS+i,expected,i<372)
    if tl.program_id(0)==0:
        own=_header_byte(SESSION,PARTY,i,H0,H1,H2,H3,ROUND,N,STAGE)
        tl.store(FRAME+i,own,i<93)
    data=tl.load(PAYLOAD+i-93,(i>=93)&(i<N+93),0)
    tl.store(FRAME+i,data,(i>=93)&(i<N+93))


def headers(session_hash_gpu,label_hash,round_id,size,stage):
    """Build the four exact SPDZ31G1 headers from cached session bytes."""
    if session_hash_gpu.dtype!=torch.uint8 or session_hash_gpu.shape!=(32,) or not session_hash_gpu.is_cuda:
        raise ValueError('session hash must contain 32 CUDA bytes')
    if not isinstance(label_hash,bytes) or len(label_hash)!=32 or stage not in (b'V',b'E'):
        raise ValueError('invalid header hash or stage')
    if not 0<=round_id<1<<64 or not 0<=size<1<<64:
        raise ValueError('header counter and length must fit uint64')
    result=torch.empty((4,93),dtype=torch.uint8,device=session_hash_gpu.device)
    words=struct.unpack('<4q',label_hash)
    _headers[(1,)](session_hash_gpu,result,*words,round_id,size,stage[0],512,num_warps=4)
    return result


def header_frame(session_hash_gpu,label_hash,round_id,stage,party,payload):
    """Build expected headers and one owned frame in a single allocation/launch.

    The frame owns a copy of its payload. Returned buffers remain unchanged
    when a later call uses the same input tensors or a different operation.
    """
    if session_hash_gpu.dtype!=torch.uint8 or session_hash_gpu.shape!=(32,) or not session_hash_gpu.is_cuda:
        raise ValueError('session hash must contain 32 CUDA bytes')
    if payload.dtype!=torch.uint8 or payload.ndim!=1 or payload.device!=session_hash_gpu.device or not payload.is_contiguous():
        raise ValueError('frame payload must be contiguous bytes on the session device')
    if not isinstance(label_hash,bytes) or len(label_hash)!=32 or stage not in (b'V',b'E') or party not in range(4):
        raise ValueError('invalid frame hash, stage or party')
    if not 0<=round_id<1<<64:raise ValueError('header counter must fit uint64')
    size=payload.numel()
    storage=torch.empty(372+93+size,dtype=torch.uint8,device=payload.device)
    expected=storage[:372].reshape(4,93)
    frame=storage[372:]
    _header_frame[(triton.cdiv(max(372,93+size),256),)](session_hash_gpu,payload,expected,frame,
        *struct.unpack('<4q',label_hash),round_id,party,size,stage[0],256,num_warps=4)
    return expected,frame
