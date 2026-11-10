"""Checked CUDA opening for the experimental BLAKE3 device protocol.

The first round sends local slot zero clockwise. The second sends the BLAKE3
hash of local slot two counterclockwise. Reconstruction starts only after the
received missing component matches its independently supplied backup tag.
"""
import math
import torch
import triton
import triton.language as tl


@triton.jit
def _encode32(A, B, OUT, N: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0)*BLOCK + tl.arange(0,BLOCK)
    a = tl.load(A+i,i < N,0).to(tl.uint32)
    b = tl.load(B+i,i < N,0).to(tl.uint32)
    tl.store(OUT+i,a,i < N)
    tl.store(OUT+N+i,b,i < N)


@triton.jit
def _reconstruct(MISSING, A, B, C, OUT, N: tl.constexpr,
                 BINARY: tl.constexpr, WIDTH: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0)*BLOCK + tl.arange(0,BLOCK)
    value = tl.load(MISSING+i,i < N,0).to(tl.uint64)
    a = tl.load(A+i,i < N,0).to(tl.uint64)
    b = tl.load(B+i,i < N,0).to(tl.uint64)
    c = tl.load(C+i,i < N,0).to(tl.uint64)
    if BINARY:
        value = value ^ a ^ b ^ c
    else:
        value = value+a+b+c
    if WIDTH == 32:
        if BINARY:
            result = value.to(tl.uint32).to(tl.int64)
        else:
            result = value.to(tl.int32).to(tl.int64)
    else:
        result = value.to(tl.int64)
    tl.store(OUT+i,result,i < N)


def _validate_frame(incoming, peer, size, device):
    if set(incoming) != {peer}:
        raise ValueError('device opening received an unexpected peer set')
    value = incoming[peer]
    if (not isinstance(value,torch.Tensor) or value.dtype != torch.uint8
            or value.device != device or value.ndim != 1
            or not value.is_contiguous() or value.numel() != size):
        raise ValueError('device opening received an invalid frame')
    return value


def _exchange(rt, outgoing, sizes):
    method = getattr(rt,'_exchange_device',None)
    if method is None:
        method = rt.transport.exchange_device
    return method(outgoing,sizes)


def _finish(x, missing, width):
    result = torch.empty(x.shape,dtype=torch.int64,device=x.device)
    count = math.prod(x.shape)
    if count:
        dtype = torch.uint32 if width == 32 else torch.int64
        # Device frames may start at an arbitrary byte-buffer offset.
        if missing.storage_offset() % (width//8):
            missing = missing.clone()
        parts = tuple(part.contiguous().reshape(-1) for part in x.parts)
        _reconstruct[(triton.cdiv(count,256),)](missing.view(dtype),*parts,result,count,
                                               x.binary,width,256)
    return result


def open_device(rt,x):
    """Reveal one Rep4 share after two GPU BLAKE3-checked protocol rounds.

    Arithmetic ring32 results are signed; binary32 results are canonical words
    in [0,2**32). The function consumes no PRG draws. Empty public shapes retain
    both rounds, including an explicit empty clockwise payload and a 32-byte
    counterclockwise BLAKE3 tag. Byte transports must preserve that public
    zero-length receive entry when reconstructing their device frames.
    """
    if x.device.type != 'cuda':
        raise ValueError('device opening requires CUDA shares')
    from .gpu_hash import digest_batch
    count = math.prod(x.shape)
    width = rt._wire_bits(x.binary)
    if width not in (32,64):
        raise ValueError('device opening requires 32-bit or 64-bit words')
    device = x.device
    first = x.parts[0].contiguous().reshape(-1)
    backup = x.parts[2].contiguous().reshape(-1)
    if width == 32:
        encoded = torch.empty((2,count),dtype=torch.uint32,device=device)
        if count:
            _encode32[(triton.cdiv(count,256),)](first,backup,encoded,count,256)
        payload, backup_wire = (part.view(torch.uint8) for part in encoded.unbind(0))
    else:
        payload, backup_wire = first.view(torch.uint8),backup.view(torch.uint8)
    sender,receiver = (rt.party-1)%4,(rt.party+1)%4
    incoming = _exchange(rt,{receiver:payload},{sender:count*(width//8)})
    missing = _validate_frame(incoming,sender,count*(width//8),device)
    # Equal-length independent messages share the GPU hashing dispatch.
    tags = digest_batch(torch.stack((backup_wire,missing)),
                        block_tree=getattr(rt,'hash_block_tree',True))
    checks = _exchange(rt,{sender:tags[0]},{receiver:32})
    actual = _validate_frame(checks,receiver,32,device)
    if not torch.equal(actual,tags[1]):
        error = ValueError('device opening BLAKE3 component hash mismatch')
        abort = getattr(rt.transport,'abort',None)
        if abort is not None:
            abort(error)
        raise error
    return _finish(x,missing,width)
