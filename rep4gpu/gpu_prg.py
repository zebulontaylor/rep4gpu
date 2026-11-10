"""GPU-resident ChaCha20 masks with separate keys for explicit public domains.

The state uses ChaCha's original 64-bit block counter and 64-bit nonce. A draw
starts at block zero; its nonce is the caller's unique 64-bit draw counter.
"""
from __future__ import annotations

import hashlib
import operator
import struct

import torch
import triton
import triton.language as tl


DOMAIN_TAG = b"Rep4GPU-DevicePRGv1\0"


def _domain_bytes(domain):
    if isinstance(domain,str):
        return domain.encode("utf-8")
    if isinstance(domain,(bytes,bytearray,memoryview)):
        return bytes(domain)
    raise TypeError("a PRG domain must be bytes or text")


def key_bytes(seed,domain):
    """Derive a 256-bit domain key with an explicit length-framed domain."""
    if not isinstance(seed,(bytes,bytearray,memoryview)):
        raise TypeError("PRG seeds must be bytes")
    seed=bytes(seed)
    if len(seed)<32:
        raise ValueError("PRG seeds must contain at least 256 bits")
    domain=_domain_bytes(domain)
    return hashlib.sha256(seed+DOMAIN_TAG+struct.pack("<Q",len(domain))+domain).digest()


class DomainKeys:
    """Cache each domain's eight uint32 key words on one CUDA device."""

    def __init__(self,seed,device="cuda"):
        if not isinstance(seed,(bytes,bytearray,memoryview)):
            raise TypeError("PRG seeds must be bytes")
        self._seed=bytes(seed)
        if len(self._seed)<32:
            raise ValueError("PRG seeds must contain at least 256 bits")
        device=torch.device(device)
        if device.type!="cuda":
            raise ValueError("GPU PRG keys require a CUDA device")
        self.device=torch.device("cuda",torch.cuda.current_device()) if device.index is None else device
        self._keys={}

    def key(self,domain):
        domain=_domain_bytes(domain)
        if domain not in self._keys:
            words=struct.unpack("<8I",key_bytes(self._seed,domain))
            self._keys[domain]=torch.tensor(words,device=self.device,dtype=torch.uint32)
        return self._keys[domain]


@triton.jit
def _rotl(x,N:tl.constexpr):
    return (x<<N)|(x>>(32-N))


@triton.jit
def _quarter(a,b,c,d):
    a=a+b
    d=_rotl(d^a,16)
    c=c+d
    b=_rotl(b^c,12)
    a=a+b
    d=_rotl(d^a,8)
    c=c+d
    b=_rotl(b^c,7)
    return a,b,c,d


@triton.jit(do_not_specialize=["NONCE","BLOCK_START"])
def _chacha_words(KEY,OUT,NONCE,BLOCK_START,COUNT:tl.constexpr,BITS:tl.constexpr,BLOCK:tl.constexpr):
    block=tl.program_id(0).to(tl.uint64)*BLOCK+tl.arange(0,BLOCK).to(tl.uint64)
    counter=block+BLOCK_START.to(tl.uint64)
    nonce=NONCE.to(tl.uint64)
    s0=tl.full((BLOCK,),0x61707865,tl.uint32)
    s1=tl.full((BLOCK,),0x3320646e,tl.uint32)
    s2=tl.full((BLOCK,),0x79622d32,tl.uint32)
    s3=tl.full((BLOCK,),0x6b206574,tl.uint32)
    s4=tl.full((BLOCK,),0,tl.uint32)+tl.load(KEY+0)
    s5=tl.full((BLOCK,),0,tl.uint32)+tl.load(KEY+1)
    s6=tl.full((BLOCK,),0,tl.uint32)+tl.load(KEY+2)
    s7=tl.full((BLOCK,),0,tl.uint32)+tl.load(KEY+3)
    s8=tl.full((BLOCK,),0,tl.uint32)+tl.load(KEY+4)
    s9=tl.full((BLOCK,),0,tl.uint32)+tl.load(KEY+5)
    s10=tl.full((BLOCK,),0,tl.uint32)+tl.load(KEY+6)
    s11=tl.full((BLOCK,),0,tl.uint32)+tl.load(KEY+7)
    s12=counter.to(tl.uint32)
    s13=(counter>>32).to(tl.uint32)
    s14=tl.full((BLOCK,),0,tl.uint32)+nonce.to(tl.uint32)
    s15=tl.full((BLOCK,),0,tl.uint32)+(nonce>>32).to(tl.uint32)
    x0,x1,x2,x3=s0,s1,s2,s3
    x4,x5,x6,x7=s4,s5,s6,s7
    x8,x9,x10,x11=s8,s9,s10,s11
    x12,x13,x14,x15=s12,s13,s14,s15
    # Column and diagonal rounds follow RFC 8439 section 2.3; only the
    # counter/nonce allocation differs from its IETF state layout.
    for _ in range(10):
        x0,x4,x8,x12=_quarter(x0,x4,x8,x12)
        x1,x5,x9,x13=_quarter(x1,x5,x9,x13)
        x2,x6,x10,x14=_quarter(x2,x6,x10,x14)
        x3,x7,x11,x15=_quarter(x3,x7,x11,x15)
        x0,x5,x10,x15=_quarter(x0,x5,x10,x15)
        x1,x6,x11,x12=_quarter(x1,x6,x11,x12)
        x2,x7,x8,x13=_quarter(x2,x7,x8,x13)
        x3,x4,x9,x14=_quarter(x3,x4,x9,x14)
    words=(x0+s0,x1+s1,x2+s2,x3+s3,x4+s4,x5+s5,x6+s6,x7+s7,
           x8+s8,x9+s9,x10+s10,x11+s11,x12+s12,x13+s13,x14+s14,x15+s15)
    if BITS==32:
        for i in tl.static_range(16):
            idx=block*16+i
            tl.store(OUT+idx,words[i].to(tl.uint64).to(tl.int64),idx<COUNT)
    else:
        for i in tl.static_range(8):
            idx=block*8+i
            word=words[2*i].to(tl.uint64)|(words[2*i+1].to(tl.uint64)<<32)
            tl.store(OUT+idx,word.to(tl.int64),idx<COUNT)


def _generate(key_tensor,nonce,count,bits=64,*,block_start=0):
    nonce=operator.index(nonce)
    count=operator.index(count)
    block_start=operator.index(block_start)
    if bits not in (32,64):
        raise ValueError("bits must be 32 or 64")
    if not 0<=nonce<2**64:
        raise ValueError("draw nonce must fit in 64 bits")
    if not 0<=count<2**63:
        raise ValueError("element count must be nonnegative and fit in 63 bits")
    blocks=triton.cdiv(count,16 if bits==32 else 8)
    if not 0<=block_start<2**64 or blocks>2**64-block_start:
        raise ValueError("ChaCha block counter exhausted")
    if not isinstance(key_tensor,torch.Tensor) or key_tensor.dtype!=torch.uint32 or key_tensor.shape!=(8,) or not key_tensor.is_cuda:
        raise ValueError("key must contain eight uint32 CUDA words")
    if not key_tensor.is_contiguous():
        raise ValueError("key words must be contiguous")
    out=torch.empty((count,),device=key_tensor.device,dtype=torch.int64)
    if count:
        with torch.cuda.device(key_tensor.device):
            _chacha_words[(triton.cdiv(blocks,128),)](
                key_tensor,out,nonce,block_start,count,bits,128,num_warps=4)
    return out


def generate(key_tensor,nonce,count,bits=64):
    """Generate a fresh draw directly into an int64 CUDA tensor.

    A 32-bit draw contains canonical unsigned values in [0,2**32); 64-bit
    draws retain signed int64 bit patterns. Serialization is little-endian,
    matching ChaCha20 applied to zeros with IV ``struct.pack('<QQ',0,nonce)``.
    Draws use fewer than 2**32 blocks. This also matches cryptography releases
    that interpret the same IV as a 32-bit counter plus a 96-bit nonce: their
    initial state is identical and the low block counter never overflows.
    The caller must never reuse a key/nonce pair for distinct mask draws.
    """
    count=operator.index(count)
    if bits not in (32,64):
        raise ValueError("bits must be 32 or 64")
    if triton.cdiv(count,16 if bits==32 else 8)>=2**32:
        raise ValueError("a draw must use fewer than 2**32 ChaCha blocks")
    return _generate(key_tensor,nonce,count,bits)


def generate_bytes(key_tensor,nonce,count):
    """Return a CUDA uint8 view of exactly ``count`` ChaCha20 stream bytes."""
    count=operator.index(count)
    if count<0:
        raise ValueError("byte count must be nonnegative")
    words=generate(key_tensor,nonce,triton.cdiv(count,8),64)
    return words.view(torch.uint8)[:count]
