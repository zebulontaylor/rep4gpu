"""Exact GPU matrix products in rings with 8, 16, 32, or 64 bits."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import triton
import triton.language as tl


@triton.jit
def _tensor_mm(A, B, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
               BITS: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
               BK: tl.constexpr, PREPARED: tl.constexpr = False):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    kk = tl.arange(0, BK)
    out = tl.full((BM, BN), 0, tl.uint64)
    for start in range(tl.cdiv(K, BK)):
        a = tl.load(A + rows[:, None] * K + start * BK + kk[None, :],
                    (rows[:, None] < M) & (start * BK + kk[None, :] < K), 0).to(tl.uint64)
        if not PREPARED:
            b = tl.load(B + (start * BK + kk[:, None]) * N + cols[None, :],
                        (start * BK + kk[:, None] < K) & (cols[None, :] < N), 0).to(tl.uint64)
            b += 0x8080808080808080
        # Adding this constant turns byte extraction into balanced signed digits.
        a += 0x8080808080808080
        for diagonal in tl.static_range(BITS // 8):
            acc = tl.full((BM, BN), 0, tl.int32)
            for i in tl.static_range(diagonal + 1):
                ai = (((a >> (8 * i)) & 255).to(tl.int32) - 128).to(tl.int8)
                if PREPARED:
                    bi = tl.load(B + (diagonal-i)*K*N + (start*BK+kk[:,None])*N + cols[None,:],
                                 (start*BK+kk[:,None]<K)&(cols[None,:]<N), 0)
                else:
                    bi = (((b >> (8 * (diagonal-i))) & 255).to(tl.int32) - 128).to(tl.int8)
                acc = tl.dot(ai, bi, acc, out_dtype=tl.int32)
            out += acc.to(tl.int64).to(tl.uint64) << (8 * diagonal)
    if BITS < 64:
        out = out & ((1 << BITS)-1)
    tl.store(C + rows[:, None] * N + cols[None, :], out.to(tl.int64),
             (rows[:, None] < M) & (cols[None, :] < N))


@triton.jit
def _decode_mm(A, B, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
               BITS: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    kk = tl.arange(0, BK)
    DTYPE: tl.constexpr = tl.uint32 if BITS <= 32 else tl.uint64
    sums = tl.full((BK, BN), 0, DTYPE)
    for start in range(tl.cdiv(K, BK)):
        a = tl.load(A + row*K + start*BK+kk, start*BK+kk < K, 0).to(DTYPE)
        b = tl.load(B + (start*BK+kk[:,None])*N+cols[None,:],
                    (start*BK+kk[:,None]<K)&(cols[None,:]<N), 0).to(DTYPE)
        sums += a[:,None]*b
    out = tl.sum(sums, 0)
    if BITS < 64:
        out &= (1 << BITS)-1
    tl.store(C+row*N+cols, out.to(tl.int64), cols<N)


@triton.jit
def _pack_weights(B, P, SIZE: tl.constexpr, LIMBS: tl.constexpr, BLOCK: tl.constexpr):
    idx = tl.program_id(0)*BLOCK + tl.arange(0, BLOCK)
    limb = tl.program_id(1)
    word = tl.load(B+idx, idx<SIZE, 0).to(tl.uint64) + 0x8080808080808080
    digit = (((word >> (8*limb)) & 255).to(tl.int32)-128).to(tl.int8)
    tl.store(P+limb*SIZE+idx,digit,idx<SIZE)


@dataclass(frozen=True)
class PreparedWeight:
    """Balanced int8 limbs for a static matrix; rebuild after changing weights."""
    limbs: torch.Tensor
    bits: int
    original: torch.Tensor | None = None

    @property
    def shape(self):
        return self.limbs.shape[1:]

    @property
    def device(self):
        return self.limbs.device


@dataclass(frozen=True)
class CompactWeight:
    """One signed int32 word per weight, interpreted modulo 2**32."""
    words: torch.Tensor
    bits: int = 32

    def __post_init__(self):
        if self.bits != 32 or self.words.dtype != torch.int32 or self.words.ndim != 2 or not self.words.is_cuda:
            raise ValueError("compact weights require a two-dimensional int32 CUDA tensor in the 32-bit ring")
        if not self.words.is_contiguous():
            raise ValueError("compact weight words must be contiguous")

    @property
    def shape(self):
        return self.words.shape

    @property
    def device(self):
        return self.words.device


def prepare_weight(b: torch.Tensor, bits: int = 64, *, keep_original: bool = False,
                   storage: str = "limbs") -> PreparedWeight | CompactWeight:
    """Pack static weights; default storage uses balanced int8 limbs.

    ``storage="words32"`` stores one int32 word per value and requires bits=32.
    Both 32-bit storage modes use four bytes per value. Compact words avoid
    byte-plane reconstruction in decode and retain no source tensor.
    ``keep_original=True`` also retains raw data with the default limb storage.
    """
    if bits not in (8,16,32,64):
        raise ValueError("bits must be 8, 16, 32, or 64")
    allowed = (torch.int64, {8: torch.uint8, 16: torch.uint16}[bits]) if bits <= 16 else (torch.int64,)
    if b.ndim != 2 or b.dtype not in allowed or not b.is_cuda:
        raise ValueError("weight must be a CUDA matrix with int64 storage or matching ring8/16 packed storage")
    if storage not in ("limbs", "words32"):
        raise ValueError("storage must be limbs or words32")
    if storage == "words32":
        if bits != 32 or keep_original:
            raise ValueError("words32 storage requires bits=32 and keep_original=False")
        return CompactWeight(b.to(torch.int32).contiguous())
    b = b.contiguous()
    packed = torch.empty((bits//8,*b.shape),dtype=torch.int8,device=b.device)
    if b.numel():
        with torch.cuda.device(b.device):
            _pack_weights[(triton.cdiv(b.numel(),256),bits//8)](b,packed,b.numel(),bits//8,256)
    return PreparedWeight(packed,bits,b if keep_original else None)


def ring_matmul(a: torch.Tensor, b: torch.Tensor | PreparedWeight | CompactWeight, bits: int = 64,
                *, backend: str = "auto") -> torch.Tensor:
    """Multiply int64 CUDA matrices exactly modulo ``2**bits``.

    A 64-bit result uses its signed int64 bit pattern. Narrow results are
    canonicalized to [0, 2**bits). Inputs may contain any int64 bit pattern.
    ``tensor`` uses balanced signed bytes and exact int8 tensor-core products.
    """
    if bits not in (8, 16, 32, 64):
        raise ValueError("bits must be 8, 16, 32, or 64")
    compact = isinstance(b, CompactWeight)
    if compact:
        if bits != 32:
            raise ValueError("compact weights require bits=32")
        b = b.words
    prepared = isinstance(b, PreparedWeight)
    if prepared and b.bits != bits:
        raise ValueError("prepared weight ring does not match bits")
    allowed = (torch.int64, torch.uint8) if bits == 8 else (torch.int64,)
    if a.dtype not in allowed or (not prepared and b.dtype not in allowed and not compact):
        raise TypeError("ring_matmul requires int64 inputs or ring8 uint8 inputs")
    if a.ndim != 2 or (not prepared and b.ndim != 2) or a.shape[1] != b.shape[0]:
        raise ValueError("expected compatible two-dimensional matrices")
    if not a.is_cuda or (not prepared and not b.is_cuda) or a.device != b.device:
        raise ValueError("inputs must be on the same CUDA device")
    if backend not in ("auto", "decode", "tensor"):
        raise ValueError("backend must be auto, decode, or tensor")
    a = a.contiguous()
    if not prepared:
        b = b.contiguous()
    m, k = a.shape
    n = b.shape[1]
    c = torch.empty((m, n), dtype=torch.int64, device=a.device)
    if m == 0 or n == 0:
        return c
    if k == 0:
        return c.zero_()
    if backend == "auto":
        backend = "decode" if m == 1 and k >= 2048 and not prepared else "tensor"
    with torch.cuda.device(a.device):
        if backend == "decode" and prepared:
            raise ValueError("prepared weights require tensor backend")
        if backend == "decode":
            _decode_mm[(m, triton.cdiv(n, 32))](a,b,c,m,n,k,bits,32,32,num_warps=4)
        else:
            _tensor_mm[(triton.cdiv(m,16),triton.cdiv(n,32))](
                a,b.limbs if prepared else b,c,m,n,k,bits,16,32,64,prepared,num_warps=4)
    return c

@triton.jit
def _local_products(X0, X1, X2, Y0, Y1, Y2, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
               BITS: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
               BK: tl.constexpr, PREPARED: tl.constexpr = False):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    kk = tl.arange(0, BK)
    out = tl.full((BM, BN), 0, tl.uint64)
    for term in tl.static_range(2):
        for start in range(tl.cdiv(K, BK)):
            if tl.program_id(2) == 0:
                A = X0
                A2 = X1
                if term == 0:
                    B = Y0
                else:
                    B = Y1
            elif tl.program_id(2) == 1:
                A = X1
                A2 = X2
                if term == 0:
                    B = Y1
                else:
                    B = Y2
            else:
                if term == 0:
                    A = X0
                    B = Y2
                else:
                    A = X2
                    B = Y0
                A2 = X0
            a = tl.load(A + rows[:, None] * K + start * BK + kk[None, :],
                        (rows[:, None] < M) & (start * BK + kk[None, :] < K), 0).to(tl.uint64)
            if term == 0:
                if tl.program_id(2) < 2:
                    a += tl.load(A2 + rows[:,None]*K + start*BK+kk[None,:],
                                 (rows[:,None]<M)&(start*BK+kk[None,:]<K),0).to(tl.uint64)
            if not PREPARED:
                b = tl.load(B + (start * BK + kk[:, None]) * N + cols[None, :],
                            (start * BK + kk[:, None] < K) & (cols[None, :] < N), 0).to(tl.uint64)
                b += 0x8080808080808080
            # Adding this constant turns byte extraction into balanced signed digits.
            a += 0x8080808080808080
            for diagonal in tl.static_range(BITS // 8):
                acc = tl.full((BM, BN), 0, tl.int32)
                for i in tl.static_range(diagonal + 1):
                    ai = (((a >> (8 * i)) & 255).to(tl.int32) - 128).to(tl.int8)
                    if PREPARED:
                        bi = tl.load(B + (diagonal-i)*K*N + (start*BK+kk[:,None])*N + cols[None,:],
                                     (start*BK+kk[:,None]<K)&(cols[None,:]<N), 0)
                    else:
                        bi = (((b >> (8 * (diagonal-i))) & 255).to(tl.int32) - 128).to(tl.int8)
                    acc = tl.dot(ai, bi, acc, out_dtype=tl.int32)
                out += acc.to(tl.int64).to(tl.uint64) << (8 * diagonal)
    if BITS < 64:
        out = out & ((1 << BITS)-1)
    tl.store(C + tl.program_id(2)*M*N + rows[:, None] * N + cols[None, :], out.to(tl.int64),
             (rows[:, None] < M) & (cols[None, :] < N))


@triton.jit
def _local_decode(X0,X1,X2,Y0,Y1,Y2,C,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,
                  BITS:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
    row=tl.program_id(0)
    cols=tl.program_id(1)*BN+tl.arange(0,BN)
    kk=tl.arange(0,BK)
    DT:tl.constexpr=tl.uint32 if BITS<=32 else tl.uint64
    s0=tl.full((BK,BN),0,DT)
    s1=tl.full((BK,BN),0,DT)
    s4=tl.full((BK,BN),0,DT)
    for start in range(tl.cdiv(K,BK)):
        ki=start*BK+kk
        xp=row*K+ki
        mask=ki<K
        x0=tl.load(X0+xp,mask,0).to(DT)[:,None]
        x1=tl.load(X1+xp,mask,0).to(DT)[:,None]
        x2=tl.load(X2+xp,mask,0).to(DT)[:,None]
        yp=ki[:,None]*N+cols[None,:]
        mask=(ki[:,None]<K)&(cols[None,:]<N)
        y0=tl.load(Y0+yp,mask,0).to(DT)
        y1=tl.load(Y1+yp,mask,0).to(DT)
        y2=tl.load(Y2+yp,mask,0).to(DT)
        s0+=(x0+x1)*y0+x0*y1
        s1+=(x1+x2)*y1+x1*y2
        s4+=x0*y2+x2*y0
    v0=tl.sum(s0,0).to(tl.uint64)
    v1=tl.sum(s1,0).to(tl.uint64)
    v4=tl.sum(s4,0).to(tl.uint64)
    if BITS<64:
        v0&=(1 << BITS)-1
        v1&=(1 << BITS)-1
        v4&=(1 << BITS)-1
    cp=row*N+cols
    tl.store(C+cp,v0.to(tl.int64),cols<N)
    tl.store(C+M*N+cp,v1.to(tl.int64),cols<N)
    tl.store(C+2*M*N+cp,v4.to(tl.int64),cols<N)


def local_products(x0, x1, x2, y0, y1, y2, bits=64):
    """Compute the three local replicated-share contributions in one launch.

    Returns ``(v0,v1,v4)`` where ``v0=(x0+x1)@y0+x0@y1``,
    ``v1=(x1+x2)@y1+x1@y2`` and ``v4=x0@y2+x2@y0`` in the ring.
    Weight arguments can all be ``PreparedWeight`` objects for the same ring.
    """
    if bits not in (8,16,32,64):
        raise ValueError("bits must be 8, 16, 32, or 64")
    xs = (x0,x1,x2)
    ys = (y0,y1,y2)
    compact = isinstance(y0,CompactWeight)
    if any(isinstance(y,CompactWeight)!=compact for y in ys):
        raise TypeError("weight formats must match")
    if compact:
        if bits != 32:
            raise ValueError("compact weights require bits=32")
        ys = tuple(y.words for y in ys)
        y0,y1,y2 = ys
    prepared = isinstance(y0,PreparedWeight)
    if any(isinstance(y,PreparedWeight)!=prepared for y in ys):
        raise TypeError("weight formats must match")
    allowed = (torch.int64, {8: torch.uint8, 16: torch.uint16}[bits]) if bits <= 16 else (torch.int64,)
    if any(x.dtype not in allowed or x.dtype != x0.dtype or x.ndim!=2 or not x.is_cuda for x in xs):
        raise ValueError("input shares require matching CUDA matrices with int64 or matching ring8/16 packed storage")
    if any(x.shape!=x0.shape or x.device!=x0.device for x in xs):
        raise ValueError("input share shapes and devices must match")
    if any(y.shape!=y0.shape or y.device!=x0.device for y in ys):
        raise ValueError("weight share shapes and devices must match")
    if prepared:
        if any(y.bits!=bits for y in ys):
            raise ValueError("prepared weight ring does not match bits")
    elif any((y.dtype not in allowed and not compact) or y.dtype != y0.dtype or y.ndim!=2 or not y.is_cuda for y in ys):
        raise ValueError("weight shares require matching CUDA matrices with int64 or matching ring8/16 packed storage")
    m,k=x0.shape
    n=y0.shape[1]
    if y0.shape[0]!=k:
        raise ValueError("input and weight matrix dimensions do not match")
    out=torch.empty((3,m,n),device=x0.device,dtype=x0.dtype)
    if m and n and k:
        xs=tuple(x.contiguous() for x in xs)
        with torch.cuda.device(x0.device):
            single_column = bits == 16 and n == 1
            narrow_classifier = bits in (8, 16) and n == 4 and k == 512
            decode_bn = 1 if single_column else 4 if narrow_classifier else 32
            decode_bk = (min(1024, max(32, triton.next_power_of_2(k))) if single_column
                         else 128 if narrow_classifier else 32)
            # Four-column classifiers reuse input loads across all three local
            # products and avoid padding each tensor-core output tile to 32 columns.
            if (m==1 or single_column or narrow_classifier) and (not prepared or all(y.original is not None for y in ys)):
                raw_ys=tuple(y.original if prepared else y.contiguous() for y in ys)
                _local_decode[(m,triton.cdiv(n,decode_bn))](*xs,*raw_ys,out,m,n,k,bits,decode_bn,decode_bk,num_warps=4)
            elif (m==1 or single_column or narrow_classifier) and prepared:
                _local_decode_packed[(m,triton.cdiv(n,decode_bn))](
                    *xs,*(y.limbs for y in ys),out,m,n,k,bits,decode_bn,decode_bk,num_warps=4)
            else:
                ys=tuple(y.limbs if prepared else y.contiguous() for y in ys)
                _local_products[(triton.cdiv(m,16),triton.cdiv(n,32),3)](
                    *xs,*ys,out,m,n,k,bits,16,32,64,prepared,num_warps=4)
    elif k==0:
        out.zero_()
    return out.unbind(0)


@triton.jit
def _load_packed_word(P, offsets, mask, SIZE:tl.constexpr, BITS:tl.constexpr):
    word=tl.full(offsets.shape,0,tl.uint64)
    for limb in tl.static_range(BITS//8):
        digit=tl.load(P+limb*SIZE+offsets,mask,0).to(tl.int64).to(tl.uint64)
        word+=digit<<(8*limb)
    return word


@triton.jit
def _local_decode_packed(X0,X1,X2,Y0,Y1,Y2,C,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,
                         BITS:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
    row=tl.program_id(0)
    cols=tl.program_id(1)*BN+tl.arange(0,BN)
    kk=tl.arange(0,BK)
    DT:tl.constexpr=tl.uint32 if BITS<=32 else tl.uint64
    s0=tl.full((BK,BN),0,DT)
    s1=tl.full((BK,BN),0,DT)
    s4=tl.full((BK,BN),0,DT)
    for start in range(tl.cdiv(K,BK)):
        ki=start*BK+kk
        xp=row*K+ki
        mask=ki<K
        x0=tl.load(X0+xp,mask,0).to(DT)[:,None]
        x1=tl.load(X1+xp,mask,0).to(DT)[:,None]
        x2=tl.load(X2+xp,mask,0).to(DT)[:,None]
        yp=ki[:,None]*N+cols[None,:]
        mask=(ki[:,None]<K)&(cols[None,:]<N)
        y0=_load_packed_word(Y0,yp,mask,K*N,BITS).to(DT)
        y1=_load_packed_word(Y1,yp,mask,K*N,BITS).to(DT)
        y2=_load_packed_word(Y2,yp,mask,K*N,BITS).to(DT)
        s0+=(x0+x1)*y0+x0*y1
        s1+=(x1+x2)*y1+x1*y2
        s4+=x0*y2+x2*y0
    v0=tl.sum(s0,0).to(tl.uint64)
    v1=tl.sum(s1,0).to(tl.uint64)
    v4=tl.sum(s4,0).to(tl.uint64)
    if BITS<64:
        v0&=(1 << BITS)-1
        v1&=(1 << BITS)-1
        v4&=(1 << BITS)-1
    cp=row*N+cols
    tl.store(C+cp,v0.to(tl.int64),cols<N)
    tl.store(C+M*N+cp,v1.to(tl.int64),cols<N)
    tl.store(C+2*M*N+cp,v4.to(tl.int64),cols<N)
