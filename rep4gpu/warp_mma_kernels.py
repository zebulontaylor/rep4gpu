"""Exact byte-limb tensor products with directly loaded warp MMA fragments.

Fragment coordinates follow NVIDIA's PTX m16n8k32 integer mapping:
https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-fragment-mma-16832
"""
from dataclasses import dataclass

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from .kernels import PreparedWeight, prepare_weight


@dataclass(frozen=True)
class MMAWeight:
    words: torch.Tensor
    k: int
    n: int
    bits: int

    @property
    def shape(self):
        return (self.k,self.n)

    @property
    def device(self):
        return self.words.device


def prepare_mma_weight(b, bits=64):
    """Transpose and pad balanced weight bytes for direct warp fragment loads."""
    p=b if isinstance(b,PreparedWeight) else prepare_weight(b,bits)
    if p.bits!=bits:
        raise ValueError("weight ring does not match bits")
    k,n=p.shape
    kp=triton.cdiv(k,32)*32
    transposed=p.limbs.transpose(1,2).contiguous()
    if kp!=k:
        transposed=F.pad(transposed,(0,kp-k))
    return MMAWeight(transposed.view(torch.int32),k,n,bits)


@triton.jit
def _pack_input_words(X0,X1,X2,P,M:tl.constexpr,K:tl.constexpr,KW:tl.constexpr,
                      LIMBS:tl.constexpr,BLOCK:tl.constexpr):
    idx=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    row=idx//KW
    kw=idx%KW
    limb=tl.program_id(1)
    for channel in tl.static_range(5):
        packed=tl.full((BLOCK,),0,tl.uint32)
        for byte in tl.static_range(4):
            ki=4*kw+byte
            mask=(row<M)&(ki<K)
            if channel==0 or channel==3:
                x=tl.load(X0+row*K+ki,mask,0).to(tl.uint64)
            elif channel==1 or channel==4:
                x=tl.load(X1+row*K+ki,mask,0).to(tl.uint64)
            else:
                x=tl.load(X2+row*K+ki,mask,0).to(tl.uint64)
            if channel==3:
                x+=tl.load(X1+row*K+ki,mask,0).to(tl.uint64)
            elif channel==4:
                x+=tl.load(X2+row*K+ki,mask,0).to(tl.uint64)
            x+=0x8080808080808080
            digit=(((x>>(8*limb))&255).to(tl.int32)-128).to(tl.int8).to(tl.uint8).to(tl.uint32)
            packed|=digit<<(8*byte)
        tl.store(P+(channel*LIMBS+limb)*M*KW+idx,packed,row<M)


@triton.jit
def _mma(a0,a1,a2,a3,b0,b1,c0,c1,c2,c3):
    return tl.inline_asm_elementwise(
        "mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 "
        "{$0,$1,$2,$3}, {$4,$5,$6,$7}, {$8,$9}, {$10,$11,$12,$13};",
        constraints="=r,=r,=r,=r,r,r,r,r,r,r,r,r,r,r",
        args=[a0,a1,a2,a3,b0,b1,c0,c1,c2,c3],
        dtype=(tl.int32,tl.int32,tl.int32,tl.int32),is_pure=True,pack=1)


@triton.jit
def _warp_local(A,Y0,Y1,Y2,C,M:tl.constexpr,N:tl.constexpr,KW:tl.constexpr,
                BITS:tl.constexpr,WARPS:tl.constexpr):
    tid=tl.arange(0,WARPS*32)
    lane=tid%32
    warp=tid//32
    group=lane//4
    quad=lane%4
    row=tl.program_id(0)*16+group
    bcol=tl.program_id(1)*(WARPS*8)+warp*8+group
    ccol=tl.program_id(1)*(WARPS*8)+warp*8+quad*2
    output=tl.program_id(2)
    c0_0=tl.full((WARPS*32,),0,tl.int32)
    c0_1=tl.full((WARPS*32,),0,tl.int32)
    c0_2=tl.full((WARPS*32,),0,tl.int32)
    c0_3=tl.full((WARPS*32,),0,tl.int32)
    c1_0=tl.full((WARPS*32,),0,tl.int32)
    c1_1=tl.full((WARPS*32,),0,tl.int32)
    c1_2=tl.full((WARPS*32,),0,tl.int32)
    c1_3=tl.full((WARPS*32,),0,tl.int32)
    c2_0=tl.full((WARPS*32,),0,tl.int32)
    c2_1=tl.full((WARPS*32,),0,tl.int32)
    c2_2=tl.full((WARPS*32,),0,tl.int32)
    c2_3=tl.full((WARPS*32,),0,tl.int32)
    c3_0=tl.full((WARPS*32,),0,tl.int32)
    c3_1=tl.full((WARPS*32,),0,tl.int32)
    c3_2=tl.full((WARPS*32,),0,tl.int32)
    c3_3=tl.full((WARPS*32,),0,tl.int32)
    c4_0=tl.full((WARPS*32,),0,tl.int32)
    c4_1=tl.full((WARPS*32,),0,tl.int32)
    c4_2=tl.full((WARPS*32,),0,tl.int32)
    c4_3=tl.full((WARPS*32,),0,tl.int32)
    c5_0=tl.full((WARPS*32,),0,tl.int32)
    c5_1=tl.full((WARPS*32,),0,tl.int32)
    c5_2=tl.full((WARPS*32,),0,tl.int32)
    c5_3=tl.full((WARPS*32,),0,tl.int32)
    c6_0=tl.full((WARPS*32,),0,tl.int32)
    c6_1=tl.full((WARPS*32,),0,tl.int32)
    c6_2=tl.full((WARPS*32,),0,tl.int32)
    c6_3=tl.full((WARPS*32,),0,tl.int32)
    c7_0=tl.full((WARPS*32,),0,tl.int32)
    c7_1=tl.full((WARPS*32,),0,tl.int32)
    c7_2=tl.full((WARPS*32,),0,tl.int32)
    c7_3=tl.full((WARPS*32,),0,tl.int32)
    for term in tl.static_range(2):
        if output==0:
            if term==0:
                ac=3
                B=Y0
            else:
                ac=0
                B=Y1
        elif output==1:
            if term==0:
                ac=4
                B=Y1
            else:
                ac=1
                B=Y2
        else:
            if term==0:
                ac=0
                B=Y2
            else:
                ac=2
                B=Y0
        for start in range(KW//8):
            ki=start*8+quad
            if BITS>0:
                for i in tl.static_range(1):
                    ap=(ac*(BITS//8)+i)*M*KW
                    bp=(0-i)*N*KW
                    a0=tl.load(A+ap+row*KW+ki,row<M,0)
                    a1=tl.load(A+ap+(row+8)*KW+ki,row+8<M,0)
                    a2=tl.load(A+ap+row*KW+ki+4,row<M,0)
                    a3=tl.load(A+ap+(row+8)*KW+ki+4,row+8<M,0)
                    b0=tl.load(B+bp+bcol*KW+ki,bcol<N,0)
                    b1=tl.load(B+bp+bcol*KW+ki+4,bcol<N,0)
                    c0_0,c0_1,c0_2,c0_3=_mma(a0,a1,a2,a3,b0,b1,c0_0,c0_1,c0_2,c0_3)
            if BITS>8:
                for i in tl.static_range(2):
                    ap=(ac*(BITS//8)+i)*M*KW
                    bp=(1-i)*N*KW
                    a0=tl.load(A+ap+row*KW+ki,row<M,0)
                    a1=tl.load(A+ap+(row+8)*KW+ki,row+8<M,0)
                    a2=tl.load(A+ap+row*KW+ki+4,row<M,0)
                    a3=tl.load(A+ap+(row+8)*KW+ki+4,row+8<M,0)
                    b0=tl.load(B+bp+bcol*KW+ki,bcol<N,0)
                    b1=tl.load(B+bp+bcol*KW+ki+4,bcol<N,0)
                    c1_0,c1_1,c1_2,c1_3=_mma(a0,a1,a2,a3,b0,b1,c1_0,c1_1,c1_2,c1_3)
            if BITS>16:
                for i in tl.static_range(3):
                    ap=(ac*(BITS//8)+i)*M*KW
                    bp=(2-i)*N*KW
                    a0=tl.load(A+ap+row*KW+ki,row<M,0)
                    a1=tl.load(A+ap+(row+8)*KW+ki,row+8<M,0)
                    a2=tl.load(A+ap+row*KW+ki+4,row<M,0)
                    a3=tl.load(A+ap+(row+8)*KW+ki+4,row+8<M,0)
                    b0=tl.load(B+bp+bcol*KW+ki,bcol<N,0)
                    b1=tl.load(B+bp+bcol*KW+ki+4,bcol<N,0)
                    c2_0,c2_1,c2_2,c2_3=_mma(a0,a1,a2,a3,b0,b1,c2_0,c2_1,c2_2,c2_3)
            if BITS>24:
                for i in tl.static_range(4):
                    ap=(ac*(BITS//8)+i)*M*KW
                    bp=(3-i)*N*KW
                    a0=tl.load(A+ap+row*KW+ki,row<M,0)
                    a1=tl.load(A+ap+(row+8)*KW+ki,row+8<M,0)
                    a2=tl.load(A+ap+row*KW+ki+4,row<M,0)
                    a3=tl.load(A+ap+(row+8)*KW+ki+4,row+8<M,0)
                    b0=tl.load(B+bp+bcol*KW+ki,bcol<N,0)
                    b1=tl.load(B+bp+bcol*KW+ki+4,bcol<N,0)
                    c3_0,c3_1,c3_2,c3_3=_mma(a0,a1,a2,a3,b0,b1,c3_0,c3_1,c3_2,c3_3)
            if BITS>32:
                for i in tl.static_range(5):
                    ap=(ac*(BITS//8)+i)*M*KW
                    bp=(4-i)*N*KW
                    a0=tl.load(A+ap+row*KW+ki,row<M,0)
                    a1=tl.load(A+ap+(row+8)*KW+ki,row+8<M,0)
                    a2=tl.load(A+ap+row*KW+ki+4,row<M,0)
                    a3=tl.load(A+ap+(row+8)*KW+ki+4,row+8<M,0)
                    b0=tl.load(B+bp+bcol*KW+ki,bcol<N,0)
                    b1=tl.load(B+bp+bcol*KW+ki+4,bcol<N,0)
                    c4_0,c4_1,c4_2,c4_3=_mma(a0,a1,a2,a3,b0,b1,c4_0,c4_1,c4_2,c4_3)
            if BITS>40:
                for i in tl.static_range(6):
                    ap=(ac*(BITS//8)+i)*M*KW
                    bp=(5-i)*N*KW
                    a0=tl.load(A+ap+row*KW+ki,row<M,0)
                    a1=tl.load(A+ap+(row+8)*KW+ki,row+8<M,0)
                    a2=tl.load(A+ap+row*KW+ki+4,row<M,0)
                    a3=tl.load(A+ap+(row+8)*KW+ki+4,row+8<M,0)
                    b0=tl.load(B+bp+bcol*KW+ki,bcol<N,0)
                    b1=tl.load(B+bp+bcol*KW+ki+4,bcol<N,0)
                    c5_0,c5_1,c5_2,c5_3=_mma(a0,a1,a2,a3,b0,b1,c5_0,c5_1,c5_2,c5_3)
            if BITS>48:
                for i in tl.static_range(7):
                    ap=(ac*(BITS//8)+i)*M*KW
                    bp=(6-i)*N*KW
                    a0=tl.load(A+ap+row*KW+ki,row<M,0)
                    a1=tl.load(A+ap+(row+8)*KW+ki,row+8<M,0)
                    a2=tl.load(A+ap+row*KW+ki+4,row<M,0)
                    a3=tl.load(A+ap+(row+8)*KW+ki+4,row+8<M,0)
                    b0=tl.load(B+bp+bcol*KW+ki,bcol<N,0)
                    b1=tl.load(B+bp+bcol*KW+ki+4,bcol<N,0)
                    c6_0,c6_1,c6_2,c6_3=_mma(a0,a1,a2,a3,b0,b1,c6_0,c6_1,c6_2,c6_3)
            if BITS>56:
                for i in tl.static_range(8):
                    ap=(ac*(BITS//8)+i)*M*KW
                    bp=(7-i)*N*KW
                    a0=tl.load(A+ap+row*KW+ki,row<M,0)
                    a1=tl.load(A+ap+(row+8)*KW+ki,row+8<M,0)
                    a2=tl.load(A+ap+row*KW+ki+4,row<M,0)
                    a3=tl.load(A+ap+(row+8)*KW+ki+4,row+8<M,0)
                    b0=tl.load(B+bp+bcol*KW+ki,bcol<N,0)
                    b1=tl.load(B+bp+bcol*KW+ki+4,bcol<N,0)
                    c7_0,c7_1,c7_2,c7_3=_mma(a0,a1,a2,a3,b0,b1,c7_0,c7_1,c7_2,c7_3)
    v0=tl.full((WARPS*32,),0,tl.uint64)
    if BITS>0:
        v0+=c0_0.to(tl.int64).to(tl.uint64)<<0
    if BITS>8:
        v0+=c1_0.to(tl.int64).to(tl.uint64)<<8
    if BITS>16:
        v0+=c2_0.to(tl.int64).to(tl.uint64)<<16
    if BITS>24:
        v0+=c3_0.to(tl.int64).to(tl.uint64)<<24
    if BITS>32:
        v0+=c4_0.to(tl.int64).to(tl.uint64)<<32
    if BITS>40:
        v0+=c5_0.to(tl.int64).to(tl.uint64)<<40
    if BITS>48:
        v0+=c6_0.to(tl.int64).to(tl.uint64)<<48
    if BITS>56:
        v0+=c7_0.to(tl.int64).to(tl.uint64)<<56
    if BITS==32:
        v0&=0xffffffff
    tl.store(C+output*M*N+(row+0)*N+ccol+0,v0.to(tl.int64),(row+0<M)&(ccol+0<N))
    v1=tl.full((WARPS*32,),0,tl.uint64)
    if BITS>0:
        v1+=c0_1.to(tl.int64).to(tl.uint64)<<0
    if BITS>8:
        v1+=c1_1.to(tl.int64).to(tl.uint64)<<8
    if BITS>16:
        v1+=c2_1.to(tl.int64).to(tl.uint64)<<16
    if BITS>24:
        v1+=c3_1.to(tl.int64).to(tl.uint64)<<24
    if BITS>32:
        v1+=c4_1.to(tl.int64).to(tl.uint64)<<32
    if BITS>40:
        v1+=c5_1.to(tl.int64).to(tl.uint64)<<40
    if BITS>48:
        v1+=c6_1.to(tl.int64).to(tl.uint64)<<48
    if BITS>56:
        v1+=c7_1.to(tl.int64).to(tl.uint64)<<56
    if BITS==32:
        v1&=0xffffffff
    tl.store(C+output*M*N+(row+0)*N+ccol+1,v1.to(tl.int64),(row+0<M)&(ccol+1<N))
    v2=tl.full((WARPS*32,),0,tl.uint64)
    if BITS>0:
        v2+=c0_2.to(tl.int64).to(tl.uint64)<<0
    if BITS>8:
        v2+=c1_2.to(tl.int64).to(tl.uint64)<<8
    if BITS>16:
        v2+=c2_2.to(tl.int64).to(tl.uint64)<<16
    if BITS>24:
        v2+=c3_2.to(tl.int64).to(tl.uint64)<<24
    if BITS>32:
        v2+=c4_2.to(tl.int64).to(tl.uint64)<<32
    if BITS>40:
        v2+=c5_2.to(tl.int64).to(tl.uint64)<<40
    if BITS>48:
        v2+=c6_2.to(tl.int64).to(tl.uint64)<<48
    if BITS>56:
        v2+=c7_2.to(tl.int64).to(tl.uint64)<<56
    if BITS==32:
        v2&=0xffffffff
    tl.store(C+output*M*N+(row+8)*N+ccol+0,v2.to(tl.int64),(row+8<M)&(ccol+0<N))
    v3=tl.full((WARPS*32,),0,tl.uint64)
    if BITS>0:
        v3+=c0_3.to(tl.int64).to(tl.uint64)<<0
    if BITS>8:
        v3+=c1_3.to(tl.int64).to(tl.uint64)<<8
    if BITS>16:
        v3+=c2_3.to(tl.int64).to(tl.uint64)<<16
    if BITS>24:
        v3+=c3_3.to(tl.int64).to(tl.uint64)<<24
    if BITS>32:
        v3+=c4_3.to(tl.int64).to(tl.uint64)<<32
    if BITS>40:
        v3+=c5_3.to(tl.int64).to(tl.uint64)<<40
    if BITS>48:
        v3+=c6_3.to(tl.int64).to(tl.uint64)<<48
    if BITS>56:
        v3+=c7_3.to(tl.int64).to(tl.uint64)<<56
    if BITS==32:
        v3&=0xffffffff
    tl.store(C+output*M*N+(row+8)*N+ccol+1,v3.to(tl.int64),(row+8<M)&(ccol+1<N))


def local_products_mma(x0,x1,x2,y0,y1,y2,bits=64,*,warps=4):
    """Compute exact local products with packed warp fragments and no CTA barriers.

    Static weights must be packed with ``prepare_mma_weight``. Input packing is
    part of this call. K <= 8191 bounds all full-K signed int32 accumulators.
    """
    xs=(x0,x1,x2)
    ys=(y0,y1,y2)
    if bits not in (32,64) or warps not in (4,8):
        raise ValueError("expected 32/64 bits and 4/8 warps")
    if any(not isinstance(y,MMAWeight) or y.bits!=bits for y in ys):
        raise ValueError("weights must be prepared in the requested ring")
    if any(x.ndim!=2 or x.dtype!=torch.int64 or not x.is_cuda or x.shape!=x0.shape or x.device!=x0.device for x in xs):
        raise ValueError("input shares must be matching int64 CUDA matrices")
    if any(y.shape!=y0.shape or y.device!=x0.device for y in ys):
        raise ValueError("weight share shapes and devices must match")
    m,k=x0.shape
    n=y0.n
    if y0.k!=k or k>8191:
        raise ValueError("matrix dimensions must match and K <=8191")
    kw=triton.cdiv(k,32)*8
    packed=torch.empty((5,bits//8,m,kw),device=x0.device,dtype=torch.int32)
    out=torch.empty((3,m,n),device=x0.device,dtype=torch.int64)
    xs=tuple(x.contiguous() for x in xs)
    if m and n and k:
        with torch.cuda.device(x0.device):
            _pack_input_words[(triton.cdiv(m*kw,128),bits//8)](*xs,packed,m,k,kw,bits//8,128)
            _warp_local[(triton.cdiv(m,16),triton.cdiv(n,warps*8),3)](
                packed,*(y.words for y in ys),out,m,n,kw,bits,warps,num_warps=warps,num_stages=1)
    elif k==0:
        out.zero_()
    return out.unbind(0)
