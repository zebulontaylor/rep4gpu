"""Experimental exact ring32 Strassen products with static weight sums."""
from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from .kernels import CompactWeight, _tensor_mm


@triton.jit
def _split_inputs(A, OUT, M:tl.constexpr, K:tl.constexpr, BLOCK:tl.constexpr):
    index=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    product=tl.program_id(1)
    parent=tl.program_id(2)
    A+=parent*M*K
    size:tl.constexpr=(M//2)*(K//2)
    r,c=index//(K//2),index%(K//2)
    base=r*K+c
    if product==0:
        left,right,sign=base,base+(M//2)*K+K//2,1
    elif product==1:
        left,right,sign=base+(M//2)*K,base+(M//2)*K+K//2,1
    elif product==2:
        left,right,sign=base,base,0
    elif product==3:
        left,right,sign=base+(M//2)*K+K//2,base,0
    elif product==4:
        left,right,sign=base,base+K//2,1
    elif product==5:
        left,right,sign=base+(M//2)*K,base,-1
    else:
        left,right,sign=base+K//2,base+(M//2)*K+K//2,-1
    value=tl.load(A+left,index<size,0).to(tl.uint32)
    other=tl.load(A+right,(index<size)&(sign!=0),0).to(tl.uint32)
    value+=other*sign.to(tl.uint32)
    tl.store(OUT+(parent*7+product)*size+index,value.to(tl.int64),index<size)


@triton.jit
def _join_outputs(P,OUT,M:tl.constexpr,N:tl.constexpr,BLOCK:tl.constexpr):
    index=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    size:tl.constexpr=(M//2)*(N//2)
    parent=tl.program_id(1)
    P+=parent*7*size
    OUT+=parent*M*N
    p0=tl.load(P+index,index<size,0).to(tl.uint32)
    p1=tl.load(P+1*size+index,index<size,0).to(tl.uint32)
    p2=tl.load(P+2*size+index,index<size,0).to(tl.uint32)
    p3=tl.load(P+3*size+index,index<size,0).to(tl.uint32)
    p4=tl.load(P+4*size+index,index<size,0).to(tl.uint32)
    p5=tl.load(P+5*size+index,index<size,0).to(tl.uint32)
    p6=tl.load(P+6*size+index,index<size,0).to(tl.uint32)
    base=(index//(N//2))*N+index%(N//2)
    tl.store(OUT+base,(p0+p3-p4+p6).to(tl.int64),index<size)
    tl.store(OUT+base+N//2,(p2+p4).to(tl.int64),index<size)
    tl.store(OUT+base+(M//2)*N,(p1+p3).to(tl.int64),index<size)
    tl.store(OUT+base+(M//2)*N+N//2,(p0-p1+p2+p5).to(tl.int64),index<size)


@triton.jit
def _batched_leaves(A,B,C,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,BM:tl.constexpr):
    batch=tl.program_id(2)
    _tensor_mm(A+batch*M*K,B+batch*K*N,C+batch*M*N,M,N,K,32,BM,128,64,False)


@dataclass(frozen=True)
class StrassenWeight:
    """Contiguous int32 CUDA leaf matrices for a static ring32 right operand.

    ``shape`` is the original (K,N); ``words`` has shape
    (7**levels,K//2**levels,N//2**levels). Rebuild after changing the weights.
    One or two levels require both original dimensions to be divisible by 2**levels.
    """
    words:torch.Tensor
    shape:tuple
    levels:int

    def __post_init__(self):
        if self.levels not in (1,2) or len(self.shape)!=2 or any(
                not isinstance(v,int) or v<0 or v%(1<<self.levels) for v in self.shape):
            raise ValueError('Strassen dimensions must be divisible by 2**levels for levels 1 or 2')
        expected=(7**self.levels,*(v//(1<<self.levels) for v in self.shape))
        if self.words.dtype!=torch.int32 or not self.words.is_cuda or not self.words.is_contiguous() or tuple(self.words.shape)!=expected:
            raise ValueError('Strassen leaf weights must be matching contiguous int32 CUDA matrices')

    @property
    def bits(self):
        return 32

    @property
    def device(self):
        return self.words.device

    @property
    def storage_bytes(self):
        return self.words.untyped_storage().nbytes()


def prepare_strassen_weight(weight,levels=1):
    """Prepare static combinations; retained bytes grow by (7/4)**levels."""
    if levels not in (1,2):
        raise ValueError('Strassen experiments support one or two levels')
    if not isinstance(weight,CompactWeight):
        raise TypeError('Strassen experiments require compact ring32 weights')
    k,n=weight.shape
    divisor=1<<levels
    if k%divisor or n%divisor:
        raise ValueError('weight dimensions must be divisible by the Strassen subdivision')
    def split(b,depth):
        if not depth:
            return [b]
        rows,cols=b.shape
        b00,b01=b[:rows//2,:cols//2],b[:rows//2,cols//2:]
        b10,b11=b[rows//2:,:cols//2],b[rows//2:,cols//2:]
        parts=(b00+b11,b00,b01-b11,b10-b00,b11,b00+b01,b10+b11)
        return [leaf for part in parts for leaf in split(part,depth-1)]
    return StrassenWeight(torch.stack(split(weight.words,levels)),(k,n),levels)


def ring_matmul_strassen(a,weight):
    """Run every dynamic sum, batched leaf product and reconstruction on CUDA.

    Strassen's identities use only addition and multiplication, so they hold
    modulo 2**32 for every input bit pattern, without division or range bounds.
    The input row count must be divisible by 2**weight.levels; other row counts,
    including single-token decode, raise ValueError rather than changing format.
    """
    if not isinstance(weight,StrassenWeight) or a.dtype!=torch.int64 or a.ndim!=2 or not a.is_cuda:
        raise ValueError('provide an int64 CUDA matrix and prepared Strassen weight')
    m,k=a.shape
    n=weight.shape[1]
    if a.device!=weight.device or k!=weight.shape[0] or m%(1<<weight.levels):
        raise ValueError('input dimensions/device must match Strassen preparation')
    if not m or not n or not k:
        out=torch.empty((m,n),device=a.device,dtype=torch.int64)
        return out.zero_() if not k else out
    a=a.contiguous()
    batch=1
    for _ in range(weight.levels):
        packed=torch.empty((batch*7,m//2,k//2),dtype=torch.int32,device=a.device)
        _split_inputs[(triton.cdiv((m//2)*(k//2),256),7,batch)](a,packed,m,k,256)
        a=packed
        m,k,n,batch=m//2,k//2,n//2,batch*7
    products=torch.empty((batch,m,n),device=a.device,dtype=torch.int32)
    bm=32 if m>=32 else 16
    _batched_leaves[(triton.cdiv(m,bm),triton.cdiv(n,128),batch)](
        a,weight.words,products,m,n,k,bm,num_warps=4,num_stages=1)
    for _ in range(weight.levels):
        m,n,batch=m*2,n*2,batch//7
        output=torch.empty((batch,m,n),device=a.device,dtype=torch.int64 if batch==1 else torch.int32)
        _join_outputs[(triton.cdiv((m//2)*(n//2),256),batch)](products,output,m,n,256)
        products=output
    return products[0]


@triton.jit
def _split_three(X0,X1,X2,OUT,M:tl.constexpr,K:tl.constexpr,PARENTS:tl.constexpr,BLOCK:tl.constexpr):
    index=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    product=tl.program_id(1)
    component=tl.program_id(2)//PARENTS
    parent=tl.program_id(2)%PARENTS
    if component==0:
        A=X0
    elif component==1:
        A=X1
    else:
        A=X2
    A+=parent*M*K
    OUT+=component*PARENTS*7*(M//2)*(K//2)
    size:tl.constexpr=(M//2)*(K//2)
    r,c=index//(K//2),index%(K//2)
    base=r*K+c
    if product==0:
        left,right,sign=base,base+(M//2)*K+K//2,1
    elif product==1:
        left,right,sign=base+(M//2)*K,base+(M//2)*K+K//2,1
    elif product==2:
        left,right,sign=base,base,0
    elif product==3:
        left,right,sign=base+(M//2)*K+K//2,base,0
    elif product==4:
        left,right,sign=base,base+K//2,1
    elif product==5:
        left,right,sign=base+(M//2)*K,base,-1
    else:
        left,right,sign=base+K//2,base+(M//2)*K+K//2,-1
    value=tl.load(A+left,index<size,0).to(tl.uint32)
    other=tl.load(A+right,(index<size)&(sign!=0),0).to(tl.uint32)
    value+=other*sign.to(tl.uint32)
    tl.store(OUT+(parent*7+product)*size+index,value.to(tl.int64),index<size)


@triton.jit
def _batched_local_leaves(X0, X1, X2, Y0, Y1, Y2, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
               BITS: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
               BK: tl.constexpr, BATCH:tl.constexpr, PREPARED: tl.constexpr = False, OPAQUE:tl.constexpr=False):
    component=tl.program_id(2)%3
    batch=tl.program_id(2)//3
    X0+=batch*M*K
    X1+=batch*M*K
    X2+=batch*M*K
    Y0+=batch*K*N
    Y1+=batch*K*N
    Y2+=batch*K*N
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    kk = tl.arange(0, BK)
    DT:tl.constexpr=tl.uint32 if OPAQUE else tl.uint64
    BALANCE:tl.constexpr=0x80808080 if OPAQUE else 0x8080808080808080
    out = tl.full((BM, BN), 0, DT)
    for term in tl.static_range(2):
        for start in range(tl.cdiv(K, BK)):
            if component == 0:
                A = X0
                A2 = X1
                if term == 0:
                    B = Y0
                else:
                    B = Y1
            elif component == 1:
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
                        (rows[:, None] < M) & (start * BK + kk[None, :] < K), 0).to(DT)
            if term == 0:
                if component < 2:
                    a += tl.load(A2 + rows[:,None]*K + start*BK+kk[None,:],
                                 (rows[:,None]<M)&(start*BK+kk[None,:]<K),0).to(DT)
            if not PREPARED:
                b = tl.load(B + (start * BK + kk[:, None]) * N + cols[None, :],
                            (start * BK + kk[:, None] < K) & (cols[None, :] < N), 0).to(DT)
                b += BALANCE
            # Adding this constant turns byte extraction into balanced signed digits.
            a += BALANCE
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
                if OPAQUE:
                    out = tl.inline_asm_elementwise(
                        "add.u32 $0, $1, $2;", constraints="=r,r,r",
                        args=[out,acc.to(tl.uint32) << (8 * diagonal)],
                        dtype=tl.uint32,is_pure=True,pack=1)
                else:
                    out += acc.to(tl.int64).to(DT) << (8 * diagonal)
    if BITS == 32:
        out = out & 0xffffffff
    tl.store(C + (component*BATCH+batch)*M*N + rows[:, None] * N + cols[None, :], out.to(tl.int64),
             (rows[:, None] < M) & (cols[None, :] < N))




def local_products_strassen(xs,weights,*,opaque=False):
    """Compute three exact local contributions using batched matrix Strassen.

    Input rows must be divisible by 2**levels. Weights must have identical original
    shapes, levels and devices, and match the input K dimension. Results use
    canonical int64 ring32 values. Opaque reconstruction uses bounded per-tile
    int32 MMA diagonals and a separate modular add, with a K tile of 32.
    """
    xs,weights=tuple(xs),tuple(weights)
    if len(xs)!=3 or len(weights)!=3 or any(not isinstance(w,StrassenWeight) for w in weights):
        raise ValueError('provide three input shares and three Strassen weights')
    if any(x.ndim!=2 or x.dtype!=torch.int64 or not x.is_cuda for x in xs):
        raise ValueError('input shares must be two-dimensional int64 CUDA matrices')
    m,k=xs[0].shape
    n=weights[0].shape[1]
    levels=weights[0].levels
    if any(x.dtype!=torch.int64 or x.shape!=xs[0].shape or x.device!=weights[0].device for x in xs):
        raise ValueError('input shares must be matching int64 matrices on the weight device')
    if any(w.shape!=weights[0].shape or w.device!=weights[0].device or w.levels!=levels for w in weights):
        raise ValueError('prepared weights must have matching dimensions/device/levels')
    if k!=weights[0].shape[0] or m%(1<<levels):
        raise ValueError('matrix dimensions must match the Strassen subdivision')
    if not m or not n or not k:
        out=torch.empty((3,m,n),device=xs[0].device,dtype=torch.int64)
        if not k:
            out.zero_()
        return tuple(out.unbind(0))
    batch=1
    inputs=tuple(x.contiguous() for x in xs)
    for _ in range(levels):
        packed=torch.empty((3,batch*7,m//2,k//2),dtype=torch.int32,device=xs[0].device)
        _split_three[(triton.cdiv((m//2)*(k//2),256),7,batch*3)](*inputs,packed,m,k,batch,256)
        inputs=tuple(packed.unbind(0))
        m,k,n,batch=m//2,k//2,n//2,batch*7
    products=torch.empty((3,batch,m,n),dtype=torch.int32,device=xs[0].device)
    bm=32 if m>=32 else 16
    bk=32 if opaque else 64
    _batched_local_leaves[(triton.cdiv(m,bm),triton.cdiv(n,128),batch*3)](
        *inputs,*(w.words for w in weights),products,m,n,k,32,bm,128,bk,batch,False,opaque,
        num_warps=4,num_stages=1)
    for _ in range(levels):
        m,n,batch=m*2,n*2,batch//7
        output=torch.empty((3,batch,m,n),device=xs[0].device,dtype=torch.int64 if batch==1 else torch.int32)
        _join_outputs[(triton.cdiv((m//2)*(n//2),256),batch*3)](products,output,m,n,256)
        products=output
    return tuple(products[:,0].unbind(0))
