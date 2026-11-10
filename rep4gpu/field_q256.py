"""Exact CUDA prototype for q=2**256-189 with little-endian radix-2**16 limbs.

Inputs and outputs are contiguous int32 tensors shaped [M,K,16], [K,N,16],
and [M,N,16]. Every limb is in [0,65536); inputs represent canonical field
values. The scalar convolution supports K<=4096 and does not use floating point.
"""
import torch
import triton
import triton.language as tl

PRIME=2**256-189


@triton.jit
def _convolution(A,B,C,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,BN:tl.constexpr):
    row=tl.program_id(0).to(tl.int64)
    cols=tl.program_id(1).to(tl.int64)*BN+tl.arange(0,BN)
    diagonal=tl.program_id(2)
    acc=tl.full((BN,),0,tl.int64)
    for ki in range(K):
        for i in range(16):
            j=diagonal-i
            if (j>=0)&(j<16):
                a=tl.load(A+(row*K+ki)*16+i).to(tl.int64)
                b=tl.load(B+(ki*N+cols)*16+j,cols<N,0).to(tl.int64)
                acc+=a*b
    tl.store(C+(row*N+cols)*32+diagonal,acc,cols<N)


@triton.jit
def _finish(C,OUT,NOUT:tl.constexpr,BN:tl.constexpr):
    outputs=tl.program_id(0).to(tl.int64)*BN+tl.arange(0,BN)
    limbs=tl.arange(0,16)
    low=tl.load(C+outputs[None,:]*32+limbs[:,None],outputs[None,:]<NOUT,0)
    high=tl.load(C+outputs[None,:]*32+limbs[:,None]+16,(outputs[None,:]<NOUT)&(limbs[:,None]<15),0)
    value=low+189*high
    # Each group of16 rounds propagates a carry around the complete radix ring.
    # K<=4096 bounds coefficients below2**56. Four groups cover its initial
    # high part, one possible overflow, and the final carry propagation.
    for iteration in range(64):
        carry=value>>16
        previous=tl.gather(carry,tl.broadcast_to(((limbs+15)%16)[:,None],(16,BN)),axis=0)
        value=(value&65535)+tl.where(limbs[:,None]==0,previous*189,previous)
    near_modulus=tl.sum(tl.where(limbs[:,None]==0,value>=65347,value==65535).to(tl.int32),axis=0)==16
    value=tl.where(near_modulus[None,:],tl.where(limbs[:,None]==0,value-65347,0),value)
    tl.store(OUT+outputs[None,:]*16+limbs[:,None],value,outputs[None,:]<NOUT)


def field_matmul_q256(a,b):
    """Multiply canonical q256 matrices; return exact int32 radix-65536 limbs.

    This experimental SIMT implementation allocates32 int64 coefficients per
    output before carrying. The caller establishes canonical input ranges.
    """
    if (not isinstance(a,torch.Tensor) or not isinstance(b,torch.Tensor) or a.dtype!=torch.int32
            or b.dtype!=torch.int32 or not a.is_cuda or a.device!=b.device
            or a.ndim!=3 or b.ndim!=3 or a.shape[-1]!=16 or b.shape[-1]!=16):
        raise ValueError('q256 CUDA inputs must be int32 matrices with16 little-endian limbs')
    m,k,_=a.shape
    if b.shape[0]!=k or k>4096:
        raise ValueError('q256 inner dimensions must match and K<=4096')
    n=b.shape[1]
    a,b=a.contiguous(),b.contiguous()
    if not k:
        return torch.zeros((m,n,16),device=a.device,dtype=torch.int32)
    coefficients=torch.empty((m,n,32),device=a.device,dtype=torch.int64)
    out=torch.empty((m,n,16),device=a.device,dtype=torch.int32)
    if m and n:
        with torch.cuda.device(a.device):
            if k:
                _convolution[(m,triton.cdiv(n,16),31)](a,b,coefficients,m,n,k,16,num_warps=4)
            _finish[(triton.cdiv(m*n,16),)](coefficients,out,m*n,16,num_warps=4)
    return out
