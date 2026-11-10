"""Exact q256 tensor products from centered bytes and integer corrections."""
import torch
import triton
import triton.language as tl

from .field_q256 import _finish
from .field_q256_tensor import _cross_packed


@triton.jit
def _pack_a(SOURCE,DIGITS,SUMS,K:tl.constexpr,BLOCK:tl.constexpr):
    row=tl.program_id(0).to(tl.int64)
    digit=tl.program_id(1)
    ki=tl.arange(0,BLOCK)
    word=tl.load(SOURCE+(row*K+ki)*16+digit//2,ki<K,0).to(tl.int32)
    value=((word>>(8*(digit%2)))&255)-128
    value=tl.where(ki<K,value,0)
    tl.store(DIGITS+(row*32+digit)*K+ki,value.to(tl.int8),ki<K)
    tl.store(SUMS+row*32+digit,tl.sum(value,0))


@triton.jit
def _pack_b(SOURCE,DIGITS,SUMS,N:tl.constexpr,K:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
    digit=tl.program_id(0)
    cols=tl.program_id(1).to(tl.int64)*BN+tl.arange(0,BN)
    kk=tl.arange(0,BK)
    total=tl.full((BN,),0,tl.int32)
    for block in range(tl.cdiv(K,BK)):
        ki=block*BK+kk
        mask=(ki[:,None]<K)&(cols[None,:]<N)
        word=tl.load(SOURCE+(ki[:,None].to(tl.int64)*N+cols[None,:])*16+digit//2,mask,0).to(tl.int32)
        value=((word>>(8*(digit%2)))&255)-128
        value=tl.where(mask,value,0)
        total+=tl.sum(value,0)
        tl.store(DIGITS+ki[:,None].to(tl.int64)*(32*N)+digit*N+cols[None,:],value.to(tl.int8),mask)
    tl.store(SUMS+digit*N+cols,total,cols<N)


@triton.jit
def _fold_bytes(C,ASUM,BSUM,COEFFICIENTS,N:tl.constexpr,K:tl.constexpr,BN:tl.constexpr):
    row=tl.program_id(0).to(tl.int64)
    cols=tl.program_id(1).to(tl.int64)*BN+tl.arange(0,BN)
    limb=tl.program_id(2)
    acc=tl.full((BN,),0,tl.int64)
    for piece in tl.static_range(4):
        diagonal=2*limb+(piece%2)+(piece//2)*32
        diagonal_sum=tl.full((BN,),0,tl.int64)
        for i in range(32):
            j=diagonal-i
            if (j>=0)&(j<32):
                signed=tl.load(C+(row*32+i)*(32*N)+j*N+cols,cols<N,0).to(tl.int64)
                asum=tl.load(ASUM+row*32+i).to(tl.int64)
                bsum=tl.load(BSUM+j*N+cols,cols<N,0).to(tl.int64)
                # Both operands were centered by128 before the signed MMA.
                unsigned=signed+128*(asum+bsum)+16384*K
                diagonal_sum+=unsigned
        scale=189 if piece>=2 else 1
        acc+=(diagonal_sum<<(8*(piece%2)))*scale
    offset=(row*N+cols)*32+limb
    tl.store(COEFFICIENTS+offset,acc,cols<N)
    tl.store(COEFFICIENTS+offset+16,0,cols<N)


class PreparedQ256Bytes:
    """Owned centered-byte weight snapshot and exact per-column sums."""
    def __init__(self,digits,sums,k,n):
        self.digits=digits
        self.sums=sums
        self.k=k
        self.n=n


def _validate(source):
    if (not isinstance(source,torch.Tensor) or source.dtype not in (torch.int32,torch.uint16)
            or not source.is_cuda or source.ndim!=3 or source.shape[-1]!=16):
        raise ValueError('q256 sources require CUDA int32/uint16 radix65536 matrices')


def prepare_q256_bytes(b):
    """Pack canonical field weights; original tensor storage is not retained."""
    _validate(b)
    k,n,_=b.shape
    if k>4096:raise ValueError('q256 byte tensor products require K<=4096')
    digits=torch.empty((k,32*n),device=b.device,dtype=torch.int8)
    sums=torch.empty((32*n,),device=b.device,dtype=torch.int32)
    if n:
        with torch.cuda.device(b.device):
            _pack_b[(32,triton.cdiv(n,32))](b.contiguous(),digits,sums,n,k,32,64,num_warps=4)
    return PreparedQ256Bytes(digits,sums,k,n)


def field_matmul_q256_bytes(a,b):
    """Multiply canonical q256 limb matrices with exact centered-byte MMA.

    Shapes and canonical limb contract match ``field_matmul_q256``. B may be an
    owned prepared snapshot. Packing A and raw B remains part of this call.
    Each signed dot has absolute value<=K*16384; its exact integer correction
    reconstructs unsigned-byte dots before carry folding. K<=4096 bounds both
    the int32 MMA and the int64 convolution arithmetic.
    """
    _validate(a)
    m,k,_=a.shape
    prepared=b if isinstance(b,PreparedQ256Bytes) else prepare_q256_bytes(b)
    n=prepared.n
    if k!=prepared.k or k>4096 or prepared.digits.device!=a.device:
        raise ValueError('q256 weight dimensions or device mismatch')
    if not k:return torch.zeros((m,n,16),device=a.device,dtype=torch.int32)
    adigits=torch.empty((m*32,k),device=a.device,dtype=torch.int8)
    asum=torch.empty((m*32,),device=a.device,dtype=torch.int32)
    cross=torch.empty((m*32,n*32),device=a.device,dtype=torch.int32)
    coefficients=torch.empty((m,n,32),device=a.device,dtype=torch.int64)
    out=torch.empty((m,n,16),device=a.device,dtype=torch.int32)
    if m and n:
        with torch.cuda.device(a.device):
            _pack_a[(m,32)](a.contiguous(),adigits,asum,k,triton.next_power_of_2(k),num_warps=4)
            _cross_packed[(triton.cdiv(m*32,32),triton.cdiv(n*32,128))](adigits,prepared.digits,cross,m*32,n*32,k,32,128,64,num_warps=4,num_stages=1)
            _fold_bytes[(m,triton.cdiv(n,32),16)](cross,asum,prepared.sums,coefficients,n,k,32,num_warps=4)
            _finish[(triton.cdiv(m*n,16),)](coefficients,out,m*n,16,num_warps=4)
    return out
