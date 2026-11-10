"""Exact q256 decode through positive7-bit tensor-core limb products."""
import torch
import triton
import triton.language as tl

from .field_q256 import _finish


@triton.jit
def _cross(A,B,C,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,TM:tl.constexpr,TN:tl.constexpr,TK:tl.constexpr):
    rows=tl.program_id(0)*TM+tl.arange(0,TM)
    cols=tl.program_id(1)*TN+tl.arange(0,TN)
    original_m=rows//37
    adigit=rows%37
    original_n=cols%N
    bdigit=cols//N
    abit=adigit*7
    bbit=bdigit*7
    alimb=abit//16
    blimb=bbit//16
    ashift=abit%16
    bshift=bbit%16
    kk=tl.arange(0,TK)
    acc=tl.full((TM,TN),0,tl.int32)
    for block in range(tl.cdiv(K,TK)):
        ki=block*TK+kk
        ap=(original_m[:,None].to(tl.int64)*K+ki[None,:])*16+alimb[:,None]
        bp=(ki[:,None].to(tl.int64)*N+original_n[None,:])*16+blimb[None,:]
        am=(rows[:,None]<M*37)&(ki[None,:]<K)
        bm=(cols[None,:]<N*37)&(ki[:,None]<K)
        alo=tl.load(A+ap,am,0).to(tl.uint32)
        ahi=tl.load(A+ap+1,am&(alimb[:,None]<15),0).to(tl.uint32)
        blo=tl.load(B+bp,bm,0).to(tl.uint32)
        bhi=tl.load(B+bp+1,bm&(blimb[None,:]<15),0).to(tl.uint32)
        a=(((alo>>ashift[:,None])|(ahi<<(16-ashift[:,None])))&127).to(tl.int8)
        b=(((blo>>bshift[None,:])|(bhi<<(16-bshift[None,:])))&127).to(tl.int8)
        # Every dot is nonnegative and bounded by K*127**2 < INT32_MAX.
        acc=tl.dot(a,b,acc,out_dtype=tl.int32)
    tl.store(C+rows[:,None].to(tl.int64)*(37*N)+cols[None,:],acc,(rows[:,None]<37*M)&(cols[None,:]<37*N))


@triton.jit
def _fold(C,COEFFICIENTS,M:tl.constexpr,N:tl.constexpr,BN:tl.constexpr):
    row=tl.program_id(0).to(tl.int64)
    cols=tl.program_id(1).to(tl.int64)*BN+tl.arange(0,BN)
    limb=tl.program_id(2)
    acc=tl.full((BN,),0,tl.int64)
    for diagonal in range(73):
        bit=7*diagonal
        if (bit//16)%16==limb:
            diagonal_sum=tl.full((BN,),0,tl.int64)
            for i in range(37):
                j=diagonal-i
                if (j>=0)&(j<37):
                    value=tl.load(C+(row*37+i)*(37*N)+j*N+cols,cols<N,0).to(tl.int64)
                    diagonal_sum+=value
            scale=tl.where(bit>=256,189,1)
            acc+=(diagonal_sum<<(bit%16))*scale
    offset=(row*N+cols)*32+limb
    tl.store(COEFFICIENTS+offset,acc,cols<N)
    tl.store(COEFFICIENTS+offset+16,0,cols<N)


def field_matmul_q256_tensor_inline(a,b):
    """Multiply int32 radix65536 q256 matrices through exact int8 products.

    Input contract matches ``field_matmul_q256``. The temporary cross product
    uses1369 int32 words per output, so this prototype targets decode shapes.
    K<=4096 bounds every tensor-core accumulator and every int64 carry fold.
    """
    if (not isinstance(a,torch.Tensor) or not isinstance(b,torch.Tensor) or a.dtype not in (torch.int32,torch.uint16)
            or b.dtype not in (torch.int32,torch.uint16) or not a.is_cuda or a.device!=b.device
            or a.ndim!=3 or b.ndim!=3 or a.shape[-1]!=16 or b.shape[-1]!=16):
        raise ValueError('q256 inputs require int32/uint16 CUDA16-limb matrices')
    m,k,_=a.shape
    if b.shape[0]!=k or k>4096:
        raise ValueError('inner dimensions must match and K<=4096')
    n=b.shape[1]
    if not k:return torch.zeros((m,n,16),device=a.device,dtype=torch.int32)
    a,b=a.contiguous(),b.contiguous()
    cross=torch.empty((m*37,n*37),device=a.device,dtype=torch.int32)
    coefficients=torch.empty((m,n,32),device=a.device,dtype=torch.int64)
    out=torch.empty((m,n,16),device=a.device,dtype=torch.int32)
    if m and n:
        with torch.cuda.device(a.device):
            _cross[(triton.cdiv(m*37,32),triton.cdiv(n*37,64))](a,b,cross,m,n,k,32,64,64,num_warps=4,num_stages=1)
            _fold[(m,triton.cdiv(n,32),16)](cross,coefficients,m,n,32,num_warps=4)
            _finish[(triton.cdiv(m*n,16),)](coefficients,out,m*n,16,num_warps=4)
    return out


@triton.jit
def _pack_digits(SOURCE,OUT,COLS:tl.constexpr,BN:tl.constexpr):
    row=tl.program_id(0).to(tl.int64)
    cols=tl.program_id(1).to(tl.int64)*BN+tl.arange(0,BN)
    digit=tl.arange(0,64)
    bit=digit*7
    limb=bit//16
    shift=bit%16
    offset=(row*COLS+cols[None,:])*16+limb[:,None]
    mask=(digit[:,None]<37)&(cols[None,:]<COLS)
    low=tl.load(SOURCE+offset,mask,0).to(tl.uint32)
    high=tl.load(SOURCE+offset+1,mask&(limb[:,None]<15),0).to(tl.uint32)
    value=(((low>>shift[:,None])|(high<<(16-shift[:,None])))&127).to(tl.int8)
    tl.store(OUT+(row*37+digit[:,None])*COLS+cols[None,:],value,mask)


@triton.jit
def _cross_packed(A,B,C,MR:tl.constexpr,NC:tl.constexpr,K:tl.constexpr,TM:tl.constexpr,TN:tl.constexpr,TK:tl.constexpr):
    rows=tl.program_id(0)*TM+tl.arange(0,TM)
    cols=tl.program_id(1)*TN+tl.arange(0,TN)
    kk=tl.arange(0,TK)
    acc=tl.full((TM,TN),0,tl.int32)
    for block in range(tl.cdiv(K,TK)):
        ki=block*TK+kk
        a=tl.load(A+rows[:,None].to(tl.int64)*K+ki[None,:],(rows[:,None]<MR)&(ki[None,:]<K),0)
        b=tl.load(B+ki[:,None].to(tl.int64)*NC+cols[None,:],(ki[:,None]<K)&(cols[None,:]<NC),0)
        acc=tl.dot(a,b,acc,out_dtype=tl.int32)
    tl.store(C+rows[:,None].to(tl.int64)*NC+cols[None,:],acc,(rows[:,None]<MR)&(cols[None,:]<NC))


class PreparedQ256Weight:
    """Owned exact37-digit snapshot; contains no original weight tensor."""
    def __init__(self,digits,k,n):
        self.digits=digits
        self.k=k
        self.n=n


def _pack_matrix(source):
    rows,cols,_=source.shape
    out=torch.empty((rows,37,cols),device=source.device,dtype=torch.int8)
    if rows and cols:
        with torch.cuda.device(source.device):
            _pack_digits[(rows,triton.cdiv(cols,32))](source.contiguous(),out,cols,32,num_warps=4)
    return out


def prepare_q256_weight(b):
    """Snapshot canonical radix65536 weights into exact signed-int8 digits."""
    if (not isinstance(b,torch.Tensor) or b.dtype not in (torch.int32,torch.uint16)
            or not b.is_cuda or b.ndim!=3 or b.shape[-1]!=16 or b.shape[0]>4096):
        raise ValueError('q256 weight requires CUDA16-limb int32/uint16 with K<=4096')
    k,n,_=b.shape
    return PreparedQ256Weight(_pack_matrix(b).reshape(k,37*n),k,n)




def field_matmul_q256_tensor(a,b):
    """Exact q256 decode, optionally using a prepared weight snapshot.

    A uses canonical int32/uint16 radix65536 limbs. Raw B follows the same
    contract or can be prepared once with ``prepare_q256_weight``. Packing A is
    always charged; raw B packing is charged on each call. Prepared snapshots
    contain37 bytes per field element and must follow model ownership rules.
    """
    if (not isinstance(a,torch.Tensor) or a.dtype not in (torch.int32,torch.uint16)
            or not a.is_cuda or a.ndim!=3 or a.shape[-1]!=16):
        raise ValueError('q256 input requires CUDA16-limb int32/uint16')
    m,k,_=a.shape
    prepared=b if isinstance(b,PreparedQ256Weight) else prepare_q256_weight(b)
    n=prepared.n
    if k!=prepared.k or k>4096 or prepared.digits.device!=a.device:
        raise ValueError('q256 prepared dimensions or device mismatch')
    if not k:return torch.zeros((m,n,16),device=a.device,dtype=torch.int32)
    adigits=_pack_matrix(a).reshape(m*37,k)
    cross=torch.empty((m*37,n*37),device=a.device,dtype=torch.int32)
    coefficients=torch.empty((m,n,32),device=a.device,dtype=torch.int64)
    out=torch.empty((m,n,16),device=a.device,dtype=torch.int32)
    if m and n:
        with torch.cuda.device(a.device):
            _cross_packed[(triton.cdiv(m*37,32),triton.cdiv(n*37,64))](adigits,prepared.digits,cross,m*37,n*37,k,32,64,64,num_warps=4,num_stages=1)
            _fold[(m,triton.cdiv(n,32),16)](cross,coefficients,m,n,32,num_warps=4)
            _finish[(triton.cdiv(m*n,16),)](coefficients,out,m*n,16,num_warps=4)
    return out
