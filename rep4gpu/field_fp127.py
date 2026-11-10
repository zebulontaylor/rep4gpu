"""Exact tensor-core matrix products over F127, p=2**127-1.

Canonical tensors use8 little-endian radix65536 limbs, stored as uint16 or
int32; limb7 is below32768 and the all-ones encoding of p is excluded. The
caller establishes these value ranges. Matrix metadata and prepared snapshot
versions are checked on every call. Native FP127's16-byte encoding has the same
little-endian layout. No native preprocessing or protocol state is imported.
"""
import torch
import triton
import triton.language as tl

PRIME=2**127-1
MAX_K=8192


@triton.jit
def _pack_left(SOURCE,DIGITS,SUMS,K:tl.constexpr,BLOCK:tl.constexpr):
    row=tl.program_id(0).to(tl.int64)
    digit=tl.program_id(1)
    ki=tl.arange(0,BLOCK)
    word=tl.load(SOURCE+(row*K+ki)*8+digit//2,ki<K,0).to(tl.int32)
    value=((word>>(8*(digit%2)))&255)-128
    value=tl.where(ki<K,value,0)
    tl.store(DIGITS+(row*16+digit)*K+ki,value.to(tl.int8),ki<K)
    tl.store(SUMS+row*16+digit,tl.sum(value,0))


@triton.jit
def _pack_right(SOURCE,DIGITS,SUMS,N:tl.constexpr,K:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
    digit=tl.program_id(0)
    cols=tl.program_id(1).to(tl.int64)*BN+tl.arange(0,BN)
    kk=tl.arange(0,BK)
    total=tl.full((BN,),0,tl.int32)
    for block in range(tl.cdiv(K,BK)):
        ki=block*BK+kk
        mask=(ki[:,None]<K)&(cols[None,:]<N)
        word=tl.load(SOURCE+(ki[:,None].to(tl.int64)*N+cols[None,:])*8+digit//2,mask,0).to(tl.int32)
        value=((word>>(8*(digit%2)))&255)-128
        value=tl.where(mask,value,0)
        total+=tl.sum(value,0)
        tl.store(DIGITS+ki[:,None].to(tl.int64)*(16*N)+digit*N+cols[None,:],value.to(tl.int8),mask)
    tl.store(SUMS+digit*N+cols,total,cols<N)


@triton.jit
def _cross(A,B,C,MR:tl.constexpr,NC:tl.constexpr,K:tl.constexpr,TM:tl.constexpr,TN:tl.constexpr,TK:tl.constexpr):
    rows=tl.program_id(0)*TM+tl.arange(0,TM)
    cols=tl.program_id(1)*TN+tl.arange(0,TN)
    kk=tl.arange(0,TK)
    acc=tl.full((TM,TN),0,tl.int32)
    for block in range(tl.cdiv(K,TK)):
        ki=block*TK+kk
        a=tl.load(A+rows[:,None].to(tl.int64)*K+ki[None,:],(rows[:,None]<MR)&(ki[None,:]<K),0)
        b=tl.load(B+ki[:,None].to(tl.int64)*NC+cols[None,:],(ki[:,None]<K)&(cols[None,:]<NC),0)
        # K<=8192 bounds signed dot accumulators in absolute value by2**27.
        acc=tl.dot(a,b,acc,out_dtype=tl.int32)
    tl.store(C+rows[:,None].to(tl.int64)*NC+cols[None,:],acc,(rows[:,None]<MR)&(cols[None,:]<NC))


@triton.jit
def _fold(C,ASUM,BSUM,COEFFICIENTS,N:tl.constexpr,K:tl.constexpr,BN:tl.constexpr):
    row=tl.program_id(0).to(tl.int64)
    cols=tl.program_id(1).to(tl.int64)*BN+tl.arange(0,BN)
    limb=tl.program_id(2)
    acc=tl.full((BN,),0,tl.int64)
    for piece in tl.static_range(4):
        diagonal=2*limb+(piece%2)+(piece//2)*16
        diagonal_sum=tl.full((BN,),0,tl.int64)
        for i in range(16):
            j=diagonal-i
            if (j>=0)&(j<16):
                signed=tl.load(C+(row*16+i)*(16*N)+j*N+cols,cols<N,0).to(tl.int64)
                asum=tl.load(ASUM+row*16+i).to(tl.int64)
                bsum=tl.load(BSUM+j*N+cols,cols<N,0).to(tl.int64)
                unsigned=signed+128*(asum+bsum)+16384*K
                diagonal_sum+=unsigned
        # A high diagonal's exponent8*d folds to8*d-127: shift1 or9.
        shift=8*(piece%2)+(1 if piece>=2 else 0)
        acc+=diagonal_sum<<shift
    tl.store(COEFFICIENTS+(row*N+cols)*8+limb,acc,cols<N)


@triton.jit
def _carry(c0,c1,c2,c3,c4,c5,c6,c7):
    c1+=c0>>16;c0=c0&65535
    c2+=c1>>16;c1=c1&65535
    c3+=c2>>16;c2=c2&65535
    c4+=c3>>16;c3=c3&65535
    c5+=c4>>16;c4=c4&65535
    c6+=c5>>16;c5=c5&65535
    c7+=c6>>16;c6=c6&65535
    c0+=c7>>15;c7=c7&32767
    return c0,c1,c2,c3,c4,c5,c6,c7


@triton.jit
def _finish(C,OUT,COUNT:tl.constexpr,BLOCK:tl.constexpr):
    at=tl.program_id(0).to(tl.int64)*BLOCK+tl.arange(0,BLOCK)
    c0=tl.load(C+at*8+0,at<COUNT,0)
    c1=tl.load(C+at*8+1,at<COUNT,0)
    c2=tl.load(C+at*8+2,at<COUNT,0)
    c3=tl.load(C+at*8+3,at<COUNT,0)
    c4=tl.load(C+at*8+4,at<COUNT,0)
    c5=tl.load(C+at*8+5,at<COUNT,0)
    c6=tl.load(C+at*8+6,at<COUNT,0)
    c7=tl.load(C+at*8+7,at<COUNT,0)
    # Folded coefficients<2**43. Pass1 leaves numeric value<2**127+2**28;
    # pass2 leaves value<=p; pass3 normalizes the low limb after the wrap.
    for _ in tl.static_range(3):
        c0,c1,c2,c3,c4,c5,c6,c7=_carry(c0,c1,c2,c3,c4,c5,c6,c7)
    modulus=(c0==65535)&(c1==65535)&(c2==65535)&(c3==65535)&(c4==65535)&(c5==65535)&(c6==65535)&(c7==32767)
    tl.store(OUT+at*8+0,tl.where(modulus,0,c0),at<COUNT)
    tl.store(OUT+at*8+1,tl.where(modulus,0,c1),at<COUNT)
    tl.store(OUT+at*8+2,tl.where(modulus,0,c2),at<COUNT)
    tl.store(OUT+at*8+3,tl.where(modulus,0,c3),at<COUNT)
    tl.store(OUT+at*8+4,tl.where(modulus,0,c4),at<COUNT)
    tl.store(OUT+at*8+5,tl.where(modulus,0,c5),at<COUNT)
    tl.store(OUT+at*8+6,tl.where(modulus,0,c6),at<COUNT)
    tl.store(OUT+at*8+7,tl.where(modulus,0,c7),at<COUNT)


def _source(value):
    if (not isinstance(value,torch.Tensor) or value.dtype not in (torch.int32,torch.uint16)
            or not value.is_cuda or value.ndim!=3 or value.shape[-1]!=8):
        raise ValueError('F127 matrices require CUDA int32/uint16 tensors with8 little-endian limbs')
    return value.contiguous()


class PreparedFP127Left:
    """Owned centered-byte left matrix, useful for a fixed public H."""
    def __init__(self,digits,sums,m,k):
        self.digits,self.sums,self.m,self.k=digits,sums,m,k
        self._versions=(digits._version,sums._version)

    def _check(self):
        if ((self.digits._version,self.sums._version)!=self._versions
                or self.digits.dtype!=torch.int8 or self.sums.dtype!=torch.int32
                or self.digits.shape!=(16*self.m,self.k) or self.sums.shape!=(16*self.m,)):
            raise ValueError('prepared F127 left snapshot was modified')

    @property
    def storage_bytes(self):return self.digits.numel()+4*self.sums.numel()


class PreparedFP127Weight:
    """Owned centered-byte right matrix; retains no original input tensor."""
    def __init__(self,digits,sums,k,n):
        self.digits,self.sums,self.k,self.n=digits,sums,k,n
        self._versions=(digits._version,sums._version)

    def _check(self):
        if ((self.digits._version,self.sums._version)!=self._versions
                or self.digits.dtype!=torch.int8 or self.sums.dtype!=torch.int32
                or self.digits.shape!=(self.k,16*self.n) or self.sums.shape!=(16*self.n,)):
            raise ValueError('prepared F127 right snapshot was modified')

    @property
    def storage_bytes(self):return self.digits.numel()+4*self.sums.numel()


def prepare_fp127_left(a):
    """Snapshot canonical H for repeated H@S; packing is separate setup work."""
    a=_source(a)
    m,k,_=a.shape
    if k>MAX_K:raise ValueError('F127 tensor products support K<=8192')
    digits=torch.empty((m*16,k),device=a.device,dtype=torch.int8)
    sums=torch.zeros((m*16,),device=a.device,dtype=torch.int32) if not k else torch.empty((m*16,),device=a.device,dtype=torch.int32)
    if m and k:
        with torch.cuda.device(a.device):
            _pack_left[(m,16)](a,digits,sums,k,triton.next_power_of_2(k),num_warps=4)
    return PreparedFP127Left(digits,sums,m,k)


def prepare_fp127_weight(b):
    """Snapshot canonical right weights; every element occupies16 packed bytes."""
    b=_source(b)
    k,n,_=b.shape
    if k>MAX_K:raise ValueError('F127 tensor products support K<=8192')
    digits=torch.empty((k,16*n),device=b.device,dtype=torch.int8)
    sums=torch.empty((16*n,),device=b.device,dtype=torch.int32)
    if n:
        with torch.cuda.device(b.device):
            _pack_right[(16,triton.cdiv(n,32))](b,digits,sums,n,k,32,64,num_warps=4)
    return PreparedFP127Weight(digits,sums,k,n)


def field_matmul_fp127(a,b):
    """Return canonical int32[M,N,8] limbs for an exact F127 matrix product.

    Either operand may be its corresponding prepared snapshot. Fixed public H
    can use ``prepare_fp127_left``; fixed right weights use
    ``prepare_fp127_weight``. Raw operand packing is included in each call.
    Prepared storage is checked for mutation and contains no source tensor.
    The256*M*N int32 cross products are a temporary allocation, not output MACs.
    """
    left=a if isinstance(a,PreparedFP127Left) else prepare_fp127_left(a)
    right=b if isinstance(b,PreparedFP127Weight) else prepare_fp127_weight(b)
    left._check();right._check()
    m,k,n=left.m,left.k,right.n
    if k!=right.k or k>MAX_K or left.digits.device!=right.digits.device:
        raise ValueError('F127 operand dimensions or devices mismatch')
    if not k:return torch.zeros((m,n,8),device=left.digits.device,dtype=torch.int32)
    cross=torch.empty((m*16,n*16),device=left.digits.device,dtype=torch.int32)
    coefficients=torch.empty((m,n,8),device=left.digits.device,dtype=torch.int64)
    out=torch.empty((m,n,8),device=left.digits.device,dtype=torch.int32)
    if m and n:
        with torch.cuda.device(left.digits.device):
            _cross[(triton.cdiv(m*16,32),triton.cdiv(n*16,128))](left.digits,right.digits,cross,m*16,n*16,k,32,128,64,num_warps=4,num_stages=1)
            _fold[(m,triton.cdiv(n,32),8)](cross,left.sums,right.sums,coefficients,n,k,32,num_warps=4)
            _finish[(triton.cdiv(m*n,32),)](coefficients,out,m*n,32,num_warps=4)
    return out
