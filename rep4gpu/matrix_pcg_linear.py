"""Prime31 algebra helpers for fusing a fixed public linear expansion.

These helpers implement local field identities only. They do not generate VOPE,
GGM, OT, authenticated correlations, or malicious consistency checks.
"""
import hashlib
import struct

import torch
import triton
import triton.language as tl

from .field_fast import field_matmul
from .field_kernels import PRIME,_reduce31


@triton.jit
def _accumulate(SOURCE,TOTAL,COUNT:tl.constexpr,SIGN:tl.constexpr,BLOCK:tl.constexpr):
    at=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    source=_reduce31(tl.load(SOURCE+at,at<COUNT,0).to(tl.int64))
    total=tl.load(TOTAL+at,at<COUNT,0).to(tl.int64)
    tl.store(TOTAL+at,_reduce31(total+SIGN*source),at<COUNT)


@triton.jit
def _accumulate_batch(SOURCE,TOTAL,COUNT:tl.constexpr,BATCH:tl.constexpr,SIGN_MASK:tl.constexpr,BLOCK:tl.constexpr):
    at=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    total=tl.load(TOTAL+at,at<COUNT,0).to(tl.int64)
    for batch in tl.static_range(BATCH):
        source=_reduce31(tl.load(SOURCE+batch*COUNT+at,at<COUNT,0).to(tl.int64))
        sign=1 if SIGN_MASK&(1<<batch) else -1
        total+=sign*source
    # At most64 canonical terms keep the signed accumulator below2**38.
    tl.store(TOTAL+at,_reduce31(total),at<COUNT)


@triton.jit
def _sparse_outer(POSITIONS,VALUES,B,RAW,N:tl.constexpr,T:tl.constexpr,BLOCK:tl.constexpr):
    column=tl.program_id(0).to(tl.int64)
    nz=tl.program_id(1)
    cols=tl.program_id(2)*BLOCK+tl.arange(0,BLOCK)
    row=tl.load(POSITIONS+column*T+nz).to(tl.int64)
    value=_reduce31(tl.load(VALUES+column*T+nz).to(tl.int64))
    right=_reduce31(tl.load(B+column*N+cols,cols<N,0).to(tl.int64))
    product=_reduce31(value*right)
    # One position per regular block gives at most K contributions to a row.
    # K<2**31 and product<p keep this int64 atomic sum below2**62.
    tl.atomic_add(RAW+row*N+cols,product,cols<N,sem='relaxed')


def _matrix(value,device=None):
    if (not isinstance(value,torch.Tensor) or value.dtype not in (torch.int32,torch.int64)
            or value.ndim!=2 or (device is not None and value.device!=device)):
        raise ValueError('field matrices require int32/int64 on the expected device')
    return value.contiguous()


class PublicLinearMap:
    """Owned canonical public H snapshot, bound to its shape and field by SHA256."""
    def __init__(self,h,*,prime=PRIME):
        if prime!=PRIME:
            raise ValueError('this arithmetic backend implements only prime31')
        h=_matrix(h)
        if not h.shape[0] or not h.shape[1]:raise ValueError('public H must be nonempty')
        if bool(((h<0)|(h>=PRIME)).any()):raise ValueError('public H must be canonical')
        self._h=h.clone()
        self._version=self._h._version
        self.rows,self.columns=h.shape
        raw=self._h.to(device='cpu',dtype=torch.int64).numpy().astype('<i8',copy=False).tobytes()
        self._identity=hashlib.sha256(b'MatrixPCG-public-linear-p31-v1\0'+struct.pack('<QQQ',PRIME,self.rows,self.columns)+raw).digest()

    @property
    def identity(self):return self._identity

    @property
    def device(self):return self._h.device

    def _check(self,identity):
        if not isinstance(identity,bytes) or identity!=self._identity:
            raise ValueError('public linear map identity mismatch')
        if self._h._version!=self._version or self._h.shape!=(self.rows,self.columns):
            raise ValueError('owned public linear map was modified')

    def expand(self,raw,*,map_identity):
        self._check(map_identity)
        raw=_matrix(raw,self.device)
        if raw.shape[0]!=self.columns:raise ValueError('raw expansion dimensions mismatch')
        return field_matmul(self._h,raw,packed=True)


class RegularSparseColumns:
    """Owned t-regular E: one nonzero in each public block per column.

    Positions and values both have shape[K,t]. Position[:,j] belongs to block j
    of the public map's input space. Values may be signed field representatives
    but must be nonzero modulo p. Validation finishes before atomic writes.
    """
    def __init__(self,public_map,positions,values,*,map_identity):
        public_map._check(map_identity)
        positions=_matrix(positions,public_map.device)
        values=_matrix(values,public_map.device)
        if positions.shape!=values.shape:raise ValueError('sparse position/value shapes differ')
        k,t=positions.shape
        if not t or public_map.columns%t or k>=2**31:
            raise ValueError('invalid regular blocks or int64 accumulation bound')
        block=public_map.columns//t
        lower=torch.arange(t,device=positions.device,dtype=torch.int64)*block
        if bool(((positions<lower)|(positions>=lower+block)).any()):
            raise ValueError('each sparse position must belong to its regular block')
        canonical=values.to(torch.int64).remainder(PRIME)
        if bool((canonical==0).any()):raise ValueError('regular sparse values must be nonzero')
        self._positions=positions.to(torch.int64).clone()
        self._values=canonical.clone()
        self._versions=(self._positions._version,self._values._version)
        self.map_identity=bytes(map_identity)
        self.rows=public_map.columns
        self.k,self.t=k,t
        self.device=positions.device

    def _check(self,public_map):
        public_map._check(self.map_identity)
        if (self.device!=public_map.device or self.rows!=public_map.columns
                or (self._positions._version,self._values._version)!=self._versions):
            raise ValueError('sparse snapshot map, device, or contents changed')

    def materialize(self):
        dense=torch.zeros((self.rows,self.k),device=self.device,dtype=torch.int64)
        for j in range(self.t):
            dense[self._positions[:,j],torch.arange(self.k,device=self.device)]=self._values[:,j]
        return dense


def sparse_local_product(public_map,sparse,b,*,map_identity):
    """Return E@B without forming E; each product is reduced before int64 sums."""
    public_map._check(map_identity)
    if not isinstance(sparse,RegularSparseColumns):raise ValueError('expected regular sparse columns')
    sparse._check(public_map)
    b=_matrix(b,public_map.device)
    if b.shape[0]!=sparse.k:raise ValueError('sparse local inner dimensions mismatch')
    n=b.shape[1]
    raw=torch.zeros((sparse.rows,n),device=b.device,dtype=torch.int64)
    if sparse.k and n:
        if b.is_cuda:
            with torch.cuda.device(b.device):
                _sparse_outer[(sparse.k,sparse.t,triton.cdiv(n,128))](sparse._positions,sparse._values,b,raw,n,sparse.t,128,num_warps=4)
        else:
            right=b.to(torch.int64).remainder(PRIME)
            for j in range(sparse.t):
                product=(sparse._values[:,j,None]*right).remainder(PRIME)
                raw.index_add_(0,sparse._positions[:,j],product)
    return raw


class LinearExpansionAccumulator:
    """Stream raw signed terms, then apply one bound public H.

    Every add names the exact public-map identity. Finish seals this local
    accumulator; lower-level cryptographic checks remain the caller's work.
    """
    def __init__(self,public_map,output_columns,*,map_identity):
        if not isinstance(public_map,PublicLinearMap):raise ValueError('expected a public linear map')
        public_map._check(map_identity)
        if type(output_columns) is not int or output_columns<0:raise ValueError('invalid output column count')
        self._map=public_map
        self._identity=bytes(map_identity)
        self._total=torch.zeros((public_map.columns,output_columns),device=public_map.device,dtype=torch.int32)
        self._finished=False

    def _check(self,map_identity):
        if self._finished:raise ValueError('linear accumulator is sealed')
        self._map._check(map_identity)
        if map_identity!=self._identity:raise ValueError('accumulator public map changed')

    def add(self,term,*,sign,map_identity):
        self._check(map_identity)
        if type(sign) is not int or sign not in (-1,1):raise ValueError('raw term sign must be +1 or -1')
        term=_matrix(term,self._map.device)
        if term.shape!=self._total.shape:raise ValueError('raw term dimensions mismatch')
        if term.is_cuda:
            count=term.numel()
            if count:
                _accumulate[(triton.cdiv(count,256),)](term,self._total,count,sign,256,num_warps=4)
        else:
            value=term.to(torch.int64).remainder(PRIME)
            self._total.copy_((self._total.to(torch.int64)+sign*value).remainder(PRIME).to(torch.int32))
        return self

    def add_batch(self,terms,*,signs,map_identity):
        """Accumulate1..64 consecutive raw terms from one contiguous batch.

        This performs every field addition; batching changes no correlation or
        lower-level consistency check. Signs are public +1/-1 metadata.
        """
        self._check(map_identity)
        if (not isinstance(terms,torch.Tensor) or terms.dtype not in (torch.int32,torch.int64)
                or terms.device!=self._map.device or terms.ndim!=3
                or terms.shape[1:]!=self._total.shape or not 1<=terms.shape[0]<=64):
            raise ValueError('raw batch must have shape[B,L,N] with1<=B<=64')
        signs=tuple(signs)
        if len(signs)!=terms.shape[0] or any(type(sign) is not int or sign not in (-1,1) for sign in signs):
            raise ValueError('raw batch requires one +1/-1 sign per term')
        terms=terms.contiguous()
        if terms.is_cuda:
            count=self._total.numel()
            if count:
                mask=sum(1<<i for i,sign in enumerate(signs) if sign==1)
                _accumulate_batch[(triton.cdiv(count,256),)](terms,self._total,count,len(signs),mask,256,num_warps=4)
        else:
            canonical=terms.to(torch.int64).remainder(PRIME)
            coefficients=torch.tensor(signs,dtype=torch.int64,device=terms.device).reshape(-1,1,1)
            total=self._total.to(torch.int64)+(canonical*coefficients).sum(dim=0)
            self._total.copy_(total.remainder(PRIME).to(torch.int32))
        return self

    def add_local(self,sparse,b,*,map_identity):
        self._check(map_identity)
        raw=sparse_local_product(self._map,sparse,b,map_identity=map_identity)
        return self.add(raw,sign=1,map_identity=map_identity)

    def finish(self,*,map_identity):
        self._check(map_identity)
        self._finished=True
        return self._map.expand(self._total,map_identity=map_identity)
