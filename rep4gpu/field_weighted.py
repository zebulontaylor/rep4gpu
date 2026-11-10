"""Independent-key weighted reductions of frozen prime31 MAC residuals."""
import torch
import triton
import triton.language as tl

from .field_fusion import _integer_tensors,_mac_final
from .field_kernels import _reduce31


@triton.jit
def _partial(RESIDUAL,CHI,PARTS,N:tl.constexpr,NP:tl.constexpr,BLOCK:tl.constexpr):
    key=tl.program_id(0)
    idx=tl.program_id(1)*BLOCK+tl.arange(0,BLOCK)
    residual=_reduce31(tl.load(RESIDUAL+key*N+idx,idx<N,0).to(tl.int64))
    chi=_reduce31(tl.load(CHI+key*N+idx,idx<N,0).to(tl.int64))
    weighted=_reduce31(residual*chi)
    tl.store(PARTS+key*NP+tl.program_id(1),_reduce31(tl.sum(weighted,0)))


def field_weighted_check(residual,challenge,*,packed=True):
    """Return one prime31 weighted residual per independent MAC key.

    Both inputs have shape [H,N]. The caller freezes residuals before deriving
    fresh independent challenge rows. Only chunk reductions are allocated;
    no dummy opened-value vector or alpha product is required.
    """
    residual,challenge=_integer_tensors((residual,challenge))
    if residual.ndim!=2 or challenge.shape!=residual.shape:
        raise ValueError("independent challenge[H,N] rows must match frozen residuals")
    h,n=residual.shape
    if h<1 or n>=2**31:
        raise ValueError("invalid MAC key count or accumulation bound")
    out=torch.empty((h,),device=residual.device,dtype=torch.int32 if packed else torch.int64)
    if not n:
        return out.zero_()
    parts=triton.cdiv(n,256)
    partial=torch.empty((h,parts),device=residual.device,dtype=torch.int64)
    with torch.cuda.device(residual.device):
        _partial[(h,parts)](residual,challenge,partial,n,parts,256,num_warps=4)
        _mac_final[(h,)](partial,out,parts,256,num_warps=4)
    return out
