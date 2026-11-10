"""Accumulate independent-key MAC projections into owned device buffers."""
import torch
import triton
import triton.language as tl

from .field_fusion import _integer_tensors
from .field_kernels import _reduce31
from .field_weighted import _partial


@triton.jit
def _accumulate(PARTS,TOTAL,FAILURE,REJECTED,NP:tl.constexpr,BLOCK:tl.constexpr):
    key=tl.program_id(0)
    idx=tl.arange(0,BLOCK)
    sums=tl.full((BLOCK,),0,tl.int64)
    for start in range(tl.cdiv(NP,BLOCK)):
        at=start*BLOCK+idx
        sums+=tl.load(PARTS+key*NP+at,at<NP,0).to(tl.int64)
    previous=_reduce31(tl.load(TOTAL+key).to(tl.int64))
    tl.store(TOTAL+key,_reduce31(previous+tl.sum(sums,0)))
    if key==0:
        failure=tl.load(FAILURE)
        rejected=tl.load(REJECTED)
        tl.store(FAILURE,failure|rejected)


def field_weighted_accumulate(residual,challenge,total,failure,rejected):
    """Add a projection to ``total`` and OR rejection status into ``failure``.

    Residual and independently sampled challenge have shape [H,N]. Total is an
    owned contiguous int32/int64 vector [H]; failure and rejected are int32 [1].
    The caller freezes residuals before deriving each fresh challenge, preserves
    the protocol's segment/key domains, and submits successive calls on one CUDA
    stream. This helper changes no coin, commitment, or rejection policy.

    Total and failure are modified in place and must not share storage with any
    other argument. Inputs may use arbitrary signed field representatives. The
    returned tensors are the same total and failure objects supplied by caller.
    """
    if not isinstance(residual,torch.Tensor) or not isinstance(challenge,torch.Tensor):
        raise ValueError('residual and challenge must be tensors')
    if residual.ndim!=2 or challenge.shape!=residual.shape:
        raise ValueError('independent challenge[H,N] rows must match frozen residuals')
    h,n=residual.shape
    if h<1 or n>=2**31:
        raise ValueError('invalid MAC key count or accumulation bound')
    if (not isinstance(total,torch.Tensor) or total.shape!=(h,) or total.dtype not in (torch.int32,torch.int64)
            or not total.is_contiguous() or total.device!=residual.device):
        raise ValueError('owned total must be a contiguous field vector [H]')
    for flag in (failure,rejected):
        if (not isinstance(flag,torch.Tensor) or flag.shape!=(1,) or flag.dtype!=torch.int32
                or flag.device!=residual.device or not flag.is_contiguous()):
            raise ValueError('status flags must be contiguous int32 [1] on the source device')
    sources=(residual,challenge,total,failure,rejected)
    for output_index in (2,3):
        output=sources[output_index]
        for source_index,source in enumerate(sources):
            if source_index!=output_index and output.untyped_storage().data_ptr()==source.untyped_storage().data_ptr():
                raise ValueError('owned accumulator buffers must not alias other arguments')
    residual,challenge=_integer_tensors((residual,challenge))
    parts=triton.cdiv(n,256)
    partial=torch.empty((h,parts),device=residual.device,dtype=torch.int64)
    with torch.cuda.device(residual.device):
        if n:
            _partial[(h,parts)](residual,challenge,partial,n,parts,256,num_warps=4)
        _accumulate[(h,)](partial,total,failure,rejected,parts,256,num_warps=4)
    return total,failure
