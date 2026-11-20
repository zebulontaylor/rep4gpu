"""Validated tile selection for exact GPU Rep4 local products."""
from functools import lru_cache

import torch
import triton

from .kernels import CompactWeight,_local_products,_tensor_mm,local_products,ring_matmul
from .five_product import FiveWeights


@lru_cache(maxsize=None)
def _supports_tuned(device):
    # These integer MMA tiles and shared-memory budgets are tested on SM 8.9.
    return torch.cuda.get_device_capability(device)==(8,9)


def local_products_tuned(x0,x1,x2,y0,y1,y2,bits=64, *, encoder_pipeline_stages=0):
    """Use exact ring32 modular-add reconstruction for SM 8.9 prefill batches.

    Compact ring32 weights and at least sixteen rows usually select the
    32/128/32 tile. Classifiers with 512 input features and at most 64 output
    coordinates use narrower tiles when they reduce measured work. Other
    devices, rings, formats, and smaller batches retain the baseline route.
    """
    xs=(x0,x1,x2)
    ys=(y0,y1,y2)
    if bits!=32 or not all(isinstance(y,CompactWeight) for y in ys) or x0.ndim!=2 or x0.shape[0]<16 or not _supports_tuned(x0.device):
        return local_products(*xs,*ys,bits)
    m,k=x0.shape
    n=y0.shape[1]
    if encoder_pipeline_stages and (k, n) == (512, 128):
        from .pipelined_encoder import local_products_pipelined
        return local_products_pipelined(*xs, *ys, stages=encoder_pipeline_stages)
    tile=(32,128,32)
    if k==512 and n<=32:
        if m<1024:
            return local_products(*xs,*ys,bits)
        tile=(64,32,32) if m<4096 else (32,32,64)
    elif k==512 and n<=64 and m>=4096:
        tile=(32,64,32)
    from .ring32_opaque import local_products_opaque
    return local_products_opaque(*xs,*ys,tile=tile)



def ring_matmul_tuned(a,b,bits=64):
    """Use the validated prefill tile for SM 8.9 compact ring32 matrix products."""
    if bits!=32 or not isinstance(b,CompactWeight) or a.ndim!=2 or a.shape[0]<32 or not _supports_tuned(a.device):
        return ring_matmul(a,b,bits)
    if a.dtype!=torch.int64 or not a.is_cuda or a.device!=b.device or a.shape[1]!=b.shape[0]:
        raise ValueError("expected a compatible int64 CUDA input matrix")
    m,k=a.shape
    n=b.shape[1]
    out=torch.empty((m,n),device=a.device,dtype=torch.int64)
    if m and n and k:
        with torch.cuda.device(a.device):
            _tensor_mm[(triton.cdiv(m,32),triton.cdiv(n,128))](
                a.contiguous(),b.words,out,m,n,k,32,32,128,64,False,num_warps=4,num_stages=1)
    elif k==0:
        out.zero_()
    return out


def local_products_five_tuned(xs,weights):
    """Use five prepacked ring32 products with the validated matrix tile.

    ``FiveWeights`` caches five matrices, compared with three for the six-product
    path. Input combinations and output reconstruction use exact GPU additions.
    """
    xs=tuple(xs)
    if len(xs)!=3 or not isinstance(weights,FiveWeights) or len(weights.parts)!=5:
        raise ValueError("provide three input shares and prepared five-product weights")
    if any(x.ndim!=2 or x.dtype!=torch.int64 or not x.is_cuda or x.shape!=xs[0].shape or x.device!=weights.device for x in xs):
        raise ValueError("input shares must be matching int64 CUDA matrices on the weight device")
    if xs[0].shape[1]!=weights.shape[0]:
        raise ValueError("matrix dimensions must match")
    x0,x1,x2=xs
    y01,y1,y12,y2,y012=weights.parts
    x01=x0+x1
    p0=ring_matmul_tuned(x01,y01,32)
    p1=ring_matmul_tuned(x1,y1,32)
    p2=ring_matmul_tuned(x1+x2,y12,32)
    p3=ring_matmul_tuned(x2,y2,32)
    p4=ring_matmul_tuned(x01+x2,y012,32)
    return ((p0-p1)&0xffffffff,(p2-p3)&0xffffffff,(p4-p0-p2+p1)&0xffffffff)
