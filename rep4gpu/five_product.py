"""Five-GEMM ring32 local products using validated exact matrix kernels."""
from dataclasses import dataclass

import torch

from rep4gpu.kernels import CompactWeight, prepare_weight, ring_matmul


@dataclass(frozen=True)
class FiveWeights:
    """Five compact static combinations, occupying 20 bytes per matrix entry.

    Three original compact shares occupy 12 bytes per entry. Release them after
    preparation if the original format is no longer needed.
    """
    parts: tuple

    @property
    def shape(self):
        return self.parts[0].shape

    @property
    def device(self):
        return self.parts[0].device

    @property
    def bits(self):
        return 32


def prepare_five_weights(y0,y1,y2):
    """Cache y0+y1, y1, y1+y2, y2 and y0+y1+y2 as compact ring32 words."""
    ys=(y0,y1,y2)
    compact=isinstance(y0,CompactWeight)
    if any(isinstance(y,CompactWeight)!=compact for y in ys):
        raise ValueError("weight formats must match")
    if compact:
        if any(y.shape!=y0.shape or y.device!=y0.device for y in ys):
            raise ValueError("weight shapes and devices must match")
        base=ys
    else:
        if any(y.ndim!=2 or y.dtype!=torch.int64 or not y.is_cuda or y.shape!=y0.shape or y.device!=y0.device for y in ys):
            raise ValueError("weights must be matching int64 CUDA matrices or compact ring32 weights")
        base=tuple(prepare_weight(y,32,storage="words32") for y in ys)
    a,b,c=base
    ab=CompactWeight(a.words+b.words)
    bc=CompactWeight(b.words+c.words)
    abc=CompactWeight(ab.words+c.words)
    return FiveWeights((ab,b,bc,c,abc))


def local_products_five(xs,weights):
    """Return exact v0,v1,v4 with five ring GEMMs and GPU integer additions.

    Input shares remain int64 CUDA matrices. All arithmetic is modulo 2**32;
    returned int64 matrices use canonical values in [0,2**32).
    """
    xs=tuple(xs)
    if len(xs)!=3 or not isinstance(weights,FiveWeights) or len(weights.parts)!=5:
        raise ValueError("provide three input shares and prepared five-product weights")
    if any(x.ndim!=2 or x.dtype!=torch.int64 or not x.is_cuda or x.shape!=xs[0].shape or x.device!=weights.device for x in xs):
        raise ValueError("input shares must be matching int64 CUDA matrices on the weight device")
    if xs[0].shape[1]!=weights.shape[0]:
        raise ValueError("matrix dimensions must match")
    x0,x1,x2=xs
    x01=x0+x1
    x12=x1+x2
    x012=x01+x2
    y01,y1,y12,y2,y012=weights.parts
    p0=ring_matmul(x01,y01,32)
    p1=ring_matmul(x1,y1,32)
    p2=ring_matmul(x12,y12,32)
    p3=ring_matmul(x2,y2,32)
    p4=ring_matmul(x012,y012,32)
    return ((p0-p1)&0xffffffff,(p2-p3)&0xffffffff,(p4-p0-p2+p1)&0xffffffff)
