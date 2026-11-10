"""Measure local CUDA work for weighted compact bit conversion.

Run from the repository root::

    PYTHONPATH=. .venv/bin/python benchmarks/compact_conversion_local.py

Random int64 tensors stand in for local protocol shares. The check compares
pair preparation and all six weighted reductions before timing. No network,
PRG, joint input, product resharing, or final arithmetic assembly runs here.
Triton's CUDA-event benchmark uses 30 ms warmup and 100 ms measurement per
path. The fused output buffers are allocated before timing; the reference
path creates its usual intermediate tensors during each call.
"""

import torch,triton,triton.testing
from rep4gpu.compact_conversion_gpu import _reduce,_pair_products
from rep4gpu.compact_conversion import _weighted_sum
from rep4gpu.conversion import _components
from rep4gpu.runtime import Share,make_seeds
from rep4gpu.fused_runtime import FusedParty
from rep4gpu.weighted_kernels import weighted_local_products
p,d,o=3,1048576,2
rt=FusedParty(0,None,{i:s for i,s in make_seeds().items() if i!=0},device='cuda',bits=64)
parts=tuple(torch.randint(-(1<<40),1<<40,(p,d),dtype=torch.int64,device='cuda') for _ in range(3))
planes=Share(parts,True)
q=Share(tuple(torch.randint(-(1<<40),1<<40,x.shape,dtype=x.dtype,device=x.device) for x in parts),False)
r=Share(tuple(torch.randint(-(1<<40),1<<40,x.shape,dtype=x.dtype,device=x.device) for x in parts),False)
w=torch.tensor([[1,-1048576,-1048576],[0,0,1]],device='cuda',dtype=torch.int64)
out=torch.empty((6,o,d),device='cuda',dtype=torch.int64)
pair=torch.empty_like(parts[0])
def old():
    clipped=Share(tuple(x&1 for x in parts),True)
    a,b,c,e=_components(rt,clipped,False)
    known={1:clipped.parts[0],2:clipped.parts[1],3:clipped.parts[2]}
    product=known[2]*known[3]
    left=rt.sub(rt.add(a,b),rt.mul_public(q,2))
    right=rt.sub(rt.add(c,e),rt.mul_public(r,2))
    local=weighted_local_products(left.parts,right.parts,w,bits=64)
    linear=rt._share(tuple(_weighted_sum(l+rr,w) for l,rr in zip(left.parts,right.parts)))
    return product,local,linear

def new():
    _pair_products[(triton.cdiv(p*d,256),)](*parts,pair,p*d,0,256)
    _reduce[(triton.cdiv(d,64),)](*parts,*q.parts,*r.parts,w,out,p,d,o,0,64,4,64,num_warps=4)
product,local,linear=old()
new()
assert torch.equal(product,pair)
for actual,expected in zip(out.unbind(0),(*local,*linear.parts)):
    assert torch.equal(actual,expected)
print('all six reductions and pair preparation exact')
for fn in (old,new):
    fn()
    torch.cuda.synchronize()
    print(fn.__name__,triton.testing.do_bench(fn,warmup=30,rep=100))
