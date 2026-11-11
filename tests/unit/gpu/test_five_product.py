import pytest
import torch

from rep4gpu.gpu.matmul.kernels import local_products,prepare_weight
from rep4gpu.gpu.matmul.five_product import local_products_five,prepare_five_weights


@pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA required")
@pytest.mark.parametrize("compact",[False,True])
@pytest.mark.parametrize("shape",[(1,71,39),(17,71,39),(128,128,128),(3,0,7)])
def test_five_exact_full_ring(compact,shape):
    m,k,n=shape
    torch.manual_seed(0x50000+m+k+n)
    xs=[torch.randint(-2**63,2**63-1,(m,k),device="cuda",dtype=torch.int64) for _ in range(3)]
    ys=[torch.randint(-2**63,2**63-1,(k,n),device="cuda",dtype=torch.int64) for _ in range(3)]
    xc=[x.cpu() for x in xs]
    yc=[y.cpu() for y in ys]
    reference=[((xc[0]+xc[1])@yc[0]+xc[0]@yc[1])&0xffffffff,
               ((xc[1]+xc[2])@yc[1]+xc[1]@yc[2])&0xffffffff,
               (xc[0]@yc[2]+xc[2]@yc[0])&0xffffffff]
    if compact:
        ys=[prepare_weight(y,32,storage="words32") for y in ys]
    prepared=prepare_five_weights(*ys)
    actual=local_products_five(xs,prepared)
    for a,r,b in zip(actual,reference,local_products(*xs,*ys,32)):
        torch.testing.assert_close(a.cpu(),r,rtol=0,atol=0)
        torch.testing.assert_close(b.cpu(),r,rtol=0,atol=0)
    assert len(prepared.parts)==5
    assert all(p.words.dtype==torch.int32 for p in prepared.parts)
    assert sum(p.words.numel()*p.words.element_size() for p in prepared.parts)==20*k*n
