import pytest
import torch

from rep4gpu.gpu.matmul.kernels import prepare_weight
from rep4gpu.gpu.matmul.tuned_kernels import local_products_tuned

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA required")


@pytest.mark.parametrize("bits",[32,64])
@pytest.mark.parametrize("shape",[(1,71,39),(16,71,39),(17,71,39),(128,71,39),(128,128,128),(128,0,7)])
def test_tuned_against_full_ring_cpu(bits,shape):
    m,k,n=shape
    torch.manual_seed(0x32a000+m+k+n)
    xc=[torch.randint(-2**63,2**63-1,(m,k),dtype=torch.int64) for _ in range(3)]
    yc=[torch.randint(-2**63,2**63-1,(k,n),dtype=torch.int64) for _ in range(3)]
    reference=[(xc[0]+xc[1])@yc[0]+xc[0]@yc[1],
               (xc[1]+xc[2])@yc[1]+xc[1]@yc[2],
               xc[0]@yc[2]+xc[2]@yc[0]]
    if bits==32:
        reference=[x&0xffffffff for x in reference]
    xs=[x.cuda() for x in xc]
    storage="words32" if bits==32 else "limbs"
    ys=[prepare_weight(y.cuda(),bits,storage=storage) for y in yc]
    for actual,expected in zip(local_products_tuned(*xs,*ys,bits),reference):
        torch.testing.assert_close(actual.cpu(),expected,rtol=0,atol=0)


def test_saturating_fusion_regression():
    # This seed catches unsafe fusion of arbitrary ring32 values into a saturating MMA accumulator.
    torch.manual_seed(3280909)
    xc=[torch.randint(-2**63,2**63-1,(128,71),dtype=torch.int64) for _ in range(3)]
    yc=[torch.randint(-2**63,2**63-1,(71,39),dtype=torch.int64) for _ in range(3)]
    xs=[x.cuda() for x in xc]
    ys=[prepare_weight(y.cuda(),32,storage="words32") for y in yc]
    expected=[((xc[0]+xc[1])@yc[0]+xc[0]@yc[1])&0xffffffff,
              ((xc[1]+xc[2])@yc[1]+xc[1]@yc[2])&0xffffffff,
              (xc[0]@yc[2]+xc[2]@yc[0])&0xffffffff]
    for actual,want in zip(local_products_tuned(*xs,*ys,32),expected):
        torch.testing.assert_close(actual.cpu(),want,rtol=0,atol=0)


@pytest.mark.parametrize("shape",[(1,71,39),(17,71,39),(128,71,39),(128,128,128)])
def test_tuned_five_against_cpu(shape):
    from rep4gpu.gpu.matmul.five_product import prepare_five_weights
    from rep4gpu.gpu.matmul.tuned_kernels import local_products_five_tuned

    m,k,n=shape
    torch.manual_seed(0x5a000+m+k+n)
    xc=[torch.randint(-2**63,2**63-1,(m,k),dtype=torch.int64) for _ in range(3)]
    yc=[torch.randint(-2**63,2**63-1,(k,n),dtype=torch.int64) for _ in range(3)]
    xs=[x.cuda() for x in xc]
    ys=[prepare_weight(y.cuda(),32,storage="words32") for y in yc]
    weights=prepare_five_weights(*ys)
    expected=[((xc[0]+xc[1])@yc[0]+xc[0]@yc[1])&0xffffffff,
              ((xc[1]+xc[2])@yc[1]+xc[1]@yc[2])&0xffffffff,
              (xc[0]@yc[2]+xc[2]@yc[0])&0xffffffff]
    for actual,want in zip(local_products_five_tuned(xs,weights),expected):
        torch.testing.assert_close(actual.cpu(),want,rtol=0,atol=0)
