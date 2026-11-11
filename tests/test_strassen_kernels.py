"""Matrix-level identities must hold for arbitrary ring32 bit patterns."""
import pytest
import torch
from rep4gpu.kernels import prepare_weight
from rep4gpu.strassen_kernels import prepare_strassen_weight, ring_matmul_strassen

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


@pytest.mark.parametrize('levels',[1,2])
@pytest.mark.parametrize('shape',[(4,12,20),(16,64,128),(128,128,256)])
def test_strassen_arbitrary_ring32(levels,shape):
    m,k,n=shape
    torch.manual_seed(722+m+k+n)
    a=torch.randint(-2**63,2**63-1,(m,k),dtype=torch.int64)
    b=torch.randint(-2**63,2**63-1,(k,n),dtype=torch.int64)
    expected=(a@b)&0xffffffff
    weight=prepare_weight(b.cuda(),32,storage='words32')
    prepared=prepare_strassen_weight(weight,levels)
    actual=ring_matmul_strassen(a.cuda(),prepared)
    torch.testing.assert_close(actual.cpu(),expected,rtol=0,atol=0)
    assert prepared.storage_bytes==k*n*4*(7**levels)//(4**levels)


@pytest.mark.parametrize('levels',[1,2])
@pytest.mark.parametrize('shape',[(4,12,20),(128,128,256),(8,0,8),(0,8,8),(8,8,0)])
@pytest.mark.parametrize('opaque',[False,True])
def test_strassen_local_products(levels,shape,opaque):
    from rep4gpu.strassen_kernels import local_products_strassen
    m,k,n=shape
    torch.manual_seed(8193+m+k+n)
    xs=[torch.randint(-2**63,2**63-1,(m,k),dtype=torch.int64) for _ in range(3)]
    ys=[torch.randint(-2**63,2**63-1,(k,n),dtype=torch.int64) for _ in range(3)]
    expected=[((xs[0]+xs[1])@ys[0]+xs[0]@ys[1])&0xffffffff,
              ((xs[1]+xs[2])@ys[1]+xs[1]@ys[2])&0xffffffff,
              (xs[0]@ys[2]+xs[2]@ys[0])&0xffffffff]
    weights=[prepare_strassen_weight(prepare_weight(y.cuda(),32,storage='words32'),levels) for y in ys]
    actual=local_products_strassen([x.cuda() for x in xs],weights,opaque=opaque)
    for got,want in zip(actual,expected):
        torch.testing.assert_close(got.cpu(),want,rtol=0,atol=0)
