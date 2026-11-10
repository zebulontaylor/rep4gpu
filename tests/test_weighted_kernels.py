import pytest
import torch

from rep4gpu.weighted_kernels import weighted_local_products

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA required")


def reference(xs,ys,weights,bits):
    x0,x1,x2=[x.cpu() for x in xs]
    y0,y1,y2=[y.cpu() for y in ys]
    values=((x0+x1)*y0+x0*y1,(x1+x2)*y1+x1*y2,x0*y2+x2*y0)
    weights=weights.cpu()
    results=[]
    for value in values:
        if weights.ndim==1:
            result=(value*weights.reshape(-1,*([1]*(value.ndim-1)))).sum(0)
        else:
            result=(value.unsqueeze(0)*weights.reshape(*weights.shape,*([1]*(value.ndim-1)))).sum(1)
        if bits==32:
            result&=0xffffffff
        results.append(result)
    return results


@pytest.mark.parametrize("bits",[32,64])
@pytest.mark.parametrize("planes",[0,1,7,32,64])
@pytest.mark.parametrize("outputs",[None,2])
def test_weighted_full_ring(bits,planes,outputs):
    shape=(planes,3,19)
    xs=[torch.randint(-2**63,2**63-1,shape,device="cuda",dtype=torch.int64) for _ in range(3)]
    ys=[torch.randint(-2**63,2**63-1,shape,device="cuda",dtype=torch.int64) for _ in range(3)]
    wshape=(planes,) if outputs is None else (outputs,planes)
    weights=torch.randint(-2**63,2**63-1,wshape,device="cuda",dtype=torch.int64)
    expected=reference(xs,ys,weights,bits)
    for actual,want in zip(weighted_local_products(xs,ys,weights,bits),expected):
        torch.testing.assert_close(actual.cpu(),want,rtol=0,atol=0)


@pytest.mark.parametrize("bits",[32,64])
def test_signed_bit_weights(bits):
    planes=64
    shape=(planes,37)
    xs=[torch.randint(-2**63,2**63-1,shape,device="cuda",dtype=torch.int64) for _ in range(3)]
    ys=[torch.randint(-2**63,2**63-1,shape,device="cuda",dtype=torch.int64) for _ in range(3)]
    weights=[1<<i for i in range(64)]
    wt=torch.tensor([w if w<2**63 else w-2**64 for w in weights],dtype=torch.int64)
    for actual,want in zip(weighted_local_products(xs,ys,weights,bits),reference(xs,ys,wt,bits)):
        torch.testing.assert_close(actual.cpu(),want,rtol=0,atol=0)


def test_scalar_output():
    xs=[torch.arange(7,device="cuda",dtype=torch.int64)+i for i in range(3)]
    ys=[torch.arange(7,device="cuda",dtype=torch.int64)-i for i in range(3)]
    weights=torch.arange(-3,4,device="cuda",dtype=torch.int64)
    for actual,want in zip(weighted_local_products(xs,ys,weights),reference(xs,ys,weights,64)):
        assert actual.shape==()
        torch.testing.assert_close(actual.cpu(),want,rtol=0,atol=0)
