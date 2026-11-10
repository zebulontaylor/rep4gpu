import numpy as np
import pytest
import torch

from rep4gpu.field_fusion import authenticated_local,field_mac_check
from rep4gpu.field_kernels import PRIME

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA required")


def mm(a,b):
    return (a.astype(object)@b.astype(object))%PRIME


@pytest.mark.parametrize("h",[1,5,10])
@pytest.mark.parametrize("packed",[False,True])
def test_authenticated_matrix(h,packed):
    rng=np.random.default_rng(0x3100+h)
    m,k,n=17,71,39
    shapes=[(m,k),(k,n),(m,k),(h,m,k),(k,n),(h,k,n),(m,n),(h,m,n),(h,)]
    data=[rng.integers(0,PRIME,s,dtype=np.int64) for s in shapes]
    d,e,av,am,bv,bm,cv,cm,alpha=data
    public=mm(d,e)
    for party in (0,2):
        value=(cv.astype(object)+mm(d,bv)+mm(av,e)+(public if party==0 else 0))%PRIME
        macs=np.stack([(cm[j].astype(object)+mm(d,bm[j])+mm(am[j],e)+int(alpha[j])*public)%PRIME for j in range(h)])
        # Mixed storage exercises all pointer branches, especially MAC/value planes.
        tensors=[torch.tensor(x,device="cuda",dtype=torch.int32 if i%2 else torch.int64) for i,x in enumerate(data)]
        actual_v,actual_m=authenticated_local(*tensors,party,packed=packed)
        assert actual_v.dtype==actual_m.dtype==(torch.int32 if packed else torch.int64)
        np.testing.assert_array_equal(actual_v.cpu().numpy(),np.asarray(value,dtype=np.int64))
        np.testing.assert_array_equal(actual_m.cpu().numpy(),np.asarray(macs,dtype=np.int64))


@pytest.mark.parametrize("h,n",[(1,1),(5,257),(10,65539)])
@pytest.mark.parametrize("packed",[False,True])
def test_independent_mac_reduction(h,n,packed):
    rng=np.random.default_rng(1900+n)
    macs=rng.integers(0,PRIME,(h,n),dtype=np.int64)
    opened=rng.integers(0,PRIME,n,dtype=np.int64)
    alpha=rng.integers(0,PRIME,h,dtype=np.int64)
    chi=rng.integers(0,PRIME,(h,n),dtype=np.int64)
    expected=np.asarray([sum((int(macs[j,i])-int(alpha[j])*int(opened[i]))*int(chi[j,i]) for i in range(n))%PRIME for j in range(h)],dtype=np.int64)
    inputs=[torch.tensor(x,device="cuda",dtype=torch.int32 if j%2==0 else torch.int64) for j,x in enumerate((macs,opened,alpha,chi))]
    actual=field_mac_check(*inputs,packed=packed)
    np.testing.assert_array_equal(actual.cpu().numpy(),expected)


def test_mac_challenges_must_have_independent_rows():
    with pytest.raises(ValueError,match="independent"):
        field_mac_check(torch.zeros((10,19),device="cuda",dtype=torch.int32),
                        torch.zeros(19,device="cuda",dtype=torch.int32),
                        torch.ones(10,device="cuda",dtype=torch.int32),
                        torch.ones(19,device="cuda",dtype=torch.int32))


def test_signed_mac_reduction_boundaries():
    values=np.asarray([-2**63,2**63-1,-1,0,1,PRIME,PRIME-1],dtype=np.int64)
    macs=np.stack((values,values[::-1])).copy()
    opened=values.copy()
    alpha=np.asarray([-2**63,2**63-1],dtype=np.int64)
    chi=np.stack((values[::-1],values)).copy()
    expected=np.asarray([sum((int(macs[j,i])-int(alpha[j])*int(opened[i]))*int(chi[j,i]) for i in range(len(values)))%PRIME for j in range(2)],dtype=np.int64)
    actual=field_mac_check(*(torch.from_numpy(x).cuda() for x in (macs,opened,alpha,chi)))
    np.testing.assert_array_equal(actual.cpu().numpy(),expected)


def test_authenticated_full_signed_int64():
    rng=np.random.default_rng(0x31ffff)
    h,m,k,n=2,3,9,7
    shapes=[(m,k),(k,n),(m,k),(h,m,k),(k,n),(h,k,n),(m,n),(h,m,n),(h,)]
    data=[rng.bit_generator.random_raw(np.prod(shape)).view(np.int64).reshape(shape) for shape in shapes]
    d,e,av,am,bv,bm,cv,cm,alpha=data
    public=mm(d,e)
    value=(cv.astype(object)+mm(d,bv)+mm(av,e)+public)%PRIME
    macs=np.stack([(cm[j].astype(object)+mm(d,bm[j])+mm(am[j],e)+int(alpha[j])*public)%PRIME for j in range(h)])
    actual=authenticated_local(*(torch.from_numpy(x).cuda() for x in data),0)
    np.testing.assert_array_equal(actual[0].cpu().numpy(),np.asarray(value,dtype=np.int64))
    np.testing.assert_array_equal(actual[1].cpu().numpy(),np.asarray(macs,dtype=np.int64))
