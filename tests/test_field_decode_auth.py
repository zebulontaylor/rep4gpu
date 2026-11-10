import numpy as np
import pytest
import torch

from rep4gpu.field_decode_auth import authenticated_local_decode
from rep4gpu.field_kernels import PRIME

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA required")


@pytest.mark.parametrize("m,h",[(1,1),(1,10),(3,10)])
@pytest.mark.parametrize("packed",[False,True])
def test_independent_big_integer_decode(m,h,packed):
    k,n=71,39
    rng=np.random.default_rng(913+h+m)
    shapes=[(m,k),(k,n),(m,k),(h,m,k),(k,n),(h,k,n),(m,n),(h,m,n),(h,)]
    data=[rng.integers(0,PRIME,s,dtype=np.int64) for s in shapes]
    d,e,av,am,bv,bm,cv,cm,alpha=data
    matmul=lambda a,b:(a.astype(object)@b.astype(object))%PRIME
    public=matmul(d,e)
    for party in (0,3):
        want_v=(cv.astype(object)+matmul(d,bv)+matmul(av,e)+(public if party==0 else 0))%PRIME
        want_m=np.stack([(cm[j].astype(object)+matmul(d,bm[j])+matmul(am[j],e)+int(alpha[j])*public)%PRIME for j in range(h)])
        tensors=[torch.tensor(x,device="cuda",dtype=torch.int32 if j%2 else torch.int64) for j,x in enumerate(data)]
        value,macs=authenticated_local_decode(*tensors,party,packed=packed)
        np.testing.assert_array_equal(value.cpu().numpy(),np.asarray(want_v,dtype=np.int64))
        np.testing.assert_array_equal(macs.cpu().numpy(),np.asarray(want_m,dtype=np.int64))


def test_full_signed_int64_decode():
    h,m,k,n=2,1,9,7
    rng=np.random.default_rng(813)
    shapes=[(m,k),(k,n),(m,k),(h,m,k),(k,n),(h,k,n),(m,n),(h,m,n),(h,)]
    data=[rng.bit_generator.random_raw(np.prod(s)).view(np.int64).reshape(s) for s in shapes]
    d,e,av,am,bv,bm,cv,cm,alpha=data
    mm=lambda a,b:(a.astype(object)@b.astype(object))%PRIME
    public=mm(d,e)
    value=(cv.astype(object)+mm(d,bv)+mm(av,e)+public)%PRIME
    macs=np.stack([(cm[j].astype(object)+mm(d,bm[j])+mm(am[j],e)+int(alpha[j])*public)%PRIME for j in range(h)])
    got=authenticated_local_decode(*(torch.from_numpy(x).cuda() for x in data),0)
    np.testing.assert_array_equal(got[0].cpu().numpy(),np.asarray(value,dtype=np.int64))
    np.testing.assert_array_equal(got[1].cpu().numpy(),np.asarray(macs,dtype=np.int64))
