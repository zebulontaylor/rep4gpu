import numpy as np
import pytest
import torch

from rep4gpu.field_kernels import PRIME
from rep4gpu.field_prepared import prepared_right_online
from rep4gpu.field_prepared_canonical import prepared_right_online_canonical


@pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA required")
@pytest.mark.parametrize("m,h",[(1,1),(1,10),(3,10)])
@pytest.mark.parametrize("packed",[False,True])
def test_canonical_decode_big_integer(m,h,packed):
    k,n=137,71
    rng=np.random.default_rng(0x31c0+m+h)
    shapes=[(m,k),(k,n),(h,k,n),(m,n),(h,m,n)]
    data=[rng.integers(0,PRIME,s,dtype=np.int64) for s in shapes]
    d,wv,wm,fv,fm=data
    mm=lambda a,b:(a.astype(object)@b.astype(object))%PRIME
    want_v=(fv.astype(object)+mm(d,wv))%PRIME
    want_m=np.stack([(fm[j].astype(object)+mm(d,wm[j]))%PRIME for j in range(h)])
    tensors=[torch.tensor(x,device="cuda",dtype=torch.int32 if j%2==0 else torch.int64) for j,x in enumerate(data)]
    got=prepared_right_online_canonical(*tensors,packed=packed)
    baseline=prepared_right_online(*tensors,packed=packed)
    for a,b,want in zip(got,baseline,(want_v,want_m)):
        torch.testing.assert_close(a,b,rtol=0,atol=0)
        np.testing.assert_array_equal(a.cpu().numpy(),np.asarray(want,dtype=np.int64))


@pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA required")
@pytest.mark.parametrize("k",[0,1,8193])
def test_maximum_canonical_products(k):
    h,m,n=10,1,17
    shapes=[(m,k),(k,n),(h,k,n),(m,n),(h,m,n)]
    tensors=[torch.full(s,PRIME-1,device="cuda",dtype=torch.int32) for s in shapes]
    got=prepared_right_online_canonical(*tensors)
    expected=(PRIME-1+k)%PRIME
    for value in got:
        torch.testing.assert_close(value,torch.full_like(value,expected),rtol=0,atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA required")
def test_general_signed_fallback_is_unchanged():
    h,m,k,n=2,1,9,7
    rng=np.random.default_rng(1779)
    shapes=[(m,k),(k,n),(h,k,n),(m,n),(h,m,n)]
    data=[rng.bit_generator.random_raw(np.prod(s)).view(np.int64).reshape(s) for s in shapes]
    d,wv,wm,fv,fm=data
    mm=lambda a,b:(a.astype(object)@b.astype(object))%PRIME
    want=[(fv.astype(object)+mm(d,wv))%PRIME,
          np.stack([(fm[j].astype(object)+mm(d,wm[j]))%PRIME for j in range(h)])]
    actual=prepared_right_online(*(torch.from_numpy(x).cuda() for x in data))
    for a,b in zip(actual,want):
        np.testing.assert_array_equal(a.cpu().numpy(),np.asarray(b,dtype=np.int64))
