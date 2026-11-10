import numpy as np
import pytest
import torch

from rep4gpu.field_fast import field_matmul,PRIME

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA required")


@pytest.mark.parametrize("dtype",[torch.int32,torch.int64])
@pytest.mark.parametrize("packed",[False,True])
@pytest.mark.parametrize("backend",["tensor","decode"])
def test_exact_inputs_outputs(dtype,packed,backend):
    rng=np.random.default_rng(1771)
    low,high=(-2**31,2**31) if dtype==torch.int32 else (-2**63,2**63-1)
    a=rng.integers(low,high,(17,71),dtype=np.int64)
    b=rng.integers(low,high,(71,39),dtype=np.int64)
    out=field_matmul(torch.tensor(a,device="cuda",dtype=dtype),torch.tensor(b,device="cuda",dtype=dtype),packed=packed,backend=backend)
    expected=np.asarray((a.astype(object)@b.astype(object))%PRIME,dtype=np.int64)
    assert out.dtype==(torch.int32 if packed else torch.int64)
    np.testing.assert_array_equal(out.cpu().numpy(),expected)


def test_full_int64_boundaries():
    v=torch.tensor([-2**63,2**63-1,-1,0,1,PRIME-1,PRIME,PRIME//2,PRIME//2+1],dtype=torch.int64)
    a=v.repeat(17,3)
    b=v.repeat(19,3).T.contiguous()
    expected=np.asarray((a.numpy().astype(object)@b.numpy().astype(object))%PRIME,dtype=np.int64)
    np.testing.assert_array_equal(field_matmul(a.cuda(),b.cuda(),packed=True).cpu().numpy(),expected)
