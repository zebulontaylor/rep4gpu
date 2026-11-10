import numpy as np
import pytest
import torch

from rep4gpu.field_weighted import field_weighted_check
from rep4gpu.field_kernels import PRIME

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA required")


@pytest.mark.parametrize("shape",[(1,0),(1,1),(10,257),(10,65539)])
@pytest.mark.parametrize("packed",[False,True])
def test_exact_weighted_residuals(shape,packed):
    rng=np.random.default_rng(713+shape[1])
    h,n=shape
    residual=rng.bit_generator.random_raw(h*n).view(np.int64).reshape(shape)
    chi=rng.bit_generator.random_raw(h*n).view(np.int64).reshape(shape)
    expected=np.asarray([sum(int(residual[j,i])*int(chi[j,i]) for i in range(n))%PRIME for j in range(h)],dtype=np.int64)
    actual=field_weighted_check(torch.from_numpy(residual).cuda(),torch.from_numpy(chi).cuda(),packed=packed)
    assert actual.dtype==(torch.int32 if packed else torch.int64)
    np.testing.assert_array_equal(actual.cpu().numpy(),expected)


def test_compact_and_independent_rows():
    r=torch.tensor([[0,PRIME-1,1],[1,2,3]],device="cuda",dtype=torch.int32)
    chi=torch.tensor([[1,2,3],[4,5,6]],device="cuda",dtype=torch.int32)
    expected=np.asarray([sum(int(a)*int(b) for a,b in zip(x,y))%PRIME for x,y in zip(r.cpu().tolist(),chi.cpu().tolist())],dtype=np.int64)
    np.testing.assert_array_equal(field_weighted_check(r,chi).cpu().numpy(),expected)
    with pytest.raises(ValueError,match="independent"):
        field_weighted_check(r,chi[0])
