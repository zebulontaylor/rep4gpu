import numpy as np
import pytest
import torch

from rep4gpu.field_kernels import PRIME,field_matmul


def reference(a,b):
    return np.asarray((a.astype(object)@b.astype(object))%PRIME,dtype=np.int64)


@pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA required")
@pytest.mark.parametrize("backend",["tensor","decode"])
@pytest.mark.parametrize("shape",[(1,71,39),(17,137,71),(128,128,128),(3,0,7)])
def test_full_field(backend,shape):
    rng=np.random.default_rng(1203)
    m,k,n=shape
    a=rng.integers(0,PRIME,(m,k),dtype=np.int64)
    b=rng.integers(0,PRIME,(k,n),dtype=np.int64)
    actual=field_matmul(torch.from_numpy(a).cuda(),torch.from_numpy(b).cuda(),backend=backend)
    np.testing.assert_array_equal(actual.cpu().numpy(),reference(a,b))


@pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA required")
@pytest.mark.parametrize("backend",["tensor","decode"])
def test_boundaries_and_noncanonical(backend):
    v=np.asarray([0,1,PRIME-1,PRIME,PRIME+1,PRIME//2,PRIME//2+1,-1,-PRIME,-2**63,2**63-1],dtype=np.int64)
    a=np.tile(v,(17,3))
    b=np.tile(v,(23,3)).T.copy()
    actual=field_matmul(torch.from_numpy(a).cuda(),torch.from_numpy(b).cuda(),backend=backend)
    np.testing.assert_array_equal(actual.cpu().numpy(),reference(a,b))


def test_cpu_fallback():
    a=torch.tensor([[PRIME-1,PRIME//2,-1]],dtype=torch.int64)
    b=torch.tensor([[PRIME-1],[PRIME//2],[PRIME+1]],dtype=torch.int64)
    np.testing.assert_array_equal(field_matmul(a,b).numpy(),reference(a.numpy(),b.numpy()))


@pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA required")
@pytest.mark.parametrize("backend",["tensor","decode"])
def test_random_signed_int64_normalization(backend):
    rng=np.random.default_rng(719)
    a=rng.bit_generator.random_raw(17*71).view(np.int64).reshape(17,71)
    b=rng.bit_generator.random_raw(71*39).view(np.int64).reshape(71,39)
    actual=field_matmul(torch.from_numpy(a).cuda(),torch.from_numpy(b).cuda(),backend=backend)
    np.testing.assert_array_equal(actual.cpu().numpy(),reference(a,b))
