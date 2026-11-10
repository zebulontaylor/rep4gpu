import numpy as np
import pytest
import torch

from rep4gpu.field_kernels import PRIME
from rep4gpu.field_prepared import prepared_right_factor,prepared_right_online


def reference(a,b):
    return (a.astype(object)@b.astype(object))%PRIME


@pytest.mark.parametrize("device",["cpu","cuda"])
@pytest.mark.parametrize("m,h",[(1,1),(1,10),(17,10)])
@pytest.mark.parametrize("packed",[False,True])
def test_factor_online_exact(device,m,h,packed):
    if device=="cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    k,n=71,39
    rng=np.random.default_rng(1631+h+m)
    shapes=[(m,k),(h,m,k),(m,n),(h,m,n),(k,n),(m,k),(k,n),(h,k,n)]
    data=[rng.integers(0,PRIME,s,dtype=np.int64) for s in shapes]
    av,am,cv,cm,e,d,wv,wm=data
    fv=(cv.astype(object)+reference(av,e))%PRIME
    fm=np.stack([(cm[j].astype(object)+reference(am[j],e))%PRIME for j in range(h)])
    zv=(fv+reference(d,wv))%PRIME
    zm=np.stack([(fm[j]+reference(d,wm[j]))%PRIME for j in range(h)])
    sources=[torch.tensor(x,device=device,dtype=torch.int32 if j%2==0 else torch.int64) for j,x in enumerate(data)]
    av,am,cv,cm,e,d,wv,wm=sources
    got_fv,got_fm=prepared_right_factor(av,am,cv,cm,e,packed=packed)
    got_zv,got_zm=prepared_right_online(d,wv,wm,got_fv,got_fm,packed=packed)
    for actual,want in [(got_fv,fv),(got_fm,fm),(got_zv,zv),(got_zm,zm)]:
        assert actual.dtype==(torch.int32 if packed else torch.int64)
        np.testing.assert_array_equal(actual.cpu().numpy(),np.asarray(want,dtype=np.int64))


@pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA required")
@pytest.mark.parametrize("backend",["decode","tensor"])
def test_full_signed_int64_prepared(backend):
    h,m,k,n=2,3,9,7
    rng=np.random.default_rng(0x31ffff)
    shapes=[(m,k),(h,m,k),(m,n),(h,m,n),(k,n),(m,k),(k,n),(h,k,n)]
    data=[rng.bit_generator.random_raw(np.prod(s)).view(np.int64).reshape(s) for s in shapes]
    av,am,cv,cm,e,d,wv,wm=data
    fv=(cv.astype(object)+reference(av,e))%PRIME
    fm=np.stack([(cm[j].astype(object)+reference(am[j],e))%PRIME for j in range(h)])
    zv=(fv+reference(d,wv))%PRIME
    zm=np.stack([(fm[j]+reference(d,wm[j]))%PRIME for j in range(h)])
    av,am,cv,cm,e,d,wv,wm=[torch.from_numpy(x).cuda() for x in data]
    got_fv,got_fm=prepared_right_factor(av,am,cv,cm,e,backend=backend)
    got_zv,got_zm=prepared_right_online(d,wv,wm,got_fv,got_fm,backend=backend)
    for actual,want in [(got_fv,fv),(got_fm,fm),(got_zv,zv),(got_zm,zm)]:
        np.testing.assert_array_equal(actual.cpu().numpy(),np.asarray(want,dtype=np.int64))


@pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA required")
@pytest.mark.parametrize("backend",["decode","tensor"])
def test_residue_near_int32_max_never_enters_saturating_mma(backend):
    h,m,k,n=10,17,71,39
    a=torch.full((m,k),127,device="cuda",dtype=torch.int32)
    am=torch.full((h,m,k),127,device="cuda",dtype=torch.int32)
    b=torch.full((k,n),127,device="cuda",dtype=torch.int32)
    bm=torch.full((h,k,n),127,device="cuda",dtype=torch.int32)
    c=torch.full((m,n),PRIME-1,device="cuda",dtype=torch.int32)
    cm=torch.full((h,m,n),PRIME-1,device="cuda",dtype=torch.int32)
    expected=(PRIME-1+k*127*127)%PRIME
    for outputs in (prepared_right_factor(a,am,c,cm,b,backend=backend),
                    prepared_right_online(a,b,bm,c,cm,backend=backend)):
        for output in outputs:
            torch.testing.assert_close(output,torch.full_like(output,expected),rtol=0,atol=0)
