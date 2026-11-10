import numpy as np
import pytest
import torch

from rep4gpu.field_kernels import PRIME
from rep4gpu.field_weighted import field_weighted_check
from rep4gpu.field_weighted_accumulate import field_weighted_accumulate

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


@pytest.mark.parametrize('h,n',[(7,0),(7,1),(7,257),(10,65539)])
@pytest.mark.parametrize('packed',[False,True])
def test_projection_accumulation_bigint(h,n,packed):
    rng=np.random.default_rng(3719+n+h)
    total_data=rng.bit_generator.random_raw(h).view(np.int64)
    dtype=torch.int32 if packed else torch.int64
    if packed:total_data=(total_data%PRIME).astype(np.int32)
    total=torch.tensor(total_data,device='cuda',dtype=dtype)
    failure=torch.tensor([2],device='cuda',dtype=torch.int32)
    expected=total_data.astype(object)
    frozen=[]
    for rejected_value in (0,1,-2147483648):
        residual=rng.bit_generator.random_raw(h*n).view(np.int64).reshape(h,n)
        challenge=rng.bit_generator.random_raw(h*n).view(np.int64).reshape(h,n)
        expected=np.asarray([(int(expected[j])+sum(int(a)*int(b) for a,b in zip(residual[j],challenge[j])))%PRIME for j in range(h)],dtype=object)
        r=torch.from_numpy(residual).cuda()
        chi=torch.from_numpy(challenge).cuda()
        rejected=torch.tensor([rejected_value],device='cuda',dtype=torch.int32)
        snapshots=(r.clone(),chi.clone(),rejected.clone())
        result=field_weighted_accumulate(r,chi,total,failure,rejected)
        assert result[0] is total and result[1] is failure
        np.testing.assert_array_equal(total.cpu().numpy(),np.asarray(expected,dtype=np.int64))
        for actual,before in zip((r,chi,rejected),snapshots):
            torch.testing.assert_close(actual,before,rtol=0,atol=0)
        frozen.append(rejected_value)
        assert failure.item()==int(np.bitwise_or.reduce(np.asarray([2,*frozen],dtype=np.int32)))


def test_compact_rows_match_existing_helper():
    r=torch.randint(PRIME,(7,1031),device='cuda',dtype=torch.int32)
    chi=torch.randint(PRIME,(7,1031),device='cuda',dtype=torch.int32)
    total=torch.full((7,),PRIME-1,device='cuda',dtype=torch.int32)
    expected=(field_weighted_check(r,chi).long()+total.long()).remainder(PRIME).int()
    field_weighted_accumulate(r,chi,total,torch.zeros(1,device='cuda',dtype=torch.int32),torch.zeros(1,device='cuda',dtype=torch.int32))
    torch.testing.assert_close(total,expected,rtol=0,atol=0)


def test_rejects_shared_challenge_and_aliases():
    r=torch.ones((7,9),device='cuda',dtype=torch.int32)
    chi=r.clone()
    total=torch.zeros(7,device='cuda',dtype=torch.int32)
    failure=torch.zeros(1,device='cuda',dtype=torch.int32)
    rejected=failure.clone()
    with pytest.raises(ValueError,match='independent'):
        field_weighted_accumulate(r,chi[0],total,failure,rejected)
    with pytest.raises(ValueError,match='alias'):
        field_weighted_accumulate(r,chi,total,failure,failure)
    with pytest.raises(ValueError,match='alias'):
        field_weighted_accumulate(r,chi,r.flatten()[:7],failure,rejected)


def test_signed_extremes_and_prior_total():
    values=np.asarray([-(2**63),2**63-1,-1,0,1,PRIME,PRIME-1],dtype=np.int64)
    residual=np.tile(values,(7,1))
    challenge=np.tile(values[::-1],(7,1)).copy()
    expected=np.asarray([(int(v)+sum(int(a)*int(b) for a,b in zip(residual[j],challenge[j])))%PRIME for j,v in enumerate(values)],dtype=np.int64)
    total=torch.from_numpy(values.copy()).cuda()
    failure=torch.tensor([-1],device='cuda',dtype=torch.int32)
    field_weighted_accumulate(torch.from_numpy(residual).cuda(),torch.from_numpy(challenge).cuda(),total,failure,torch.zeros(1,device='cuda',dtype=torch.int32))
    np.testing.assert_array_equal(total.cpu().numpy(),expected)
    assert failure.item()==-1
