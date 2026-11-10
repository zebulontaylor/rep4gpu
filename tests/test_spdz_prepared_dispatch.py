"""Check the prepared-only canonical decode dispatch without allocating a GPU."""
import pytest
import torch

from rep4gpu.spdz_distributed import AuthShare
from rep4gpu.spdz_prepared_right import _online


class _CudaMetadata(torch.Tensor):
    @property
    def is_cuda(self):
        return True


def _metadata(shape):
    return torch.empty(shape,device='meta',dtype=torch.int32).as_subclass(_CudaMetadata)


@pytest.mark.parametrize('m,k,expected',[(1,7,'canonical'),(4,7,'canonical'),(5,7,'general'),
                                         (1,2**31-1,'canonical'),(1,2**31,'general')])
def test_gpu_prepared_dispatch_respects_decode_and_accumulation_bounds(monkeypatch,m,k,expected):
    calls = []
    result = (torch.zeros((m,1),dtype=torch.int32),torch.zeros((10,m,1),dtype=torch.int32))
    def canonical(*args,**kwargs):
        calls.append('canonical')
        return result
    def general(*args,**kwargs):
        calls.append('general')
        return result
    monkeypatch.setattr('rep4gpu.field_prepared_canonical.prepared_right_online_canonical',canonical)
    monkeypatch.setattr('rep4gpu.field_prepared.prepared_right_online',general)
    weights = AuthShare(_metadata((k,1)),_metadata((10,k,1)))
    factor = AuthShare(_metadata((m,1)),_metadata((10,m,1)))
    actual = _online(_metadata((m,k)),weights,factor)
    assert calls == [expected]
    assert actual.values is result[0] and actual.macs is result[1]


def test_cpu_prepared_arithmetic_does_not_enter_either_gpu_helper(monkeypatch):
    def forbidden(*args,**kwargs):
        raise AssertionError('CPU path entered a GPU helper')
    monkeypatch.setattr('rep4gpu.field_prepared_canonical.prepared_right_online_canonical',forbidden)
    monkeypatch.setattr('rep4gpu.field_prepared.prepared_right_online',forbidden)
    d = torch.tensor([[2,3]],dtype=torch.int64)
    weights = AuthShare(torch.tensor([[5],[7]],dtype=torch.int64),torch.full((10,2,1),11,dtype=torch.int64))
    factor = AuthShare(torch.tensor([[13]],dtype=torch.int64),torch.full((10,1,1),17,dtype=torch.int64))
    actual = _online(d,weights,factor)
    assert torch.equal(actual.values,torch.tensor([[44]],dtype=torch.int64))
    assert torch.equal(actual.macs,torch.full((10,1,1),72,dtype=torch.int64))
