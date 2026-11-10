"""Exercise prepared model capabilities with immediate and deferred GPU MACs."""
import functools

import pytest
import torch

import test_spdz_prepared_right as checks


CASES = [
    ('test_prepared_right_matches_exact_product_and_opens_only_query_difference',(10,(1,3,7))),
    ('test_prepared_right_matches_exact_product_and_opens_only_query_difference',(10,(3,5,2))),
    ('test_preparation_owns_immutable_correlation_and_model_snapshots',()),
    *[('test_prepared_capability_failure_consumes_handle_and_poison_session',(kind,))
      for kind in ('reuse','model','shape','key','session')],
    ('test_prepared_handles_cannot_be_constructed_serialized_or_forged',()),
    ('test_corrupted_opening_cannot_produce_ready_handle_or_online_output',('prepare',)),
    ('test_corrupted_opening_cannot_produce_ready_handle_or_online_output',('online',)),
    ('test_original_triple_cannot_be_used_after_model_preparation',()),
    ('test_prepared_capabilities_cannot_transfer_between_live_roles',()),
]


@pytest.mark.parametrize('backend',['full','compressed','deferred'])
@pytest.mark.parametrize('name,args',CASES,ids=[name.removeprefix('test_')+str(args) for name,args in CASES])
def test_prepared_gpu_model_capability(monkeypatch,backend,name,args):
    if not torch.cuda.is_available():
        pytest.skip('CUDA required')
    from rep4gpu.spdz_device import DeviceDistributedParty
    from rep4gpu.spdz_compression import DeviceCompressedParty
    from rep4gpu.spdz_deferred import DeviceDeferredCompressedParty
    party_type = dict(full=DeviceDistributedParty,compressed=DeviceCompressedParty,
                      deferred=DeviceDeferredCompressedParty)[backend]
    monkeypatch.setattr(checks,'_run',functools.partial(checks._run,device='cuda',party_type=party_type))
    if name == 'test_prepared_right_matches_exact_product_and_opens_only_query_difference':
        from rep4gpu.spdz_prepared_right import prepare_right_matmul,matmul_prepared_right
        keys,(m,k,n) = args
        def operation(p,x,w,triple):
            prepared = prepare_right_matmul(p,w,triple,model_id=b'gpu-model-v1')
            assert triple.used and not prepared.used
            output = matmul_prepared_right(p,x,prepared,model_id=b'gpu-model-v1')
            assert prepared.used
            opened = p.open(output)
            assert not getattr(p,'pending_values',0)
            return opened
        results,roles,_,expected = checks._run(operation,keys=keys,m=m,k=k,n=n)
        assert all(isinstance(value,torch.Tensor) and torch.equal(value,expected) for value in results)
        assert all(not role.poisoned for role in roles)
    else:
        getattr(checks,name)(*args)
