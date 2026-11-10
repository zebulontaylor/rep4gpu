"""Disjoint one-use query rows from one model-dependent batch."""
import pytest
import torch
from rep4gpu.spdz_distributed import AuthShare,ProtocolError
from rep4gpu.spdz_prepared_right import partition_prepared_right,matmul_prepared_right,_states
from test_spdz_prepared_right import _run,_prepare


@pytest.mark.parametrize('counts',[(1,1,1,1),(1,3),(2,2),(4,)])
def test_partition_rows_match_independent_queries(counts):
    def operation(p,x,w,t):
        parent=_prepare(p,w,t,model_id=b'batch')
        before=p.round
        children=partition_prepared_right(p,parent,counts,model_id=b'batch')
        assert p.round==before and parent.used
        assert len({child.triple_id for child in children})==len(counts)
        assert all(_states[child].weights is _states[parent].weights for child in children)
        offset=0;outputs=[]
        for child,count in zip(children,counts):
            query=AuthShare(x.values[offset:offset+count],x.macs[:,offset:offset+count])
            outputs.append(p.open(matmul_prepared_right(p,query,child,model_id=b'batch')))
            offset+=count
        return torch.cat(outputs)
    results,roles,_,expected=_run(operation,m=4)
    assert all(isinstance(result,torch.Tensor) and torch.equal(result,expected) for result in results)
    assert all(not p.poisoned for p in roles)


@pytest.mark.parametrize('counts',[(),(0,4),(-1,5),(True,3),(1,2),(5,),('4',)])
def test_invalid_partition_burns_parent_and_poison(counts):
    def operation(p,x,w,t):
        parent=_prepare(p,w,t,model_id=b'batch')
        before=p.round
        with pytest.raises(ValueError):partition_prepared_right(p,parent,counts,model_id=b'batch')
        assert parent.used and p.poisoned and p.round==before
        return True
    results,_,_,_=_run(operation,m=4)
    assert results==[True]*4


@pytest.mark.parametrize('kind',['parent-reuse','child-reuse','key','session','model','parent-storage','child-storage','shared-weight'])
def test_partition_capability_guards(kind):
    def operation(p,x,w,t):
        parent=_prepare(p,w,t,model_id=b'batch')
        children=partition_prepared_right(p,parent,(1,1),model_id=b'batch')
        child=children[0]
        query=AuthShare(x.values[:1],x.macs[:,:1])
        if kind=='child-reuse':
            matmul_prepared_right(p,query,child,model_id=b'batch')
            p._test_query_ready.wait(timeout=10)
        if kind=='key':p.alpha_share=p.alpha_share.clone()
        if kind=='session':p.session=b'wrong'
        if kind=='parent-storage':
            _states[parent].a.values.fill_(0)
            _states[parent].factor.macs.fill_(0)
            # Independent child row snapshots survive mutations to consumed A/F.
            return p.open(matmul_prepared_right(p,query,child,model_id=b'batch'))
        if kind=='child-storage':_states[child].a.values.fill_(0)
        if kind=='shared-weight':_states[parent].weights.macs.fill_(0)
        before=p.round
        with pytest.raises((ValueError,ProtocolError)):
            if kind=='parent-reuse':partition_prepared_right(p,parent,(2,),model_id=b'batch')
            else:matmul_prepared_right(p,query,child,model_id=b'wrong' if kind=='model' else b'batch')
        assert p.poisoned and p.round==before
        if kind!='parent-reuse':assert child.used
        return True
    results,_,_,expected=_run(operation)
    if kind=='parent-storage':assert all(torch.equal(result,expected[:1]) for result in results)
    else:assert results==[True]*4


def test_different_partitions_fail_before_candidate_or_output():
    from unittest.mock import patch
    def operation(p,x,w,t):
        parent=_prepare(p,w,t,model_id=b'batch')
        child=partition_prepared_right(p,parent,(1,3) if p.party==0 else (1,1,2),model_id=b'batch')[0]
        return matmul_prepared_right(p,AuthShare(x.values[:1],x.macs[:,:1]),child,model_id=b'batch')
    with patch('rep4gpu.spdz_prepared_right._online',side_effect=AssertionError('unchecked candidate used')):
        results,roles,_,_=_run(operation,m=4)
    assert all(isinstance(result,BaseException) and not isinstance(result,AssertionError) for result in results)
    assert all(p.poisoned for p in roles)


@pytest.mark.parametrize('backend',['full','deferred'])
def test_device_partition_rows_match_product(backend):
    if not torch.cuda.is_available():pytest.skip('CUDA required')
    from rep4gpu.spdz_device import DeviceDistributedParty
    from rep4gpu.spdz_deferred import DeviceDeferredCompressedParty
    party_type=DeviceDistributedParty if backend=='full' else DeviceDeferredCompressedParty
    def operation(p,x,w,t):
        parent=_prepare(p,w,t,model_id=b'CUDA-batch')
        children=partition_prepared_right(p,parent,(1,2),model_id=b'CUDA-batch')
        outputs=[];offset=0
        for child,count in zip(children,(1,2)):
            query=AuthShare(x.values[offset:offset+count],x.macs[:,offset:offset+count])
            outputs.append(p.open(matmul_prepared_right(p,query,child,model_id=b'CUDA-batch')))
            offset+=count
        return torch.cat(outputs)
    results,roles,_,expected=_run(operation,m=3,k=5,n=7,device='cuda',party_type=party_type)
    assert all(isinstance(result,torch.Tensor) and torch.equal(result,expected) for result in results)
    assert all(not p.poisoned for p in roles)
