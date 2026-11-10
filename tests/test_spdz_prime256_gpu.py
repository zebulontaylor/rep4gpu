"""Exact GPU value/MAC products inside the unchanged checked q256 protocol."""
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import hashlib
import secrets
import pytest
import torch
from rep4gpu.spdz_prime256 import PRIME,LOWGEAR_PRIME,QShare,clear_matmul
from rep4gpu.spdz_prime256_gpu import GPUPrime256Party,pack_limbs,unpack_limbs
from rep4gpu.spdz_prime256_prepared import import_prepared_rows,matmul_prepared_rows,_states
from rep4gpu.transport import IdealNetwork
from test_spdz_prime256 import _share


def test_limb_encoding_extrema_matches_independent_integer_bytes():
    values=[0,1,65535,65536,2**255,PRIME-1]
    encoded=pack_limbs(values,(2,3))
    assert encoded.numpy().astype('<u2').tobytes()==b''.join(v.to_bytes(32,'little') for v in values)
    assert unpack_limbs(encoded)==values
    with pytest.raises(ValueError):pack_limbs([PRIME],(1,1))
    encoded[0,0,0]=65536
    with pytest.raises(RuntimeError,match='limb'):unpack_limbs(encoded)


def test_gpu_adapter_rejects_other_field_before_cuda_allocation():
    class NoCommunication:
        def exchange(self,*args,**kwargs):raise AssertionError('unexpected network operation')
    with pytest.raises(ValueError,match='require q='):
        GPUPrime256Party(0,NoCommunication(),7,session=b'wrong-field',field_prime=LOWGEAR_PRIME,
                         _allow_insecure_test_transport=True)


def _run(operation,*,k=2,n=3):
    if not torch.cuda.is_available():pytest.skip('CUDA required')
    alpha=[secrets.randbelow(PRIME) for _ in range(4)]
    xp=[0,PRIME-1,7,PRIME-2,2**255,19];wp=[PRIME-1,1,71,2**255,0,PRIME-2]
    x=[xp[i%len(xp)] for i in range(3*k)];w=[wp[i%len(wp)] for i in range(k*n)]
    a=[secrets.randbelow(PRIME) for _ in x]
    xs,ws,aa,ff=(_share(values,shape,alpha) for values,shape in ((x,(3,k)),(w,(k,n)),(a,(3,k)),(clear_matmul(a,w,(3,k),(k,n)),(3,n))))
    identifier=secrets.token_bytes(32);model_id=hashlib.sha256(b'GPU-fixed-model').digest()
    network=IdealNetwork(timeout=60)
    parties=[GPUPrime256Party(p,network.endpoint(p),alpha[p],session=b'checked-GPU-products',
                             _allow_insecure_test_transport=True) for p in range(4)]
    def role(p):
        try:
            batch=SimpleNamespace(weights=ws[p],a=aa[p],factor=ff[p],identifier=identifier,model_id=model_id,used=False)
            handles=import_prepared_rows(parties[p],batch,(1,2),model_id=model_id)
            return operation(parties[p],xs[p],handles,model_id)
        except BaseException as error:network.abort(error);return error
    with ThreadPoolExecutor(4) as pool:results=list(pool.map(role,range(4)))
    return results,parties,clear_matmul(x,w,(3,k),(k,n))


def test_four_checked_gpu_parties_match_full_ring_bigint_and_share_one_weight_snapshot():
    def operation(p,x,handles,model):
        p.prepare_handles(handles)
        assert len(p._gpu_weights)==1 and p.gpu_setup_ms>0
        output=[];offset=0
        for handle,count in zip(handles,(1,2)):
            query=QShare(x.values[offset*2:(offset+count)*2],x.macs[offset*2:(offset+count)*2],(count,2))
            output+=p.open(matmul_prepared_rows(p,query,handle,model_id=model));offset+=count
        assert p.gpu_product_calls==2 and p.gpu_query_product_ms>0
        return output
    results,parties,expected=_run(operation)
    assert results==[expected]*4 and all(not p.poisoned for p in parties)


@pytest.mark.parametrize('fault',['unprepared','snapshot','query-mac'])
def test_gpu_failure_retires_query_before_any_output(fault):
    def operation(p,x,handles,model):
        if fault!='unprepared':p.prepare_handles(handles)
        if fault=='snapshot':next(iter(p._gpu_weights.values())).prepared.digits.fill_(0)
        query=QShare(x.values[:2],x.macs[:2],(1,2))
        if fault=='query-mac' and p.party==0:query.macs[0]=(query.macs[0]+1)%PRIME
        result=matmul_prepared_rows(p,query,handles[0],model_id=model)
        raise AssertionError('fault produced an unchecked result')
    results,parties,_=_run(operation)
    assert all(isinstance(result,BaseException) and not isinstance(result,AssertionError) for result in results)
    assert all(p.poisoned for p in parties)



@pytest.mark.parametrize('k,n',[(2,0),(0,3),(0,0)])
def test_empty_gpu_products_finish_events_and_match_checked_bigint(k,n):
    def operation(p,x,handles,model):
        p.prepare_handles(handles)
        result=[];offset=0
        for handle,count in zip(handles,(1,2)):
            query=QShare(x.values[offset*k:(offset+count)*k],x.macs[offset*k:(offset+count)*k],(count,k))
            output=matmul_prepared_rows(p,query,handle,model_id=model)
            assert output.shape==(count,n)
            result+=p.open(output);offset+=count
        assert p.gpu_product_calls==2 and p._gpu_stop.query()
        assert all(sample['device_stream_interval_ms']>=0 for sample in p.gpu_product_samples)
        return result
    results,parties,expected=_run(operation,k=k,n=n)
    assert results==[expected]*4 and all(not party.poisoned for party in parties)
