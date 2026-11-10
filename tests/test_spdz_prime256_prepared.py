"""Independent exact arithmetic and consumption tests for native fixed-right rows."""
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import hashlib
import threading
import secrets
import pytest
from rep4gpu.spdz_prime256 import PRIME,QShare,Prime256Party,clear_matmul
from rep4gpu.spdz_prime256_prepared import import_prepared_rows,matmul_prepared_rows,_states
from rep4gpu.transport import IdealNetwork
from test_spdz_prime256 import _share


def _run(operation):
    alpha=[secrets.randbelow(PRIME) for _ in range(4)]
    x=[0,1,PRIME-1,7,PRIME-2,19];w=[PRIME-1,7,11,0,1,31]
    a=[secrets.randbelow(PRIME) for _ in x]
    xs,ws,aa,ff=(_share(values,shape,alpha) for values,shape in ((x,(3,2)),(w,(2,3)),(a,(3,2)),(clear_matmul(a,w,(3,2),(2,3)),(3,3))))
    identifier=secrets.token_bytes(32);model_id=hashlib.sha256(b'fixed-model').digest()
    batches=[SimpleNamespace(weights=ws[p],a=aa[p],factor=ff[p],identifier=identifier,model_id=model_id,used=False) for p in range(4)]
    network=IdealNetwork(timeout=10)
    parties=[Prime256Party(p,network.endpoint(p),alpha[p],session=b'prepared-native-tests',_allow_insecure_test_transport=True) for p in range(4)]
    def role(p):
        try:return operation(parties[p],xs[p],batches[p],model_id)
        except BaseException as error:network.abort(error);return error
    with ThreadPoolExecutor(4) as pool:results=list(pool.map(role,range(4)))
    return results,parties,clear_matmul(x,w,(3,2),(2,3))


@pytest.mark.parametrize('counts',[(1,1,1),(1,2),(3,)])
def test_fixed_right_rows_match_bigint_oracle_and_own_snapshots(counts):
    def operation(p,x,batch,model):
        handles=import_prepared_rows(p,batch,counts,model_id=model)
        assert batch.used and p.round==0
        assert all(_states[h].weights is _states[handles[0]].weights for h in handles)
        for share in (batch.weights,batch.a,batch.factor):share.values[:]=[0]*len(share.values);share.macs[:]=[0]*len(share.macs)
        output=[];offset=0
        for handle,count in zip(handles,counts):
            query=QShare(x.values[offset*2:(offset+count)*2],x.macs[offset*2:(offset+count)*2],(count,2))
            output+=p.open(matmul_prepared_rows(p,query,handle,model_id=model));offset+=count
        return output
    results,parties,expected=_run(operation)
    assert results==[expected]*4 and all(not p.poisoned for p in parties)


@pytest.mark.parametrize('fault',['parent-reuse','child-reuse','model','key','session','partition','query-mac'])
def test_fixed_right_guard_faults_poison_before_output(fault):
    ready=threading.Barrier(4)
    def operation(p,x,batch,model):
        counts=(1,2) if fault!='partition' or p.party==0 else (1,1,1)
        handle=import_prepared_rows(p,batch,counts,model_id=model)[0]
        if fault=='parent-reuse':return import_prepared_rows(p,batch,(3,),model_id=model)
        query=QShare(x.values[:2],x.macs[:2],(1,2))
        if fault=='child-reuse':
            p.open(matmul_prepared_rows(p,query,handle,model_id=model));ready.wait(timeout=10)
        if fault=='key':p.alpha_share=(p.alpha_share+1)%PRIME
        if fault=='session':p.session=b'other'
        if fault=='query-mac' and p.party==0:query.macs[0]=(query.macs[0]+1)%PRIME
        return matmul_prepared_rows(p,query,handle,model_id=b'wrong' if fault=='model' else model)
    results,parties,_=_run(operation)
    assert all(isinstance(result,BaseException) for result in results)
    assert all(p.poisoned for p in parties)


@pytest.mark.parametrize('counts',[(),(0,3),(True,2),(1,1),(4,)])
def test_fixed_right_invalid_partition_consumes_batch(counts):
    def operation(p,x,batch,model):
        with pytest.raises(ValueError):import_prepared_rows(p,batch,counts,model_id=model)
        assert batch.used and p.poisoned
        return True
    results,_,_=_run(operation)
    assert results==[True]*4


@pytest.mark.parametrize('operation',['import','query'])
def test_prepared_q256_rejects_other_approved_field(operation):
    from rep4gpu.spdz_prime256 import LOWGEAR_PRIME
    class NoCommunication:
        def exchange(self,*args,**kwargs):raise AssertionError('field guard must precede communication')
    party=Prime256Party(0,NoCommunication(),7,session=b'other-field',field_prime=LOWGEAR_PRIME,
                        _allow_insecure_test_transport=True)
    batch=SimpleNamespace(used=False)
    with pytest.raises(ValueError,match='require the q='):
        if operation=='import':import_prepared_rows(party,batch,(1,),model_id=b'model')
        else:matmul_prepared_rows(party,None,None,model_id=b'model')
    assert party.poisoned and not batch.used and party.round==0
