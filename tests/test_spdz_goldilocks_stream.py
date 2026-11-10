"""Exact Goldilocks prepared-row arithmetic and one-use protocol tests."""
from concurrent.futures import ThreadPoolExecutor
import random
from types import SimpleNamespace

import pytest

from rep4gpu.spdz_goldilocks_stream import (
    PRIME, prepare_fixed_weights, matmul_rows, partition_square_rows, square_rows, _rows,
)
from rep4gpu.spdz_prime256 import Prime256Party, QShare
from rep4gpu.transport import IdealNetwork


def shared(values, shape, alpha, rng):
    coordinates=[]
    for plain in (values, [sum(alpha)*v%PRIME for v in values]):
        pieces=[[rng.randrange(PRIME) for _ in plain] for _ in range(3)]
        pieces.append([(v-sum(part[j] for part in pieces))%PRIME for j,v in enumerate(plain)])
        coordinates.append(pieces)
    return [QShare(coordinates[0][p],coordinates[1][p],shape) for p in range(4)]


def product(a,b,m,k,n):
    return [sum(a[i*k+t]*b[t*n+j] for t in range(k))%PRIME for i in range(m) for j in range(n)]


def run(operation, mutate=None):
    rng=random.Random(43102);alpha=[rng.randrange(PRIME) for _ in range(4)]
    x=[0,PRIME-1,2**63,7,PRIME-2,19];w=[PRIME-1,7,11,0,1,31]
    a=[rng.randrange(PRIME) for _ in x];b=[rng.randrange(PRIME) for _ in w]
    shares=[shared(v,s,alpha,rng) for v,s in [(x,(3,2)),(w,(2,3)),(a,(3,2)),(b,(2,3)),(product(a,b,3,2,3),(3,3))]]
    mask=[rng.randrange(PRIME) for _ in range(9)]
    ma=shared(mask,(3,3),alpha,rng);mc=shared([v*v%PRIME for v in mask],(3,3),alpha,rng)
    triples=[SimpleNamespace(a=shares[2][p],b=shares[3][p],c=shares[4][p],identifier=b't'*32,used=False) for p in range(4)]
    squares=[SimpleNamespace(a=ma[p],a_squared=mc[p],identifier=b's'*32,used=False) for p in range(4)]
    if mutate:mutate(shares,triples,squares)
    network=IdealNetwork(timeout=5)
    parties=[Prime256Party(p,network.endpoint(p),alpha[p],session=b'gold-stream-tests',field_prime=PRIME,_allow_insecure_test_transport=True) for p in range(4)]
    def role(p):
        try:return operation(parties[p],shares[0][p],shares[1][p],triples[p],squares[p])
        except BaseException as error:network.abort(error);return error
    with ThreadPoolExecutor(4) as pool:outputs=list(pool.map(role,range(4)))
    return outputs,parties,product(x,w,3,2,3)


@pytest.mark.parametrize('counts',[(1,1,1),(1,2),(3,)])
def test_generic_triple_preparation_disjoint_queries_and_square_partition(counts):
    def operation(p,x,w,t,s):
        handles=prepare_fixed_weights(p,w,t,counts,model_id=b'model')
        squares=partition_square_rows(p,s,counts)
        assert t.used and s.used and all(_rows[h].weights is _rows[handles[0]].weights for h in handles)
        # Changes to the caller's old buffers cannot affect child capabilities.
        for item in (w,t.a,t.b,t.c,s.a,s.a_squared):
            item.values[:]=[0]*len(item.values);item.macs[:]=[0]*len(item.macs)
        result=[];start=0
        for count,h,sq in zip(counts,handles,squares):
            q=QShare(x.values[start*2:(start+count)*2],x.macs[start*2:(start+count)*2],(count,2))
            result += p.open(square_rows(p,matmul_rows(p,q,h,model_id=b'model'),sq))
            start+=count
        assert all(h.used for h in (*handles,*squares))
        return result
    outputs,parties,expected=run(operation)
    assert outputs==[[v*v%PRIME for v in expected]]*4
    assert all(not p.poisoned for p in parties)


@pytest.mark.parametrize('fault',['parent','child','square-parent','square-child','model','key','session','partition','query-mac'])
def test_capability_faults_poison_session(fault):
    def operation(p,x,w,t,s):
        handles=prepare_fixed_weights(p,w,t,(1,1,1),model_id=b'model')
        masks=partition_square_rows(p,s,(1,1,1))
        q=QShare(x.values[:2],x.macs[:2],(1,2))
        if fault=='parent':return prepare_fixed_weights(p,w,t,(3,),model_id=b'model')
        if fault=='square-parent':return partition_square_rows(p,s,(3,))
        if fault=='child':
            matmul_rows(p,q,handles[0],model_id=b'model')
        if fault=='square-child':
            zero=QShare([0]*3,[0]*3,(1,3));square_rows(p,zero,masks[0]);return square_rows(p,zero,masks[0])
        if fault=='key':p.alpha_share=(p.alpha_share+1)%PRIME
        if fault=='session':p.session=b'other'
        if fault=='partition':return prepare_fixed_weights(p,w,SimpleNamespace(a=t.a,b=t.b,c=t.c,used=False,identifier=b'z'*32),(1,),model_id=b'model')
        if fault=='query-mac' and p.party==0:q.macs[0]=(q.macs[0]+1)%PRIME
        return matmul_rows(p,q,handles[0],model_id=b'wrong' if fault=='model' else b'model')
    outputs,parties,_=run(operation)
    assert all(isinstance(value,BaseException) for value in outputs)
    assert all(p.poisoned for p in parties)


def test_bad_preparation_weight_mac_aborts_before_handles_are_issued():
    def mutate(shares,*_):shares[1][0].macs[0]=(shares[1][0].macs[0]+1)%PRIME
    outputs,parties,_=run(lambda p,x,w,t,s:prepare_fixed_weights(p,w,t,(1,1,1),model_id=b'model'),mutate)
    assert all(isinstance(value,BaseException) for value in outputs)
    assert all(p.poisoned for p in parties)


def test_preparation_labels_bind_complete_partition_before_factor_use():
    outputs,parties,_=run(lambda p,x,w,t,s:prepare_fixed_weights(p,w,t,(1,2) if p.party==0 else (1,1,1),model_id=b'model'))
    assert all(isinstance(value,BaseException) for value in outputs)
    assert all(p.poisoned for p in parties)


def test_dependent_queries_use_checked_previous_output_argmax():
    def operation(p,x,w,t,s):
        handles=prepare_fixed_weights(p,w,t,(1,1,1),model_id=b'model')
        current=QShare(x.values[:2],x.macs[:2],(1,2));outputs=[]
        for h in handles:
            opened=p.open(matmul_rows(p,current,h,model_id=b'model'))
            outputs.append(opened)
            signed=[v-PRIME if v>PRIME//2 else v for v in opened]
            # Fold the three output classes into two input coordinates.
            token=max(range(3),key=lambda j:signed[j])%2
            current=p.add_public(QShare([0,0],[0,0],(1,2)),[int(i==token) for i in range(2)])
        return outputs
    results,_,_=run(operation)
    current=[0,PRIME-1];w=[PRIME-1,7,11,0,1,31];expected=[]
    for _ in range(3):
        row=product(current,w,1,2,3);expected.append(row)
        signed=[v-PRIME if v>PRIME//2 else v for v in row]
        token=max(range(3),key=lambda j:signed[j])%2
        current=[int(i==token) for i in range(2)]
    assert results==[expected]*4


def test_stream_manifest_allocates_only_one_initial_input_row():
    from scripts.run_spdz_goldilocks_stream import request_for
    request=request_for(128)
    assert [matrix['m'] for matrix in request['matrices']]==[128,128]
    assert request['squares']==[dict(name='hidden',shape=[128,2])]
    assert [x for x in request['inputs'] if x['owner']==0]==[dict(name='seed',owner=0,shape=[1,8])]
    assert sum(x['shape'][0]*x['shape'][1] for x in request['inputs'])==50


def test_protected_difference_hook_receives_only_burned_bound_capabilities():
    from rep4gpu.spdz_goldilocks_stream import _validate_masked_capability
    def operation(p,x,w,t,s):
        handles=prepare_fixed_weights(p,w,t,(1,1,1),model_id=b'model')
        masks=partition_square_rows(p,s,(1,1,1));calls=[]
        def hook(difference,handle,*,kind,label):
            state=_validate_masked_capability(p,handle,kind,label)
            assert state.used and difference.shape==state.a.shape
            calls.append(kind)
            return p.open_many((difference,),label=label)[0]
        p._open_goldilocks_difference=hook
        query=QShare(x.values[:2],x.macs[:2],(1,2))
        result=p.open(square_rows(p,matmul_rows(p,query,handles[0],model_id=b'model'),masks[0]))
        assert calls==['matmul','square']
        return result
    outputs,_,expected=run(operation)
    assert outputs==[[v*v%PRIME for v in expected[:3]]]*4


@pytest.mark.parametrize('fault',['unburned','kind','label','key','epoch'])
def test_deferred_authorization_rejects_invalid_capability(fault):
    from rep4gpu.spdz_goldilocks_stream import _validate_masked_capability
    def operation(p,x,w,t,s):
        handle=prepare_fixed_weights(p,w,t,(1,1,1),model_id=b'model')[0]
        state=_rows[handle]
        if fault!='unburned':
            q=QShare(x.values[:2],x.macs[:2],(1,2));matmul_rows(p,q,handle,model_id=b'model')
        if fault=='key':p.alpha_share=(p.alpha_share+1)%PRIME
        if fault=='epoch':p.session=b'foreign'
        with p.operation():
            return _validate_masked_capability(p,handle,'square' if fault=='kind' else 'matmul',
                                              b'wrong' if fault=='label' else state.identifier+b'/D')
    outputs,parties,_=run(operation)
    assert all(isinstance(value,BaseException) for value in outputs)
    assert all(p.poisoned for p in parties)


def test_fixed_weight_setup_flushes_pending_before_burning_new_triple():
    def operation(p,x,w,t,s):
        called=[]
        def flush():
            assert not t.used
            called.append(True)
        p.flush_pending=flush
        prepare_fixed_weights(p,w,t,(1,1,1),model_id=b'model')
        assert called==[True] and t.used
        return True
    outputs,_,_=run(operation)
    assert outputs==[True]*4


def test_immediate_square_refactor_preserves_every_byte_against_base_party_square():
    from rep4gpu.spdz_goldilocks_stream import _squares
    def make_operation(golden):
        def operation(p,x,w,t,s):
            frames=[];exchange=p.transport.exchange
            def record(outgoing):
                frames.append(tuple(sorted(outgoing.items())))
                return exchange(outgoing)
            p.transport.exchange=record
            p._commit_salt=lambda:b'fixed disposable fixture salt!!!'[:32].ljust(32,b'!')
            rows=prepare_fixed_weights(p,w,t,(1,1,1),model_id=b'model')
            masks=partition_square_rows(p,s,(1,1,1))
            q=QShare(x.values[:2],x.macs[:2],(1,2))
            value=matmul_rows(p,q,rows[0],model_id=b'model')
            if golden:
                state=_squares[masks[0]]
                mask=SimpleNamespace(a=state.a,a_squared=state.factor,identifier=state.identifier,used=False)
                value=p.square(value,mask)
            else:value=square_rows(p,value,masks[0])
            return p.open(value),frames
        return operation
    actual,_,_=run(make_operation(False));golden,_,_=run(make_operation(True))
    assert actual==golden
    assert all(len(frames)==24 for _,frames in actual)
