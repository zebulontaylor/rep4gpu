"""Fixed-circuit deferred checks must precede every public output value."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
import random
from types import SimpleNamespace

import pytest

from rep4gpu.spdz_distributed import ProtocolError
from rep4gpu.spdz_prime256 import QShare,Prime256Party
from rep4gpu.spdz_goldilocks_stream import (PRIME,prepare_fixed_weights,matmul_rows,
    partition_square_rows,square_rows,_rows,_squares)
from rep4gpu.spdz_goldilocks_deferred import DeferredGoldilocksParty
from rep4gpu.transport import IdealNetwork
from test_spdz_goldilocks_stream import shared,product


def run(operation,*,tmp_path=None,cap=2**20):
    rng=random.Random(107931);alpha=[rng.randrange(PRIME) for _ in range(4)]
    x=[PRIME-1,2**63];w1=[PRIME-1,7,11,2**63,1,31];w2=[PRIME-2,3,19,0,PRIME-1,2**63]
    xs=shared(x,(1,2),alpha,rng);ws1=shared(w1,(2,3),alpha,rng);ws2=shared(w2,(3,2),alpha,rng)
    def triple(k,n,identifier):
        a=[rng.randrange(PRIME) for _ in range(2*k)];b=[rng.randrange(PRIME) for _ in range(k*n)]
        aa,bb,cc=[shared(v,s,alpha,rng) for v,s in [(a,(2,k)),(b,(k,n)),(product(a,b,2,k,n),(2,n))]]
        return [SimpleNamespace(a=aa[p],b=bb[p],c=cc[p],identifier=identifier,used=False) for p in range(4)]
    t1,t2=triple(2,3,b'a'*32),triple(3,2,b'b'*32)
    a=[rng.randrange(PRIME) for _ in range(6)]
    aa=shared(a,(2,3),alpha,rng);cc=shared([v*v%PRIME for v in a],(2,3),alpha,rng)
    squares=[SimpleNamespace(a=aa[p],a_squared=cc[p],identifier=b's'*32,used=False) for p in range(4)]
    network=IdealNetwork(timeout=10);parties=[]
    if tmp_path is not None:
        epoch=tmp_path/'epoch';epoch.mkdir()
        (epoch/'public-manifest.json').write_text('{}')
    for p in range(4):
        role_path=None
        if tmp_path is not None:
            role_path=tmp_path/f'party{p}';role_path.mkdir()
            (role_path/'public-manifest.json').write_text(json.dumps(dict(key_epoch_root=str(epoch))))
        parties.append(DeferredGoldilocksParty(p,network.endpoint(p),alpha[p],session=b'deferred-fullfield',
            max_pending_values=cap[p] if isinstance(cap,tuple) else cap,key_epoch_role_path=role_path,_allow_insecure_test_transport=True))
    def role(p):
        try:
            party=parties[p]
            h1=prepare_fixed_weights(party,ws1[p],t1[p],(1,1),model_id=b'first')
            h2=prepare_fixed_weights(party,ws2[p],t2[p],(1,1),model_id=b'second')
            sq=partition_square_rows(party,squares[p],(1,1))
            assert party.pending_values==0 and party.round==6
            return operation(party,xs[p],h1,h2,sq)
        except BaseException as error:network.abort(error);return error
    with ThreadPoolExecutor(4) as pool:results=list(pool.map(role,range(4)))
    return results,parties,(x,w1,w2)


def circuit(p,x,h1,h2,sq,index=0,*,fault=None):
    if fault in ('first','first-value') and p.party==0:
        coordinate=x.values if fault.endswith('-value') else x.macs
        coordinate[0]=(coordinate[0]+1)%PRIME
    h=matmul_rows(p,x,h1[index],model_id=b'first')
    if fault in ('square','square-value') and p.party==0:
        coordinate=h.values if fault.endswith('-value') else h.macs
        coordinate[0]=(coordinate[0]+1)%PRIME
    h=square_rows(p,h,sq[index])
    if fault in ('second','second-value') and p.party==0:
        coordinate=h.values if fault.endswith('-value') else h.macs
        coordinate[0]=(coordinate[0]+1)%PRIME
    return matmul_rows(p,h,h2[index],model_id=b'second')


def test_full_field_dependent_sequence_uses_16_exchanges_and_fresh_rows():
    def operation(p,x,h1,h2,sq):
        outputs=[]
        for index in range(2):
            before=p.transport.round
            result=circuit(p,x,h1,h2,sq,index)
            assert len(p._pending)==3 and p.pending_values==8
            assert p.transport.round-before==6
            opened=p.open(result);outputs.append(opened)
            assert p.transport.round-before==16 and p.pending_values==0
            signed=[v-PRIME if v>PRIME//2 else v for v in opened]
            token=max(range(2),key=lambda i:signed[i])
            x=p.add_public(QShare([0,0],[0,0],(1,2)),[int(i==token) for i in range(2)])
        assert all(h.used for h in (*h1,*h2,*sq))
        return outputs
    results,parties,(x,w1,w2)=run(operation)
    expected=[]
    for _ in range(2):
        h=product(x,w1,1,2,3);h=[v*v%PRIME for v in h]
        output=product(h,w2,1,3,2);expected.append(output)
        signed=[v-PRIME if v>PRIME//2 else v for v in output]
        token=max(range(2),key=lambda i:signed[i]);x=[int(i==token) for i in range(2)]
    assert results==[expected]*4 and all(not p.poisoned for p in parties)


@pytest.mark.parametrize('fault',['first','square','second','first-value','square-value','second-value','cancel-entries','reorder'])
def test_faults_complete_only_masked_values_then_retire_epoch_before_any_output(tmp_path,fault):
    output_frames=[[] for _ in range(4)]
    def operation(p,x,h1,h2,sq):
        before=p.transport.round
        result=circuit(p,x,h1,h2,sq,fault=fault)
        assert p.transport.round-before==6 and len(p._pending)==3
        if p.party==0 and fault=='cancel-entries':
            for index,delta in [(0,1),(1,-1)]:
                entry=p._pending[index];values=list(entry.residual);values[0]=(values[0]+delta)%PRIME
                p._pending[index]=replace(entry,residual=tuple(values))
        if p.party==0 and fault=='reorder':p._pending.reverse()
        broadcast=p.broadcast
        def observe(payload,*,label):
            if label.startswith(b'SPDZq256-open-v1') and label.endswith(b'values'):output_frames[p.party].append(label)
            return broadcast(payload,label=label)
        p.broadcast=observe
        return p.open(result)
    results,parties,_=run(operation,tmp_path=tmp_path)
    assert all(isinstance(result,BaseException) for result in results)
    assert all(not frames for frames in output_frames)
    assert all(p.poisoned and p.pending_values==0 and not p._pending for p in parties)
    assert (tmp_path/'epoch/key-epoch-retired').is_file()


def test_residual_snapshot_owns_source_lists_and_candidates():
    def operation(p,x,h1,h2,sq):
        original=p._open_goldilocks_difference
        saved=[]
        def mutate(share,handle,*,kind,label):
            candidate=original(share,handle,kind=kind,label=label)
            snapshot=p._pending[-1]
            share.macs[:]=[0]*len(share.macs);share.values[:]=[0]*len(share.values)
            copy=list(candidate);candidate[:]=[0]*len(candidate)
            assert p._pending[-1] is snapshot
            saved.append(tuple(snapshot.residual))
            return copy
        p._open_goldilocks_difference=mutate
        output=p.open(circuit(p,x,h1,h2,sq))
        assert len(saved)==3 and not p._pending
        return output
    results,parties,(x,w1,w2)=run(operation)
    expected=product([v*v%PRIME for v in product(x,w1,1,2,3)],w2,1,3,2)
    assert results==[expected]*4 and all(not p.poisoned for p in parties)


@pytest.mark.parametrize('fault',['unknown','unused','duplicate'])
def test_only_once_burned_capabilities_authorize_deferred_values(fault):
    def operation(p,x,h1,h2,sq):
        handle=h1[0];state=_rows[handle]
        if fault=='unknown':handle=object()
        if fault=='duplicate':matmul_rows(p,x,handle,model_id=b'first')
        return p._open_goldilocks_difference(x,handle,kind='matmul',label=state.identifier+b'/D')
    results,parties,_=run(operation)
    assert all(isinstance(result,BaseException) for result in results)
    assert all(p.poisoned for p in parties)


def test_public_open_and_generic_square_never_queue_unmasked_output():
    def operation(p,x,h1,h2,sq):
        before=p.transport.round
        assert p.open(x) is not None and not p._pending
        assert p.transport.round-before==6
        return True
    results,parties,_=run(operation)
    assert results==[True]*4 and all(not p.poisoned for p in parties)



@pytest.mark.parametrize('stage',['commit','reveal'])
def test_consistently_corrupted_batch_messages_cannot_release_output(tmp_path,stage):
    frames=[[] for _ in range(4)]
    def operation(p,x,h1,h2,sq):
        result=circuit(p,x,h1,h2,sq)
        broadcast=p.broadcast
        def fault(payload,*,label):
            if label.startswith(b'SPDZq256-open-v1') and label.endswith(b'values'):frames[p.party].append(label)
            if p.party==0 and label.startswith(b'Goldilocks-deferred-MAC-v1') and label.endswith(('/'+stage).encode()):
                payload=payload[:-1]+bytes([payload[-1]^1])
            return broadcast(payload,label=label)
        p.broadcast=fault
        return p.open(result)
    results,parties,_=run(operation,tmp_path=tmp_path)
    assert all(isinstance(result,BaseException) for result in results)
    assert all(not row for row in frames)
    assert all(p.poisoned and not p._pending for p in parties)
    assert (tmp_path/'epoch/key-epoch-retired').exists()


@pytest.mark.parametrize('cap',[1,4])
def test_public_queue_cap_flushes_without_changing_exact_output(cap):
    def operation(p,x,h1,h2,sq):return p.open(circuit(p,x,h1,h2,sq))
    results,parties,(x,w1,w2)=run(operation,cap=cap)
    expected=product([v*v%PRIME for v in product(x,w1,1,2,3)],w2,1,3,2)
    assert results==[expected]*4 and all(not p.poisoned for p in parties)


def test_successful_session_does_not_retire_native_epoch(tmp_path):
    results,parties,_=run(lambda p,x,h1,h2,sq:p.open(circuit(p,x,h1,h2,sq)),tmp_path=tmp_path)
    assert all(isinstance(result,list) for result in results)
    assert all(not p.poisoned and p._retirement_error is None for p in parties)
    assert not (tmp_path/'epoch/key-epoch-retired').exists()



def test_different_public_queue_caps_abort_before_any_candidate():
    results,parties,_=run(lambda p,x,h1,h2,sq:matmul_rows(p,x,h1[0],model_id=b'first'),cap=(4,8,8,8))
    assert all(isinstance(result,BaseException) for result in results)
    assert all(p.poisoned and not p._pending for p in parties)


def test_generic_square_keeps_its_immediate_check_and_cannot_defer():
    rng=random.Random(91);alpha=[rng.randrange(PRIME) for _ in range(4)]
    x=[PRIME-1,2**63];a=[rng.randrange(PRIME) for _ in x]
    xs=shared(x,(1,2),alpha,rng);aa=shared(a,(1,2),alpha,rng)
    cc=shared([v*v%PRIME for v in a],(1,2),alpha,rng)
    network=IdealNetwork(timeout=5)
    parties=[DeferredGoldilocksParty(p,network.endpoint(p),alpha[p],session=b'generic-square-immediate',
                                   _allow_insecure_test_transport=True) for p in range(4)]
    def role(p):
        try:
            mask=SimpleNamespace(a=aa[p],a_squared=cc[p],identifier=b'g'*32,used=False)
            result=parties[p].square(xs[p],mask)
            assert parties[p].pending_values==0 and parties[p].transport.round==6
            return parties[p].open(result)
        except BaseException as error:network.abort(error);return error
    with ThreadPoolExecutor(4) as pool:results=list(pool.map(role,range(4)))
    assert results==[[v*v%PRIME for v in x]]*4
    assert all(not p.poisoned for p in parties)



@pytest.mark.parametrize('alias',['ordinary','base-open','base-open-many'])
def test_public_open_aliases_flush_before_any_output_frame(tmp_path,alias):
    frames=[[] for _ in range(4)]
    def operation(p,x,h1,h2,sq):
        output=circuit(p,x,h1,h2,sq,fault='first')
        broadcast=p.broadcast
        def observe(payload,*,label):
            if label.startswith(b'SPDZq256-open-v1') and label.endswith(b'values'):frames[p.party].append(label)
            return broadcast(payload,label=label)
        p.broadcast=observe
        if alias=='base-open-many':return Prime256Party.open_many(p,(output,),label=b'explicit-base')[0]
        if alias=='base-open':return Prime256Party.open(p,output)
        return p.open(output)
    results,parties,_=run(operation,tmp_path=tmp_path)
    assert all(isinstance(result,BaseException) for result in results)
    assert all(not row for row in frames)
    assert all(p.poisoned for p in parties)
    assert (tmp_path/'epoch/key-epoch-retired').exists()



@pytest.mark.parametrize('base_alias',[False,True])
def test_lazy_public_share_iterable_cannot_create_pending_after_output_barrier(tmp_path,base_alias):
    frames=[[] for _ in range(4)]
    def operation(p,x,h1,h2,sq):
        broadcast=p.broadcast
        def observe(payload,*,label):
            if label.startswith(b'SPDZq256-open-v1') and label.endswith(b'values'):frames[p.party].append(label)
            return broadcast(payload,label=label)
        p.broadcast=observe
        def lazy():yield circuit(p,x,h1,h2,sq,fault='second')
        if base_alias:return Prime256Party.open_many(p,lazy())
        return p.open_many(lazy())
    results,parties,_=run(operation,tmp_path=tmp_path)
    assert all(isinstance(result,BaseException) for result in results)
    assert all(not row for row in frames)
    assert all(p.poisoned for p in parties)
    assert (tmp_path/'epoch/key-epoch-retired').exists()
