from concurrent.futures import ThreadPoolExecutor
import shutil
import pytest
import torch
from rep4gpu.spdz import TrustedDealer,PRIME,field_matmul
from rep4gpu.spdz_distributed import DistributedParty,MatrixTriple,split_share,split_triple
from rep4gpu.transport import IdealNetwork


def _fixture():
    dealer=TrustedDealer(parties=4,mac_keys=5)
    x=torch.tensor([[7,-11,3],[-1,17,91]],dtype=torch.int64).remainder(PRIME)
    y=torch.tensor([[1,-1],[71,8],[-9,17]],dtype=torch.int64).remainder(PRIME)
    return dealer,x,y,split_share(dealer.share(x)),split_share(dealer.share(y)),split_triple(dealer.triple(2,3,2))


def _run(operation,mutate=None,wire_fault=None):
    dealer,x,y,xs,ys,triples=_fixture()
    network=IdealNetwork(timeout=5)
    parties=[]
    for p in range(4):
        transport=network.endpoint(p)
        if wire_fault is not None:
            exchange=transport.exchange
            captured={}
            def faulty(outgoing,p=p,transport=transport,exchange=exchange,captured=captured):
                outgoing=wire_fault(p,transport.round,dict(outgoing),captured)
                return exchange(outgoing)
            transport.exchange=faulty
        parties.append(DistributedParty(p,transport,dealer.alpha_shares[:,p].clone(),
                                        parties=4,session=b'isolated-unit-test',
                                        _allow_insecure_test_transport=True))
        if mutate is not None:
            mutate(p,xs[p],ys[p],triples[p])
    def work(p):
        try:
            return operation(parties[p],xs[p],ys[p],triples[p])
        except BaseException as error:
            network.abort(error)
            return error
    with ThreadPoolExecutor(4) as pool:
        result=list(pool.map(work,range(4)))
    return result,parties,field_matmul(x,y)


def test_distributed_checked_beaver_matches_field_reference():
    outputs,parties,expected=_run(lambda p,x,y,t:p.open(p.matmul(x,y,t)))
    assert all(isinstance(value,torch.Tensor) and torch.equal(value,expected) for value in outputs)
    assert all(not party.poisoned for party in parties)
    assert all(party.alpha_share.shape==(5,) for party in parties)


def test_distributed_open_many_mixed_shapes():
    outputs,parties,_=_run(lambda p,x,y,t:p.open_many([x,y]))
    assert all(isinstance(value,list) and len(value)==2 for value in outputs)
    assert all(torch.equal(outputs[0][j],value[j]) for value in outputs for j in (0,1))


@pytest.mark.parametrize('kind',['value','mac'])
def test_corrupt_local_share_aborts_and_retires_session(kind):
    def mutate(p,x,y,t):
        if p==0:
            target=x.values if kind=='value' else x.macs
            target.reshape(-1)[0]=(target.reshape(-1)[0]+1)%PRIME
    outputs,parties,_=_run(lambda p,x,y,t:p.open(x),mutate=mutate)
    assert all(isinstance(value,BaseException) for value in outputs)
    assert all(party.poisoned for party in parties)


@pytest.mark.parametrize('fault_round',[0,4])
def test_value_equivocation_and_committed_residual_mutation_abort(fault_round):
    def fault(p,round_index,outgoing,captured):
        if p==0 and round_index==fault_round:
            peer=next(iter(outgoing))
            data=bytearray(outgoing[peer]);data[-1]^=1
            outgoing[peer]=bytes(data)
        return outgoing
    outputs,parties,_=_run(lambda p,x,y,t:p.open(x),wire_fault=fault)
    assert all(isinstance(value,BaseException) for value in outputs)
    assert all(party.poisoned for party in parties)


def test_replayed_previous_open_frame_is_rejected():
    def fault(p,round_index,outgoing,captured):
        if p==0 and round_index==0:
            captured.update(outgoing)
        if p==0 and round_index==6:
            outgoing=dict(captured)
        return outgoing
    def operation(p,x,y,t):
        p.open(x)
        return p.open(x)
    outputs,parties,_=_run(operation,wire_fault=fault)
    assert all(isinstance(value,BaseException) for value in outputs)
    assert all(party.poisoned for party in parties)


def test_consumed_triple_identifier_cannot_be_rewrapped():
    def operation(p,x,y,t):
        p.matmul(x,y,t)
        assert t.used
        replacement=MatrixTriple(t.a,t.b,t.c,used=False,identifier=t.identifier)
        with pytest.raises(ValueError,match='single-use|used|spent|reuse'):
            p.matmul(x,y,replacement)
        return p.poisoned
    outputs,_,_=_run(operation)
    assert outputs==[True]*4


def test_trusted_preprocessing_splits_have_only_local_components():
    dealer,x,y,xs,ys,triples=_fixture()
    assert all(share.values.shape==x.shape and share.macs.shape==(5,*x.shape) for share in xs)
    assert all(share.values.shape==y.shape and share.macs.shape==(5,*y.shape) for share in ys)
    assert len({triple.identifier for triple in triples})==1
    assert all(not triple.used for triple in triples)


@pytest.mark.skipif(shutil.which('openssl') is None,reason='openssl required for TLS test identities')
def test_four_isolated_tls_workers(tmp_path):
    from scripts.run_spdz_distributed import run
    report=run(tmp_path/'distributed.json',m=2,k=3,n=2,timeout=45)
    assert report['passed'] and report['return_codes']==[0]*4
    assert report['global_payload_bytes']>report['global_online_payload_bytes']>0
    assert report['online_critical_path_ms']>0 and report['preprocessing_ms']>0
    assert all(p['authenticated_tls'] and p['local_only_preprocessing'] and p['triple_consumed'] for p in report['parties'])
    assert all(not p['other_owners_plaintext_supplied'] for p in report['parties'])
    assert all(p['alpha_share_shape']==[5] and p['local_mac_shape']==[5,2,3] for p in report['parties'])
    assert not list(tmp_path.rglob('*.key')) and not list(tmp_path.rglob('*.pt'))


@pytest.mark.parametrize('phase',[b'commit',b'reveal',b'values'])
def test_consistent_malicious_payload_reaches_authentication_guards(phase):
    import struct
    def operation(p,x,y,t):
        if p.party==0:
            broadcast=p.broadcast
            def malicious(payload,*,label):
                if label.endswith(phase):
                    if phase==b'values':
                        payload=struct.pack('<I',PRIME)+payload[4:]
                    else:
                        data=bytearray(payload);data[-1]^=1;payload=bytes(data)
                return broadcast(payload,label=label)
            p.broadcast=malicious
        return p.open(x)
    outputs,parties,_=_run(operation)
    assert all(isinstance(value,BaseException) for value in outputs)
    assert all(party.poisoned for party in parties)
    if phase!=b'values':
        assert any('commitment' in str(error) for error in outputs)
    else:
        assert any('canonical' in str(error) for error in outputs)


def test_stacked_mac_local_arithmetic_equals_individual_macs():
    dealer,x,y,xs,ys,triples=_fixture()
    generator=torch.Generator().manual_seed(370)
    d=torch.randint(PRIME,x.shape,generator=generator)
    e=torch.randint(PRIME,y.shape,generator=generator)
    for p in range(4):
        transport=IdealNetwork().endpoint(p)
        runtime=DistributedParty(p,transport,dealer.alpha_shares[:,p],session=b'local-comparison',
                                 _allow_insecure_test_transport=True)
        reference=runtime._matmul_local(triples[p],d,e)
        runtime.stack_macs=True
        actual=runtime._matmul_local(triples[p],d,e)
        assert torch.equal(actual.values,reference.values)
        assert torch.equal(actual.macs,reference.macs)


@pytest.mark.parametrize('timeout',[0,-1,float('nan'),float('inf')])
def test_runner_rejects_invalid_deadline(tmp_path,timeout):
    from scripts.run_spdz_distributed import run
    with pytest.raises(ValueError,match='finite and positive'):
        run(tmp_path/'unused.json',timeout=timeout)


@pytest.mark.parametrize('round_index',[0,2,4])
def test_identical_wire_mutation_to_every_peer_aborts(round_index):
    import struct
    def fault(p,current,outgoing,captured):
        if p==0 and current==round_index:
            result={}
            for peer,payload in outgoing.items():
                data=bytearray(payload)
                if round_index==0:
                    data[93:97]=struct.pack('<I',PRIME)
                else:
                    data[-1]^=1
                result[peer]=bytes(data)
            return result
        return outgoing
    outputs,parties,_=_run(lambda p,x,y,t:p.open(x),wire_fault=fault)
    assert all(isinstance(value,BaseException) for value in outputs)
    assert all(party.poisoned for party in parties)
