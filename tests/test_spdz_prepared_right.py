"""Check single-use static-model preparation and its independent online phase."""
from concurrent.futures import ThreadPoolExecutor
import copy
import pickle
import threading

import pytest
import torch

from rep4gpu.spdz import PRIME,TrustedDealer,field_matmul
from rep4gpu.spdz_distributed import DistributedParty,MatrixTriple,split_share,split_triple
from rep4gpu.spdz_prepared_right import PreparedRightMatmul,prepare_right_matmul as _production_prepare,matmul_prepared_right
from rep4gpu.transport import IdealNetwork


def _prepare(party,*args,**kwargs):
    prepared = _production_prepare(party,*args,**kwargs)
    # Intentional post-preparation faults must not race a peer's final setup
    # receipt. This barrier is test orchestration, not a protocol guarantee.
    party._test_preparation_ready.wait(timeout=10)
    return prepared


def _run(operation,*,keys=10,m=2,k=3,n=2,device='cpu',party_type=None):
    dealer = TrustedDealer(parties=4,mac_keys=keys,device=device)
    x = (torch.arange(m*k,dtype=torch.int64,device=device).reshape(m,k)-3).remainder(PRIME)
    w = (torch.arange(k*n,dtype=torch.int64,device=device).reshape(k,n)-2).remainder(PRIME)
    xs,ws = split_share(dealer.share(x)),split_share(dealer.share(w))
    triples = split_triple(dealer.triple(m,k,n))
    network = IdealNetwork(timeout=10)
    if party_type is None:
        party_type = DistributedParty
    endpoints = [network.endpoint(p) for p in range(4)]
    if torch.device(device).type == 'cuda':
        from rep4gpu.device_transport import DeviceIdealEndpoint
        endpoints = [DeviceIdealEndpoint(endpoint,device=device) for endpoint in endpoints]
    roles = [party_type(p,endpoints[p],dealer.alpha_shares[:,p],
                        session=b'prepared-right-tests',_allow_insecure_test_transport=True)
             for p in range(4)]
    prepared_barrier = threading.Barrier(4)
    query_barrier = threading.Barrier(4)
    for role in roles:
        role._test_preparation_ready = prepared_barrier
        role._test_query_ready = query_barrier
    def worker(p):
        try:
            return operation(roles[p],xs[p],ws[p],triples[p])
        except BaseException as error:
            network.abort(error)
            return error
    with ThreadPoolExecutor(4) as pool:
        results = list(pool.map(worker,range(4)))
    return results,roles,triples,field_matmul(x,w)


@pytest.mark.parametrize('keys',[5,10])
@pytest.mark.parametrize('shape',[(1,3,7),(3,5,2)])
def test_prepared_right_matches_exact_product_and_opens_only_query_difference(keys,shape):
    m,k,n = shape
    def operation(p,x,w,t):
        layouts = []
        original = p.open_many
        def record(shares,*,label=b''):
            layouts.append(tuple(tuple(share.values.shape) for share in shares))
            return original(shares,label=label)
        p.open_many = record
        prepared = _prepare(p,w,t,model_id=b'model-v1')
        assert t.used and not prepared.used
        assert prepared.model_id == b'model-v1'
        assert prepared.input_shape == (m,k)
        assert prepared.output_shape == (m,n)
        assert layouts == [((k,n),)]
        result = matmul_prepared_right(p,x,prepared,model_id=b'model-v1')
        assert prepared.used
        assert layouts == [((k,n),),((m,k),)]
        return p.open(result)
    results,roles,_,expected = _run(operation,keys=keys,m=m,k=k,n=n)
    assert all(isinstance(value,torch.Tensor) and torch.equal(value,expected) for value in results)
    assert all(not p.poisoned for p in roles)


def test_preparation_owns_immutable_correlation_and_model_snapshots():
    def operation(p,x,w,t):
        prepared = _prepare(p,w,t,model_id=b'weights-snapshot')
        for share in (w,t.a,t.b,t.c):
            share.values.fill_(0)
            share.macs.fill_(0)
        return p.open(matmul_prepared_right(p,x,prepared,model_id=b'weights-snapshot'))
    results,roles,_,expected = _run(operation)
    assert all(isinstance(value,torch.Tensor) and torch.equal(value,expected) for value in results)
    assert all(not p.poisoned for p in roles)


@pytest.mark.parametrize('kind',['reuse','model','shape','key','session'])
def test_prepared_capability_failure_consumes_handle_and_poison_session(kind):
    def operation(p,x,w,t):
        prepared = _prepare(p,w,t,model_id=b'model-v1')
        if kind == 'reuse':
            matmul_prepared_right(p,x,prepared,model_id=b'model-v1')
            # Finish the valid first use at every role before injecting the
            # reuse fault, whose abort otherwise interrupts a peer's receipt.
            p._test_query_ready.wait(timeout=10)
        if kind == 'key':
            p.alpha_share = p.alpha_share.clone()
        if kind == 'session':
            p.session = b'another-session'
        query = x if kind != 'shape' else type(x)(x.values[:1],x.macs[:,:1])
        before = p.round
        with pytest.raises((ValueError,RuntimeError)):
            matmul_prepared_right(p,query,prepared,model_id=b'wrong-model' if kind == 'model' else b'model-v1')
        assert prepared.used
        assert p.poisoned
        assert p.round == before
        return True
    results,_,_,_ = _run(operation)
    assert results == [True]*4


def test_prepared_handles_cannot_be_constructed_serialized_or_forged():
    with pytest.raises(TypeError):
        PreparedRightMatmul()
    def operation(p,x,w,t):
        prepared = _prepare(p,w,t,model_id=b'model-v1')
        for serializer in (copy.copy,pickle.dumps):
            with pytest.raises(TypeError):
                serializer(prepared)
        fake = object.__new__(PreparedRightMatmul)
        before = p.round
        with pytest.raises(ValueError,match='unknown'):
            matmul_prepared_right(p,x,fake,model_id=b'model-v1')
        assert p.round == before and p.poisoned
        return True
    results,_,_,_ = _run(operation)
    assert results == [True]*4


@pytest.mark.parametrize('stage',['prepare','online'])
def test_corrupted_opening_cannot_produce_ready_handle_or_online_output(stage):
    def operation(p,x,w,t):
        if stage == 'prepare' and p.party == 0:
            w.macs[9,0,0] = (w.macs[9,0,0]+1)%PRIME
        prepared = _prepare(p,w,t,model_id=b'model-v1')
        if stage == 'online' and p.party == 0:
            x.macs[9,0,0] = (x.macs[9,0,0]+1)%PRIME
        return p.open(matmul_prepared_right(p,x,prepared,model_id=b'model-v1'))
    if stage == 'prepare':
        from unittest.mock import patch
        def forbidden(*args,**kwargs):
            raise AssertionError('unchecked E reached model factor preparation')
        with patch('rep4gpu.spdz_prepared_right._factor',side_effect=forbidden):
            results,roles,triples,_ = _run(operation)
        assert not any(isinstance(value,AssertionError) for value in results)
    else:
        results,roles,triples,_ = _run(operation)
    assert all(isinstance(value,BaseException) for value in results)
    assert all(p.poisoned for p in roles)
    assert all(t.used for t in triples)


def test_original_triple_cannot_be_used_after_model_preparation():
    def operation(p,x,w,t):
        _prepare(p,w,t,model_id=b'model-v1')
        fresh_wrapper = MatrixTriple(t.a,t.b,t.c,identifier=t.identifier)
        before = p.round
        with pytest.raises(ValueError,match='single-use'):
            p.matmul(x,w,fresh_wrapper)
        assert p.round == before and p.poisoned
        return True
    results,_,_,_ = _run(operation)
    assert results == [True]*4


def test_pending_checks_cannot_be_hidden_inside_model_preparation():
    def operation(p,x,w,t):
        p.pending_values = 1
        with pytest.raises(RuntimeError,match='pending'):
            _prepare(p,w,t,model_id=b'model-v1')
        assert t.used and p.poisoned and p.round == 0
        return True
    results,_,_,_ = _run(operation)
    assert results == [True]*4


def test_prepared_fixture_clone_is_confined_to_exact_replay_transport():
    from benchmarks.spdz_prepared_fixture import replay_prepared_right
    from rep4gpu.transport import ReplayTransport
    def operation(p,x,w,t):
        prepared = _prepare(p,w,t,model_id=b'model-v1')
        with pytest.raises(TypeError,match='replay'):
            replay_prepared_right(p,prepared)
        replay = DistributedParty(p.party,ReplayTransport([]),p.alpha_share,session=p.session,
                                   _allow_insecure_test_transport=True)
        cloned = replay_prepared_right(replay,prepared)
        assert not cloned.used and not prepared.used
        assert cloned.input_shape == prepared.input_shape
        assert cloned.model_id == prepared.model_id
        wrong_shape = type(x)(x.values[:1],x.macs[:,:1])
        with pytest.raises(ValueError,match='dimensions'):
            matmul_prepared_right(replay,wrong_shape,cloned,model_id=b'model-v1')
        assert cloned.used and not prepared.used
        return True
    results,_,_,_ = _run(operation)
    assert results == [True]*4


def test_prepared_capabilities_cannot_transfer_between_live_roles():
    import threading
    barrier = threading.Barrier(4)
    handles = {}
    def operation(p,x,w,t):
        handles[p.party] = _prepare(p,w,t,model_id=b'model-v1')
        barrier.wait(timeout=5)
        foreign = handles[(p.party+1)%4]
        before = p.round
        with pytest.raises(RuntimeError,match='another party'):
            matmul_prepared_right(p,x,foreign,model_id=b'model-v1')
        assert foreign.used and p.poisoned and p.round == before
        return True
    results,_,_,_ = _run(operation)
    assert results == [True]*4


def test_online_prepared_matmul_uses_runtime_masked_opening_policy():
    def operation(p,x,w,t):
        calls = []
        ordinary = p.open_many
        def masked(shares,*,label):
            calls.append(label)
            return ordinary(shares,label=label)
        p._masked_open_many = masked
        prepared = _prepare(p,w,t,model_id=b'model-v1')
        assert not calls
        result = matmul_prepared_right(p,x,prepared,model_id=b'model-v1')
        assert len(calls) == 1
        return p.open(result)
    results,_,_,expected = _run(operation)
    assert all(isinstance(value,torch.Tensor) and torch.equal(value,expected) for value in results)


def test_prepared_fixture_rejects_unregistered_replay_subclass():
    from benchmarks.spdz_prepared_fixture import replay_prepared_right
    from rep4gpu.transport import ReplayTransport
    class UnregisteredReplay(ReplayTransport):
        pass
    def operation(p,x,w,t):
        prepared = _prepare(p,w,t,model_id=b'model-v1')
        target = DistributedParty(p.party,UnregisteredReplay([]),p.alpha_share,session=p.session,
                                   _allow_insecure_test_transport=True)
        with pytest.raises(TypeError,match='exact fixed replay'):
            replay_prepared_right(target,prepared)
        assert not prepared.used
        return True
    results,_,_,_ = _run(operation)
    assert results == [True]*4


def test_prepared_fixture_accepts_only_the_named_streamed_replay_class():
    from benchmarks.spdz_inference import StreamedDeviceReplay
    from benchmarks.spdz_prepared_fixture import replay_prepared_right
    def operation(p,x,w,t):
        prepared = _prepare(p,w,t,model_id=b'model-v1')
        transport = StreamedDeviceReplay([],device='cpu',check_outgoing=True)
        target = DistributedParty(p.party,transport,p.alpha_share,session=p.session,
                                   _allow_insecure_test_transport=True)
        cloned = replay_prepared_right(target,prepared)
        assert not cloned.used and not prepared.used
        assert cloned.model_id == prepared.model_id
        return True
    results,_,_,_ = _run(operation)
    assert results == [True]*4


@pytest.mark.parametrize('change',['identity','version'])
def test_prepared_fixture_rejects_changed_source_key_object(change):
    from benchmarks.spdz_prepared_fixture import replay_prepared_right
    from rep4gpu.transport import ReplayTransport
    def operation(p,x,w,t):
        prepared = _prepare(p,w,t,model_id=b'model-v1')
        if change == 'identity':
            p.alpha_share = p.alpha_share.clone()
        else:
            p.alpha_share.add_(0)
        replay = DistributedParty(p.party,ReplayTransport([]),p.alpha_share,session=p.session,
                                   _allow_insecure_test_transport=True)
        with pytest.raises(ValueError,match='source MAC-key'):
            replay_prepared_right(replay,prepared)
        assert not prepared.used
        return True
    results,_,_,_ = _run(operation)
    assert results == [True]*4
