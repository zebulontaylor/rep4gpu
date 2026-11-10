"""Independent ten-key fault tests for checked device Beaver operations."""
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

from rep4gpu.spdz import PRIME, TrustedDealer, field_matmul
from rep4gpu.spdz_distributed import MatrixTriple, split_share, split_triple
from rep4gpu.transport import IdealNetwork


def _run(operation, *, mutate=None, wire_fault=None, party_type=None):
    if not torch.cuda.is_available():
        pytest.skip('CUDA required')
    from rep4gpu.spdz_device import DeviceDistributedParty
    from rep4gpu.device_transport import DeviceIdealEndpoint
    party_type = DeviceDistributedParty if party_type is None else party_type

    dealer = TrustedDealer(parties=4, mac_keys=10, device='cuda')
    x = torch.tensor([[PRIME-1,2,3],[7,11,13]], dtype=torch.int64, device='cuda')
    y = torch.tensor([[1,2],[3,PRIME-2],[5,7]], dtype=torch.int64, device='cuda')
    xs,ys = split_share(dealer.share(x)),split_share(dealer.share(y))
    triples = split_triple(dealer.triple(2,3,2))
    network = IdealNetwork(timeout=15)
    parties = []
    for p in range(4):
        underlying = network.endpoint(p)
        transport = DeviceIdealEndpoint(underlying)
        if wire_fault is not None:
            exchange = transport.exchange_device
            captured = {}
            def faulty(outgoing, incoming_sizes, *, p=p, underlying=underlying,
                       exchange=exchange, captured=captured):
                outgoing = wire_fault(p,underlying.round,dict(outgoing),captured)
                return exchange(outgoing,incoming_sizes)
            transport.exchange_device = faulty
        party = party_type(p,transport,dealer.alpha_shares[:,p],
                           session=b'ten-key-independent-faults',
                           _allow_insecure_test_transport=True)
        if hasattr(party,'prepare_broadcast'):
            party.prepare_broadcast(32)
            for count in (x.numel(),x.numel()+y.numel(),x.shape[0]*y.shape[1]):
                party.prepare_broadcast(4*count,0)
                party.prepare_broadcast(32+40*count,32)
        parties.append(party)
        if mutate is not None:
            mutate(p,xs[p],ys[p],triples[p])

    def worker(p):
        try:
            return operation(parties[p],xs[p],ys[p],triples[p])
        except BaseException as error:
            network.abort(error)
            return error
    with ThreadPoolExecutor(4) as pool:
        results = list(pool.map(worker,range(4)))
    return results,parties,triples,field_matmul(x,y)


def _forbid_local_product(party):
    def forbidden(*args, **kwargs):
        raise AssertionError('unchecked opening reached local Beaver arithmetic')
    party._matmul_local = forbidden


def _assert_poisoned(results,parties):
    assert all(isinstance(result,BaseException) for result in results)
    assert not any(isinstance(result,AssertionError) for result in results)
    assert all(party.poisoned for party in parties)


def test_ten_key_device_beaver_matches_exact_field_product():
    results,parties,triples,expected = _run(lambda p,x,y,t:p.open(p.matmul(x,y,t)))
    assert all(isinstance(result,torch.Tensor) and torch.equal(result,expected) for result in results)
    assert all(not party.poisoned for party in parties)
    assert all(triple.used for triple in triples)


@pytest.mark.parametrize('kind',['value','tenth-mac'])
def test_corrupt_ten_key_share_is_rejected_before_local_product(kind):
    def mutate(p,x,y,triple):
        if p == 0:
            target = x.values if kind == 'value' else x.macs[9]
            target.reshape(-1)[0] = (target.reshape(-1)[0]+1) % PRIME
    def operation(p,x,y,triple):
        _forbid_local_product(p)
        return p.matmul(x,y,triple)
    results,parties,triples,_ = _run(operation,mutate=mutate)
    _assert_poisoned(results,parties)
    assert all(triple.used for triple in triples)


@pytest.mark.parametrize('round_index',range(6))
@pytest.mark.parametrize('consistent',[False,True])
def test_each_opening_phase_mutation_aborts_before_local_product(round_index,consistent):
    def fault(p,round_id,outgoing,captured):
        if p == 0 and round_id == round_index:
            targets = list(outgoing) if consistent else [next(iter(outgoing))]
            for peer in targets:
                changed = outgoing[peer].clone()
                changed[-1] ^= 1
                outgoing[peer] = changed
        return outgoing
    targets = {1,2,3} if consistent else {1}
    def operation(p,x,y,triple):
        if round_index != 5 or p.party in targets:
            _forbid_local_product(p)
        return p.matmul(x,y,triple)
    results,parties,triples,_ = _run(operation,wire_fault=fault)
    if round_index == 5:
        # Consistency with abort does not promise globally atomic delivery.
        # A receiver of an intact last echo can finish while another rejects.
        for receiver in targets:
            assert isinstance(results[receiver],BaseException)
            assert not isinstance(results[receiver],AssertionError)
            assert parties[receiver].poisoned
        assert not any(isinstance(result,AssertionError) for result in results)
    else:
        _assert_poisoned(results,parties)
    assert all(triple.used for triple in triples)


def test_replayed_previous_opening_frame_aborts_before_local_product():
    def fault(p,round_id,outgoing,captured):
        if p == 0 and round_id == 0:
            captured.update({peer:data.clone() for peer,data in outgoing.items()})
        if p == 0 and round_id == 6:
            return dict(captured)
        return outgoing
    def operation(p,x,y,triple):
        p.open(x)
        _forbid_local_product(p)
        return p.matmul(x,y,triple)
    results,parties,_,_ = _run(operation,wire_fault=fault)
    _assert_poisoned(results,parties)


def test_copied_spent_identifier_aborts_without_another_exchange():
    import threading
    completed = threading.Barrier(4)
    def operation(p,x,y,triple):
        p.matmul(x,y,triple)
        completed.wait(timeout=15)
        before = p.round
        replacement = MatrixTriple(triple.a,triple.b,triple.c,identifier=triple.identifier)
        with pytest.raises(ValueError,match='single-use|used|spent|reuse'):
            p.matmul(x,y,replacement)
        assert p.round == before
        assert p.poisoned
        return True
    results,_,_,_ = _run(operation)
    assert results == [True]*4


@pytest.mark.parametrize('stage',[b'values',b'commit',b'reveal'])
def test_consistent_malicious_payload_cannot_bypass_residual_guards(stage):
    def operation(p,x,y,triple):
        _forbid_local_product(p)
        if p.party == 0:
            broadcast = p.broadcast_device
            def malicious(payload, *, label, **kwargs):
                if label.endswith(stage):
                    payload = payload.clone()
                    payload[-1] ^= 1
                return broadcast(payload,label=label,**kwargs)
            p.broadcast_device = malicious
        return p.matmul(x,y,triple)
    results,parties,triples,_ = _run(operation)
    _assert_poisoned(results,parties)
    assert all(triple.used for triple in triples)


def test_production_device_open_draws_a_new_salt_for_each_opening():
    def operation(p,x,y,triple):
        salts = []
        original = p._commit_salt
        def record_salt():
            value = original()
            salts.append(value)
            return value
        p._commit_salt = record_salt
        first,second = p.open(x),p.open(x)
        assert torch.equal(first,second)
        assert len(salts) == 2
        assert all(isinstance(value,bytes) and len(value) == 32 for value in salts)
        assert salts[0] != salts[1]
        return True
    results,parties,_,_ = _run(operation)
    assert results == [True]*4
    assert all(not p.poisoned for p in parties)


def test_canonical_cache_rechecks_mutated_tensor_version_before_sending():
    sends = {}
    def operation(p,x,y,triple):
        p.validate_share(x)
        original = p.transport.exchange_device
        def count_send(outgoing,incoming_sizes):
            sends[p.party] = sends.get(p.party,0)+1
            return original(outgoing,incoming_sizes)
        p.transport.exchange_device = count_send
        if p.party == 0:
            x.macs[9,0,0] = PRIME
        return p.open(x)
    results,parties,_,_ = _run(operation)
    _assert_poisoned(results,parties)
    assert sends.get(0,0) == 0


def test_noncontiguous_ten_key_share_opening_matches_plaintext():
    def operation(p,x,y,triple):
        from rep4gpu.spdz_distributed import AuthShare
        view = AuthShare(x.values.transpose(0,1),x.macs.transpose(1,2))
        return p.open(view)
    results,parties,_,_ = _run(operation)
    expected = torch.tensor([[PRIME-1,7],[2,11],[3,13]],dtype=torch.int64,device='cuda')
    assert all(isinstance(value,torch.Tensor) and torch.equal(value,expected) for value in results)
    assert all(not party.poisoned for party in parties)
