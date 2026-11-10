"""Checked-output and queue-lifetime tests for deferred masked openings."""
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

from rep4gpu.spdz import PRIME, TrustedDealer, field_matmul
from rep4gpu.spdz_distributed import AuthShare, split_share, split_triple
from rep4gpu.spdz_deferred import DeviceDeferredCompressedParty
from rep4gpu.device_transport import DeviceIdealEndpoint
from rep4gpu.transport import IdealNetwork


def _move(share):
    return AuthShare(share.values.to('cuda').int(), share.macs.to('cuda').int())


def _run(dealer, work, *, cap=2**25, threshold=0, fault=None):
    network = IdealNetwork(timeout=30)
    device = torch.device('cuda', torch.cuda.current_device())
    parties, events = [], [[] for _ in range(4)]
    phase = ['internal']*4
    for p in range(4):
        party = DeviceDeferredCompressedParty(p, DeviceIdealEndpoint(network.endpoint(p), device=device),
                    dealer.alpha_shares[:, p].to(device), session=b'deferred-test',
                    max_pending_values=cap[p] if isinstance(cap, list) else cap,
                    output_check_threshold=threshold[p] if isinstance(threshold, list) else threshold,
                    _allow_insecure_test_transport=True)
        gpu_broadcast, cpu_broadcast = party.broadcast_device, party.broadcast
        def gpu(payload, *, label, p=p, original=gpu_broadcast, **kwargs):
            events[p].append((phase[p], 'device', label))
            if fault is not None:
                payload = fault(p, 'device', label, payload)
            return original(payload, label=label, **kwargs)
        def cpu(payload, *, label, p=p, original=cpu_broadcast):
            events[p].append((phase[p], 'bytes', label))
            if fault is not None:
                payload = fault(p, 'bytes', label, payload)
            return original(payload, label=label)
        party.broadcast_device, party.broadcast = gpu, cpu
        parties.append(party)
    def role(p):
        try:
            return work(p, parties[p], phase)
        except BaseException as error:
            network.abort(error)
            return error
    with ThreadPoolExecutor(4) as pool:
        results = list(pool.map(role, range(4)))
    return results, parties, events


def _output_value_events(events):
    return [label for phase, _, label in events
            if phase == 'output' and label.startswith((b'SPDZ31G-independent-MAC-v1', b'SPDZ31G-open-v1')) and label.endswith((b'/values', b'values'))]


@pytest.mark.parametrize('keys', [7, 10])
@pytest.mark.parametrize('fault_kind', ['masked-value', 'coin-reveal', 'residual-reveal', 'peer-disconnect'])
@pytest.mark.parametrize('threshold', [0, 4096])
def test_pending_failure_aborts_before_first_unmasked_output_frame(fault_kind, threshold, keys):
    if not torch.cuda.is_available():
        pytest.skip('CUDA required')
    dealer = TrustedDealer(mac_keys=keys)
    x, y = torch.tensor([[0]]), torch.tensor([[7919]])
    xs, ys = split_share(dealer.share(x)), split_share(dealer.share(y))
    triples = split_triple(dealer.triple(1, 1, 1))
    tentative = [None]*4
    def fault(p, kind, label, payload):
        if p != 0:
            return payload
        if fault_kind == 'masked-value' and kind == 'device' and label.startswith(b'SPDZ31G-deferred-masked-v1'):
            payload = payload.clone()
            words = payload.view(torch.int32)
            words[0] = (words[0].to(torch.int64) + 1).remainder(PRIME).to(torch.int32)
        if b'SPDZ31G-deferred-check-v1' in label:
            if fault_kind == 'coin-reveal' and kind == 'bytes' and b'/coin/' in label and label.endswith(b'/reveal'):
                payload = payload[:-1] + bytes([payload[-1] ^ 1])
            if fault_kind == 'residual-reveal' and kind == 'device' and label.endswith(b'/reveal'):
                payload = payload.clone()
                payload[0] ^= 1
            if fault_kind == 'peer-disconnect' and kind == 'device' and label.endswith(b'/commit'):
                raise ConnectionError('synthetic peer disconnect during deferred flush')
        return payload
    def work(p, party, phase):
        triple = triples[p]
        triple.a, triple.b, triple.c = _move(triple.a), _move(triple.b), _move(triple.c)
        product = party.matmul(_move(xs[p]), _move(ys[p]), triple)
        assert party.pending_values == 2 and len(party._pending) == 1
        tentative[p] = product.values.cpu()
        phase[p] = 'output'
        return party.open(product)
    results, parties, events = _run(dealer, work, threshold=threshold, fault=fault)
    assert all(isinstance(result, BaseException) for result in results)
    assert all(party.poisoned and party.pending_values == 0 and not party._pending for party in parties)
    assert all(not _output_value_events(role_events) for role_events in events)
    if fault_kind == 'masked-value':
        # These synthetic local shares would reveal y if output shares were
        # sent before authenticating the earlier masked opening.
        assert torch.equal(sum(value.to(torch.int64) for value in tentative).remainder(PRIME), y)


@pytest.mark.parametrize('threshold', [0, 4096])
def test_three_operation_fixed_circuit_checks_queue_then_output(threshold):
    if not torch.cuda.is_available():
        pytest.skip('CUDA required')
    from rep4gpu.spdz_polynomial import make_square_masks, square
    dealer = TrustedDealer(mac_keys=10)
    x = torch.tensor([[3, 7, 11], [13, 17, 19]])
    w = torch.tensor([[2, 3], [5, 7], [11, 13]])
    v = torch.tensor([[17], [19]])
    xs, ws, vs = (split_share(dealer.share(value)) for value in (x, w, v))
    first = split_triple(dealer.triple(2, 3, 2))
    masks = make_square_masks(dealer, (2, 2))
    second = split_triple(dealer.triple(2, 2, 1))
    def work(p, party, phase):
        t1, t2, mask = first[p], second[p], masks[p]
        for triple in (t1, t2):
            triple.a, triple.b, triple.c = _move(triple.a), _move(triple.b), _move(triple.c)
        mask.a, mask.a_squared = _move(mask.a), _move(mask.a_squared)
        z = party.matmul(_move(xs[p]), _move(ws[p]), t1)
        z = square(party, z, mask)
        z = party.matmul(z, _move(vs[p]), t2)
        assert len(party._pending) == 3 and party.pending_values == 22
        phase[p] = 'output'
        output = party.open(z).cpu()
        assert not party._pending and party._flush_index == 1
        return output
    results, parties, events = _run(dealer, work, threshold=threshold)
    hidden = field_matmul(x, w)
    expected = field_matmul((hidden*hidden).remainder(PRIME), v)
    assert all(isinstance(value, torch.Tensor) and torch.equal(value, expected) for value in results)
    broadcasts = 12 if threshold == 0 else 10
    assert all(not party.poisoned and party.round == broadcasts for party in parties)
    assert all(len(_output_value_events(order)) == 1 for order in events)
    # Three masked broadcasts + one batch check + one checked output. The
    # square mask identifier is bound to its masked-opening label.
    assert all(len(order) == broadcasts for order in events)


@pytest.mark.parametrize('cap', [2, 1])
def test_public_queue_cap_flushes_before_next_or_oversized_masked_opening(cap):
    if not torch.cuda.is_available():
        pytest.skip('CUDA required')
    dealer = TrustedDealer(mac_keys=10)
    a = split_share(dealer.share(torch.tensor([[7, 9]])))
    b = split_share(dealer.share(torch.tensor([[11, 13]])))
    out = split_share(dealer.share(torch.tensor([[42]])))
    def work(p, party, phase):
        party._masked_open_many((_move(a[p]),), label=b'first-test-mask')
        if cap == 2:
            assert party.pending_values == 2
        else:
            assert party.pending_values == 0
        party._masked_open_many((_move(b[p]),), label=b'second-test-mask')
        if cap == 2:
            assert party._flush_index == 1 and party.pending_values == 2
        else:
            assert party._flush_index == 0 and party.pending_values == 0
        phase[p] = 'output'
        return party.open(_move(out[p])).cpu()
    results, parties, _ = _run(dealer, work, cap=cap)
    assert all(isinstance(value, torch.Tensor) and value.item() == 42 for value in results)
    assert all(party.pending_values == 0 and not party.poisoned for party in parties)
    assert all(party.round == 15 for party in parties)


def test_queue_freezes_residuals_before_input_and_candidate_mutations():
    if not torch.cuda.is_available():
        pytest.skip('CUDA required')
    dealer = TrustedDealer(mac_keys=10)
    local = split_share(dealer.share(torch.tensor([[7, 9]])))
    out = split_share(dealer.share(torch.tensor([[42]])))
    def work(p, party, phase):
        share = _move(local[p])
        candidate = party._masked_open_many((share,), label=b'freeze-test-mask')[0]
        pending = party._pending[0]
        before = pending.residual.clone()
        assert isinstance(pending.schema, bytes)
        assert pending.residual.data_ptr() != share.macs.data_ptr()
        share.macs.fill_(19)
        share.values.fill_(23)
        candidate.fill_(29)
        assert torch.equal(pending.residual, before)
        phase[p] = 'output'
        return party.open(_move(out[p])).cpu()
    results, parties, _ = _run(dealer, work)
    assert all(isinstance(value, torch.Tensor) and value.item() == 42 for value in results)
    assert all(not party.poisoned for party in parties)


def test_public_cap_disagreement_fails_closed():
    if not torch.cuda.is_available():
        pytest.skip('CUDA required')
    dealer = TrustedDealer(mac_keys=10)
    local = split_share(dealer.share(torch.tensor([[7, 9]])))
    def work(p, party, phase):
        return party._masked_open_many((_move(local[p]),), label=b'mismatched-cap')
    results, parties, _ = _run(dealer, work, cap=[1, 2, 2, 2])
    assert all(isinstance(result, BaseException) for result in results)
    assert all(party.poisoned for party in parties)


@pytest.mark.parametrize('cap', [0, -1, 2**31, True, 1.5, None])
def test_public_cap_validation_precedes_gpu_setup(cap):
    with pytest.raises(ValueError, match='max_pending_values'):
        DeviceDeferredCompressedParty(max_pending_values=cap)


def test_oversized_masked_opening_flushes_existing_queue_before_broadcast():
    if not torch.cuda.is_available():
        pytest.skip('CUDA required')
    dealer = TrustedDealer(mac_keys=10)
    prefix = split_share(dealer.share(torch.tensor([[7]])))
    oversized = split_share(dealer.share(torch.tensor([[11, 13, 17]])))
    out = split_share(dealer.share(torch.tensor([[42]])))
    def work(p, party, phase):
        party._masked_open_many((_move(prefix[p]),), label=b'prefix-mask')
        assert party.pending_values == 1
        party._masked_open_many((_move(oversized[p]),), label=b'oversized-mask')
        assert party.pending_values == 0 and party._flush_index == 1
        phase[p] = 'output'
        return party.open(_move(out[p])).cpu()
    results, parties, _ = _run(dealer, work, cap=2)
    assert all(isinstance(value, torch.Tensor) and value.item() == 42 for value in results)
    assert all(not party.poisoned and party.round == 15 for party in parties)


def test_elementwise_masked_hook_queues_and_checks_fresh_product():
    if not torch.cuda.is_available():
        pytest.skip('CUDA required')
    from rep4gpu.spdz import MatrixTriple as CentralTriple, _random
    from rep4gpu.spdz_distributed import split_elementwise_triple
    dealer = TrustedDealer(mac_keys=10)
    x = torch.tensor([[0, 1, PRIME - 1], [71, PRIME - 2, 91]])
    y = torch.tensor([[1, PRIME - 1, 91], [0, 71, PRIME - 2]])
    xs, ys = split_share(dealer.share(x)), split_share(dealer.share(y))
    a, b = _random(tuple(x.shape), 'cpu'), _random(tuple(x.shape), 'cpu')
    triples = split_elementwise_triple(CentralTriple(dealer.share(a), dealer.share(b), dealer.share((a*b).remainder(PRIME))))
    def work(p, party, phase):
        triple = triples[p]
        triple.a, triple.b, triple.c = _move(triple.a), _move(triple.b), _move(triple.c)
        product = party.mul(_move(xs[p]), _move(ys[p]), triple)
        assert party.pending_values == 12 and len(party._pending) == 1
        phase[p] = 'output'
        return party.open(product).cpu()
    results, parties, _ = _run(dealer, work)
    expected = (x*y).remainder(PRIME)
    assert all(isinstance(value, torch.Tensor) and torch.equal(value, expected) for value in results)
    assert all(not party.poisoned and party.round == 10 for party in parties)



@pytest.mark.parametrize('count', [0, 4, 5])
def test_hybrid_output_threshold_boundary_and_empty_output(count):
    if not torch.cuda.is_available():
        pytest.skip('CUDA required')
    dealer = TrustedDealer(mac_keys=10)
    masked = split_share(dealer.share(torch.tensor([[7]])))
    clear = torch.arange(count).reshape(1, count)
    output = split_share(dealer.share(clear))
    def work(p, party, phase):
        party._masked_open_many((_move(masked[p]),), label=b'fresh-test-mask')
        phase[p] = 'output'
        return party.open(_move(output[p])).cpu()
    results, parties, events = _run(dealer, work, threshold=4)
    expected_mode = 'full-vector' if count <= 4 else 'compressed'
    expected_rounds = 8 if count <= 4 else 10
    assert all(isinstance(value, torch.Tensor) and torch.equal(value, clear) for value in results)
    assert all(not party.poisoned and party.last_output_check_mode == expected_mode and party.round == expected_rounds
               for party in parties)
    assert all(len(_output_value_events(order)) == 1 for order in events)


@pytest.mark.parametrize('keys', [7, 10])
@pytest.mark.parametrize('threshold', [0, 4])
def test_hybrid_output_keeps_its_own_mac_authentication(threshold, keys):
    if not torch.cuda.is_available():
        pytest.skip('CUDA required')
    dealer = TrustedDealer(mac_keys=keys)
    masked = split_share(dealer.share(torch.tensor([[7]])))
    output = split_share(dealer.share(torch.tensor([[42]])))
    output[0].macs[keys - 1, 0, 0] = (output[0].macs[keys - 1, 0, 0] + 1) % PRIME
    def work(p, party, phase):
        party._masked_open_many((_move(masked[p]),), label=b'fresh-test-mask')
        phase[p] = 'output'
        return party.open(_move(output[p]))
    results, parties, events = _run(dealer, work, threshold=threshold)
    assert all(isinstance(value, BaseException) for value in results)
    assert all(party.poisoned for party in parties)
    assert all(len(_output_value_events(order)) == 1 for order in events)


def test_output_threshold_disagreement_is_bound_before_masked_continuation():
    if not torch.cuda.is_available():
        pytest.skip('CUDA required')
    dealer = TrustedDealer(mac_keys=10)
    local = split_share(dealer.share(torch.tensor([[7]])))
    def work(p, party, phase):
        return party._masked_open_many((_move(local[p]),), label=b'mismatched-output-policy')
    results, parties, _ = _run(dealer, work, threshold=[0, 4096, 4096, 4096])
    assert all(isinstance(value, BaseException) for value in results)
    assert all(party.poisoned for party in parties)



@pytest.mark.parametrize('threshold', [0, 4])
def test_seventh_pending_mac_coordinate_is_checked_before_output(threshold):
    if not torch.cuda.is_available():
        pytest.skip('CUDA required')
    dealer = TrustedDealer(mac_keys=7)
    masked = split_share(dealer.share(torch.tensor([[7]])))
    output = split_share(dealer.share(torch.tensor([[42]])))
    masked[0].macs[6, 0, 0] = (masked[0].macs[6, 0, 0] + 1) % PRIME
    def work(p, party, phase):
        party._masked_open_many((_move(masked[p]),), label=b'seventh-coordinate-test')
        phase[p] = 'output'
        return party.open(_move(output[p]))
    results, parties, events = _run(dealer, work, threshold=threshold)
    assert all(isinstance(value, BaseException) for value in results)
    assert all(party.poisoned and not party._pending for party in parties)
    assert all(not _output_value_events(order) for order in events)
