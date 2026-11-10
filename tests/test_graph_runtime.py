import threading

import pytest
import torch

from rep4gpu.runtime import Party


def test_graph_draw_reservations_never_wrap_or_partially_commit():
    party = object.__new__(Party)
    party._counter_lock = threading.Lock()
    party.counters = {0: 2**64-1, 1: 8}
    with pytest.raises(ValueError, match='exhausted'):
        party._reserve_nonces((1, 0, 0))
    assert party.counters == {0: 2**64-1, 1: 8}
    assert party._reserve_nonces((0,)) == (2**64-1,)
    with pytest.raises(ValueError, match='exhausted'):
        party._reserve_nonces((0,))
    assert party.counters[0] == 2**64


@pytest.mark.parametrize('bits', [32, 64])
@pytest.mark.parametrize('count', [1, 17, 1025])
def test_graph_prg_reads_fresh_nonce_inputs(bits, count):
    if not torch.cuda.is_available():
        pytest.skip('CUDA required')
    from rep4gpu.gpu_prg import DomainKeys
    from rep4gpu.gpu_prg_batch import generate_rows
    from rep4gpu.gpu_prg_graph import generate_rows_into

    cache = DomainKeys(bytes(range(32)))
    keys = tuple(None if i == 2 else cache.key(bytes([i])) for i in range(6))
    nonces = torch.zeros(6, dtype=torch.uint64, device='cuda')
    output = torch.empty((6, count), dtype=torch.int64, device='cuda')
    generate_rows_into(keys, nonces, output, bits)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        generate_rows_into(keys, nonces, output, bits)
    previous = None
    for base in (0, 1, 2**32-1, 2**32+1, 2**63, 2**64-1):
        values = [None if key is None else base for key in keys]
        nonces.copy_(torch.tensor([v or 0 for v in values], dtype=torch.uint64, device='cuda'))
        graph.replay()
        expected = generate_rows(keys, values, count, bits)
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        if previous is not None:
            assert not torch.equal(output[0], previous)
        previous = output[0].clone()


@pytest.mark.parametrize('bits', [32, 64])
@pytest.mark.parametrize('binary', [False, True])
def test_prepared_products_match_fresh_eager_transcripts_and_keep_outputs(bits, binary):
    if not torch.cuda.is_available():
        pytest.skip('CUDA required')
    from concurrent.futures import ThreadPoolExecutor
    from rep4gpu.device_inference import DeviceInferenceParty
    from rep4gpu.graph_runtime import GraphDeviceParty
    from rep4gpu.device_transport import DeviceIdealEndpoint, DeviceRecorder
    from rep4gpu.runtime import make_seeds
    from rep4gpu.transport import IdealNetwork

    seeds = make_seeds()
    shape = (17,)

    def run(kind):
        network = IdealNetwork(timeout=90)
        records = [DeviceRecorder(DeviceIdealEndpoint(network.endpoint(p))) for p in range(4)]
        roles = [kind(p, records[p], {e:s for e,s in seeds.items() if e != p}, bits=bits) for p in range(4)]
        if kind is GraphDeviceParty:
            for role in roles:
                role.prepare_product(shape, binary, True)
        def worker(role):
            try:
                outputs, snapshots = [], []
                for value in (3, 5):
                    a = role.public(torch.full(shape, value, dtype=torch.int64, device=role.device), binary=binary)
                    b = role.public(torch.full(shape, value+1, dtype=torch.int64, device=role.device), binary=binary)
                    product = role.mul(a, b)
                    outputs.append(product)
                    snapshots.append(tuple(t.clone() for t in product.parts))
                torch.cuda.synchronize()
                for output, snapshot in zip(outputs, snapshots):
                    assert all(torch.equal(a,b) for a,b in zip(output.parts, snapshot))
                return snapshots, dict(role.counters)
            except BaseException as error:
                network.abort(error)
                raise
        with ThreadPoolExecutor(4) as pool:
            results = list(pool.map(worker, roles))
        return results, records

    expected, eager = run(DeviceInferenceParty)
    actual, graphs = run(GraphDeviceParty)
    for (actual_values, actual_counters), (expected_values, expected_counters), a, b in zip(actual, expected, graphs, eager):
        assert actual_counters == expected_counters
        for result, reference in zip(actual_values, expected_values):
            assert all(torch.equal(x,y) for x,y in zip(result, reference))
        assert len(a.transcript) == len(b.transcript) == 2
        for frame, reference in zip(a.transcript, b.transcript):
            assert frame.kind == reference.kind
            assert frame.outgoing == reference.outgoing
            assert frame.incoming == reference.incoming
        assert a.transcript[0].outgoing != a.transcript[1].outgoing


def test_corrupt_graph_receives_abort_before_assembly_or_next_message(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip('CUDA required')
    from concurrent.futures import ThreadPoolExecutor
    import rep4gpu.graph_runtime as module
    from rep4gpu.device_inference import DeviceInferenceParty
    from rep4gpu.device_transport import DeviceFrame, DeviceIdealEndpoint, DeviceRecorder, DeviceReplay
    from rep4gpu.runtime import make_seeds
    from rep4gpu.transport import IdealNetwork

    seeds = make_seeds()
    network = IdealNetwork(timeout=90)
    records = [DeviceRecorder(DeviceIdealEndpoint(network.endpoint(p))) for p in range(4)]
    shape = (17,)

    def operation(role):
        tensor = torch.full(shape, 7, dtype=torch.int64, device=role.device)
        a = role.public(tensor, binary=False)
        return role.mul(a, a)

    def record(p):
        try:
            role = DeviceInferenceParty(p, records[p], {e:s for e,s in seeds.items() if e != p})
            operation(role)
        except BaseException as error:
            network.abort(error)
            raise
    with ThreadPoolExecutor(4) as pool:
        list(pool.map(record, range(4)))

    def reject_assembly(*args, **kwargs):
        raise AssertionError('received data reached assembly before verification')
    monkeypatch.setattr(module, '_assemble', reject_assembly)
    for party, recording in enumerate(records):
        frame = recording.transcript[0]
        # Each sender's incoming region and each backup tag is independently
        # corrupted. No subsequent exchange or assembly is allowed.
        for peer, data in frame.incoming.items():
            incoming = dict(frame.incoming)
            incoming[peer] = bytes([data[0] ^ 1]) + data[1:]
            replay = DeviceReplay([DeviceFrame(frame.kind, frame.outgoing, incoming)], check_outgoing=True)
            role = module.GraphDeviceParty(party, replay, {e:s for e,s in seeds.items() if e != party})
            role.prepare_product(shape, False, True)
            with pytest.raises(ValueError, match='hash mismatch'):
                operation(role)
            assert replay.round == 1
            assert role.multiplications == 0


def test_prepared_graph_ffn_matches_consecutive_fresh_live_inferences():
    if not torch.cuda.is_available():
        pytest.skip('CUDA required')
    from concurrent.futures import ThreadPoolExecutor
    from rep4gpu.activations import FixedPoint
    from rep4gpu.device_inference import DeviceInferenceParty
    from rep4gpu.graph_runtime import GraphDeviceParty
    from rep4gpu.device_transport import DeviceIdealEndpoint, DeviceRecorder
    from rep4gpu.runtime import make_seeds
    from rep4gpu.transport import IdealNetwork

    seeds = make_seeds()
    plans = [set() for _ in range(4)]
    class PlanningParty(DeviceInferenceParty):
        def _fused_product(self, local, binary, device, pointwise=None):
            plans[self.party].add((tuple(local[0].shape), bool(binary), pointwise is not None))
            return super()._fused_product(local, binary, device, pointwise)

    scale = 1 << 14
    x = torch.tensor([[1,-2,3],[-3,1,2]], device='cuda', dtype=torch.int64)*(scale//8)
    w1 = torch.tensor([[1,2,-1,3],[-2,1,3,1],[1,-3,2,1]], device='cuda', dtype=torch.int64)*(scale//16)
    w2 = torch.tensor([[1,-1],[2,1],[-1,3],[2,-2]], device='cuda', dtype=torch.int64)*(scale//16)
    inputs = (x, w1, w2)
    cx, cw1, cw2 = (value.cpu() for value in inputs)
    expected = (torch.clamp((cx@cw1)>>14, min=0)@cw2)>>14

    def run(kind):
        network = IdealNetwork(timeout=90)
        records = [DeviceRecorder(DeviceIdealEndpoint(network.endpoint(p))) for p in range(4)]
        roles = [kind(p, records[p], {e:s for e,s in seeds.items() if e != p}, bits=32) for p in range(4)]
        if kind is GraphDeviceParty:
            for role in roles:
                for shape,binary,pointwise in sorted(plans[role.party]):
                    role.prepare_product(shape,binary,pointwise)
        def worker(role):
            try:
                results = []
                for _ in range(2):
                    fp = FixedPoint(role, fractional_bits=14)
                    shared = [role.input(owner, value if role.party == owner else None, value.shape)
                              for owner,value in zip((0,1,1),inputs)]
                    hidden = fp.matmul_relu(shared[0],shared[1])
                    output = fp.matmul(hidden,shared[2])
                    opened = role.open(output).cpu()
                    assert torch.equal(opened,expected)
                    results.append((tuple(t.cpu() for t in output.parts),dict(role.counters)))
                return results
            except BaseException as error:
                network.abort(error)
                raise
        with ThreadPoolExecutor(4) as pool:
            results = list(pool.map(worker,roles))
        return results,records

    expected_results,eager_records = run(PlanningParty)
    actual_results,graph_records = run(GraphDeviceParty)
    for reference,actual,eager,graph in zip(expected_results,actual_results,eager_records,graph_records):
        for (reference_parts,reference_counters),(actual_parts,actual_counters) in zip(reference,actual):
            assert actual_counters == reference_counters
            assert all(torch.equal(a,b) for a,b in zip(actual_parts,reference_parts))
        assert len(eager.transcript) == len(graph.transcript)
        for a,b in zip(eager.transcript,graph.transcript):
            assert a.kind == b.kind
            assert a.outgoing == b.outgoing
            assert a.incoming == b.incoming
