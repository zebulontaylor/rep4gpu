"""Packed final carry gates reuse the sign prefix without changing exact floors."""
from concurrent.futures import ThreadPoolExecutor
import random

import pytest
import torch

from rep4gpu.runtime import Share, make_seeds
from rep4gpu.transport import IdealNetwork

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def legacy_truncation_bits(rt, a, b, shift):
    from rep4gpu.carry_tree_swar import carry_bits
    low, high, before = carry_bits(rt, a, b, (shift, rt.bits, rt.bits-1))
    sign = Share(tuple(((left >> (rt.bits-1)) & 1) ^
                       ((right >> (rt.bits-1)) & 1) ^ carry
                       for left, right, carry in zip(a.parts, b.parts, before.parts)), True)
    return low, high, sign


def run_four(bits, work, binary_bits=None):
    from rep4gpu.device_inference import DeviceInferenceParty
    from rep4gpu.device_transport import DeviceIdealEndpoint
    torch.set_num_threads(1)
    network, seeds = IdealNetwork(timeout=180), make_seeds()
    device = torch.device('cuda:0')

    def guarded(role):
        try:
            endpoint = network.endpoint(role)
            p = DeviceInferenceParty(role, DeviceIdealEndpoint(endpoint, device),
                {i:s for i,s in seeds.items() if i != role}, device=device,
                bits=bits, binary_bits=bits if binary_bits is None else binary_bits,
                optimized=True, gpu_open=True,
                rounding='floor', sparse_products=True)
            return work(p, endpoint)
        except BaseException as error:
            network.abort(error)
            raise
    with ThreadPoolExecutor(4) as pool:
        return list(pool.map(guarded, range(4)))


def fixtures(bits, count):
    mask, half = (1 << bits)-1, 1 << (bits-1)
    edges = [(mask, 1), (half, half), (half-1, 1), (0, 0),
             (mask, mask), (half, 0), (1, half-1), (half+1, mask)]
    rng = random.Random(17+bits+count)
    pairs = (edges+[(rng.getrandbits(bits), rng.getrandbits(bits))
                   for _ in range(count)])[:count]
    def signed(v):
        return v-(1 << bits) if v & half else v
    a = torch.tensor([signed(x) for x,y in pairs], dtype=torch.int64)
    b = torch.tensor([signed(y) for x,y in pairs], dtype=torch.int64)
    return a, b, pairs


@pytest.mark.parametrize('bits,binary_bits', [(32,32), (64,64), (32,64)])
@pytest.mark.parametrize('count', [1, 65, 257])
def test_shared_prefix_matches_full_ring_integer_oracle(bits, binary_bits, count):
    from rep4gpu.carry_tree_swar import truncation_bits
    from rep4gpu.compact_conversion_gpu import weighted_b2a_gpu
    from rep4gpu.exact_truncation import trunc
    a, b, pairs = fixtures(bits, count)
    shifts = sorted({1, 14, 24, bits-1})
    mask = (1 << bits)-1

    def work(p, endpoint):
        sa = p.input(0, a if p.party == 0 else None, a.shape, binary=True)
        sb = p.input(1, b if p.party == 1 else None, b.shape, binary=True)
        sx = p.input(0, a if p.party == 0 else None, a.shape)
        rows = []
        for shift in shifts:
            carries = truncation_bits(p, sa, sb, shift)
            decoded = [p.open(v).cpu() for v in carries]
            output = trunc(p, sx, shift, carry_impl=truncation_bits,
                           convert_impl=weighted_b2a_gpu)
            rows.append((decoded, p.open(output).cpu()))
        return rows
    for result in run_four(bits, work, binary_bits):
        for shift, (actual, floor) in zip(shifts, result):
            low = torch.tensor([((x & ((1 << shift)-1))+
                                 (y & ((1 << shift)-1))) >> shift for x,y in pairs])
            high = torch.tensor([(x+y) >> bits for x,y in pairs])
            sign = torch.tensor([((x+y) & mask) >> (bits-1) for x,y in pairs])
            assert all(torch.equal(got, want) for got,want in zip(actual, (low,high,sign)))
            assert torch.equal(floor, a >> shift)


def measure_truncation(bits=64, elements=257, shift=24):
    from rep4gpu.carry_tree_swar import truncation_bits
    from rep4gpu.compact_conversion_gpu import weighted_b2a_gpu
    from rep4gpu.exact_truncation import trunc
    a, _, _ = fixtures(bits, elements)
    measurements = {}
    for name, carry in [('before',legacy_truncation_bits), ('after',truncation_bits)]:
        def work(p, endpoint):
            share = p.input(0, a if p.party == 0 else None, a.shape)
            start, rounds = endpoint.bytes_sent, endpoint.round
            output = trunc(p, share, shift, carry_impl=carry, convert_impl=weighted_b2a_gpu)
            payload, rounds = endpoint.bytes_sent-start, endpoint.round-rounds
            return payload, rounds, p.open(output).cpu()
        results = run_four(bits, work)
        assert all(torch.equal(row[2], a >> shift) for row in results)
        measurements[name] = {'bytes':sum(row[0] for row in results),
                              'rounds':results[0][1]}
    return measurements


@pytest.mark.parametrize('bits,shift', [(32,14),(64,24)])
def test_shared_prefix_reduces_truncation_traffic(bits, shift):
    result = measure_truncation(bits=bits, shift=shift)
    assert result['after']['bytes'] < result['before']['bytes']
    assert result['after']['rounds'] == result['before']['rounds']+1
