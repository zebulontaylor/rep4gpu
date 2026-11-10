import concurrent.futures

import pytest
import torch

from rep4gpu.compact_conversion import b2a, weighted_b2a
from rep4gpu.conversion import b2a as reference_b2a
from rep4gpu.runtime import Party, make_seeds
from rep4gpu.transport import IdealNetwork


def run_four(fn, bits=64):
    seeds, network = make_seeds(), IdealNetwork()
    parties = [Party(p, network.endpoint(p), {i:s for i,s in seeds.items() if i != p}, bits=bits)
               for p in range(4)]
    with concurrent.futures.ThreadPoolExecutor(4) as pool:
        return list(pool.map(fn, parties))


@pytest.mark.parametrize('ring', [32, 64])
@pytest.mark.parametrize('selected,signed', [(1, False), (7, True), (23, False), (32, True)])
def test_word_conversion_matches_reference(ring, selected, signed):
    x = torch.tensor([-(1<<63), -1, -1234567, 0, 1, 1234567, (1<<63)-1])
    def work(p):
        binary = p.input(0, x if p.party==0 else None, x.shape, binary=True)
        new = b2a(p, binary, bits=selected, signed=signed)
        ref = reference_b2a(p, binary, bits=selected, signed=signed)
        return p.open(new), p.open(ref)
    for actual, expected in run_four(work, ring):
        assert torch.equal(actual, expected)


@pytest.mark.parametrize('ring', [32, 64])
def test_full_ring_and_payload(ring):
    generator = torch.Generator().manual_seed(442)
    x = torch.empty(67, dtype=torch.int64).random_(generator=generator)
    x[::2] = -x[::2]-1
    def work(p):
        binary = p.input(3, x if p.party==3 else None, x.shape, binary=True)
        before = p.transport.bytes_sent
        result = b2a(p, binary)
        sent = p.transport.bytes_sent-before
        before = p.transport.bytes_sent
        ref = reference_b2a(p, binary)
        reference_sent = p.transport.bytes_sent-before
        return p.open(result), p.open(ref), sent, reference_sent
    outputs = run_four(work, ring)
    for actual, expected, _, _ in outputs:
        assert torch.equal(actual, expected)
    assert sum(o[2] for o in outputs) < sum(o[3] for o in outputs)/4


@pytest.mark.parametrize('ring', [32, 64])
def test_arbitrary_weights_and_two_outputs(ring):
    planes = torch.tensor([[0, 1, 1, 0], [1, 0, 1, 0], [1, 1, 0, 0]])
    weights = torch.tensor([[1, -(1 << 20), -(1 << 20)], [0, 0, 1]])
    def work(p):
        secret = p.input(1, planes if p.party==1 else None, planes.shape, binary=True)
        return p.open(weighted_b2a(p, secret, weights)), p.open(weighted_b2a(p, secret, weights[0]))
    for outputs, single in run_four(work, ring):
        assert torch.equal(outputs, weights @ planes)
        assert torch.equal(single, (weights @ planes)[0])


def test_validation_requires_binary_planes_and_matching_weights():
    class NoTransport:
        def exchange(self, outgoing):
            raise AssertionError('invalid arguments must not communicate')
    seeds = make_seeds()
    party = Party(0, NoTransport(), {i:s for i,s in seeds.items() if i != 0})
    arithmetic = party.public(torch.zeros((3, 2), dtype=torch.int64))
    with pytest.raises(ValueError, match='binary'):
        weighted_b2a(party, arithmetic, [1, 2, 3])
    binary = party.public(torch.zeros((3, 2), dtype=torch.int64), binary=True)
    with pytest.raises(ValueError, match='weights'):
        weighted_b2a(party, binary, [1, 2])
