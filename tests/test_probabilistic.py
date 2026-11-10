import concurrent.futures

import pytest
import torch

from rep4gpu.probabilistic import trunc_pr
from rep4gpu.runtime import Party, make_seeds
from rep4gpu.transport import IdealNetwork


def run_four(fn, bits=64):
    seeds = make_seeds()
    network = IdealNetwork()
    parties = [Party(p, network.endpoint(p), {i: s for i, s in seeds.items() if i != p}, bits=bits)
               for p in range(4)]
    with concurrent.futures.ThreadPoolExecutor(4) as pool:
        return list(pool.map(fn, parties))


@pytest.mark.parametrize('bits', [32, 64])
@pytest.mark.parametrize('shift', [1, 12, 29])
def test_bounded_signed_rounding(bits, shift):
    limit = 1 << (bits-2)
    generator = torch.Generator().manual_seed(164)
    x = torch.randint(-limit+1, limit, (1001,), generator=generator)
    x[:7] = torch.tensor([-limit+1, -4097, -1, 0, 1, 4097, limit-1])
    def work(p):
        secret = p.input(0, x if p.party == 0 else None, x.shape)
        before = p.multiplications
        result = trunc_pr(p, secret, shift)
        return p.open(result), p.multiplications-before
    outputs = run_four(work, bits)
    floor = x >> shift
    ceil = floor + ((x & ((1 << shift)-1)) != 0).long()
    for result, count in outputs:
        assert torch.all((result == floor) | (result == ceil))
        assert count == 1
        assert torch.equal(result, outputs[0][0])


@pytest.mark.parametrize('bits', [32, 64])
def test_exhaustive_low_mask_rounding_probability(bits):
    shift = 4
    residues = torch.arange(1 << shift)
    x = torch.tensor([-33, -17, -16, -15, -1, 0, 1, 15, 16, 17, 33])[:, None].expand(-1, 16)
    def work(p):
        secret = p.input(1, x if p.party == 1 else None, x.shape)
        original_random = p._random
        replaced = set()
        def controlled(excluded, shape, device, binary=False):
            fresh = original_random(excluded, shape, device, binary)
            if excluded in (2, 3) and excluded not in replaced:
                replaced.add(excluded)
                return residues.expand_as(x).clone() if excluded == 2 else torch.zeros_like(x)
            return fresh
        p._random = controlled
        return p.open(trunc_pr(p, secret, shift))
    for result in run_four(work, bits):
        expected = (x + residues) >> shift
        assert torch.equal(result, expected)
        assert torch.equal((result > (x >> shift)).sum(dim=1), x[:, 0] & 15)


def test_masked_opening_hash_is_checked_before_resharing():
    class BadTransport:
        def exchange(self, outgoing):
            return {0: bytes(8), 1: bytes(32)}
    seeds = make_seeds()
    party = Party(2, BadTransport(), {i: s for i, s in seeds.items() if i != 2})
    with pytest.raises(ValueError, match='hash mismatch'):
        trunc_pr(party, party.public(torch.tensor([0])), 12)


@pytest.mark.parametrize('bits', [32, 64])
def test_shift_validation_and_identity(bits):
    class NoTransport:
        def exchange(self, outgoing):
            raise AssertionError('validation must not communicate')
    seeds = make_seeds()
    party = Party(0, NoTransport(), {i: s for i, s in seeds.items() if i != 0}, bits=bits)
    x = party.public(torch.tensor([-1, 0, 1]))
    assert trunc_pr(party, x, 0) is x
    for shift in [-1, bits-1, 1.5]:
        with pytest.raises(ValueError):
            trunc_pr(party, x, shift)
    with pytest.raises(ValueError, match='arithmetic'):
        trunc_pr(party, party.public(0, binary=True), 1)
