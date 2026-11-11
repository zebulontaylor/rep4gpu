import concurrent.futures

import pytest
import torch

from rep4gpu.arithmetic.activations import FixedPoint
from rep4gpu.arithmetic.conversion import a2b, b2a, signed_lt
from rep4gpu.protocol.runtime import Party, make_seeds
from rep4gpu.protocol.transport import IdealNetwork


def run_four(fn, device='cpu'):
    seeds = make_seeds()
    network = IdealNetwork()
    parties = [Party(p, network.endpoint(p), {i: s for i, s in seeds.items() if i != p}, device)
               for p in range(4)]
    with concurrent.futures.ThreadPoolExecutor(4) as pool:
        return list(pool.map(fn, parties))


VALUES = torch.tensor([-(1 << 63), -(1 << 63) + 1, -(1 << 62), -4097,
                       -4096, -1, 0, 1, 4095, 4096, (1 << 62), (1 << 63) - 1])


@pytest.mark.parametrize('shift', [0, 1, 12, 32, 63])
def test_full_ring_truncation(shift):
    def work(p):
        x = p.input(0, VALUES if p.party == 0 else None, VALUES.shape)
        return p.open(p.trunc(x, shift))
    for actual in run_four(work):
        assert torch.equal(actual, VALUES >> shift)


def test_roundtrip_random():
    gen = torch.Generator().manual_seed(723)
    x = torch.empty(97, dtype=torch.int64).random_(generator=gen)
    x[::2] = -x[::2] - 1
    def work(p):
        secret = p.input(2, x if p.party == 2 else None, x.shape)
        binary = a2b(p, secret)
        return p.open(binary), p.open(b2a(p, binary))
    for binary, arithmetic in run_four(work):
        assert torch.equal(binary, x)
        assert torch.equal(arithmetic, x)


def test_signed_comparison_overflow():
    x, y = torch.meshgrid(VALUES, VALUES, indexing='ij')
    def work(p):
        a = p.input(0, x if p.party == 0 else None, x.shape)
        b = p.input(1, y if p.party == 1 else None, y.shape)
        return p.open(signed_lt(p, a, b))
    for actual in run_four(work):
        assert torch.equal(actual, (x < y).long())


def test_relu_edges():
    def work(p):
        x = p.input(3, VALUES if p.party == 3 else None, VALUES.shape)
        return p.open(FixedPoint(p, 12).relu(x))
    for actual in run_four(work):
        assert torch.equal(actual, VALUES.clamp_min(0))


@pytest.mark.parametrize('name', ['gelu', 'silu'])
def test_smooth_secret_integration(name):
    x = torch.tensor([-6., -3., -1., 0., .5, 1., 3., 6.])
    encoded = (x * (1 << 16)).long()
    def work(p):
        secret = p.input(1, encoded if p.party == 1 else None, encoded.shape)
        return p.open(getattr(FixedPoint(p, 16), name)(secret)).double() / (1 << 16)
    target = getattr(torch.nn.functional, name)(x)
    for actual in run_four(work):
        torch.testing.assert_close(actual, target.double(), atol=.075 if name == 'silu' else .008, rtol=0)
