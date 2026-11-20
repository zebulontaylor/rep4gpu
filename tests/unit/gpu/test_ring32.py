"""32-bit ring arithmetic, conversion, and payload regression tests."""
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

from rep4gpu.arithmetic.conversion import a2b, b2a, signed_lt
from rep4gpu.protocol.runtime import Party, make_seeds, tensor_bytes, bytes_tensor
from rep4gpu.transports.byte import IdealNetwork

VALUES = torch.tensor([-(1 << 31), -(1 << 31)+1, -4097, -2049, -2048, -1, 0, 1, 2048, 4096, (1 << 31)-1])


def signed32(x):
    return ((x & 0xffffffff) ^ 0x80000000) - 0x80000000


def run_four(work):
    seeds = make_seeds()
    net = IdealNetwork()
    parties = [Party(p, net.endpoint(p), {e: seed for e, seed in seeds.items() if e != p}, bits=32) for p in range(4)]
    with ThreadPoolExecutor(4) as pool:
        return list(pool.map(work, parties))


def test_wire_width_and_uniform_components():
    source = torch.tensor([0, 0xffffffff, 0x80000000])
    encoded = tensor_bytes(source, bits=32)
    assert len(encoded) == 12
    assert torch.equal(bytes_tensor(encoded, source.shape, 'cpu', bits=32), source)
    def work(p):
        x = p.input(0, VALUES if p.party == 0 else None, VALUES.shape)
        assert all(bool(torch.all((part >= 0) & (part <= 0xffffffff))) for part in x.parts)
        return p.open(x)
    for actual in run_four(work):
        assert torch.equal(actual, VALUES)


def test_full_ring_product_and_payload():
    y = torch.flip(VALUES, [0])
    def work(p):
        x = p.input(0, VALUES if p.party == 0 else None, VALUES.shape)
        b = p.input(1, y if p.party == 1 else None, y.shape)
        before = p.transport.bytes_sent
        result = p.mul(x, b)
        assert all(bool(torch.all((part >= 0) & (part <= 0xffffffff))) for part in result.parts)
        return p.open(result), p.transport.bytes_sent - before
    results = run_four(work)
    for actual, _ in results:
        assert torch.equal(actual, signed32(VALUES*y))
    # Includes multiplication followed by checked opening.
    assert sum(count for _, count in results) == 6*4*len(VALUES)+192 + 4*4*len(VALUES)+128


@pytest.mark.parametrize('shift', [0, 1, 12, 31])
def test_conversion_sign_truncation(shift):
    def work(p):
        x = p.input(2, VALUES if p.party == 2 else None, VALUES.shape)
        word = a2b(p, x)
        return p.open(word), p.open(b2a(p, word)), p.open(p.trunc(x, shift)), p.open(p.sign_bit(x)), p.open(p.trunc_relu(x, shift))
    for word, actual, truncated, sign, relu in run_four(work):
        assert torch.equal(word, VALUES & 0xffffffff)
        assert torch.equal(actual, VALUES)
        assert torch.equal(truncated, VALUES >> shift)
        assert torch.equal(sign, (VALUES < 0).long())
        assert torch.equal(relu, (VALUES >> shift).clamp_min(0))


def test_signed_comparison_overflow():
    x, y = torch.meshgrid(VALUES, VALUES, indexing='ij')
    def work(p):
        a = p.input(0, x if p.party == 0 else None, x.shape)
        b = p.input(1, y if p.party == 1 else None, y.shape)
        return p.open(signed_lt(p, a, b))
    for actual in run_four(work):
        assert torch.equal(actual, (x < y).long())


def test_matrix_overflow_and_public_operations():
    x = torch.tensor([[0x7fffffff, -7, 8193], [-0x80000000, 4096, -4097]])
    y = torch.tensor([[3, -1], [-5, 1 << 30], [2048, -8192]])
    def work(p):
        a = p.input(0, x if p.party == 0 else None, x.shape)
        b = p.input(3, y if p.party == 3 else None, y.shape)
        result = p.matmul(a, b)
        adjusted = p.sub(p.add(result, p.public(-1, like=result)), p.public(7, like=result))
        return p.open(result), p.open(p.mul_public(adjusted, 3))
    for actual, adjusted in run_four(work):
        assert torch.equal(actual, signed32(x@y))
        assert torch.equal(adjusted, signed32((x@y-8)*3))


def test_binary_domain_stays_full_word():
    x = torch.tensor([-(1<<63), -1, 0x7fffffff])
    def work(p):
        a = p.input(0, x if p.party == 0 else None, x.shape, binary=True)
        b = p.public(-1, like=a)
        before = p.transport.bytes_sent
        result = p.mul(a, b)
        sent = p.transport.bytes_sent - before
        return p.open(result), sent
    output = run_four(work)
    for actual, _ in output:
        assert torch.equal(actual, x)
    assert sum(sent for _, sent in output) == 6*8*len(x)+192
