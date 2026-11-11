import concurrent.futures
import hashlib
import struct
import pytest
import torch
from rep4gpu.protocol.prg import random_bytes
from rep4gpu.protocol.runtime import Party, make_seeds
from rep4gpu.protocol.transport import IdealNetwork


def test_shake_reference_bytes():
    seed = bytes(range(32))
    for ring_bits, binary in ((64, False), (64, True), (32, False), (32, True)):
        width = 64 if binary else ring_bits
        context = b'Rep4GPU-mask-v1' + struct.pack('<QQ', 19, 7)
        if ring_bits == 32:
            context += bytes([width])
        expected = hashlib.shake_256(seed + context).digest(7 * width // 8)
        assert random_bytes(seed, 19, 7, ring_bits=ring_bits, binary=binary) == expected


def test_aes_determinism_and_domain_separation():
    pytest.importorskip('cryptography')
    seed = bytes(range(32))
    original = random_bytes(seed, 0, 7, backend='aes')
    assert len(original) == 56
    assert original == random_bytes(seed, 0, 7, backend='aes')
    candidates = [random_bytes(seed, 1, 7, backend='aes'),
                  random_bytes(bytes(reversed(seed)), 0, 7, backend='aes'),
                  random_bytes(seed, 0, 8, backend='aes')[:56],
                  random_bytes(seed, 0, 7, binary=True, backend='aes'),
                  random_bytes(seed, 0, 7, ring_bits=32, binary=True, backend='aes')]
    assert all(value != original for value in candidates)
    assert random_bytes(seed, 0, 0, backend='aes') == b''


@pytest.mark.parametrize('kwargs', [dict(counter=1 << 64, count=1), dict(counter=-1, count=1),
                                  dict(counter=0, count=1 << 60), dict(counter=0, count=-1)])
def test_prg_counter_bounds(kwargs):
    with pytest.raises(ValueError):
        random_bytes(bytes(32), **kwargs, backend='aes')


def test_aes_holders_share_fresh_draws():
    pytest.importorskip('cryptography')
    seeds = make_seeds()
    network = IdealNetwork()
    parties = [Party(p, network.endpoint(p), {e:s for e,s in seeds.items() if e != p}, prg='aes') for p in range(4)]
    for excluded in range(4):
        holders = [p for p in parties if p.party != excluded]
        first = [p._random_bytes(excluded, (3,)) for p in holders]
        second = [p._random_bytes(excluded, (3,)) for p in holders]
        assert len(set(first)) == len(set(second)) == 1
        assert first[0] != second[0]
        assert all(p.counters[excluded] == 2 for p in holders)
        with pytest.raises(KeyError):
            parties[excluded]._random_bytes(excluded, (3,))


@pytest.mark.parametrize('bits,binary', [(64, False), (64, True), (32, False), (32, True)])
def test_aes_runtime_product(bits, binary):
    pytest.importorskip('cryptography')
    seeds = make_seeds()
    network = IdealNetwork()
    x = torch.tensor([-(1 << 63), -1, 7, 17], dtype=torch.int64)
    y = torch.tensor([-1, 7, 2**62, -91], dtype=torch.int64)
    def work(party):
        runtime = Party(party, network.endpoint(party), {e:s for e,s in seeds.items() if e != party}, bits=bits, prg='aes')
        a = runtime.input(0, x if party == 0 else None, x.shape, binary=binary)
        b = runtime.input(1, y if party == 1 else None, y.shape, binary=binary)
        return runtime.open(runtime.mul(a, b))
    expected = x & y if binary else x * y
    if bits == 32 and not binary:
        expected = ((expected & 0xffffffff) ^ 0x80000000) - 0x80000000
    with concurrent.futures.ThreadPoolExecutor(4) as pool:
        actual = list(pool.map(work, range(4)))
    assert all(torch.equal(value, expected) for value in actual)


@pytest.mark.parametrize('backend', ['shake', 'aes'])
def test_binary32_conversion_and_traffic(backend):
    if backend == 'aes':
        pytest.importorskip('cryptography')
    from rep4gpu.arithmetic.conversion import a2b, b2a
    seeds = make_seeds()
    network = IdealNetwork()
    value = torch.tensor([-(1 << 31), -1, 0, 17, (1 << 31)-1], dtype=torch.int64)
    def work(party):
        transport = network.endpoint(party)
        runtime = Party(party, transport, {e:s for e,s in seeds.items() if e != party},
                        bits=32, binary_bits=32, prg=backend)
        x = runtime.input(0, value if party == 0 else None, value.shape)
        binary = a2b(runtime, x)
        assert all(bool(torch.all((part >= 0) & (part <= 0xffffffff))) for part in binary.parts)
        before = transport.bytes_sent
        product = runtime.mul(binary, binary)
        sent = transport.bytes_sent - before
        output = runtime.open(b2a(runtime, product))
        return output, sent
    with concurrent.futures.ThreadPoolExecutor(4) as pool:
        actual = list(pool.map(work, range(4)))
    assert all(torch.equal(output, value) for output, _ in actual)
    assert sum(sent for _, sent in actual) == 6 * value.numel() * 4 + 6 * 32


def test_binary32_prg_domain_and_configuration():
    seed = bytes(range(32))
    a = random_bytes(seed, 0, 7, ring_bits=32, binary=True, binary_bits=32)
    assert len(a) == 28
    assert a != random_bytes(seed, 0, 7, ring_bits=32)
    with pytest.raises(ValueError, match='32-bit ring'):
        Party(0, None, {e:seed for e in (1, 2, 3)}, binary_bits=32)
