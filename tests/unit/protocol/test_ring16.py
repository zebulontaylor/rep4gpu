"""Ring16 arithmetic, checked messages, and cross-ring mask separation."""
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

from rep4gpu.arithmetic.conversion import a2b, b2a
from rep4gpu.protocol.prg import random_bytes
from rep4gpu.protocol.runtime import Party, bytes_tensor, make_seeds, tensor_bytes
from rep4gpu.transports.byte import IdealNetwork


def signed16(value):
    return ((value & 0xffff) ^ 0x8000) - 0x8000


def run_four(function, binary_bits=32, prg='aes', mutate=None):
    seeds, network = make_seeds(), IdealNetwork(timeout=30)
    roles = []
    for p in range(4):
        transport = network.endpoint(p)
        if mutate is not None:
            mutate(p, transport)
        roles.append(Party(p, transport, {e: s for e, s in seeds.items() if e != p},
                           bits=16, binary_bits=binary_bits, prg=prg))

    def worker(rt):
        try:
            return function(rt)
        except BaseException as error:
            network.abort(error)
            raise

    with ThreadPoolExecutor(4) as pool:
        return list(pool.map(worker, roles))


@pytest.mark.parametrize('owner', range(4))
@pytest.mark.parametrize('binary_bits', [32, 64])
def test_full_width_inputs_products_and_sign_labels(owner, binary_bits):
    generator = torch.Generator().manual_seed(23)
    x = torch.empty((7, 5), dtype=torch.int64).random_(generator=generator)
    y = torch.empty((5, 3), dtype=torch.int64).random_(generator=generator)
    x[::2] = -x[::2] - 1
    y[::2] = -y[::2] - 1
    x[0] = torch.tensor([-(1 << 63), -32768, -1, 0, 32767])

    def work(rt):
        before = rt.transport.bytes_sent
        a = rt.input(owner, x if rt.party == owner else None, x.shape)
        input_bytes = rt.transport.bytes_sent - before
        b_owner = (owner + 1) % 4
        b = rt.input(b_owner, y if rt.party == b_owner else None, y.shape)
        before = rt.transport.bytes_sent
        product = rt.matmul(a, b)
        product_bytes = rt.transport.bytes_sent - before
        for value in (a, b, product, rt.mul_public(a, -100000)):
            assert all(bool(torch.all((part >= 0) & (part <= 0xffff))) for part in value.parts)
        labels = rt.sub(rt.public(1, like=a), rt.sign_bit(a))
        return (rt.open(a), rt.open(product), rt.open(labels),
                input_bytes, product_bytes)

    outputs = run_four(work, binary_bits, 'aes' if owner % 2 else 'shake')
    for opened, product, labels, _, _ in outputs:
        assert torch.equal(opened, signed16(x))
        assert torch.equal(product, signed16(x @ y))
        assert torch.equal(labels, (signed16(x) >= 0).long())
    assert sum(row[3] for row in outputs) == 2 * x.numel() * 2 + 4 * 32
    assert sum(row[4] for row in outputs) == 6 * 7 * 3 * 2 + 6 * 32


@pytest.mark.parametrize('binary_bits', [32, 64])
def test_conversion_and_signed_floor_boundaries(binary_bits):
    x = torch.tensor([-32768, -32767, -257, -1, 0, 1, 257, 32767])

    def work(rt):
        a = rt.input(0, x if rt.party == 0 else None, x.shape)
        binary = a2b(rt, a)
        assert all(bool(torch.all((part >= 0) & (part <= 0xffff))) for part in binary.parts)
        return (rt.open(binary), rt.open(b2a(rt, binary)), rt.open(rt.sign_bit(a)),
                *(rt.open(rt.trunc(a, shift)) for shift in (1, 7, 15)))

    for word, back, sign, *shifted in run_four(work, binary_bits):
        assert torch.equal(word, x & 0xffff)
        assert torch.equal(back, x)
        assert torch.equal(sign, (x < 0).long())
        for actual, shift in zip(shifted, (1, 7, 15)):
            assert torch.equal(actual, x >> shift)


@pytest.mark.parametrize('backend', ['shake', 'aes'])
def test_mask_width_and_domain_separation(backend):
    seed = bytes(range(32))
    arithmetic = random_bytes(seed, 4, 16, ring_bits=16, backend=backend)
    binary32 = random_bytes(seed, 4, 16, ring_bits=16, binary=True, binary_bits=32, backend=backend)
    binary64 = random_bytes(seed, 4, 16, ring_bits=16, binary=True, binary_bits=64, backend=backend)
    assert tuple(map(len, (arithmetic, binary32, binary64))) == (32, 64, 128)
    assert len({arithmetic, binary32[:32], binary64[:32]}) == 3
    for bits, binary, width in ((32, False, 64), (32, True, 32), (32, True, 64), (64, False, 64), (64, True, 64)):
        previous = random_bytes(seed, 4, 16, ring_bits=bits, binary=binary, binary_bits=width, backend=backend)
        assert all(value[:32] != previous[:32] for value in (arithmetic, binary32, binary64))


def test_wire_encoding_and_configuration_validation():
    x = torch.tensor([-(1 << 63), -32768, -1, 0, 32767, 65536]).reshape(2, 3).t()
    encoded = tensor_bytes(x, 16)
    assert len(encoded) == x.numel() * 2
    assert encoded == x.numpy().astype('<u2').tobytes()
    assert torch.equal(bytes_tensor(encoded, x.shape, 'cpu', 16), x & 0xffff)
    assert bytes_tensor(b'', (0, 7), 'cpu', 16).shape == (0, 7)
    for invalid in (24, 128):
        with pytest.raises(ValueError, match='wire bits'):
            tensor_bytes(x, invalid)
        with pytest.raises(ValueError, match='wire bits'):
            bytes_tensor(b'', (0,), 'cpu', invalid)
        with pytest.raises(ValueError, match='ring bits'):
            random_bytes(bytes(32), 0, 1, ring_bits=invalid)
    for data, shape in ((b'\x00', (1,)), (b'\x00\x00', (0,))):
        with pytest.raises(ValueError, match='byte count'):
            bytes_tensor(data, shape, 'cpu', 16)
    seeds = {e: bytes(32) for e in (1, 2, 3)}
    with pytest.raises(ValueError, match='CPU parties only'):
        Party(0, None, seeds, bits=16, device='cuda')
    with pytest.raises(ValueError, match='binary words'):
        Party(0, None, seeds, bits=16, binary_bits=16)


def test_ring16_corrupted_joint_product_is_rejected():
    def mutate(p, endpoint):
        if p != 0:
            return
        original = endpoint.exchange

        def corrupted(outgoing):
            outgoing = dict(outgoing)
            if 3 in outgoing and outgoing[3]:
                data = bytearray(outgoing[3])
                data[0] ^= 1
                outgoing[3] = bytes(data)
            return original(outgoing)

        endpoint.exchange = corrupted

    def work(rt):
        a = rt.public(torch.tensor([32767, -32768, 0, 1]))
        return rt.open(rt.mul(a, a))

    with pytest.raises(Exception, match='hash mismatch|network aborted'):
        run_four(work, mutate=mutate)
