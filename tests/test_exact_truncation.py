import concurrent.futures

import pytest
import torch

from rep4gpu import conversion
from rep4gpu.exact_truncation import trunc, trunc_relu, trunc_four_component
from rep4gpu.runtime import Party, make_seeds
from rep4gpu.transport import IdealNetwork


def run_four(fn, bits=64):
    seeds, network = make_seeds(), IdealNetwork()
    parties = [Party(p, network.endpoint(p), {i:s for i,s in seeds.items() if i != p}, bits=bits)
               for p in range(4)]
    with concurrent.futures.ThreadPoolExecutor(4) as pool:
        return list(pool.map(fn, parties))


@pytest.mark.parametrize('bits', [32, 64])
@pytest.mark.parametrize('shift', [0, 1, 12, 29, 31])
def test_full_ring_shift_and_relu(bits, shift):
    minimum = -(1 << (bits-1))
    generator = torch.Generator().manual_seed(427)
    if bits == 64:
        random = torch.empty(101, dtype=torch.int64).random_(generator=generator)
        random[::2] = -random[::2]-1
    else:
        random = torch.randint(minimum, -minimum, (101,), generator=generator)
    x = torch.cat((torch.tensor([minimum, minimum+1, -4097, -1, 0, 1, 4097, -minimum-1]), random))
    def work(p):
        secret = p.input(2, x if p.party==2 else None, x.shape)
        return p.open(trunc(p, secret, shift)), p.open(trunc_relu(p, secret, shift))
    for actual, clipped in run_four(work, bits):
        assert torch.equal(actual, x >> shift)
        assert torch.equal(clipped, (x >> shift).clamp_min(0))


def test_maximum_shift_and_bandwidth_reduction():
    x = torch.arange(-32, 32)
    def work(p):
        secret = p.input(0, x if p.party==0 else None, x.shape)
        before = p.transport.bytes_sent
        new = trunc(p, secret, 12)
        new_bytes = p.transport.bytes_sent-before
        before = p.transport.bytes_sent
        word = conversion.a2b(p, secret)
        shifted = conversion._map(word, lambda t: (t >> 12) & ((1 << 52)-1))
        old = conversion.b2a(p, shifted, bits=52, signed=True)
        old_bytes = p.transport.bytes_sent-before
        maximum_shift = trunc(p, secret, 63)
        before = p.transport.bytes_sent
        reference = trunc_four_component(p, secret, 12)
        reference_bytes = p.transport.bytes_sent-before
        assert torch.equal(p.open(reference), x >> 12)
        return p.open(new), p.open(old), p.open(maximum_shift), new_bytes, old_bytes, reference_bytes
    outputs = run_four(work)
    for new, old, maximum_shift, _, _, _ in outputs:
        assert torch.equal(new, old)
        assert torch.equal(maximum_shift, x >> 63)
    assert sum(result[3] for result in outputs) < sum(result[4] for result in outputs)

    assert sum(result[3] for result in outputs) < sum(result[5] for result in outputs)
