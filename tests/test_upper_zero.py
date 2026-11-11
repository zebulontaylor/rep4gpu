import concurrent.futures

import pytest
import torch

from rep4gpu.runtime import Party, make_seeds
from rep4gpu.transport import IdealNetwork
from rep4gpu.upper_zero import upper_zero_residual, upper_zero_violation


def run_four(fn, bits=64):
    seeds, network = make_seeds(), IdealNetwork()
    parties = [Party(p, network.endpoint(p), {i: s for i, s in seeds.items() if i != p}, bits=bits)
               for p in range(4)]
    with concurrent.futures.ThreadPoolExecutor(4) as pool:
        return list(pool.map(fn, parties))


@pytest.mark.parametrize('width', [1, 27, 29, 63])
def test_arbitrary_ring_ranges(width):
    limit = 1 << width
    boundary = [-(1 << 63), -1, 0, 1, limit-1]
    if width < 63:
        boundary += [limit, limit+1]
    generator = torch.Generator().manual_seed(171)
    random = torch.empty(33, dtype=torch.int64).random_(generator=generator)
    random[::2] = -random[::2]-1
    values = torch.cat((torch.tensor(boundary, dtype=torch.int64), random))
    expected = torch.tensor([int(not 0 <= int(x) < limit) for x in values])

    def work(rt):
        secret = rt.input(1, values if rt.party == 1 else None, values.shape)
        residual = upper_zero_residual(rt, secret, width)
        violation = upper_zero_violation(rt, secret, width)
        return rt.open(residual), rt.open(violation)

    for residual, violation in run_four(work):
        assert torch.equal(residual, (values >> width) << width)
        assert torch.equal(violation, expected)


def test_high_valuation_residual_is_rejected():
    values = torch.tensor([-(1 << 63)], dtype=torch.int64)

    def work(rt):
        secret = rt.input(0, values if rt.party == 0 else None, values.shape)
        return rt.open(upper_zero_violation(rt, secret, 29))

    for actual in run_four(work):
        assert actual.item() == 1
    # Half of uniform ring challenges annihilate this nonzero residual.
    assert sum(((challenge * (1 << 63)) % (1 << 64)) == 0
               for challenge in range(256)) == 128


@pytest.mark.parametrize('width', [0, 64, True, 1.5])
def test_invalid_width(width):
    def work(rt):
        with pytest.raises(ValueError, match='width'):
            upper_zero_residual(rt, rt.public(0), width)
    run_four(work)


def test_reject_other_representations():
    def work(rt):
        with pytest.raises(ValueError, match='ring64 arithmetic'):
            upper_zero_residual(rt, rt.public(0, binary=True), 29)
    run_four(work)
    run_four(work, bits=32)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('width', [27, 29])
def test_gpu_packed_equalities(width):
    from rep4gpu.fused_runtime import FusedParty
    values = torch.tensor([-(1 << 63), -1, 0, 1, (1 << width)-1,
                           1 << width, (1 << width)+1], device='cuda').repeat(11)
    seeds, network = make_seeds(), IdealNetwork()

    def work(party):
        rt = FusedParty(party, network.endpoint(party),
                        {i: s for i, s in seeds.items() if i != party}, device='cuda')
        secret = rt.input(2, values if party == 2 else None, values.shape)
        return rt.open(upper_zero_violation(rt, secret, width)).cpu()

    with concurrent.futures.ThreadPoolExecutor(4) as pool:
        outputs = list(pool.map(work, range(4)))
    expected = ((values < 0) | (values >= (1 << width))).long().cpu()
    assert all(torch.equal(actual, expected) for actual in outputs)
