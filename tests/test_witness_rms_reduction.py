"""Tiled RMS square reduction keeps raw sums bounded and resharing scalar-sized."""
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

from rep4gpu.runtime import Party, make_seeds
from rep4gpu.transport import IdealNetwork
from rep4gpu.witness_activations import WitnessSession, rmsnorm_program
from rep4gpu.witness_prover import rmsnorm_witnesses


def _run(callback):
    torch.set_num_threads(1)
    seeds, network = make_seeds(), IdealNetwork(timeout=90)

    def work(role):
        party = Party(role, network.endpoint(role),
                      {i: seed for i, seed in seeds.items() if i != role})
        try:
            return callback(party)
        except BaseException as error:
            network.abort(error)
            raise

    with ThreadPoolExecutor(4) as pool:
        return list(pool.map(work, range(4)))


def _share(party, value):
    return party.input(0, value if party.party == 0 else None, value.shape)


@pytest.mark.parametrize('width', [3, 512, 513, 4096])
def test_tiled_rms_against_independent_integer_mean(width):
    scale = 1 << 24
    x = torch.tensor(([4*scale, -4*scale, scale//3, 2719] * ((width+3)//4))[:width]).reshape(1, width)
    sums = [sum(int(v)**2 for v in x[0, start:start+512]) >> 24
            for start in range(0, width, 512)]
    total = sum(sums)
    mean = ((total * round(scale/width)) >> 24 if width & (width-1)
            else total >> (width.bit_length()-1))
    variance = mean + round(1e-5*scale)
    expected_root = round((variance/scale)**-0.5 * scale)
    claim = rmsnorm_witnesses(x)['inverse_sqrt']
    assert int(claim) == expected_root

    def work(party):
        session = WitnessSession(party)
        session.evaluate(rmsnorm_program(), {'x': _share(party, x)},
                         {'inverse_sqrt': _share(party, claim)})
        return party.open(session.finish()[0])

    for output in _run(work):
        assert torch.equal(output, (x*expected_root) >> 24)


def test_square_reduction_traffic_and_rounding_improve():
    scale = 1 << 24
    # Every square has a fractional part; summing first preserves that precision.
    x = torch.full((2, 512), scale//3, dtype=torch.int64)

    def work(party):
        shared = _share(party, x)
        start = party.transport.bytes_sent
        old_squares = party.trunc(party.mul(shared, shared), 24)
        old = party._share(p.sum(-1, keepdim=True) for p in old_squares.parts)
        old_bytes = party.transport.bytes_sent-start
        start = party.transport.bytes_sent
        new = party.trunc(party._product(shared, shared,
                                        lambda a, b: (a*b).sum(-1, keepdim=True)), 24)
        new_bytes = party.transport.bytes_sent-start
        return party.open(old), party.open(new), old_bytes, new_bytes

    rows = _run(work)
    old_expected = ((x*x) >> 24).sum(-1, keepdim=True)
    new_expected = (x*x).sum(-1, keepdim=True) >> 24
    assert bool((new_expected > old_expected).all())
    assert sum(row[3] for row in rows) < sum(row[2] for row in rows)/100
    for old, new, _, _ in rows:
        assert torch.equal(old, old_expected)
        assert torch.equal(new, new_expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_cuda_rms_crosses_tile_boundary():
    from rep4gpu.device_inference import DeviceInferenceParty
    from rep4gpu.device_transport import DeviceIdealEndpoint

    torch.set_num_threads(1)
    device, scale = torch.device('cuda:0'), 1 << 24
    x = torch.full((1, 513), 4*scale, dtype=torch.int64)
    x[0, -1] = -(scale//3)
    root = rmsnorm_witnesses(x)['inverse_sqrt']
    seeds, network = make_seeds(), IdealNetwork(timeout=120)

    def work(role):
        party = DeviceInferenceParty(
            role, DeviceIdealEndpoint(network.endpoint(role), device),
            {i: seed for i, seed in seeds.items() if i != role},
            device=device, bits=64, binary_bits=64, optimized=True,
            gpu_open=True, rounding='floor')
        try:
            session = WitnessSession(party)
            session.evaluate(rmsnorm_program(), {'x': _share(party, x)},
                             {'inverse_sqrt': _share(party, root)})
            return party.open(session.finish()[0]).cpu()
        except BaseException as error:
            network.abort(error)
            raise

    with ThreadPoolExecutor(4) as pool:
        outputs = list(pool.map(work, range(4)))
    assert all(torch.equal(output, (x*root) >> 24) for output in outputs)
