from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

from rep4gpu.activations import FixedPoint
from rep4gpu.inference import InferenceParty
from rep4gpu.runtime import make_seeds
from rep4gpu.transport import IdealNetwork


@pytest.mark.parametrize('bits', [32, 64])
@pytest.mark.parametrize('rounding', ['floor', 'probabilistic'])
def test_inference_runtime_rescale(bits, rounding):
    seeds, network = make_seeds(), IdealNetwork()
    values = torch.tensor([-12345, -4096, -1, 0, 1, 4096, 12345])
    def worker(party):
        p = InferenceParty(party, network.endpoint(party),
                           {e: seed for e, seed in seeds.items() if e != party},
                           device='cpu', bits=bits, prg='shake', rounding=rounding)
        x = p.input(0, values if party == 0 else None, values.shape)
        return p.open(p.trunc(x, 12)), p.open(p.trunc_relu(x, 12))
    with ThreadPoolExecutor(4) as pool:
        outputs = list(pool.map(worker, range(4)))
    for quotient, clipped in outputs:
        if rounding == 'floor':
            assert torch.equal(quotient, values >> 12)
            assert torch.equal(clipped, (values >> 12).clamp_min(0))
        else:
            assert bool(((quotient >= values >> 12) & (quotient <= (values + 4095) >> 12)).all())
            assert bool(((clipped >= (values >> 12).clamp_min(0)) &
                         (clipped <= ((values + 4095) >> 12).clamp_min(0))).all())
    assert all(torch.equal(outputs[0][0], pair[0]) and torch.equal(outputs[0][1], pair[1]) for pair in outputs)
