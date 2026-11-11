"""Direct CUDA witness execution agrees with the exact-floor prover arithmetic."""
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

from rep4gpu.arithmetic.numerical import NumericalFixedPoint
from rep4gpu.protocol.runtime import make_seeds
from rep4gpu.protocol.transport import IdealNetwork
from rep4gpu.proofs.witness_activations import (
    WitnessSession, input_witnesses, numerical_program, rmsnorm_program,
    softmax_program,
)
from rep4gpu.proofs.witness_prover import (
    _PlainFloor, numerical_witnesses, rmsnorm_witnesses, softmax_witnesses,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('kind', ['silu', 'rmsnorm', 'softmax'])
def test_direct_cuda_witness_activation(kind):
    from rep4gpu.models.device_inference import DeviceInferenceParty
    from rep4gpu.protocol.device_transport import DeviceIdealEndpoint

    torch.set_num_threads(1)
    scale = 1 << 24
    encoded = torch.round(torch.tensor([[-4., -0.5, 0., 4.],
                                       [0., 0., 0., 0.]], dtype=torch.float64)*scale).long()
    if kind == 'silu':
        program = numerical_program(kind)
        claims = numerical_witnesses(kind, {'x': encoded})

        expected = claims['output']
    elif kind == 'rmsnorm':
        program = rmsnorm_program()
        claims = rmsnorm_witnesses(encoded)
        expected = (encoded*claims['inverse_sqrt']) >> 24
    else:
        program = softmax_program()
        claims = softmax_witnesses(encoded)
        exponentials = NumericalFixedPoint(_PlainFloor()).exp(encoded-claims['shift'])
        expected = (exponentials*claims['reciprocal']) >> 24

    device = torch.device('cuda:0')
    network = IdealNetwork(timeout=120)
    seeds = make_seeds()
    shapes = {name: value.shape for name, value in claims.items()}

    def work(role):
        try:
            party = DeviceInferenceParty(
                role, DeviceIdealEndpoint(network.endpoint(role), device=device),
                {i: seed for i, seed in seeds.items() if i != role},
                device=device, bits=64, binary_bits=64, optimized=True,
                gpu_open=True, rounding='floor')
            x = party.input(0, encoded.to(device) if role == 0 else None, encoded.shape)
            witnesses = input_witnesses(party, program, 1, shapes,
                                       claims if role == 1 else None)
            session = WitnessSession(party)
            session.evaluate(program, {'x': x}, witnesses)
            output, = session.finish()
            return party.open(output).cpu()
        except BaseException as error:
            network.abort(error)
            raise

    with ThreadPoolExecutor(4) as pool:
        results = list(pool.map(work, range(4)))
    assert all(torch.equal(result, expected) for result in results)
