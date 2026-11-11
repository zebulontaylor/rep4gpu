"""Derived activation bounds permit narrower signs without weakening input checks."""
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

from rep4gpu.numerical import NumericalFixedPoint
from rep4gpu.runtime import Party, make_seeds
from rep4gpu.transport import IdealNetwork
from rep4gpu.witness_activations import InvalidWitness, WitnessSession, softmax_program
from rep4gpu.witness_prover import _PlainFloor, numerical_witnesses, softmax_witnesses


def _four(work, device):
    torch.set_num_threads(1)
    seeds, network = make_seeds(), IdealNetwork(timeout=120)

    def run(role):
        try:
            selected = {i: s for i, s in seeds.items() if i != role}
            if device == 'cpu':
                party = Party(role, network.endpoint(role), selected)
            else:
                from rep4gpu.device_inference import DeviceInferenceParty
                from rep4gpu.device_transport import DeviceIdealEndpoint
                party = DeviceInferenceParty(
                    role, DeviceIdealEndpoint(network.endpoint(role), device), selected,
                    device=device, bits=64, binary_bits=64, optimized=True,
                    gpu_open=True, rounding='floor')
            return work(party)
        except BaseException as error:
            network.abort(error)
            raise

    with ThreadPoolExecutor(4) as pool:
        return list(pool.map(run, range(4)))


def _secret(party, value):
    return party.input(0, value if party.party == 0 else None, value.shape)


@pytest.mark.parametrize('fractional_bits', [20, 24])
@pytest.mark.parametrize('device', ['cpu', 'cuda:0'])
def test_absolute_sign_at_domain_endpoints(fractional_bits, device):
    if device.startswith('cuda') and not torch.cuda.is_available():
        pytest.skip('CUDA required')
    scale = 1 << fractional_bits
    values = torch.tensor([-4*scale, -4*scale+1, -1, 0, 1, 4*scale-1, 4*scale])

    def work(party):
        absolute, sign = NumericalFixedPoint(party, fractional_bits)._abs_sign(_secret(party, values))
        return party.open(absolute).cpu(), party.open(sign).cpu()

    for absolute, sign in _four(work, device):
        assert torch.equal(absolute, values.abs())
        assert torch.equal(sign, (values < 0).long())


@pytest.mark.parametrize('fractional_bits', [20, 24])
@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_softmax_narrow_sign_keeps_negative_eight_endpoint(fractional_bits):
    scale = 1 << fractional_bits
    values = torch.tensor([[-4*scale, 0, 4*scale], [4*scale, 4*scale, 4*scale]])
    claims = softmax_witnesses(values, fractional_bits)
    fp = NumericalFixedPoint(_PlainFloor(), fractional_bits)
    expected = fp.mul(fp.exp(values-claims['shift']), claims['reciprocal'])

    def work(party):
        inputs = {'x': _secret(party, values)}
        witnesses = {name: _secret(party, value) for name, value in claims.items()}
        session = WitnessSession(party, fractional_bits)
        session.evaluate(softmax_program(), inputs, witnesses)
        return party.open(session.finish()[0]).cpu()

    for result in _four(work, 'cuda:0'):
        assert torch.equal(result, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_untrusted_high_bits_are_not_discarded_by_softmax_narrow_sign():
    scale = 1 << 24
    # All of these would alias small valid values in a 28/29-bit sign circuit.
    values = torch.tensor([[-2**63, 4*scale], [2**63-1, 4*scale],
                           [1 << 40, 4*scale], [-(1 << 40), 4*scale]])
    safe = values.clamp(-4*scale, 4*scale)
    claims = softmax_witnesses(safe)

    def work(party):
        session = WitnessSession(party)
        inputs = {'x': _secret(party, values)}
        witnesses = {name: _secret(party, value) for name, value in claims.items()}
        session.evaluate(softmax_program(), inputs, witnesses)
        with pytest.raises(InvalidWitness):
            session.finish()

    _four(work, 'cuda:0')


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('kind', ['silu', 'gelu'])
def test_plaintext_cuda_witness_builder_uses_plain_sign(kind):
    values = torch.tensor([-4, -1, 0, 1, 4], device='cuda', dtype=torch.int64) << 24
    actual = numerical_witnesses(kind, {'x': values})
    expected = numerical_witnesses(kind, {'x': values.cpu()})
    assert actual.keys() == expected.keys()
    assert all(torch.equal(actual[name].cpu(), expected[name]) for name in actual)
