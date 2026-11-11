"""Direct output claims remain bound to the recomputed SiLU exponential."""
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

from rep4gpu.runtime import Party, make_seeds
from rep4gpu.silu_exp import EXP_REMEZ6_Q24
from rep4gpu.transport import IdealNetwork
from rep4gpu.witness_activations import InvalidWitness, WitnessSession, numerical_program
from rep4gpu.witness_prover import _nearest_quotient, numerical_witnesses


SCALE = 1 << 24


def run_four(callback, device='cpu'):
    torch.set_num_threads(1)
    network, seeds = IdealNetwork(timeout=120), make_seeds()

    def run(role):
        try:
            known = {i: seed for i, seed in seeds.items() if i != role}
            if device == 'cpu':
                party = Party(role, network.endpoint(role), known)
            else:
                from rep4gpu.device_inference import DeviceInferenceParty
                from rep4gpu.device_transport import DeviceIdealEndpoint
                party = DeviceInferenceParty(role, DeviceIdealEndpoint(network.endpoint(role), device),
                    known, device=device, bits=64, binary_bits=64, optimized=True,
                    gpu_open=True, rounding='floor', sparse_products=True)
            return callback(party)
        except BaseException as error:
            network.abort(error)
            raise

    with ThreadPoolExecutor(4) as pool:
        return list(pool.map(run, range(4)))


def secret(party, value, owner=0):
    value = torch.as_tensor(value, dtype=torch.int64)
    return party.input(owner, value if party.party == owner else None, value.shape)


def test_nearest_quotient_integer_ties_and_large_signed_numerators():
    numerators = [-2**52+1, -11, -9, -7, -5, -3, -1, 0, 1, 3, 5, 7, 9, 11, 2**52-1]
    denominators = [33554431] + [2]*13 + [33554431]
    expected = []
    for numerator, denominator in zip(numerators, denominators):
        quotient, remainder = divmod(numerator, denominator)
        if 2*remainder > denominator or (2*remainder == denominator and quotient % 2):
            quotient += 1
        expected.append(quotient)
    assert torch.equal(_nearest_quotient(torch.tensor(numerators), torch.tensor(denominators)),
                       torch.tensor(expected))


def test_dense_direct_claim_accuracy_and_nearest_identity():
    torch.set_num_threads(1)
    encoded = torch.round(torch.linspace(-16, 16, 65537, dtype=torch.float64)*SCALE).long()
    output = numerical_witnesses('silu', {'x': encoded})['output']
    reduced = -encoded.abs() >> 4
    c0, c1, c2, c3, c4, c5, c6 = EXP_REMEZ6_Q24
    quadratic = reduced*reduced >> 24
    q2 = c4+((c5*reduced+c6*quadratic) >> 24)
    q1 = c2+((c3*reduced+quadratic*q2) >> 24)
    exponential = c0+((c1*reduced+quadratic*q1) >> 24)
    for _ in range(4):
        exponential = (exponential*exponential+SCALE//2) >> 24
    denominator = SCALE+exponential
    numerator = encoded*torch.where(encoded < 0, exponential, SCALE)
    error = output*denominator-numerator
    assert bool((2*error.abs() <= denominator).all())
    reference = torch.nn.functional.silu(encoded.double()/SCALE)
    assert float((output.double()/SCALE-reference).abs().max()) <= 6.41e-7


@pytest.mark.parametrize('device', ['cpu', 'cuda:0'])
@pytest.mark.parametrize('fractional_bits', [20, 24])
@pytest.mark.parametrize('kind', ['silu', 'swiglu'])
def test_honest_direct_output_and_only_acceptance_opened(device, fractional_bits, kind):
    if device.startswith('cuda') and not torch.cuda.is_available():
        pytest.skip('CUDA required')
    scale = 1 << fractional_bits
    encoded = torch.tensor([-16*scale, -4*scale, -scale, -1, 0, 1, scale, 4*scale, 16*scale])
    plain_inputs = {'x': encoded}
    if kind == 'swiglu':
        plain_inputs['value'] = torch.tensor([-4, 4, -4, 4, 0, -4, 4, -4, 4])*scale
    claim = numerical_witnesses(kind, plain_inputs, fractional_bits)['output']
    expected = claim if kind == 'silu' else (claim*plain_inputs['value']) >> fractional_bits

    def work(party):
        x, y = secret(party, encoded), secret(party, claim, owner=1)
        inputs = {'x': x}
        if kind == 'swiglu':
            inputs['value'] = secret(party, plain_inputs['value'])
        opened = []
        original_open = party.open
        def observe(value):
            result = original_open(value)
            opened.append(result.cpu())
            return result
        party.open = observe
        session = WitnessSession(party, fractional_bits)
        assert session.evaluate(numerical_program(kind), inputs, {'output': y}) is None
        assert opened == []
        output, = session.finish()
        assert len(opened) == 1 and opened[0].numel() == 1 and int(opened[0]) == 1
        return original_open(output).cpu()

    assert all(torch.equal(result, expected) for result in run_four(work, device))


@pytest.mark.parametrize('bad', [-2**63, -16*SCALE, -SCALE, 0, 16*SCALE, 2**63-1])
def test_bad_output_rejected_even_when_in_range(bad):
    def work(party):
        session = WitnessSession(party)
        session.evaluate(numerical_program('silu'), {'x': secret(party, [SCALE])},
                         {'output': secret(party, [bad], owner=1)})
        with pytest.raises(InvalidWitness):
            session.finish()
    run_four(work)


@pytest.mark.parametrize('delta', [-1, 1])
def test_one_ulp_wrong_output_at_zero_rejected(delta):
    def work(party):
        session = WitnessSession(party)
        session.evaluate(numerical_program('silu'), {'x': secret(party, [0])},
                         {'output': secret(party, [delta], owner=1)})
        with pytest.raises(InvalidWitness):
            session.finish()
    run_four(work)


@pytest.mark.parametrize('bad', [-2**63, -(17*SCALE), 17*SCALE, 2**63-1])
def test_input_bounds_are_not_replaced_by_output_bounds(bad):
    def work(party):
        session = WitnessSession(party)
        session.evaluate(numerical_program('silu'), {'x': secret(party, [bad])},
                         {'output': secret(party, [0], owner=1)})
        with pytest.raises(InvalidWitness):
            session.finish()
    run_four(work)


def test_extra_numerator_or_reciprocal_claims_are_not_accepted():
    def work(party):
        value = secret(party, [0])
        for claims in ({'output': value, 'numerator': value}, {'reciprocal': value}):
            session = WitnessSession(party)
            with pytest.raises(ValueError, match='witness names'):
                session.evaluate(numerical_program('silu'), {'x': value}, claims)
    run_four(work)


@pytest.mark.parametrize('device', ['cpu', 'cuda:0'])
def test_sign_first_clamp_rejects_every_signed_extreme(device):
    if device.startswith('cuda') and not torch.cuda.is_available():
        pytest.skip('CUDA required')
    limit = 16*SCALE
    encoded = torch.tensor([-2**63, -2**63+1, -limit-1, -limit, -1, 0,
                            1, limit, limit+1, 2**63-1])

    def work(party):
        session = WitnessSession(party)
        sign, absolute = session.signed_absolute_input(secret(party, encoded), limit)
        return party.open(sign).cpu(), party.open(absolute).cpu(), party.open(session._count).item()

    for sign, absolute, violations in run_four(work, device):
        assert torch.equal(sign, (encoded < 0).long())
        assert absolute.tolist() == [min(abs(int(x)), limit) for x in encoded]
        assert violations == 5


@pytest.mark.parametrize('device', ['cpu', 'cuda:0'])
def test_range_failures_cannot_cancel_in_batch(device):
    if device.startswith('cuda') and not torch.cuda.is_available():
        pytest.skip('CUDA required')
    # Two top-bit residuals sum to zero in ring64. Each must contribute one
    # violation, even if later output products wrap to a plausible relation.
    def work(party):
        session = WitnessSession(party)
        session.evaluate(numerical_program('silu'), {'x': secret(party, [0, 0])},
                         {'output': secret(party, [-2**63, -2**63], owner=1)})
        with pytest.raises(InvalidWitness):
            session.finish()
    run_four(work, device)


def test_narrow_sign_covers_entire_signed_width():
    if not torch.cuda.is_available():
        pytest.skip('CUDA required')
    encoded = torch.tensor([-2**27, -2**27+1, -2**26-1, -1, 0, 1, 2**27-1])

    def work(party):
        from rep4gpu.bounded_activation import signed_width_sign
        return party.open(signed_width_sign(party, secret(party, encoded), 28)).cpu()

    assert all(torch.equal(result, (encoded < 0).long()) for result in run_four(work, 'cuda:0'))
