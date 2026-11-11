"""Packed rejection aggregation preserves every input bit."""
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

from rep4gpu.arithmetic.binary_aggregation import binary_any
from rep4gpu.protocol.runtime import Party, make_seeds
from rep4gpu.protocol.transport import IdealNetwork
from rep4gpu.arithmetic.upper_zero import upper_zero_violation


def run_four(work):
    seeds, network = make_seeds(), IdealNetwork()
    def run(role):
        rt = Party(role, network.endpoint(role), {i:s for i,s in seeds.items() if i != role})
        try:
            return work(rt)
        except BaseException as error:
            network.abort(error)
            raise
    with ThreadPoolExecutor(4) as pool:
        return list(pool.map(run, range(4)))


@pytest.mark.parametrize('count', [1, 2, 63, 64, 65, 257])
@pytest.mark.parametrize('rejected', [False, True])
def test_binary_any(count, rejected):
    values = torch.zeros(count, dtype=torch.int64)
    if rejected:
        values[-1] = 1
    def work(rt):
        secret = rt.input(0, values if rt.party == 0 else None, values.shape, binary=True)
        pieces = [secret] if count == 1 else [secret[:1], secret[1:]]
        return rt.open(binary_any(rt, pieces))
    assert all(result.item() == int(rejected) for result in run_four(work))


@pytest.mark.parametrize('values', [[0, 0], [1, 1], [0, 1, 1, 0]])
def test_binary_only_session_acceptance(values):
    from rep4gpu.proofs.witness_activations import InvalidWitness, WitnessSession
    values = torch.tensor(values, dtype=torch.int64)
    def work(rt):
        secret = rt.input(0, values if rt.party == 0 else None, values.shape, binary=True)
        session = WitnessSession(rt)
        session._binary_violations.extend((secret[:1], secret[1:]))
        session._outputs.append(rt.public(torch.tensor(0)))
        try:
            session.finish()
            return True
        except InvalidWitness:
            return False
    assert run_four(work) == [not bool(values.any())]*4


@pytest.mark.parametrize('width', [1, 27, 29, 63])
def test_upper_zero_binary_result(width):
    limit = 1 << width
    values = torch.tensor([-(1 << 63), -1, 0, 1, limit-1], dtype=torch.int64)
    def work(rt):
        secret = rt.input(0, values if rt.party == 0 else None, values.shape)
        return rt.open(upper_zero_violation(rt, secret, width, binary=True))
    expected = ((values < 0) | (values >= limit)).long() if width < 63 else (values < 0).long()
    assert all(torch.equal(actual, expected) for actual in run_four(work))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('width', [17, 28, 29, 30])
def test_signed_width_binary_result(width):
    from rep4gpu.gpu.bits.bounded_activation import signed_width_sign
    from rep4gpu.protocol.fused_runtime import FusedParty
    bound = 1 << (width-1)
    values = torch.tensor([-bound, -1, 0, 1, bound-1], device='cuda').repeat(13)
    seeds, network = make_seeds(), IdealNetwork()
    def work(role):
        rt = FusedParty(role, network.endpoint(role),
                        {i:s for i,s in seeds.items() if i != role}, device='cuda')
        try:
            secret = rt.input(0, values if role == 0 else None, values.shape)
            return rt.open(signed_width_sign(rt, secret, width, binary=True)).cpu()
        except BaseException as error:
            network.abort(error)
            raise
    with ThreadPoolExecutor(4) as pool:
        outputs = list(pool.map(work, range(4)))
    assert all(torch.equal(actual, (values < 0).long().cpu()) for actual in outputs)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('bounds', [(-(1 << 63), -(1 << 63)+1), (-5, -1), (-4, 4),
                                    (0, 1), (3, 5), ((1 << 63)-4, (1 << 63)-2)])
def test_binary_interval_extreme_words(device, bounds):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA required')
    from rep4gpu.proofs.witness_activations import WitnessSession
    from tests.unit.proofs.test_witness_check_optimization import _run_four, _secret
    lower, upper = bounds
    values = torch.tensor([-(1 << 63), (1 << 63)-1, -5, -1, 0, 1, 5, lower, upper],
                          dtype=torch.int64, device=device)
    def work(rt):
        session = WitnessSession(rt)
        below, above = session._interval(_secret(rt, values), lower, upper, binary=True)
        return rt.open(below).cpu(), rt.open(above).cpu()
    for below, above in _run_four(work, device):
        assert torch.equal(below, (values < lower).long().cpu())
        assert torch.equal(above, (values > upper).long().cpu())


@pytest.mark.parametrize('tolerance', [0, 1, 5])
def test_residual_binary_rejection_at_extremes(tolerance):
    from rep4gpu.proofs.witness_activations import WitnessSession
    values = torch.tensor([-(1 << 63), (1 << 63)-1, -6, -1, 0, 1, 6], dtype=torch.int64)
    def work(rt):
        secret = rt.input(0, values if rt.party == 0 else None, values.shape)
        session = WitnessSession(rt)
        session.residual(secret, tolerance)
        below, above = session._binary_violations
        return rt.open(below), rt.open(above)
    for below, above in run_four(work):
        assert torch.equal(below, (values < -tolerance).long())
        assert torch.equal(above, (values > tolerance).long())
