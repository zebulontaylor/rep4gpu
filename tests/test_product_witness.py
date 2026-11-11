"""Product claims reject arbitrary ring errors without revealing residuals."""
import pytest
import torch

from rep4gpu.product_witness import ProductWitnessBatch
from rep4gpu.witness_activations import InvalidWitness, WitnessSession
from test_witness_check_optimization import _run_four, _secret


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('tamper', ['none', 'one', 'high', 'opposite'])
def test_product_claims(device, tamper):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA required')
    left = torch.tensor([-(1 << 63), -7, 0, 11, (1 << 63)-1], device=device)
    right = torch.tensor([-1, 9, 123, -4, 2], device=device)
    claimed = left*right
    if tamper == 'one':
        claimed[0] += 1
    elif tamper == 'high':
        claimed[0] += -(1 << 63)
    elif tamper == 'opposite':
        claimed[0] += 123
        claimed[1] -= 123
    def work(rt):
        batch = ProductWitnessBatch(rt, chunk_size=3)
        a, b, c = (_secret(rt, value) for value in (left, right, claimed))
        output = batch.add(a, b, c)
        # Mutations after registration cannot change any frozen claim.
        for share in (a, b, c):
            for part in share.parts:
                part.zero_()
        session = WitnessSession(rt)
        session._outputs.append(output)
        opened = []
        original_open = rt.open
        def tracked_open(value):
            opened.append(tuple(value.shape))
            return original_open(value)
        rt.open = tracked_open
        batch.verify(session)
        assert opened == [(4,)]
        with pytest.raises(RuntimeError, match='sealed'):
            batch.add(a, b, c)
        with pytest.raises(RuntimeError, match='sealed'):
            batch.verify(session)
        if tamper == 'none':
            result, = session.finish()
            assert opened == [(4,), ()]
            return original_open(result).cpu()
        with pytest.raises(InvalidWitness):
            session.finish()
        assert opened == [(4,), ()]
        return None
    results = _run_four(work, device)
    if tamper == 'none':
        assert all(torch.equal(result, (left*right).cpu()) for result in results)


def test_product_witness_reduces_large_batch_traffic():
    count = 16384
    left = torch.arange(count, dtype=torch.int64)-count//2
    right = left.flip(0)
    claimed = left*right
    def work(rt):
        a, b = _secret(rt, left), _secret(rt, right)
        before = rt.transport.bytes_sent
        expected = rt.mul(a, b)
        baseline = rt.transport.bytes_sent-before
        before = rt.transport.bytes_sent
        c = _secret(rt, claimed)
        batch = ProductWitnessBatch(rt)
        output = batch.add(a, b, c)
        session = WitnessSession(rt)
        session._outputs.append(output)
        batch.verify(session)
        actual, = session.finish()
        measured = rt.transport.bytes_sent-before
        assert torch.equal(rt.open(actual), rt.open(expected))
        return baseline, measured
    results = _run_four(work)
    assert sum(measured for _, measured in results) < sum(baseline for baseline, _ in results)
