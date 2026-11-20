"""Private audits open a verdict before releasing the authenticated output."""
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

from rep4gpu.protocol.runtime import Party, make_seeds
from rep4gpu.protocol.transport import IdealNetwork
from rep4gpu.proofs.audit_model import AuditModel
from rep4gpu.proofs.private_model import provision_private_model, load_private_model
from rep4gpu.proofs.rep4_audit import agreement, joint_random, verify_shared_trace
from rep4gpu.proofs.witness_activations import InvalidWitness


def run_four(work):
    torch.set_num_threads(1)
    network = IdealNetwork(timeout=20)
    seeds = make_seeds()
    parties = [Party(i, network.endpoint(i), {j: seed for j, seed in seeds.items() if j != i})
               for i in range(4)]

    def guarded(p):
        try:
            return work(p)
        except BaseException as exc:
            network.abort(exc)
            raise

    with ThreadPoolExecutor(4) as pool:
        return list(pool.map(guarded, parties))


@pytest.mark.parametrize('corrupt', [False, True])
@pytest.mark.parametrize('private_weights', [False, True])
def test_ring_trace_and_open_order(corrupt, private_weights):
    model = AuditModel({'format': 'siv-int64-relu-v1', 'width': 2,
                        'layers': [{'weight': [[2, -1], [1, 3]], 'bias': [1, -2]}] * 2})
    x = torch.tensor([[-2**63, 2**63 - 1], [3, -4]], dtype=torch.int64)
    plain = [x]
    for layer in model.layers:
        plain.append(layer.forward({'x': plain[-1]})['x'])
    manifest, bundles = provision_private_model(model)
    supplied = [t.clone() for t in plain]
    if corrupt:
        supplied[1][0, 0] += 1

    def work(p):
        checked_model = load_private_model(manifest, bundles[p.party], p.party) if private_weights else model
        # Each role receives only its three components through the checked input.
        states = {i: p.input(0, value if p.party == 0 else None, value.shape)
                  for i, value in enumerate(supplied)}
        assert all(len(value.parts) == 3 for value in states.values())
        opened = []
        original = p.open

        def observe(value):
            result = original(value)
            opened.append(result.clone())
            return result

        p.open = observe
        if corrupt:
            with pytest.raises(InvalidWitness):
                verify_shared_trace(p, checked_model, states, x, [0, 1])
            assert len(opened) == 1 and opened[0].shape == () and opened[0].item() == 0
        else:
            result = verify_shared_trace(p, checked_model, states, x, [0, 1])
            assert len(opened) == 2 and opened[0].shape == () and opened[0].item() == 1
            assert torch.equal(opened[1], result)
            assert torch.equal(result, plain[-1])
    run_four(work)


def test_coin_and_agreement():
    def work(p):
        agreement(p.transport, p.party, b'transcript')
        a = joint_random(p.transport, p.party, b'context one')
        b = joint_random(p.transport, p.party, b'context two')
        assert len(a) == len(b) == 32 and a != b
        return a, b
    results = run_four(work)
    assert all(r == results[0] for r in results)


def test_disagreement_rejected():
    def work(p):
        with pytest.raises(ValueError, match='agreement'):
            agreement(p.transport, p.party, bytes([p.party]))
    run_four(work)


def test_bad_input_and_schema():
    model = AuditModel({'format': 'siv-int64-relu-v1', 'width': 1,
                        'layers': [{'weight': [[1]], 'bias': [0]}]})
    supplied = torch.tensor([[7]], dtype=torch.int64)

    def work(p):
        state = p.input(0, supplied if p.party == 0 else None, supplied.shape)
        for states, indices in [({0: state, 1: state}, []), ({0: state}, [0]),
                                ({0: state, 1: state}, [0, 0])]:
            with pytest.raises(ValueError):
                verify_shared_trace(p, model, states, supplied, indices)
        original = p.open
        opened = []

        def observe(value):
            result = original(value)
            opened.append(result)
            return result

        p.open = observe
        with pytest.raises(InvalidWitness):
            verify_shared_trace(p, model, {0: state, 1: state}, supplied + 1, [0])
        assert len(opened) == 1 and opened[0].shape == () and opened[0].item() == 0
    run_four(work)


def test_coin_reveal_tamper_rejected():
    def work(p):
        class Tamper:
            calls = 0

            def exchange(self, outgoing):
                self.calls += 1
                # Commit broadcast plus two agreement rounds precede reveal.
                if p.party == 0 and self.calls == 4:
                    outgoing = {role: b'!' * 32 for role in outgoing}
                return p.transport.exchange(outgoing)

        with pytest.raises(ValueError):
            joint_random(Tamper(), p.party, b'audit context')
    run_four(work)
