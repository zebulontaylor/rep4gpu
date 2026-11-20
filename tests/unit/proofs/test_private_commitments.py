import copy

import pytest
import torch

from rep4gpu.proofs.audit_model import AuditModel, example_spec
from rep4gpu.proofs.private_commitments import PrivateTrace, verify_private_opening


@pytest.fixture
def trace():
    model = AuditModel(example_spec())
    inputs = {'x': torch.tensor([[1, -2, 3, 4]], dtype=torch.int64)}
    return model, inputs, PrivateTrace(model, inputs, run_id='test', checkpoint_interval=2)


def verify(model, trace, opening, party=0, boundary=0, **kwargs):
    return verify_private_opening(trace.commitment, opening, party=party,
                                  expected_boundary=boundary, model=model,
                                  run_id=kwargs.get('run_id', 'test'), shape=kwargs.get('shape', [1, 4]))


def test_honest_reconstruction_and_local_slots(trace):
    model, inputs, committed = trace
    state = inputs
    for boundary in range(len(model.layers) + 1):
        if boundary:
            state = model.layers[boundary - 1].forward(state)
        components = {}
        for party in range(4):
            opening = committed.opening(boundary, party)
            expected = [(party + offset) % 4 for offset in (1, 2, 3)]
            assert [entry['excluded'] for entry in opening['components']] == expected
            assert party not in expected
            share = verify(model, committed, opening, party, boundary)
            for index, part in zip(expected, share.parts):
                if index in components:
                    assert torch.equal(part, components[index])
                components[index] = part
        reconstructed = sum(components.values(), torch.zeros_like(state['x']))
        assert torch.equal(reconstructed, state['x'])
    assert torch.equal(committed.output['x'], state['x'])
    assert committed.output['x'].data_ptr() != state['x'].data_ptr()


@pytest.mark.parametrize('mutation', ['salt', 'state', 'path', 'index', 'boundary', 'extra', 'order'])
def test_tampered_opening_rejected(trace, mutation):
    model, _, committed = trace
    opening = committed.opening(0, 0)
    entry = opening['components'][0]
    if mutation == 'salt':
        entry['salt'] = '00' * 32
    elif mutation == 'state':
        entry['state'] = opening['components'][1]['state']
    elif mutation == 'path':
        entry['siblings'][0] = '00' * 32
    elif mutation == 'index':
        entry['excluded'] = 0
    elif mutation == 'boundary':
        opening['boundary'] = 1
    elif mutation == 'extra':
        opening['components'].append(copy.deepcopy(entry))
    else:
        opening['components'].reverse()
    with pytest.raises(ValueError):
        verify(model, committed, opening)


@pytest.mark.parametrize('field,value', [('roots', ['00' * 32] * 4), ('run_id', 'other'),
                                          ('model_id', 'other'), ('shape', [2, 4]),
                                          ('layer_names', ['other']), ('version', True),
                                          ('scheme', 'other')])
def test_tampered_context_rejected(trace, field, value):
    model, _, committed = trace
    opening = committed.opening(0, 0)
    committed.commitment[field] = value
    with pytest.raises(ValueError):
        verify(model, committed, opening)


def test_wrong_recipient_and_expected_context_rejected(trace):
    model, _, committed = trace
    opening = committed.opening(0, 0)
    for kwargs in ({'party': 1}, {'shape': [2, 4]}, {'run_id': 'other'}, {'boundary': True}):
        with pytest.raises(ValueError):
            verify(model, committed, opening, **kwargs)


def test_fresh_salts_and_components(trace):
    model, inputs, committed = trace
    other = PrivateTrace(model, inputs, run_id='test')
    assert committed.commitment['roots'] != other.commitment['roots']
    assert committed.opening(0, 0) != other.opening(0, 0)


def test_components_cannot_move_between_boundaries_or_runs(trace):
    model, inputs, committed = trace
    opening = committed.opening(1, 0)
    opening['components'][2] = committed.opening(0, 0)['components'][2]
    with pytest.raises(ValueError):
        verify(model, committed, opening, boundary=1)
    other = PrivateTrace(model, inputs, run_id='test')
    opening = committed.opening(0, 0)
    opening['components'][2] = other.opening(0, 0)['components'][2]
    with pytest.raises(ValueError):
        verify(model, committed, opening)


def test_authentication_paths_only_cover_disclosed_component_trees(trace):
    _, _, committed = trace
    for party in range(4):
        opening = committed.opening(0, party)
        assert all(entry['excluded'] != party for entry in opening['components'])
        hidden_salt = committed._salts[0][party].hex()
        assert all(entry['salt'] != hidden_salt for entry in opening['components'])
        for entry in opening['components']:
            component = entry['excluded']
            assert entry['siblings'] == [s.hex() for s in committed._trees[component].proof(0)]
