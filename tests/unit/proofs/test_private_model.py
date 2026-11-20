"""Private model provisioning binds each role's random components to its manifest."""
import copy

import pytest
import torch

from rep4gpu.proofs.audit_model import AuditModel, example_spec
from rep4gpu.proofs.private_model import (
    load_private_model, provision_private_model, validate_manifest,
)


def test_parameter_sharing_and_identity():
    original = AuditModel(example_spec())
    manifest, bundles = provision_private_model(original)
    models = [load_private_model(manifest, bundles[p], p) for p in range(4)]
    assert manifest['model_id'] != original.model_id
    assert provision_private_model(original)[0]['model_id'] != manifest['model_id']
    assert all(m.model_id == manifest['model_id'] for m in models)
    for p, bundle in enumerate(bundles):
        assert [c['excluded'] for c in bundle['components']] == [(p + i) % 4 for i in (1, 2, 3)]
    for i, parameters in enumerate(original.parameters):
        for j, value in enumerate(parameters):
            parts = models[0].parameters[i][j].parts
            # Role 1's third slot is component 0, which role 0 never receives.
            reconstructed = sum(parts) + models[1].parameters[i][j].parts[2]
            assert torch.equal(reconstructed, value)
            assert all(not torch.equal(part, value) for part in parts)
    with pytest.raises(RuntimeError, match='Rep4'):
        models[0].layers[0].forward({'x': torch.zeros((1, 4), dtype=torch.int64)})


@pytest.mark.parametrize('tamper', ['state', 'salt', 'order', 'party', 'extra', 'missing'])
def test_private_bundle_tampering_rejected(tamper):
    manifest, bundles = provision_private_model(AuditModel(example_spec()))
    bundle = copy.deepcopy(bundles[0])
    if tamper == 'state':
        bundle['components'][0]['state'] = 'AAAA'
    elif tamper == 'salt':
        bundle['components'][0]['salt'] = '00' * 32
    elif tamper == 'order':
        bundle['components'].reverse()
    elif tamper == 'party':
        bundle['party'] = 1
    elif tamper == 'extra':
        bundle['components'].append(bundles[1]['components'][2])
    else:
        bundle['components'].pop()
    with pytest.raises(ValueError):
        load_private_model(manifest, bundle, 0)


def test_manifest_tampering_and_model_substitution_rejected():
    manifest, bundles = provision_private_model(AuditModel(example_spec()))
    changed = copy.deepcopy(manifest)
    changed['width'] += 1
    with pytest.raises(ValueError, match='identity'):
        validate_manifest(changed)
    _, other = provision_private_model(AuditModel(example_spec()))
    with pytest.raises(ValueError, match='commitment'):
        load_private_model(manifest, other[0], 0)
