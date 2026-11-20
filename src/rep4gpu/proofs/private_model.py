"""Provision authenticated Rep4 model shares from a trusted model owner.

The owner approves and distributes the manifest independently of the prover.
Its identity binds salted share components, not a separately claimed plaintext
model hash. Each verifier receives only its three replicated components.
"""
from __future__ import annotations

import base64
import binascii
import copy
import hashlib
import json
import secrets

import numpy as np
import torch

from ..protocol.runtime import Share
from .audit_model import AuditModel
from .layer_commitments import Layer, decode_state, encode_state
from .private_commitments import _hex, _slots

_FORMAT = 'siv-private-int64-relu-v1'


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode('ascii')


def _identity(manifest):
    content = {key: value for key, value in manifest.items() if key != 'model_id'}
    return 'sha256:' + hashlib.sha256(b'siv.private-model.identity.v1\0' + _canonical(content)).hexdigest()


def validate_manifest(manifest):
    """Validate public dimensions, component commitments, and model identity."""
    if (not isinstance(manifest, dict)
            or set(manifest) != {'format', 'width', 'layer_count', 'component_digests', 'model_id'}
            or manifest['format'] != _FORMAT
            or type(manifest['width']) is not int or not 1 <= manifest['width'] <= 128
            or type(manifest['layer_count']) is not int or not 1 <= manifest['layer_count'] <= 64
            or not isinstance(manifest['component_digests'], list)
            or len(manifest['component_digests']) != 4):
        raise ValueError('invalid private model manifest')
    for digest in manifest['component_digests']:
        _hex(digest)
    if manifest['model_id'] != _identity(manifest):
        raise ValueError('private model identity does not match its manifest')
    return copy.deepcopy(manifest)


def _component_digest(manifest, excluded, salt, payload):
    metadata = {key: manifest[key] for key in ('format', 'width', 'layer_count')}
    return hashlib.sha256(b'siv.private-model.component.v1\0' + _canonical(metadata)
                          + bytes([excluded]) + salt + payload).hexdigest()


def provision_private_model(model):
    """Return a public manifest and four private role bundles with fresh shares.

    The caller must authenticate the manifest to each verifier and deliver each
    bundle only to its named role. Keep the original model with the owner and
    prover; do not distribute it to the verifiers.
    """
    if not isinstance(model, AuditModel) or isinstance(model, PrivateAuditModel):
        raise ValueError('provisioning requires a plaintext AuditModel')
    manifest = {'format': _FORMAT, 'width': model.width, 'layer_count': len(model.layers)}
    states = [{} for _ in range(4)]
    for index, parameters in enumerate(model.parameters):
        for name, value in zip(('weight', 'bias'), parameters):
            parts = [torch.from_numpy(np.frombuffer(secrets.token_bytes(value.numel() * 8),
                                                   dtype='<i8').copy().reshape(value.shape))
                     for _ in range(3)]
            parts.append(value - parts[0] - parts[1] - parts[2])
            for component, part in enumerate(parts):
                states[component][f'{index}.{name}'] = part
    components, digests = [], []
    for excluded, state in enumerate(states):
        salt, payload = secrets.token_bytes(32), encode_state(state)
        digests.append(_component_digest(manifest, excluded, salt, payload))
        components.append({'excluded': excluded, 'salt': salt.hex(),
                           'state': base64.b64encode(payload).decode('ascii')})
    manifest['component_digests'] = digests
    manifest['model_id'] = _identity(manifest)
    bundles = [{'version': 1, 'party': party,
                'components': [copy.deepcopy(components[c]) for c in _slots(party)]}
               for party in range(4)]
    return manifest, bundles


def _no_plaintext_forward(state):
    raise RuntimeError('private model transitions require the Rep4 runtime')


class PrivateAuditModel(AuditModel):
    """Model metadata and authenticated local parameter shares, without plaintext weights."""

    def __init__(self, manifest, parameters):
        self.width = manifest['width']
        self.model_id = manifest['model_id']
        self.layers = tuple(Layer(f'affine_relu_{i}', _no_plaintext_forward)
                            for i in range(manifest['layer_count']))
        self.parameters = tuple(parameters)


def load_private_model(manifest, bundle, party_id):
    """Authenticate this role's three components before constructing parameter shares."""
    manifest = validate_manifest(manifest)
    slots = _slots(party_id)
    if (not isinstance(bundle, dict) or set(bundle) != {'version', 'party', 'components'}
            or type(bundle['version']) is not int or bundle['version'] != 1
            or type(bundle['party']) is not int or bundle['party'] != party_id
            or not isinstance(bundle['components'], list) or len(bundle['components']) != 3):
        raise ValueError('invalid private model bundle')
    expected = {f'{i}.{name}': ((manifest['width'], manifest['width']) if name == 'weight'
                              else (manifest['width'],))
                for i in range(manifest['layer_count']) for name in ('weight', 'bias')}
    states = []
    for excluded, entry in zip(slots, bundle['components']):
        if (not isinstance(entry, dict) or set(entry) != {'excluded', 'salt', 'state'}
                or type(entry['excluded']) is not int or entry['excluded'] != excluded
                or not isinstance(entry['state'], str)):
            raise ValueError('invalid private model component schema or order')
        salt = _hex(entry['salt'])
        try:
            payload = base64.b64decode(entry['state'], validate=True)
        except (ValueError, binascii.Error) as exc:
            raise ValueError('invalid private model component encoding') from exc
        if base64.b64encode(payload).decode('ascii') != entry['state']:
            raise ValueError('noncanonical private model component encoding')
        if _component_digest(manifest, excluded, salt, payload) != manifest['component_digests'][excluded]:
            raise ValueError('private model component does not match its commitment')
        state = decode_state(payload)
        if (set(state) != set(expected)
                or any(value.dtype != torch.int64 or tuple(value.shape) != expected[name]
                       for name, value in state.items())):
            raise ValueError('private model component has invalid parameter shapes or types')
        states.append(state)
    parameters = [(Share(tuple(state[f'{i}.weight'] for state in states)),
                   Share(tuple(state[f'{i}.bias'] for state in states)))
                  for i in range(manifest['layer_count'])]
    return PrivateAuditModel(manifest, parameters)
