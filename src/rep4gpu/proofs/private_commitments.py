"""Salted commitments to prover-local Rep4 components of integer boundaries.

These roots commit to random additive components, not the plaintext trace root.
A recipient gets three components and their salts; the fourth salt stays private.
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
from .layer_commitments import CommitmentRuntime, decode_state, encode_state
from .merkle import MerkleTree, verify_merkle

_SCHEME = 'rep4-salted-components-v1'
_KEYS = {'version', 'scheme', 'model_id', 'run_id', 'layer_names', 'shape', 'roots'}


def _hex(value):
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError('expected a canonical 32-byte hex value')
    try:
        result = bytes.fromhex(value)
    except ValueError as exc:
        raise ValueError('invalid hex value') from exc
    if result.hex() != value:
        raise ValueError('expected lowercase canonical hex')
    return result


def _slots(party):
    if type(party) is not int or party not in range(4):
        raise ValueError('party must be in [0,3]')
    return tuple((party + offset) % 4 for offset in (1, 2, 3))


def validate_commitment(commitment, model, run_id, shape):
    """Validate the complete public context against locally agreed metadata."""
    if (not isinstance(shape, (list, tuple, torch.Size))
            or any(type(n) is not int or n < 0 for n in shape)):
        raise ValueError('shape must contain nonnegative integer dimensions')
    if (not isinstance(commitment, dict) or set(commitment) != _KEYS
            or type(commitment['version']) is not int or commitment['version'] != 1
            or commitment['scheme'] != _SCHEME
            or not isinstance(run_id, str) or not run_id
            or commitment['run_id'] != run_id
            or commitment['model_id'] != model.model_id
            or not isinstance(commitment['model_id'], str)
            or not isinstance(commitment['layer_names'], list)
            or commitment['layer_names'] != [layer.name for layer in model.layers]
            or not isinstance(commitment['shape'], list)
            or any(type(n) is not int or n < 0 for n in commitment['shape'])
            or commitment['shape'] != list(shape)
            or not isinstance(commitment['roots'], list) or len(commitment['roots']) != 4):
        raise ValueError('private commitment does not match the agreed context')
    for root in commitment['roots']:
        _hex(root)
    return copy.deepcopy(commitment)


def _context(commitment):
    metadata = {key: value for key, value in commitment.items() if key != 'roots'}
    canonical = json.dumps(metadata, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode('ascii')
    return hashlib.sha256(b'rep4gpu.private.context.v1\0' + canonical).digest()


def _leaf(context, component, boundary, salt, state):
    return (b'rep4gpu.private.component.v1\0' + context
            + component.to_bytes(1, 'big') + boundary.to_bytes(8, 'big') + salt + state)


class PrivateTrace:
    """Commit a prover trace and retain random components for later openings.

    Checkpoint replay authenticates each plaintext boundary locally. This small
    model implementation then retains four tensors and salts per boundary, so
    component storage grows with the full trace rather than checkpoint count.
    """
    def __init__(self, model, inputs, *, run_id, checkpoint_interval=8):
        model.validate_state(inputs)
        self.commitment = dict(version=1, scheme=_SCHEME, model_id=model.model_id,
                               run_id=run_id, layer_names=[layer.name for layer in model.layers],
                               shape=list(inputs['x'].shape), roots=['00' * 32] * 4)
        validate_commitment(self.commitment, model, run_id, inputs['x'].shape)
        context = _context(self.commitment)
        self._components, self._salts = [], []
        leaves = [[] for _ in range(4)]
        with CommitmentRuntime(model.layers, checkpoint_interval=checkpoint_interval) as runtime:
            execution = runtime.commit(inputs, model_id=model.model_id, run_id=run_id)
            for boundary in range(len(model.layers) + 1):
                state = execution.recompute(boundary)
                model.validate_state(state)
                x = state['x']
                if list(x.shape) != self.commitment['shape']:
                    raise ValueError('private trace boundaries must preserve shape')
                parts = [torch.from_numpy(np.frombuffer(secrets.token_bytes(x.numel() * 8),
                                                       dtype='<i8').copy().reshape(x.shape))
                         for _ in range(3)]
                parts.append(x - parts[0] - parts[1] - parts[2])
                salts = [secrets.token_bytes(32) for _ in range(4)]
                encoded = [encode_state({'x': part}) for part in parts]
                if boundary == len(model.layers):
                    self.output = {'x': x.detach().clone()}
                self._components.append(encoded)
                self._salts.append(salts)
                for component in range(4):
                    leaves[component].append(_leaf(context, component, boundary,
                                                   salts[component], encoded[component]))
        self._trees = [MerkleTree(component_leaves) for component_leaves in leaves]
        self.commitment['roots'] = [tree.root.hex() for tree in self._trees]

    def opening(self, boundary, party):
        """Return only the three components held by the requested Rep4 party."""
        slots = _slots(party)
        if type(boundary) is not int or not 0 <= boundary < len(self._components):
            raise ValueError('boundary is outside the committed trace')
        return dict(version=1, boundary=boundary, components=[
            dict(excluded=component, salt=self._salts[boundary][component].hex(),
                 state=base64.b64encode(self._components[boundary][component]).decode('ascii'),
                 siblings=[s.hex() for s in self._trees[component].proof(boundary)])
            for component in slots])


def verify_private_opening(commitment, opening, *, party, expected_boundary, model, run_id, shape):
    """Authenticate all three local components before constructing a Share.

    The commitment must have been agreed before selecting the boundary. Apply
    transport size limits before passing untrusted decoded JSON to this function.
    """
    commitment = validate_commitment(commitment, model, run_id, shape)
    slots = _slots(party)
    count = len(model.layers) + 1
    if (type(expected_boundary) is not int or not 0 <= expected_boundary < count
            or not isinstance(opening, dict) or set(opening) != {'version', 'boundary', 'components'}
            or type(opening['version']) is not int or opening['version'] != 1
            or type(opening['boundary']) is not int or opening['boundary'] != expected_boundary
            or not isinstance(opening['components'], list) or len(opening['components']) != 3):
        raise ValueError('invalid private opening envelope')
    context, tensors = _context(commitment), []
    for component, entry in zip(slots, opening['components']):
        if (not isinstance(entry, dict) or set(entry) != {'excluded', 'salt', 'state', 'siblings'}
                or type(entry['excluded']) is not int or entry['excluded'] != component
                or not isinstance(entry['state'], str) or not isinstance(entry['siblings'], list)):
            raise ValueError('invalid private component schema or local order')
        salt = _hex(entry['salt'])
        siblings = tuple(_hex(value) for value in entry['siblings'])
        try:
            payload = base64.b64decode(entry['state'], validate=True)
        except (ValueError, binascii.Error) as exc:
            raise ValueError('invalid private component encoding') from exc
        if base64.b64encode(payload).decode('ascii') != entry['state']:
            raise ValueError('noncanonical private component encoding')
        if not verify_merkle(_hex(commitment['roots'][component]),
                             _leaf(context, component, expected_boundary, salt, payload),
                             expected_boundary, count, siblings):
            raise ValueError('private component does not match its commitment')
        state = decode_state(payload)
        model.validate_state(state)
        if list(state['x'].shape) != commitment['shape']:
            raise ValueError('private component shape differs from the commitment')
        tensors.append(state['x'])
    return Share(tuple(tensors))
