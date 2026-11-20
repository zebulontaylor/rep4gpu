"""Private ring64 checks and jointly sampled audit challenges."""
from __future__ import annotations

import hashlib
import secrets

import torch

from ..protocol.runtime import Share
from .audit_model import AuditModel
from .private_model import PrivateAuditModel
from .witness_activations import WitnessProgram, WitnessSession


def _broadcast(transport, party, payload):
    if type(party) is not int or party not in range(4) or not isinstance(payload, bytes):
        raise ValueError('broadcast requires a role and bytes')
    incoming = transport.exchange({p: payload for p in range(4) if p != party})
    if set(incoming) != set(range(4)) - {party}:
        raise ValueError('missing broadcast peer')
    return tuple(payload if p == party else incoming[p] for p in range(4))


def agreement(transport, party, payload: bytes):
    """Require all four roles to hold the same transcript bytes, or abort."""
    digest = hashlib.sha256(b'siv.rep4.agreement.v1\0' + payload).digest()
    received = _broadcast(transport, party, digest)
    # Report the entire local view so equivocation cannot give two honest roles
    # different accepted views, even when the equivocator supplies both hashes.
    views = _broadcast(transport, party, b''.join(received))
    if any(v != b''.join(received) for v in views) or any(v != digest for v in received):
        raise ValueError('audit transcript agreement failed')


def joint_random(transport, party, context: bytes) -> bytes:
    """Commit before revealing fresh local entropy, then bind the joint coin."""
    if not isinstance(context, bytes):
        raise ValueError('coin context must be bytes')
    domain = hashlib.sha256(b'siv.rep4.coin.v1\0' + context).digest()
    nonce = secrets.token_bytes(32)
    commit = hashlib.sha256(domain + bytes([party]) + nonce).digest()
    commitments = _broadcast(transport, party, commit)
    agreement(transport, party, b''.join(commitments))
    reveals = _broadcast(transport, party, nonce)
    agreement(transport, party, b''.join(reveals))
    if any(len(r) != 32 or len(c) != 32 or
           hashlib.sha256(domain + bytes([p]) + r).digest() != c
           for p, (c, r) in enumerate(zip(commitments, reveals))):
        raise ValueError('joint coin commitment mismatch')
    coin = bytes(a ^ b ^ c ^ d for a, b, c, d in zip(*reveals))
    return hashlib.sha256(domain + coin).digest()


def verify_shared_trace(party, model: AuditModel, states: dict[int, Share],
                        expected_input: torch.Tensor, indices: list[int]) -> torch.Tensor:
    """Check sampled transitions and release only acceptance and final output.

    Boundary shares must already be authenticated against the same agreed
    salted component roots. Binding roots make honest holders' replicated
    copies consistent without a separate import exchange. All roles use
    the same approved model identity, public input, boundary schema, and sample
    list. Private model parameters must be authenticated against the approved
    manifest before this call. Model tensors must remain unchanged throughout
    the audit. Unchecked transitions carry no
    correctness claim.
    """
    if not isinstance(model, AuditModel) or party.bits != 64:
        raise ValueError('audit requires an AuditModel and ring64 party')
    model.validate_state({'x': expected_input})
    count = len(model.layers)
    if (not isinstance(indices, list) or not indices or
            any(type(i) is not int or not 0 <= i < count for i in indices) or
            len(set(indices)) != len(indices)):
        raise ValueError('samples must be distinct valid layer indices')
    required = {0, count, *indices, *(i + 1 for i in indices)}
    if not isinstance(states, dict) or any(type(k) is not int for k in states) or set(states) != required:
        raise ValueError('shared boundaries must exactly match the sample schema')
    for value in states.values():
        if (not isinstance(value, Share) or value.binary or value.device.type != 'cpu'
                or tuple(value.shape) != tuple(expected_input.shape)):
            raise ValueError('boundary must be a CPU arithmetic share with the input shape')

    def recompute(session, inputs, witnesses):
        rt = session.runtime
        # A zero ring difference is exact word equality, including at signed
        # endpoints; no non-wrapping integer identity is assumed here.
        session.residual(rt.sub(witnesses['0'], rt.public(expected_input, like=witnesses['0'])))
        for i in indices:
            x = witnesses[str(i)]
            weight, bias = model.parameters[i]
            if isinstance(model, PrivateAuditModel):
                affine = rt.add(rt.matmul(x, weight), bias)
            else:
                affine = rt._share(part @ weight for part in x.parts)
                affine = rt.add(affine, rt.public(bias, like=affine))
            positive = rt.sub(rt.public(1, like=affine), rt.sign_bit(affine))
            expected = rt.mul(affine, positive)
            session.residual(rt.sub(expected, witnesses[str(i + 1)]))
        return witnesses[str(count)]

    witnesses = {str(k): v for k, v in states.items()}
    # These are arbitrary ring words. Exact equality supplies the constraints;
    # the identity preparation bypasses the default fixed-point range clamps.
    program = WitnessProgram('ring64-audit', {k: (0., 0.) for k in witnesses},
                             recompute, lambda session, values: values)
    session = WitnessSession(party)
    session.evaluate(program, {}, witnesses)
    result, = session.finish()
    return party.open(result)
