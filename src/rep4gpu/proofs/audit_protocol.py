"""Exchange plaintext layer commitments and independently audit transitions.

Callers own sockets and set their timeouts. Every model boundary must retain
the input tensor names, shapes, and dtypes. This protocol authenticates opened
states against a fresh transcript; it does not provide private inference.
"""
from __future__ import annotations

import base64
import binascii
import json
import secrets
import struct

from .layer_commitments import (
    Commitment, CommitmentRuntime, LayerOpening, decode_state, encode_state, verify_layer,
)

MAX_FRAME_BYTES = 16 * 1024 * 1024
VERSION = 1
PLAINTEXT_SEMANTICS = (
    'Inputs, outputs, and opened intermediate states are plaintext. '
    'Merkle authentication binds states to the transcript; only checked '
    'transitions are independently recomputed. This is not a private or zero-knowledge proof.'
)


def _limit(value):
    if type(value) is not int or not 0 < value <= 0xffffffff:
        raise ValueError('frame limit must be a positive uint32')


def _read_exact(sock, count):
    chunks = bytearray()
    while len(chunks) < count:
        try:
            chunk = sock.recv(count - len(chunks))
        except OSError as exc:
            raise ValueError('failed to receive protocol frame') from exc
        if not chunk:
            raise ValueError('unexpected EOF in protocol frame')
        chunks.extend(chunk)
    return bytes(chunks)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate JSON key')
        result[key] = value
    return result


def send_message(sock, message, *, max_frame_bytes=MAX_FRAME_BYTES):
    """Send one bounded JSON object with a big-endian uint32 byte length."""
    _limit(max_frame_bytes)
    if not isinstance(message, dict):
        raise ValueError('protocol message must be an object')
    try:
        payload = json.dumps(message, allow_nan=False, separators=(',', ':')).encode('utf-8')
        if not 0 < len(payload) <= max_frame_bytes:
            raise ValueError('protocol frame exceeds size limit')
        sock.sendall(struct.pack('>I', len(payload)) + payload)
    except (OSError, TypeError, OverflowError, UnicodeError, RecursionError) as exc:
        raise ValueError('failed to send protocol frame') from exc


def receive_message(sock, *, max_frame_bytes=MAX_FRAME_BYTES):
    """Receive one JSON object, rejecting oversized frames before allocation."""
    _limit(max_frame_bytes)
    size, = struct.unpack('>I', _read_exact(sock, 4))
    if not 0 < size <= max_frame_bytes:
        raise ValueError('protocol frame exceeds size limit')
    try:
        value = json.loads(_read_exact(sock, size), object_pairs_hook=_unique_object,
                           parse_constant=lambda value: (_ for _ in ()).throw(ValueError('invalid JSON number')))
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ValueError('malformed protocol JSON') from exc
    if not isinstance(value, dict):
        raise ValueError('protocol message must be an object')
    return value


def _message(kind, **fields):
    return dict(version=VERSION, type=kind, **fields)


def _receive(sock, kind, fields):
    message = receive_message(sock)
    if (set(message) != {'version', 'type', *fields}
            or type(message['version']) is not int or message['version'] != VERSION
            or message['type'] != kind):
        raise ValueError(f'expected {kind} message')
    return message


def _boundaries(indices, count):
    return sorted({0, count, *(i for index in indices for i in (index, index + 1))})


def _validate_indices(indices, count):
    if (not isinstance(indices, list) or not (min(1, count) <= len(indices) <= count)
            or any(type(index) is not int or not 0 <= index < count for index in indices)
            or indices != sorted(set(indices))):
        raise ValueError('challenge must contain sorted distinct transition indices')


def _state(payload):
    if not isinstance(payload, str):
        raise ValueError('state payload must be base64 text')
    try:
        return decode_state(base64.b64decode(payload, validate=True))
    except (binascii.Error, UnicodeError) as exc:
        raise ValueError('invalid base64 state') from exc


def serve_session(sock, model, *, checkpoint_interval=8):
    """Prove one request, then return the verifier's accepted report."""
    request = _receive(sock, 'request', {'model_id', 'run_id', 'inputs'})
    run_id = request['run_id']
    if (request['model_id'] != model.model_id or not isinstance(run_id, str)
            or len(run_id) != 64 or any(c not in '0123456789abcdef' for c in run_id)):
        raise ValueError('request model or run context does not match')
    inputs = _state(request['inputs'])
    model.validate_state(inputs)
    with CommitmentRuntime(model.layers, checkpoint_interval=checkpoint_interval) as runtime:
        execution = runtime.commit(inputs, model_id=model.model_id, run_id=run_id)
        send_message(sock, _message('commitment', commitment=execution.commitment.to_dict()))
        challenge = _receive(sock, 'challenge', {'indices'})
        indices = challenge['indices']
        _validate_indices(indices, len(model.layers))
        for boundary in _boundaries(indices, len(model.layers)):
            send_message(sock, _message('opening', opening=execution.open(boundary).to_dict()))
        result = _receive(sock, 'result', {'report'})['report']
        expected_report = _report(
            execution.commitment, indices, execution.recompute(len(model.layers)))
        if not isinstance(result, dict) or json.dumps(result, sort_keys=True) != json.dumps(
                expected_report, sort_keys=True):
            raise ValueError('verifier result does not match session')
        return result


def _report(commitment, indices, output):
    count = len(commitment.layer_names)
    return dict(accepted=True, run_id=commitment.run_id, model_id=commitment.model_id,
                root=commitment.root.hex(), checked_transitions=indices,
                total_transitions=count,
                verification_mode='full' if len(indices) == count else 'sampled',
                output={name: value.tolist() for name, value in sorted(output.items())},
                plaintext_semantics=PLAINTEXT_SEMANTICS)


def verify_session(sock, model, inputs, *, sample_count=None):
    """Commit first, sample transitions, authenticate states, and recompute them."""
    count = len(model.layers)
    if sample_count is None:
        sample_count = count
    if type(sample_count) is not int or not min(1, count) <= sample_count <= count:
        raise ValueError('sample_count must be positive and no greater than the layer count')
    model.validate_state(inputs)
    input_bytes = encode_state(inputs)
    run_id = secrets.token_hex(32)
    send_message(sock, _message('request', model_id=model.model_id, run_id=run_id,
                               inputs=base64.b64encode(input_bytes).decode('ascii')))
    message = _receive(sock, 'commitment', {'commitment'})
    commitment = Commitment.from_dict(message['commitment'])
    if (commitment.model_id != model.model_id or commitment.run_id != run_id
            or commitment.layer_names != tuple(layer.name for layer in model.layers)):
        raise ValueError('commitment model or run context does not match')
    indices = sorted(secrets.SystemRandom().sample(range(count), sample_count))
    send_message(sock, _message('challenge', indices=indices))
    input_schema = {name: (value.dtype, tuple(value.shape)) for name, value in inputs.items()}
    states = {}
    for boundary in _boundaries(indices, count):
        message = _receive(sock, 'opening', {'opening'})
        try:
            opening = LayerOpening.from_dict(message['opening'])
        except (binascii.Error, UnicodeError) as exc:
            raise ValueError('malformed opening') from exc
        state = verify_layer(commitment, opening, expected_boundary=boundary)
        model.validate_state(state)
        if {name: (value.dtype, tuple(value.shape)) for name, value in state.items()} != input_schema:
            raise ValueError('opened state schema differs from the requested input')
        states[boundary] = state
    if encode_state(states[0]) != input_bytes:
        raise ValueError('committed input differs from requested input')
    for index in indices:
        # A callback may mutate its inputs; retain the authenticated states.
        source = decode_state(encode_state(states[index]))
        expected = model.layers[index].forward(source)
        model.validate_state(expected)
        if encode_state(expected) != encode_state(states[index + 1]):
            raise ValueError(f'incorrect computation at transition {index}')
    report = _report(commitment, indices, states[count])
    send_message(sock, _message('result', report=report))
    return report
