"""Socket framing, context binding, and independent transition checks."""
import socket
import struct
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
import torch

from rep4gpu.proofs.audit_protocol import (
    receive_message, send_message, serve_session, verify_session,
)
from rep4gpu.proofs.layer_commitments import CommitmentRuntime, Layer


def model(delta=1):
    def validate_state(state):
        if set(state) != {'x'} or state['x'].dtype != torch.int64 or state['x'].shape != (2,):
            raise ValueError('invalid model state')
    return SimpleNamespace(model_id='test-v1', validate_state=validate_state,
                           layers=tuple(Layer(str(i), lambda s: {'x': s['x'] + delta})
                                        for i in range(3)))


@pytest.mark.parametrize('samples', [None, 1, 3])
def test_round_trip(samples):
    left, right = socket.socketpair()
    with left, right, ThreadPoolExecutor() as pool:
        left.settimeout(3)
        right.settimeout(3)
        future = pool.submit(serve_session, left, model(), checkpoint_interval=2)
        report = verify_session(right, model(), {'x': torch.tensor([2, -4])}, sample_count=samples)
        assert future.result() == report
    assert report['output'] == {'x': [5, -1]}
    assert report['accepted'] is True
    assert len(report['checked_transitions']) == (3 if samples is None else samples)
    assert report['verification_mode'] == ('full' if samples in (None, 3) else 'sampled')


@pytest.mark.parametrize('payload', [b'', b'[1]', b'{"x":1,"x":2}', b'{"x":NaN}', b'\xff'])
def test_invalid_json(payload):
    left, right = socket.socketpair()
    with left, right:
        left.sendall(struct.pack('>I', len(payload)) + payload)
        with pytest.raises(ValueError):
            receive_message(right)


def test_frame_limits_and_eof():
    left, right = socket.socketpair()
    with left, right:
        with pytest.raises(ValueError):
            send_message(left, {'x': 'long'}, max_frame_bytes=2)
        left.sendall(struct.pack('>I', 100))
        with pytest.raises(ValueError):
            receive_message(right, max_frame_bytes=10)
        left.shutdown(socket.SHUT_WR)
        with pytest.raises(ValueError, match='EOF'):
            receive_message(right)


@pytest.mark.parametrize('fault', ['model', 'run', 'layers', 'tamper', 'compute'])
def test_reject_bad_prover(fault):
    left, right = socket.socketpair()
    inputs = {'x': torch.tensor([2, -4])}

    def bad_prover():
        with left:
            request = receive_message(left)
            remote = model(2 if fault == 'compute' else 1)
            with CommitmentRuntime(remote.layers) as runtime:
                execution = runtime.commit(inputs, model_id=remote.model_id, run_id=request['run_id'])
                commitment = execution.commitment.to_dict()
                if fault == 'model':
                    commitment['model_id'] = 'wrong'
                elif fault == 'run':
                    commitment['run_id'] = 'old-run'
                elif fault == 'layers':
                    commitment['layer_names'] = ['other']
                send_message(left, dict(version=1, type='commitment', commitment=commitment))
                if fault in ('model', 'run', 'layers'):
                    return
                receive_message(left)
                for boundary in range(4):
                    opening = execution.open(boundary).to_dict()
                    if fault == 'tamper' and boundary == 1:
                        opening['siblings'][0] = '00' * 32
                    send_message(left, dict(version=1, type='opening', opening=opening))

    with right, ThreadPoolExecutor() as pool:
        right.settimeout(3)
        future = pool.submit(bad_prover)
        with pytest.raises(ValueError):
            verify_session(right, model(), inputs)
        future.result()


@pytest.mark.parametrize('samples', [0, -1, 4, True, 1.0])
def test_reject_invalid_sample_count(samples):
    with pytest.raises(ValueError, match='sample_count'):
        verify_session(None, model(), {'x': torch.tensor([2, -4])}, sample_count=samples)


def test_openings_keep_input_schema_even_when_model_accepts_other_batches():
    left, right = socket.socketpair()
    inputs = {'x': torch.tensor([2, -4])}
    verifier_model = model()
    verifier_model.validate_state = lambda state: None
    prover_model = model()
    prover_model.validate_state = lambda state: None
    prover_model.layers = (*prover_model.layers[:2],
                          Layer('2', lambda state: {'x': torch.cat([state['x'], state['x']])}))
    with left, right, ThreadPoolExecutor() as pool:
        left.settimeout(1)
        right.settimeout(1)
        future = pool.submit(serve_session, left, prover_model)
        with pytest.raises(ValueError, match='schema'):
            verify_session(right, verifier_model, inputs, sample_count=1)
        right.shutdown(socket.SHUT_RDWR)
        with pytest.raises(ValueError):
            future.result()
