"""State fingerprints agree with official BLAKE3 on CPU and CUDA."""
import hashlib
import json
import struct

from blake3 import blake3
import pytest
import torch

from rep4gpu.proofs.state_digest import (
    finish_state_digest, prepare_state_digest, state_digest,
)


def reference(state):
    names = sorted(state)
    entries = [[name, '<i4' if state[name].dtype == torch.int32 else '<i8', list(state[name].shape)] for name in names]
    metadata = json.dumps(entries, separators=(',', ':'), ensure_ascii=True).encode('ascii')
    digests = [blake3(state[name].cpu().contiguous().numpy().astype(entries[i][1], copy=False).tobytes()).digest() for i, name in enumerate(names)]
    return hashlib.sha256(b'rep4gpu.state-digest.v1\x00' + struct.pack('>Q', len(metadata)) + metadata + b''.join(digests)).digest()


def sample():
    return {'view': torch.arange(30, dtype=torch.int32).reshape(5, 6).T,
            'scalar': torch.tensor(-(2**63), dtype=torch.int64),
            'empty': torch.empty((2, 0, 3), dtype=torch.int64),
            'signed': torch.tensor([-(2**31), 2**31 - 1], dtype=torch.int32)}


def test_cpu_matches_official_reference_and_sorted_order():
    state = sample()
    assert state_digest(state) == reference(state)
    assert state_digest(dict(reversed(list(state.items())))) == reference(state)
    prepared = prepare_state_digest(state)
    assert all(isinstance(d, bytes) and len(d) == 32 for d in prepared.digests)
    assert finish_state_digest(prepared.metadata, prepared.digests) == reference(state)


def test_metadata_binds_name_dtype_and_shape():
    states = [{'x': torch.tensor([1, 2], dtype=torch.int32)},
              {'y': torch.tensor([1, 2], dtype=torch.int32)},
              {'x': torch.tensor([[1, 2]], dtype=torch.int32)},
              {'x': torch.tensor([1, 2], dtype=torch.int64)}]
    assert len({state_digest(state) for state in states}) == len(states)


@pytest.mark.parametrize('state', [{}, {'': torch.tensor(1)}, {'x': torch.tensor(1.)}, {'x': torch.tensor(True)}, {1: torch.tensor(1)}, {'x': object()}])
def test_invalid_state(state):
    with pytest.raises(ValueError):
        prepare_state_digest(state)


@pytest.mark.parametrize('digest', [b'', bytes(31), bytes(33), bytearray(32)])
def test_finish_rejects_invalid_digest(digest):
    with pytest.raises(ValueError):
        finish_state_digest(b'[]', [digest])


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_cuda_matches_cpu_and_transfers_only_digests():
    state = sample()
    prepared = prepare_state_digest({name: value.cuda() for name, value in state.items()})
    assert all(d.is_cuda and d.dtype == torch.uint8 and d.shape == (32,) for d in prepared.digests)
    assert prepared.finish() == reference(state)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_cuda_graph_replay_reads_current_values():
    state = {'x': torch.arange(524288, dtype=torch.int32, device='cuda')}
    prepare_state_digest(state).finish()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        prepared = prepare_state_digest(state)
        packed = torch.cat(prepared.digests)
    state['x'].add_(7)
    graph.replay()
    raw = packed.cpu().numpy().tobytes()
    assert finish_state_digest(prepared.metadata, [raw]) == reference(state)
    assert prepared.finish() == reference(state)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_nondefault_stream_and_mixed_devices():
    stream = torch.cuda.Stream()
    state = sample()
    with torch.cuda.stream(stream):
        mixed = {name: value.cuda() if index % 2 else value for index, (name, value) in enumerate(state.items())}
        prepared = prepare_state_digest(mixed)
    assert prepared.finish() == reference(state)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_finish_without_global_wait_uses_caller_ordering():
    state = {'x': torch.arange(2048, dtype=torch.int64, device='cuda')}
    prepared = prepare_state_digest(state)
    ready = torch.cuda.Event()
    ready.record()
    transfer = torch.cuda.Stream()
    with torch.cuda.stream(transfer):
        transfer.wait_event(ready)
        actual = prepared.finish(wait=False)
    assert actual == reference(state)


def test_prepare_multiple_cpu_states_preserves_fingerprints():
    from rep4gpu.proofs.state_digest import prepare_state_digests

    states = [sample(), {'other': torch.arange(8, dtype=torch.int64)}, sample()]
    prepared = prepare_state_digests(states)
    assert isinstance(prepared, tuple)
    assert [value.finish() for value in prepared] == [reference(state) for state in states]
    assert prepare_state_digests([]) == ()
    with pytest.raises(ValueError):
        prepare_state_digests([states[0], {}])


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_prepare_multiple_cuda_states_batches_equal_byte_lengths(monkeypatch):
    import importlib
    module = importlib.import_module('rep4gpu.proofs.state_digest')
    calls = []
    original = module.gpu_digest_batch

    def record(data, **kwargs):
        calls.append(tuple(data.shape))
        return original(data, **kwargs)

    monkeypatch.setattr(module, 'gpu_digest_batch', record)
    states = [
        {'a': torch.arange(2048, dtype=torch.int32, device='cuda'),
         'b': torch.arange(1024, dtype=torch.int64, device='cuda')},
        {'view': torch.arange(4096, dtype=torch.int32, device='cuda')[::2],
         'cpu': torch.tensor(-7, dtype=torch.int64),
         'empty': torch.empty((0,), dtype=torch.int32, device='cuda')},
    ]
    prepared = module.prepare_state_digests(states)
    assert calls == [(3, 8192), (1, 0)]
    assert [value.finish() for value in prepared] == [reference(state) for state in states]


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_prepare_multiple_cuda_graph_replay_reads_current_values():
    from rep4gpu.proofs.state_digest import prepare_state_digests

    states = [{'x': torch.arange(2048, dtype=torch.int32, device='cuda')}
              for _ in range(3)]
    for value in prepare_state_digests(states):
        value.finish()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        prepared = prepare_state_digests(states)
    for index, state in enumerate(states):
        state['x'].add_(index + 1)
    graph.replay()
    assert [value.finish() for value in prepared] == [reference(state) for state in states]
