from dataclasses import replace
import json

import pytest
import torch

from rep4gpu.proofs.layer_commitments import (
    Commitment,
    CommitmentRuntime,
    Layer,
    LayerOpening,
    decode_state,
    encode_state,
    verify_layer,
)


def assert_state_equal(actual, expected):
    assert set(actual) == set(expected)
    for name, tensor in expected.items():
        assert actual[name].dtype == tensor.dtype
        assert actual[name].shape == tensor.shape
        assert torch.equal(actual[name], tensor)


def test_state_encoding_is_canonical_and_preserves_integer_values():
    matrix = torch.arange(12, dtype=torch.int32).reshape(3, 4).T
    first = {
        "matrix": matrix,
        "scalar": torch.tensor(-(2**63), dtype=torch.int64),
        "empty": torch.empty((2, 0, 3), dtype=torch.int32),
    }
    second = {name: value.contiguous() for name, value in reversed(first.items())}
    assert encode_state(first) == encode_state(second)
    assert_state_equal(decode_state(encode_state(first)), first)


@pytest.mark.parametrize("state", [{}, {"": torch.tensor(1)}, {"x": torch.tensor(1.0)}, {"x": torch.tensor(True)}, {"x": torch.tensor(1, dtype=torch.int16)}])
def test_state_encoding_rejects_invalid_states(state):
    with pytest.raises((ValueError, TypeError)):
        encode_state(state)


def test_decode_rejects_truncated_or_extended_encoding():
    encoded = encode_state({"x": torch.tensor([1, 2], dtype=torch.int64)})
    for invalid in (b"", encoded[:-1], encoded + b"\x00"):
        with pytest.raises(ValueError):
            decode_state(invalid)


def residual_layers():
    return [
        Layer("project", lambda state: {"hidden": state["x"] * 3, "residual": state["x"]}),
        Layer("add", lambda state: {"x": state["hidden"] + state["residual"]}),
        Layer("offset", lambda state: {"x": state["x"] + 7}),
    ]


def test_arbitrary_state_layers_and_every_boundary_opening():
    inputs = {"x": torch.tensor([-2, 0, 3], dtype=torch.int64)}
    execution = CommitmentRuntime(residual_layers(), checkpoint_interval=2).commit(inputs, model_id="model", run_id="run")
    expected = [inputs, {"hidden": inputs["x"] * 3, "residual": inputs["x"]}, {"x": inputs["x"] * 4}, {"x": inputs["x"] * 4 + 7}]
    assert execution.commitment.layer_names == ("project", "add", "offset")
    assert isinstance(execution.commitment.root, bytes)
    for boundary, state in enumerate(expected):
        assert_state_equal(execution.recompute(boundary), state)
        opening = execution.open(boundary)
        assert_state_equal(verify_layer(execution.commitment, opening, expected_boundary=boundary), state)


def test_transport_roundtrip():
    execution = CommitmentRuntime(residual_layers()).commit({"x": torch.tensor([5], dtype=torch.int32)}, model_id="m", run_id="r")
    commitment = Commitment.from_dict(json.loads(json.dumps(execution.commitment.to_dict())))
    opening = LayerOpening.from_dict(json.loads(json.dumps(execution.open(2).to_dict())))
    assert commitment == execution.commitment
    assert_state_equal(verify_layer(commitment, opening, expected_boundary=2), {"x": torch.tensor([20], dtype=torch.int32)})


def test_tampering_with_opening_or_context_is_rejected():
    execution = CommitmentRuntime(residual_layers()).commit({"x": torch.tensor([5], dtype=torch.int64)}, model_id="m", run_id="r")
    commitment = execution.commitment
    opening = execution.open(1)
    assert opening.siblings
    bad_sibling = bytes([opening.siblings[0][0] ^ 1]) + opening.siblings[0][1:]
    cases = [
        (commitment, replace(opening, state=encode_state({"x": torch.tensor([999])})), 1),
        (commitment, replace(opening, siblings=(bad_sibling,) + opening.siblings[1:]), 1),
        (commitment, replace(opening, siblings=opening.siblings[:-1]), 1),
        (commitment, replace(opening, boundary=2), 1),
        (commitment, opening, 2),
        (replace(commitment, model_id="other"), opening, 1),
        (replace(commitment, run_id="other"), opening, 1),
        (replace(commitment, layer_names=("other", "add", "offset")), opening, 1),
        (replace(commitment, root=b"\x00" * len(commitment.root)), opening, 1),
    ]
    for changed_commitment, changed_opening, boundary in cases:
        with pytest.raises(ValueError):
            verify_layer(changed_commitment, changed_opening, expected_boundary=boundary)


def test_sparse_checkpoints_replay_only_from_nearest_checkpoint():
    calls = [0] * 7

    def forward(index):
        def apply(state):
            calls[index] += 1
            return {"x": state["x"] + 1}
        return apply

    execution = CommitmentRuntime([Layer(str(i), forward(i)) for i in range(7)], checkpoint_interval=3).commit({"x": torch.tensor(0)}, model_id="m", run_id="r")
    assert calls == [1] * 7
    assert execution.recompute(5)["x"].item() == 5
    assert calls == [1, 1, 1, 2, 2, 1, 1]


def test_inputs_and_recomputed_tensors_are_isolated():
    inputs = {"x": torch.tensor([2, 4], dtype=torch.int64)}

    def mutate(state):
        state["x"].add_(3)
        return state

    execution = CommitmentRuntime([Layer("inplace", mutate)]).commit(inputs, model_id="m", run_id="r")
    assert inputs["x"].tolist() == [2, 4]
    inputs["x"].fill_(99)
    for boundary, expected in ((0, [2, 4]), (1, [5, 7])):
        recovered = execution.recompute(boundary)
        assert recovered["x"].tolist() == expected
        recovered["x"].zero_()
        assert execution.recompute(boundary)["x"].tolist() == expected


def test_changed_layer_callback_is_detected_during_replay():
    offset = [1]
    layers = [Layer(str(i), lambda state: {"x": state["x"] + offset[0]}) for i in range(3)]
    execution = CommitmentRuntime(layers, checkpoint_interval=3).commit({"x": torch.tensor(0)}, model_id="m", run_id="r")
    offset[0] = 2
    with pytest.raises(ValueError):
        execution.recompute(1)
    with pytest.raises(ValueError):
        execution.open(1)


@pytest.mark.parametrize("boundary", [-1, 4])
def test_out_of_range_boundaries_are_rejected(boundary):
    execution = CommitmentRuntime(residual_layers()).commit({"x": torch.tensor(0)}, model_id="m", run_id="r")
    with pytest.raises(ValueError):
        execution.recompute(boundary)
    with pytest.raises(ValueError):
        execution.open(boundary)


def test_commitment_is_frozen_and_binds_execution_context():
    from dataclasses import FrozenInstanceError

    runtime = CommitmentRuntime(residual_layers())
    inputs = {"x": torch.tensor([1], dtype=torch.int64)}
    first = runtime.commit(inputs, model_id="m", run_id="r").commitment
    assert first == runtime.commit(inputs, model_id="m", run_id="r").commitment
    assert first.root != runtime.commit(inputs, model_id="other", run_id="r").commitment.root
    assert first.root != runtime.commit(inputs, model_id="m", run_id="other").commitment.root
    with pytest.raises(FrozenInstanceError):
        first.run_id = "changed"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_prepared_cuda_matches_eager_and_preserves_old_execution():
    runtime = CommitmentRuntime(residual_layers(), checkpoint_interval=2)
    inputs = {"x": torch.tensor([1, -4], dtype=torch.int64, device="cuda")}
    prepared = runtime.prepare(inputs)
    first = prepared.commit(inputs, model_id="m", run_id="r")
    eager = runtime.commit(inputs, model_id="m", run_id="r")
    assert first.commitment == eager.commitment
    second_inputs = {"x": inputs["x"] + 10}
    second = prepared.commit(second_inputs, model_id="m", run_id="r")
    assert second.commitment != first.commitment
    assert second.commitment == runtime.commit(second_inputs, model_id="m", run_id="r").commitment
    assert first.recompute(3)["x"].tolist() == [11, -9]
    commitment = Commitment.from_dict(json.loads(json.dumps(first.commitment.to_dict())))
    opening = LayerOpening.from_dict(json.loads(json.dumps(first.open(1).to_dict())))
    assert_state_equal(verify_layer(commitment, opening, expected_boundary=1), {"hidden": torch.tensor([3, -12]), "residual": torch.tensor([1, -4])})
    for invalid in ({"other": inputs["x"]}, {"x": inputs["x"][:1]}, {"x": inputs["x"].int()}, {"x": inputs["x"].cpu()}):
        with pytest.raises(ValueError):
            prepared.commit(invalid, model_id="m", run_id="r")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_inplace_aliasing_keeps_boundary_snapshots():
    def inplace(state):
        state["x"].add_(1)
        return {"x": state["x"], "alias": state["x"]}

    runtime = CommitmentRuntime([Layer("inplace", inplace), Layer("again", inplace)], checkpoint_interval=2)
    inputs = {"x": torch.tensor([5], dtype=torch.int64, device="cuda")}
    prepared = runtime.prepare(inputs)
    for active in (runtime, prepared):
        execution = active.commit(inputs, model_id="m", run_id="r")
        assert inputs["x"].item() == 5
        for boundary in range(3):
            state = verify_layer(execution.commitment, execution.open(boundary), expected_boundary=boundary)
            assert state["x"].item() == 5 + boundary
            if boundary:
                assert state["alias"].item() == 5 + boundary


def test_decode_rejects_excessive_json_nesting_as_value_error():
    import struct

    header = b'[' * 1100 + b']' * 1100
    payload = b'SIVSTATE\x01' + struct.pack('>I', len(header)) + header
    with pytest.raises(ValueError, match='malformed state encoding'):
        decode_state(payload)


@pytest.mark.parametrize('overlapping_views', [False, True])
def test_cpu_checkpoint_preserves_aliases_during_inplace_replay(overlapping_views):
    def alias(state):
        value = state['x']
        return {'x': value, 'alias': value[1:] if overlapping_views else value}

    def mutate(state):
        state['x'].add_(7)
        return state

    layers = [Layer('identity', lambda state: state), Layer('alias', alias), Layer('mutate', mutate)]
    inputs = {'x': torch.tensor([1, 2, 3], dtype=torch.int64)}
    with CommitmentRuntime(layers, checkpoint_interval=2) as runtime:
        execution = runtime.commit(inputs, model_id='m', run_id='r')
    for _ in range(2):
        recovered = execution.recompute(3)
        assert recovered['x'].tolist() == [8, 9, 10]
        assert recovered['alias'].tolist() == ([9, 10] if overlapping_views else [8, 9, 10])
        recovered['x'].fill_(99)
    opened = verify_layer(execution.commitment, execution.open(3), expected_boundary=3)
    assert opened['alias'].tolist() == ([9, 10] if overlapping_views else [8, 9, 10])
    assert inputs['x'].tolist() == [1, 2, 3]


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
def test_prepared_execution_rejects_changed_input_aliases():
    shared = torch.tensor([1], dtype=torch.int64, device='cuda')
    with CommitmentRuntime([]) as runtime:
        prepared = runtime.prepare({'x': shared, 'y': shared})
        independent = {'x': shared + 1, 'y': shared + 2}
        with pytest.raises(ValueError, match='alias'):
            prepared.commit(independent, model_id='m', run_id='r')


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
def test_mixed_state_hashing_waits_for_nondefault_stream_producers():
    inputs = {'cpu': torch.tensor([4], dtype=torch.int64),
              'gpu': torch.tensor([7], dtype=torch.int64, device='cuda')}
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())

    def delayed_mutation(state):
        torch.cuda._sleep(10_000_000)
        state['gpu'].add_(5)
        state['cpu'].add_(2)
        return state

    with CommitmentRuntime([Layer('delayed', delayed_mutation)]) as runtime:
        with torch.cuda.stream(stream):
            execution = runtime.commit(inputs, model_id='m', run_id='r')
        for boundary, expected in ((0, {'cpu': [4], 'gpu': [7]}),
                                   (1, {'cpu': [6], 'gpu': [12]})):
            state = verify_layer(execution.commitment, execution.open(boundary), expected_boundary=boundary)
            assert {name: value.tolist() for name, value in state.items()} == expected


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
def test_prepared_mixed_dtype_aliases_compare_byte_offsets():
    backing = torch.arange(1, 7, dtype=torch.int32, device='cuda')
    inputs = {'x': backing[:2], 'y': backing.view(torch.int64)[0]}
    with CommitmentRuntime([]) as runtime:
        prepared = runtime.prepare(inputs)
        same_backing = backing + 10
        same_aliases = {'x': same_backing[:2], 'y': same_backing.view(torch.int64)[0]}
        execution = prepared.commit(same_aliases, model_id='m', run_id='r')
        eager = runtime.commit(same_aliases, model_id='m', run_id='r')
        assert execution.commitment == eager.commitment
        assert_state_equal(verify_layer(execution.commitment, execution.open(0), expected_boundary=0),
                           {name: value.cpu() for name, value in same_aliases.items()})
        changed_aliases = {'x': same_backing[1:3], 'y': same_backing.view(torch.int64)[1]}
        with pytest.raises(ValueError, match='alias'):
            prepared.commit(changed_aliases, model_id='m', run_id='r')


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
def test_prepared_preserved_inputs_contexts_and_retained_execution():
    layers = [Layer(f'add-{index}', lambda state, delta=index + 1:
                    {'x': state['x'] + delta}, preserves_inputs=True) for index in range(5)]
    inputs = {'x': torch.arange(8, dtype=torch.int64, device='cuda')}
    with CommitmentRuntime(layers, checkpoint_interval=2) as runtime:
        prepared = runtime.prepare(inputs)
        retained = []
        for delta, model, run in ((0, 'model', 'run'), (11, 'other-model', 'run'),
                                  (0, 'model', 'other-run'), (0, 'model', 'run')):
            current = {'x': inputs['x'] + delta}
            actual = prepared.commit(current, model_id=model, run_id=run)
            expected = runtime.commit(current, model_id=model, run_id=run)
            assert actual.commitment == expected.commitment
            for boundary in range(len(layers) + 1):
                assert_state_equal(actual.recompute(boundary), expected.recompute(boundary))
                assert_state_equal(verify_layer(actual.commitment, actual.open(boundary), expected_boundary=boundary),
                                   {name: value.cpu() for name, value in expected.recompute(boundary).items()})
            retained.append(actual)
        assert retained[0].commitment == retained[-1].commitment
        for boundary in range(len(layers) + 1):
            assert_state_equal(retained[0].recompute(boundary), retained[-1].recompute(boundary))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
def test_prepared_packed_checkpoints_preserve_overlapping_mixed_dtype_views():
    backing = torch.arange(1, 9, dtype=torch.int32, device='cuda')
    inputs = {'x': backing[1:7:2], 'wide': backing.view(torch.int64)[1:3],
              'empty': backing[:0]}

    def mutate(state):
        state['x'].add_(3)
        return state

    layers = [Layer(f'mutate-{index}', mutate) for index in range(4)]
    with CommitmentRuntime(layers, checkpoint_interval=2) as runtime:
        prepared = runtime.prepare(inputs)
        actual = prepared.commit(inputs, model_id='m', run_id='r')
        expected = runtime.commit(inputs, model_id='m', run_id='r')
        assert actual.commitment == expected.commitment
        for boundary in range(5):
            recovered = actual.recompute(boundary)
            assert_state_equal(recovered, expected.recompute(boundary))
            assert recovered['x'].untyped_storage().data_ptr() == recovered['wide'].untyped_storage().data_ptr()
            recovered['x'].fill_(99)
            assert_state_equal(actual.recompute(boundary), expected.recompute(boundary))
        prepared.commit(inputs, model_id='m', run_id='later')
        assert_state_equal(actual.recompute(4), expected.recompute(4))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
@pytest.mark.parametrize('preserves_inputs', [False, True])
def test_prepared_large_states_without_batch_hashing(preserves_inputs):
    inputs = {'x': torch.arange(262145, dtype=torch.int32, device='cuda')}
    layers = [Layer(f'add-{index}', lambda state: {'x': state['x'] + 7},
                    preserves_inputs=preserves_inputs) for index in range(2)]
    with CommitmentRuntime(layers, checkpoint_interval=2) as runtime:
        prepared = runtime.prepare(inputs)
        assert not prepared._batch_hashes
        actual = prepared.commit(inputs, model_id='m', run_id='r')
        expected = runtime.commit(inputs, model_id='m', run_id='r')
        assert actual.commitment == expected.commitment
        for boundary in range(3):
            assert_state_equal(actual.recompute(boundary), expected.recompute(boundary))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
def test_prepared_native_unavailable_uses_python_finalization(monkeypatch):
    from rep4gpu.proofs import native_merkle

    def unavailable(*args, **kwargs):
        raise RuntimeError('test compiler unavailable')

    monkeypatch.setattr(native_merkle, 'PreparedNativeMerkle', unavailable)
    inputs = {'x': torch.arange(8, dtype=torch.int32, device='cuda')}
    with CommitmentRuntime([Layer('add', lambda state: {'x': state['x'] + 1}, preserves_inputs=True)]) as runtime:
        prepared = runtime.prepare(inputs)
        assert prepared._native_finalizer is None
        assert prepared.native_unavailable == 'test compiler unavailable'
        for model, run in (('m', 'r'), ('changed', 'another')):
            actual = prepared.commit(inputs, model_id=model, run_id=run)
            expected = runtime.commit(inputs, model_id=model, run_id=run)
            assert actual.commitment == expected.commitment


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
@pytest.mark.parametrize('force_fallback', [False, True])
def test_prepared_commit_returns_completed_execution(monkeypatch, force_fallback):
    if force_fallback:
        from rep4gpu.proofs import native_merkle

        def unavailable(*args, **kwargs):
            raise RuntimeError('test native backend unavailable')

        monkeypatch.setattr(native_merkle, 'PreparedNativeMerkle', unavailable)
    inputs = {'x': torch.arange(32, dtype=torch.int64, device='cuda')}
    layers = [Layer(f'add-{index}', lambda state: {'x': state['x'] + 3},
                    preserves_inputs=True) for index in range(3)]
    with CommitmentRuntime(layers, checkpoint_interval=2) as runtime:
        prepared = runtime.prepare(inputs)
        if not force_fallback and prepared._native_executor is None:
            pytest.skip(prepared.native_unavailable or prepared.native_cuda_unavailable or 'native CUDA executor unavailable')
        stream = torch.cuda.current_stream()
        for iteration in range(3):
            current = {'x': inputs['x'] + iteration}
            execution = prepared.commit(current, model_id=f'model-{iteration}', run_id=f'run-{iteration}')
            assert stream.query()
            assert prepared._ready.query()
            expected = runtime.commit(current, model_id=f'model-{iteration}', run_id=f'run-{iteration}')
            assert execution.commitment == expected.commitment
            for boundary in range(4):
                assert_state_equal(execution.recompute(boundary), expected.recompute(boundary))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
def test_prepared_strided_input_uses_copy_fallback_and_completes(monkeypatch):
    inputs = {'x': torch.arange(16, dtype=torch.int32, device='cuda')}
    with CommitmentRuntime([Layer('add', lambda state: {'x': state['x'] + 1}, preserves_inputs=True)]) as runtime:
        prepared = runtime.prepare(inputs)
        assert prepared._native_executor is not None

        def forbidden(*args, **kwargs):
            pytest.fail('strided inputs must bypass the native raw-copy executor')

        monkeypatch.setattr(prepared._native_executor, 'execute_with_prepare', forbidden)
        current = {'x': torch.arange(32, dtype=torch.int32, device='cuda')[::2]}
        actual = prepared.commit(current, model_id='strided', run_id='r')
        assert torch.cuda.current_stream().query()
        assert prepared._ready.query()
        expected = runtime.commit(current, model_id='strided', run_id='r')
        assert actual.commitment == expected.commitment
        for boundary in range(2):
            assert_state_equal(actual.recompute(boundary), expected.recompute(boundary))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
def test_prepared_zero_element_states_complete_without_native_copies():
    inputs = {'x': torch.empty((2, 0, 3), dtype=torch.int64, device='cuda')}
    layers = [Layer('empty-add', lambda state: {'x': state['x'] + 1}, preserves_inputs=True)]
    with CommitmentRuntime(layers) as runtime:
        prepared = runtime.prepare(inputs)
        assert prepared._native_executor is None
        actual = prepared.commit(inputs, model_id='empty', run_id='r')
        assert torch.cuda.current_stream().query()
        assert prepared._ready.query()
        expected = runtime.commit(inputs, model_id='empty', run_id='r')
        assert actual.commitment == expected.commitment
        for boundary in range(2):
            assert_state_equal(actual.recompute(boundary), expected.recompute(boundary))
