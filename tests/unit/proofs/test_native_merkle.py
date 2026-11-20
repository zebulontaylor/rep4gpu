"""Native finalization preserves the Python state and Merkle wire format."""
import hashlib
import shutil

import pytest

from rep4gpu.proofs.native_merkle import PreparedNativeMerkle
from rep4gpu.proofs.merkle import MerkleTree, verify_merkle
from rep4gpu.proofs.state_digest import finish_state_digest


@pytest.fixture(scope='module')
def native_available():
    if shutil.which('cc') is None:
        pytest.skip('C compiler unavailable')
    try:
        PreparedNativeMerkle([b''], [0])
    except RuntimeError as error:
        if 'openssl/sha.h' in str(error) or 'cannot find -lcrypto' in str(error):
            pytest.skip(str(error))
        raise


@pytest.mark.parametrize('count', [1, 2, 3, 17, 256, 257, 513])
def test_native_matches_python(native_available, count):
    metadata = [b'[["scalar","<i8",[]]]' if index % 2 else b'x' * (65 + index) for index in range(count)]
    counts = [index % 4 for index in range(count)]
    prepared = PreparedNativeMerkle(metadata, counts)
    packed = b''.join(hashlib.sha256(str(index).encode()).digest() for index in range(sum(counts)))
    for context in (bytes(32), hashlib.sha256(b'other context').digest()):
        payloads, offset = [], 0
        for encoded, size in zip(metadata, counts):
            digests = [packed[offset + i * 32:offset + (i + 1) * 32] for i in range(size)]
            payloads.append(context + finish_state_digest(encoded, digests))
            offset += size * 32
        expected = MerkleTree(payloads)
        tree, leaves = prepared.compute(packed, context)
        assert tree._levels == expected._levels
        assert tree.root == expected.root
        assert leaves == expected._levels[0]
        for index, payload in enumerate(payloads):
            assert verify_merkle(tree.root, payload, index, count, tree.proof(index))


def test_invalid_inputs(native_available):
    prepared = PreparedNativeMerkle([b'metadata'], [1])
    for packed, context in ((bytes(31), bytes(32)), (bytes(32), bytes(31)),
                            (bytearray(32), bytes(32)), (bytes(32), bytearray(32))):
        with pytest.raises(ValueError):
            prepared.compute(packed, context)
    for metadata, counts in (([], []), ([b'a'], []), ([b'a'], [-1]), ([b'a'], [True])):
        with pytest.raises(ValueError):
            PreparedNativeMerkle(metadata, counts)


@pytest.mark.parametrize('count', [1, 2, 3, 17, 256, 257])
def test_packed_output_and_lazy_tree(native_available, count):
    from rep4gpu.proofs.merkle import PackedMerkleLayout, PackedMerkleTree

    prepared = PreparedNativeMerkle([b'metadata'] * count, [1] * count)
    data, context = bytes(count * 32), bytes(32)
    packed = prepared.compute_packed(data, context)
    expected, expected_leaves = prepared.compute(data, context)
    layout = PackedMerkleLayout(count)
    tree = layout.bind(packed)
    assert isinstance(tree, PackedMerkleTree)
    assert tree.root == expected.root
    assert tree.leaf_count == count
    assert tuple(tree.leaf_hashes) == expected_leaves
    assert tree.leaf_hashes[-1] == expected_leaves[-1]
    assert tree.leaf_hashes[::2] == expected_leaves[::2]
    for index in range(count):
        assert tree.leaf_hash(index) == expected_leaves[index]
        assert tree.proof(index) == expected.proof(index)
    assert PackedMerkleTree.from_packed(packed, count).root == tree.root
    for invalid in (-1, count, True, '0'):
        with pytest.raises(IndexError):
            tree.leaf_hash(invalid)
        with pytest.raises(IndexError):
            tree.proof(invalid)
    for invalid in (packed[:-1], packed + b'x', bytearray(packed)):
        with pytest.raises(ValueError):
            layout.bind(invalid)
    with pytest.raises(ValueError):
        prepared.decode(packed[:-1])


@pytest.mark.parametrize('count', [0, -1, True, 1 << 64, '1'])
def test_packed_layout_rejects_invalid_counts(count):
    from rep4gpu.proofs.merkle import PackedMerkleLayout

    with pytest.raises(ValueError):
        PackedMerkleLayout(count)


def test_native_cuda_execution(native_available):
    import torch

    if not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    backend = PreparedNativeMerkle([b'first metadata', b'second metadata'], [1, 1])
    inputs = torch.zeros(64, dtype=torch.uint8, device='cuda')
    checkpoint = torch.empty_like(inputs)
    host = torch.empty(64, dtype=torch.uint8, pin_memory=True)
    # Warm the exact operations before graph capture.
    checkpoint.copy_(inputs + 1)
    host.copy_(checkpoint, non_blocking=True)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(graph):
        checkpoint.copy_(inputs + 1)
        host.copy_(checkpoint, non_blocking=True)
    graph.instantiate()
    ready = torch.cuda.Event()
    ready.record()
    ready.synchronize()
    executor = backend.prepare_cuda(torch.cuda.current_device(), graph.raw_cuda_graph_exec(),
                                    ready.cuda_event, host.data_ptr(), checkpoint.data_ptr(),
                                    checkpoint.numel(), [(inputs.data_ptr(), inputs.numel())])
    stream = torch.cuda.current_stream()
    for value in (3, 17):
        source = torch.full_like(inputs, value)
        saved = torch.empty_like(checkpoint)
        context = bytes([value]) * 32
        actual = executor.execute([source.data_ptr()], saved.data_ptr(), stream.cuda_stream, context)
        expected = backend.compute_packed(bytes([value + 1]) * 64, context)
        assert actual == expected
        assert torch.equal(saved, source + 1)
        assert host.numpy().tobytes() == bytes([value + 1]) * 64
    calls = []

    def prepare():
        calls.append('prepared')
        return bytes([19]) * 32

    actual = executor.execute_with_prepare([source.data_ptr()], saved.data_ptr(), stream.cuda_stream, prepare)
    assert calls == ['prepared']
    assert actual == backend.compute_packed(bytes([18]) * 64, bytes([19]) * 32)

    def fail():
        raise LookupError('preparation failed')

    for callback, error in ((fail, LookupError), (lambda: bytes(31), ValueError)):
        with pytest.raises(error):
            executor.execute_with_prepare([source.data_ptr()], saved.data_ptr(), stream.cuda_stream, callback)
        actual = executor.execute_with_prepare([source.data_ptr()], saved.data_ptr(), stream.cuda_stream, prepare)
        assert actual == backend.compute_packed(bytes([18]) * 64, bytes([19]) * 32)
    with pytest.raises(ValueError, match='pointer count'):
        executor.execute([], saved.data_ptr(), stream.cuda_stream, bytes(32))
    with pytest.raises(ValueError, match='context'):
        executor.execute([source.data_ptr()], saved.data_ptr(), stream.cuda_stream, bytes(31))


def test_native_launch_failure_drains_and_preserves_original_status(native_available, tmp_path):
    import ctypes
    import os
    import subprocess
    from rep4gpu.proofs.native_merkle import _SOURCE, _CUDA_SOURCE

    harness = r'''
static int failure_stage,drained,popped;
static int mock_current(void** value){*value=0;return 0;}
static int mock_push(void* value){return 0;}
static int mock_pop(void** value){popped++;return 99;}
static int mock_copy(uint64_t dst,uint64_t src,size_t size,void* stream){return failure_stage==1?7:0;}
static int mock_graph(void* graph,void* stream){return failure_stage==2?8:0;}
static int mock_record(void* event,void* stream){return failure_stage==3?9:0;}
static int mock_sync(void* stream){drained++;return 88;}
int test_failure(int stage){
    failure_stage=stage;drained=0;popped=0;
    current_context=mock_current;push_context=mock_push;pop_context=mock_pop;
    copy_device=mock_copy;launch_graph=mock_graph;record_event=mock_record;sync_stream=mock_sync;
    uint64_t pointer=0,size=1;
    Execution args={0};args.owner=(void*)1;args.inputs=1;
    args.destinations=&pointer;args.sources=&pointer;args.sizes=&size;
    int status=launch_cuda(&args);
    return status*100+drained*10+popped;
}
'''
    source, binary = tmp_path / 'failure.c', tmp_path / 'failure.so'
    source.write_text(_SOURCE + _CUDA_SOURCE + harness)
    result = subprocess.run([os.environ.get('CC', 'cc'), '-shared', '-fPIC', '-O2',
                             '-Wno-deprecated-declarations', str(source), '-o', str(binary),
                             '-lcrypto', '-ldl', '-pthread'], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    library = ctypes.CDLL(str(binary))
    library.test_failure.argtypes = [ctypes.c_int]
    library.test_failure.restype = ctypes.c_int
    for stage, status in ((1, 7), (2, 8), (3, 9)):
        assert library.test_failure(stage) == status * 100 + 11
