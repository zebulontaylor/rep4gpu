"""Official CPU BLAKE3 compatibility for the experimental GPU backend."""
import numpy as np
import pytest
import torch

from rep4gpu.gpu_hash import digest, digest_batch

blake3 = pytest.importorskip('blake3').blake3
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def sample(length):
    # Official BLAKE3 test vectors use this repeating 251-byte input pattern.
    return (np.arange(length,dtype=np.uint64)%251).astype(np.uint8)


@pytest.mark.parametrize('length',[0,1,3,63,64,65,127,128,129,1023,1024,1025,
                                   2048,2049,3072,4097,6145,7168,13313,
                                   2*1024*1024,4*1024*1024+131])
def test_official_cpu_compatibility(length):
    source=sample(length)
    actual=digest(torch.from_numpy(source).cuda()).cpu().numpy().tobytes()
    assert actual==blake3(source.tobytes()).digest()


def test_known_empty_digest_and_abc():
    empty=torch.empty(0,dtype=torch.uint8,device='cuda')
    assert digest(empty).cpu().numpy().tobytes().hex()==(
        'af1349b9f5f9a1a6a0404dea36dcc9499bcb25c9adc112b7cc9a93cae41f3262')
    value=torch.tensor(list(b'abc'),dtype=torch.uint8,device='cuda')
    assert digest(value).cpu().numpy().tobytes()==blake3(b'abc').digest()


@pytest.mark.parametrize('length',[0,1,65,1025,3073,5*1024+67])
def test_equal_length_batch(length):
    rng=np.random.default_rng(20261110+length)
    source=rng.integers(0,256,(5,length),dtype=np.uint8)
    actual=digest_batch(torch.from_numpy(source).cuda()).cpu().numpy()
    for row,output in zip(source,actual):
        assert output.tobytes()==blake3(row.tobytes()).digest()


def test_unaligned_contiguous_input():
    source=sample(3074)
    storage=torch.from_numpy(source).cuda()
    value=storage[1:]
    assert value.is_contiguous() and value.data_ptr()%4!=0
    assert digest(value).cpu().numpy().tobytes()==blake3(source[1:].tobytes()).digest()


@pytest.mark.parametrize("block_tree",[False,True])
def test_graph_replay_reads_current_bytes(block_tree):
    source=sample(3*1024+19)
    value=torch.from_numpy(source).cuda()
    digest(value,block_tree=block_tree)
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output=digest(value,block_tree=block_tree)
    value.bitwise_xor_(255)
    graph.replay()
    assert output.cpu().numpy().tobytes()==blake3((source^255).tobytes()).digest()


def test_nondefault_stream_and_empty_batch():
    stream=torch.cuda.Stream()
    source=sample(5*1024+23)
    with torch.cuda.stream(stream):
        value=torch.from_numpy(source).cuda()
        output=digest(value)
    stream.synchronize()
    assert output.cpu().numpy().tobytes()==blake3(source.tobytes()).digest()
    assert digest_batch(torch.empty((0,17),dtype=torch.uint8,device='cuda')).shape==(0,32)


def test_invalid_inputs():
    with pytest.raises(TypeError):
        digest(torch.zeros(7,dtype=torch.int64,device='cuda'))
    with pytest.raises(ValueError):
        digest(torch.zeros(7,dtype=torch.uint8))
    with pytest.raises(ValueError):
        digest(torch.zeros((2,7),dtype=torch.uint8,device='cuda'))
    with pytest.raises(ValueError):
        digest(torch.zeros(14,dtype=torch.uint8,device='cuda')[::2])


@pytest.mark.parametrize('length',[0,1,63,64,65,1023,1024,1025,3073,127*1024+17,128*1024,128*1024+1,129*1024+1,255*1024+17,256*1024+1,4*1024*1024+131])
def test_block_tree_matches_official_cpu(length):
    source=sample(length)
    actual=digest(torch.from_numpy(source).cuda(),block_tree=True).cpu().numpy().tobytes()
    assert actual==blake3(source.tobytes()).digest()


@pytest.mark.parametrize('length',[0,65,1025,3*1024+17,129*1024+19])
def test_block_tree_batches(length):
    source=np.random.default_rng(length).integers(0,256,(3,length),dtype=np.uint8)
    actual=digest_batch(torch.from_numpy(source).cuda(),block_tree=True).cpu().numpy()
    for row,output in zip(source,actual):
        assert output.tobytes()==blake3(row.tobytes()).digest()


def test_cached_module_launch_on_worker_without_current_context():
    from concurrent.futures import ThreadPoolExecutor
    import ctypes as ct
    from rep4gpu.gpu_hash import _module
    digest(torch.empty(0,dtype=torch.uint8,device='cuda'))
    driver,_,_,_=_module(torch.cuda.current_device())
    def worker():
        data=torch.empty(0,dtype=torch.uint8,device='cuda')
        assert driver.cuCtxSetCurrent(ct.c_void_p())==0
        return digest(data).cpu().numpy().tobytes()
    with ThreadPoolExecutor(1) as pool:
        assert pool.submit(worker).result(timeout=10)==blake3(b'').digest()


def test_context_guard_restores_absent_worker_context():
    from concurrent.futures import ThreadPoolExecutor
    import ctypes as ct
    from rep4gpu.gpu_hash import _module,_current_context
    driver,_,_,owner=_module(torch.cuda.current_device())
    def worker():
        assert driver.cuCtxSetCurrent(ct.c_void_p())==0
        with _current_context(driver,owner):
            active=ct.c_void_p()
            assert driver.cuCtxGetCurrent(ct.byref(active))==0
            assert active.value==owner.value
        previous=ct.c_void_p()
        assert driver.cuCtxGetCurrent(ct.byref(previous))==0
        assert previous.value is None
    with ThreadPoolExecutor(1) as pool:
        pool.submit(worker).result(timeout=10)
