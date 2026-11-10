"""Exact sender-prefix packing and official BLAKE3 compatibility."""
from concurrent.futures import ThreadPoolExecutor
import ctypes as ct
import struct

import numpy as np
import pytest
import torch

from rep4gpu.spdz_hash_frames import hash_frames,pack_and_hash_frames,_module

blake3=pytest.importorskip('blake3').blake3
pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def expected(frames):
    messages=[struct.pack('<I',p)+frames[p].cpu().numpy().tobytes() for p in range(4)]
    return b''.join(messages),b''.join(blake3(message).digest() for message in messages)


@pytest.mark.parametrize('length',[0,1,59,60,61,63,64,65,92,93,94,1018,1019,1020,1021,2048,65536])
def test_pack_and_hash_official_oracle(length):
    rng=np.random.default_rng(120+length)
    frames={p:torch.from_numpy(rng.integers(0,256,length,dtype=np.uint8)).cuda() for p in range(4)}
    packed,tags=pack_and_hash_frames(frames)
    wanted,hashes=expected(frames)
    assert packed.shape==(4,length+4) and tags.shape==(128,)
    assert packed.cpu().numpy().tobytes()==wanted
    assert tags.cpu().numpy().tobytes()==hashes


@pytest.mark.parametrize('size',[0,1,32,64,927,928,1024,65536])
def test_dropin_hash_matches_existing_path(size):
    import triton
    from rep4gpu.spdz_device import _frame_inputs
    from rep4gpu.gpu_hash import digest_batch
    frames={p:torch.arange(93+size,dtype=torch.int64,device='cuda').to(torch.uint8) for p in range(4)}
    actual=hash_frames(frames,size)
    packed=torch.empty((4,97+size),dtype=torch.uint8,device='cuda')
    _frame_inputs[(triton.cdiv(97+size,256),4)](*(frames[p] for p in range(4)),packed,93+size,256)
    reference=(packed,digest_batch(packed,block_tree=True).reshape(-1))
    assert all(torch.equal(a,b) for a,b in zip(actual,reference))


def test_unaligned_storage_and_nondefault_stream():
    stream=torch.cuda.Stream()
    with torch.cuda.stream(stream):
        frames={p:torch.arange(100,dtype=torch.int64,device='cuda').to(torch.uint8)[1:] for p in range(4)}
        assert all(t.is_contiguous() and t.data_ptr()%4!=0 for t in frames.values())
        packed,tags=pack_and_hash_frames(frames)
    stream.synchronize()
    wanted,hashes=expected(frames)
    assert packed.cpu().numpy().tobytes()==wanted
    assert tags.cpu().numpy().tobytes()==hashes


def test_graph_replay_uses_current_frame_bytes():
    frames={p:torch.zeros(93+32,dtype=torch.uint8,device='cuda') for p in range(4)}
    hash_frames(frames,32)
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        packed,tags=hash_frames(frames,32)
    frames[2][-1]^=7
    graph.replay()
    wanted,hashes=expected(frames)
    assert packed.cpu().numpy().tobytes()==wanted
    assert tags.cpu().numpy().tobytes()==hashes


def test_cached_module_on_fresh_worker_context():
    frames={p:torch.zeros(93,dtype=torch.uint8,device='cuda') for p in range(4)}
    hash_frames(frames,0)
    driver,_,_,_=_module(torch.cuda.current_device())
    def worker():
        assert driver.cuCtxSetCurrent(ct.c_void_p())==0
        return hash_frames(frames,0)[1].cpu().numpy().tobytes()
    with ThreadPoolExecutor(1) as pool:
        assert pool.submit(worker).result(timeout=10)==expected(frames)[1]


def test_invalid_metadata_rejected_before_launch():
    frames={p:torch.empty(93,dtype=torch.uint8,device='cuda') for p in range(4)}
    with pytest.raises(ValueError):hash_frames(frames,-1)
    with pytest.raises(ValueError):hash_frames(frames,1)
    with pytest.raises(ValueError):pack_and_hash_frames({0:frames[0]})
    with pytest.raises(TypeError):pack_and_hash_frames({p:t.long() for p,t in frames.items()})
    with pytest.raises(ValueError):pack_and_hash_frames({p:t.cpu() for p,t in frames.items()})
    with pytest.raises(ValueError):pack_and_hash_frames({p:torch.empty(186,dtype=torch.uint8,device='cuda')[::2] for p in range(4)})


def test_pointer_table_graph_replay_reads_new_allocations():
    from rep4gpu.spdz_hash_frames import hash_frame_addresses
    length=125
    old_frames={p:torch.zeros(length,dtype=torch.uint8,device='cuda') for p in range(4)}
    pointers=torch.tensor([old_frames[p].data_ptr() for p in range(4)],dtype=torch.uint64,device='cuda')
    packed=torch.empty((4,length+4),dtype=torch.uint8,device='cuda')
    tags=torch.empty(128,dtype=torch.uint8,device='cuda')
    hash_frame_addresses(pointers,length,packed,tags)
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        hash_frame_addresses(pointers,length,packed,tags)
    current={p:torch.full((length,),p+7,dtype=torch.uint8,device='cuda') for p in range(4)}
    pointers.copy_(torch.tensor([current[p].data_ptr() for p in range(4)],dtype=torch.uint64,device='cuda'))
    graph.replay()
    wanted,hashes=expected(current)
    assert packed.cpu().numpy().tobytes()==wanted
    assert tags.cpu().numpy().tobytes()==hashes
    with pytest.raises(ValueError):hash_frame_addresses(pointers,1021,packed,tags)
    with pytest.raises(ValueError):hash_frame_addresses(pointers.long(),length,packed,tags)


def test_prepared_plan_matches_existing_plan_with_live_pointers():
    from rep4gpu.spdz_hash_frames import SmallFrameHashPlan
    from rep4gpu.spdz_device_graph import _HashPlan
    for size in (0,32,927):
        fused=SmallFrameHashPlan(size,'cuda')
        reference=_HashPlan(size,'cuda')
        for repeat in range(2):
            frames={p:torch.full((93+size,),p+repeat*11,dtype=torch.uint8,device='cuda') for p in range(4)}
            packed,tags=fused.replay(frames)
            a,b=reference.replay(frames)
            assert torch.equal(packed,a) and torch.equal(tags,b)
            wanted,hashes=expected(frames)
            assert packed.cpu().numpy().tobytes()==wanted
            assert tags.cpu().numpy().tobytes()==hashes
    with pytest.raises(ValueError):SmallFrameHashPlan(928,'cuda')


@pytest.mark.parametrize('size',[0,32,927,928])
def test_direct_digest_output_at_echo_offset93(size):
    frames={p:torch.full((93+size,),p+1,dtype=torch.uint8,device='cuda') for p in range(4)}
    echo=torch.full((221,),0xa5,dtype=torch.uint8,device='cuda')
    target=echo[93:]
    assert target.data_ptr()%4==1 and target.is_contiguous()
    packed,tags=hash_frames(frames,size,digest_output=target)
    assert tags is target
    wanted,hashes=expected(frames)
    assert packed.cpu().numpy().tobytes()==wanted
    assert target.cpu().numpy().tobytes()==hashes
    assert torch.all(echo[:93]==0xa5).item()


def test_digest_output_rejects_alias_and_invalid_metadata():
    frames={p:torch.zeros(157,dtype=torch.uint8,device='cuda') for p in range(4)}
    with pytest.raises(ValueError,match='overlap'):
        hash_frames(frames,64,digest_output=frames[0][:128])
    for target in (torch.empty(127,dtype=torch.uint8,device='cuda'),
                   torch.empty(128,dtype=torch.int32,device='cuda'),
                   torch.empty(128,dtype=torch.uint8),
                   torch.empty(256,dtype=torch.uint8,device='cuda')[::2]):
        with pytest.raises(ValueError,match='digest output'):
            hash_frames(frames,64,digest_output=target)
