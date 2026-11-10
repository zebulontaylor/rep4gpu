from concurrent.futures import ThreadPoolExecutor
import threading
from types import SimpleNamespace
import pytest
import torch
from rep4gpu.runtime import Party
from rep4gpu.device_runtime import DeviceParty


def test_cpu_mask_draws_reserve_before_crypto(monkeypatch):
    barrier = threading.Barrier(2,timeout=3)
    seen = []
    def generate(seed,counter,count,**kwargs):
        seen.append(counter)
        barrier.wait()
        return bytes(count*8)
    monkeypatch.setattr('rep4gpu.runtime.random_bytes',generate)
    runtime = Party(0,None,{e:bytes([e])*32 for e in (1,2,3)})
    with ThreadPoolExecutor(2) as pool:
        values = list(pool.map(lambda _:runtime._random_bytes(1,(3,)),range(2)))
    assert sorted(seen) == [0,1]
    assert runtime.counters[1] == 2
    assert values == [bytes(24),bytes(24)]


def test_mask_generation_failure_burns_reserved_nonce(monkeypatch):
    runtime = Party(0,None,{e:bytes([e])*32 for e in (1,2,3)})
    def fail(*args,**kwargs):
        raise RuntimeError('injected crypto failure')
    monkeypatch.setattr('rep4gpu.runtime.random_bytes',fail)
    with pytest.raises(RuntimeError,match='injected'):
        runtime._random_bytes(1,(3,))
    assert runtime.counters[1] == 1
    assert runtime._reserve_nonces((1,1,2,1)) == (1,2,0,3)
    assert runtime.counters == {1:4,2:1,3:0}


def _device_stub():
    # DeviceParty's counter/dispatch path can be tested without a GPU launch.
    runtime = DeviceParty.__new__(DeviceParty)
    runtime.device = torch.device('cuda:0')
    runtime.bits = runtime.binary_bits = 64
    runtime.counters = {1:0}
    runtime._counter_lock = threading.Lock()
    runtime._device_keys = {1:SimpleNamespace(key=lambda domain:None)}
    return runtime


def test_device_draws_reserve_before_launch(monkeypatch):
    barrier = threading.Barrier(2,timeout=3)
    seen = []
    def generate(key,nonce,count,bits):
        seen.append(nonce)
        barrier.wait()
        return torch.zeros(count,dtype=torch.int64)
    monkeypatch.setattr('rep4gpu.gpu_prg.generate',generate)
    runtime = _device_stub()
    with ThreadPoolExecutor(2) as pool:
        list(pool.map(lambda _:runtime._random(1,(3,),runtime.device),range(2)))
    assert sorted(seen) == [0,1]
    assert runtime.counters[1] == 2


def test_failed_device_launch_burns_nonce(monkeypatch):
    runtime = _device_stub()
    def fail(*args,**kwargs):
        raise RuntimeError('injected GPU launch failure')
    monkeypatch.setattr('rep4gpu.gpu_prg.generate',fail)
    with pytest.raises(RuntimeError,match='injected'):
        runtime._random(1,(3,),runtime.device)
    assert runtime.counters[1] == 1
    seen = []
    def succeed(key,nonce,count,bits):
        seen.append(nonce)
        return torch.zeros(count,dtype=torch.int64)
    monkeypatch.setattr('rep4gpu.gpu_prg.generate',succeed)
    runtime._random(1,(3,),runtime.device)
    assert seen == [1] and runtime.counters[1] == 2


def test_counter_exhaustion_does_not_partially_reserve():
    runtime = Party(0,None,{e:bytes([e])*32 for e in (1,2,3)})
    runtime.counters[2] = 1 << 64
    with pytest.raises(ValueError,match='exhausted'):
        runtime._reserve_nonces((1,2))
    assert runtime.counters[1] == 0


def test_device_mixed_domains_share_fresh_counter_order(monkeypatch):
    runtime = _device_stub()
    runtime.bits, runtime.binary_bits = 32,64
    runtime._device_keys[1] = SimpleNamespace(key=lambda domain:domain)
    seen = []
    def generate(key,nonce,count,bits):
        seen.append((key,nonce,count,bits))
        return torch.zeros(count,dtype=torch.int64)
    monkeypatch.setattr('rep4gpu.gpu_prg.generate',generate)
    runtime._random(1,(2,),runtime.device,binary=False)
    runtime._random(1,(3,),runtime.device,binary=True)
    runtime._random(1,(1,),runtime.device,binary=False)
    assert seen == [(bytes([32,64,0,32,1]),0,2,32),
                    (bytes([32,64,1,64,1]),1,3,64),
                    (bytes([32,64,0,32,1]),2,1,32)]


@pytest.mark.parametrize('party',[0,1,2,3])
def test_device_batch_reserves_schedule_before_launch(monkeypatch,party):
    runtime = _device_stub()
    runtime.party = party
    runtime.bits = runtime.binary_bits = 32
    runtime.prg_batch = True
    runtime._batch_keys = {}
    runtime.counters = {e:0 for e in range(4) if e != party}
    runtime._device_keys = {e:SimpleNamespace(key=lambda domain:domain) for e in runtime.counters}
    seen = []
    def fail(keys,nonces,count,bits,**kwargs):
        seen.append((keys,nonces,count,bits))
        raise RuntimeError('injected batch launch failure')
    monkeypatch.setattr('rep4gpu.gpu_prg_batch.generate_rows',fail)
    with pytest.raises(RuntimeError,match='injected batch'):
        runtime._mask_batch((3,),False,runtime.device)
    expected_counters = {e:0 for e in runtime.counters}
    expected_nonces,expected_keys = [],[]
    for receiver in (3,0,1,2,3,2):
        if receiver == party:
            expected_nonces.append(None)
            expected_keys.append(None)
        else:
            expected_nonces.append(expected_counters[receiver])
            expected_counters[receiver] += 1
            expected_keys.append(bytes([32,32,0,32,receiver]))
    assert seen == [(tuple(expected_keys),tuple(expected_nonces),3,32)]
    assert runtime.counters == expected_counters
