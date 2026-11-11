"""Message ownership, public schedules, and deterministic device replay."""

import concurrent.futures

import pytest
import torch


def test_byte_bridge_restores_expected_empty_socket_frames():
    from rep4gpu.device_transport import DeviceByteBridge
    class EmptySocket:
        def exchange(self, outgoing):
            assert outgoing == {1: b''}
            return {}
    bridge = DeviceByteBridge(EmptySocket(), 'cpu')
    received = bridge.exchange_device({1: torch.empty(0, dtype=torch.uint8)}, {3: 0})
    assert set(received) == {3} and received[3].numel() == 0

from rep4gpu.device_transport import (DeviceByteBridge, DeviceFrame,
                                     DeviceIdealEndpoint, DeviceRecorder, DeviceReplay)
from rep4gpu.transport import IdealNetwork


def _payload(party, peer):
    return torch.tensor([party, peer, 0, 255], dtype=torch.uint8)


def _four_role_exchange(adapter):
    network = IdealNetwork()
    endpoints = [adapter(network.endpoint(p), 'cpu') for p in range(4)]
    sources = [{} for _ in range(4)]
    def work(p):
        endpoint = endpoints[p]
        byte_in = endpoint.exchange({(p+1)%4: bytes([p, 71])})
        sources[p] = {(p+1)%4: _payload(p, (p+1)%4), (p+2)%4: torch.empty(0, dtype=torch.uint8)}
        incoming = endpoint.exchange_device(sources[p], {(p-1)%4: 4, (p-2)%4: 0})
        byte_after = endpoint.exchange({(p+3)%4: bytes([p, 19, 23])})
        return byte_in, incoming, byte_after
    with concurrent.futures.ThreadPoolExecutor(4) as pool:
        outputs = list(pool.map(work, range(4)))
    for p, (before, incoming, after) in enumerate(outputs):
        assert before == {(p-1)%4: bytes([(p-1)%4, 71])}
        assert after == {(p+1)%4: bytes([(p+1)%4, 19, 23])}
        assert torch.equal(incoming[(p-1)%4], _payload((p-1)%4, p))
        assert incoming[(p-2)%4].numel() == 0
        assert endpoints[p].bytes_sent == 9
        assert endpoints[p].messages_sent == 4
        source = sources[(p-1)%4][p]
        assert incoming[(p-1)%4].data_ptr() != source.data_ptr()
        source.fill_(13)
        assert torch.equal(incoming[(p-1)%4], _payload((p-1)%4, p))
    return outputs


@pytest.mark.parametrize('adapter', [DeviceIdealEndpoint, DeviceByteBridge])
def test_four_role_mixed_protocols_and_receiver_ownership(adapter):
    _four_role_exchange(adapter)


def test_host_bridge_and_device_adapter_deliver_identical_messages():
    ideal = _four_role_exchange(DeviceIdealEndpoint)
    bridge = _four_role_exchange(DeviceByteBridge)
    for (left_before, left, left_after), (right_before, right, right_after) in zip(ideal, bridge):
        assert left_before == right_before and left_after == right_after
        assert left.keys() == right.keys()
        assert all(torch.equal(left[p], right[p]) for p in left)


@pytest.mark.parametrize('adapter', [DeviceIdealEndpoint, DeviceByteBridge])
@pytest.mark.parametrize('schedule', [{}, {2: 4}, {1: 3}])
def test_incoming_schedule_rejects_missing_wrong_peer_and_size(adapter, schedule):
    class StaticTransport:
        def exchange(self, outgoing):
            if adapter is DeviceIdealEndpoint:
                from rep4gpu.device_transport import _Envelope
                return {1: _Envelope(torch.tensor([1, 2, 3, 4], dtype=torch.uint8))}
            return {1: bytes([1, 2, 3, 4])}
    with pytest.raises(ValueError, match='peers|expected'):
        adapter(StaticTransport(), 'cpu').exchange_device({}, schedule)


@pytest.mark.parametrize('adapter', [DeviceIdealEndpoint, DeviceByteBridge])
@pytest.mark.parametrize('payload', [torch.zeros(3, dtype=torch.int64),
                                    torch.zeros((2, 2), dtype=torch.uint8),
                                    torch.zeros(8, dtype=torch.uint8)[::2]])
def test_invalid_outgoing_tensor_rejected_before_transport(adapter, payload):
    class NoTransport:
        def exchange(self, outgoing):
            raise AssertionError('invalid messages must not be sent')
    with pytest.raises((TypeError, ValueError)):
        adapter(NoTransport(), 'cpu').exchange_device({1: payload}, {})


@pytest.mark.parametrize('adapter', [DeviceIdealEndpoint, DeviceByteBridge])
@pytest.mark.parametrize('staging', ['device', 'host'])
def test_recording_replay_checks_exact_messages_and_phases(adapter, staging):
    network = IdealNetwork()
    recorders = [DeviceRecorder(adapter(network.endpoint(p), 'cpu')) for p in range(4)]
    def work(p):
        recorder = recorders[p]
        before = recorder.exchange({(p+1)%4: bytes([p])})
        payload = _payload(p, (p+1)%4)
        incoming = recorder.exchange_device({(p+1)%4: payload}, {(p-1)%4: 4})
        payload.fill_(91)
        after = recorder.exchange({(p+2)%4: b'end'})
        return before, incoming, after
    with concurrent.futures.ThreadPoolExecutor(4) as pool:
        outputs = list(pool.map(work, range(4)))
    for p, recorder in enumerate(recorders):
        assert recorder.bytes_sent == 8 and recorder.messages_sent == 3
        replay = DeviceReplay(recorder.transcript, device='cpu', staging=staging)
        with pytest.raises(ValueError, match='consumed 0'):
            replay.assert_consumed()
        ticks = iter(range(100))
        synchronizations = []
        replay.start(clock=lambda: next(ticks), synchronize=lambda: synchronizations.append(True))
        assert replay.exchange({(p+1)%4: bytes([p])}) == outputs[p][0]
        incoming = replay.exchange_device({(p+1)%4: _payload(p, (p+1)%4)}, {(p-1)%4: 4})
        assert torch.equal(incoming[(p-1)%4], outputs[p][1][(p-1)%4])
        assert replay.exchange({(p+2)%4: b'end'}) == outputs[p][2]
        phases = replay.finish()
        assert phases == [1, 1, 1, 1]
        assert len(synchronizations) == 5
        assert replay.bytes_sent == recorder.bytes_sent
        assert replay.messages_sent == recorder.messages_sent
        replay.assert_consumed()
        with pytest.raises(ValueError, match='exhausted'):
            replay.exchange({})


@pytest.mark.parametrize('staging', ['device', 'host'])
def test_replay_rejects_outgoing_content_peer_schedule_and_frame_kind(staging):
    frames = [DeviceFrame('device', {1: b'abc'}, {2: b'defg'})]
    def replay():
        return DeviceReplay(frames, device='cpu', staging=staging)
    bad = replay()
    with pytest.raises(ValueError, match='outgoing mismatch'):
        bad.exchange_device({1: torch.tensor([97, 98, 0], dtype=torch.uint8)}, {2: 4})
    assert bad.round == 0
    with pytest.raises(ValueError, match='outgoing mismatch'):
        replay().exchange_device({3: torch.tensor([97, 98, 99], dtype=torch.uint8)}, {2: 4})
    with pytest.raises(ValueError, match='peers'):
        replay().exchange_device({1: torch.tensor([97, 98, 99], dtype=torch.uint8)}, {3: 4})
    with pytest.raises(ValueError, match='expected'):
        replay().exchange_device({1: torch.tensor([97, 98, 99], dtype=torch.uint8)}, {2: 3})
    with pytest.raises(ValueError, match='frame 0'):
        replay().exchange({})


def test_sender_can_reuse_buffer_before_receiver_returns():
    import threading
    network = IdealNetwork()
    sender_mutated = threading.Event()
    source = torch.tensor([1, 2, 3], dtype=torch.uint8)
    class DelayedReceiver:
        def __init__(self, endpoint):
            self.endpoint = endpoint
        def exchange(self, outgoing):
            incoming = self.endpoint.exchange(outgoing)
            if not sender_mutated.wait(timeout=5):
                raise AssertionError('sender did not finish its exchange')
            return incoming
    endpoints = [DeviceIdealEndpoint(DelayedReceiver(network.endpoint(p)) if p == 1
                                    else network.endpoint(p), 'cpu') for p in range(4)]
    def work(p):
        incoming = endpoints[p].exchange_device({1: source} if p == 0 else {}, {0: 3} if p == 1 else {})
        if p == 0:
            source.fill_(99)
            sender_mutated.set()
        return incoming
    with concurrent.futures.ThreadPoolExecutor(4) as pool:
        outputs = list(pool.map(work, range(4)))
    assert torch.equal(outputs[1][0], torch.tensor([1, 2, 3], dtype=torch.uint8))
    assert outputs[1][0].data_ptr() != source.data_ptr()


def test_device_ideal_rejects_byte_frame_at_device_boundary():
    class WrongProtocol:
        def exchange(self, outgoing):
            return {1: b'bytes'}
    with pytest.raises(TypeError, match='byte-protocol'):
        DeviceIdealEndpoint(WrongProtocol(), 'cpu').exchange_device({}, {1: 5})


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is required for stream ordering')
def test_cuda_nondefault_stream_events_and_sender_reuse():
    import threading
    network = IdealNetwork()
    reused = threading.Event()
    source = torch.zeros(4096, dtype=torch.uint8, device='cuda')
    torch.cuda.synchronize()
    class DelayedReceiver:
        def __init__(self, endpoint):
            self.endpoint = endpoint
        def exchange(self, outgoing):
            incoming = self.endpoint.exchange(outgoing)
            if not reused.wait(timeout=5):
                raise AssertionError('sender did not enqueue buffer reuse')
            return incoming
    endpoints = [DeviceIdealEndpoint(DelayedReceiver(network.endpoint(p)) if p == 1
                                    else network.endpoint(p), 'cuda') for p in range(4)]
    def work(p):
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            if p == 0:
                torch.cuda._sleep(5_000_000)
                source.fill_(11)
            incoming = endpoints[p].exchange_device({1: source} if p == 0 else {},
                                                    {0: source.numel()} if p == 1 else {})
            if p == 0:
                source.fill_(99)
                reused.set()
            result = incoming[0] + 3 if p == 1 else None
        stream.synchronize()
        return result
    with concurrent.futures.ThreadPoolExecutor(4) as pool:
        outputs = list(pool.map(work, range(4)))
    assert torch.equal(outputs[1].cpu(), torch.full((4096,), 14, dtype=torch.uint8))
    assert torch.equal(source.cpu(), torch.full((4096,), 99, dtype=torch.uint8))


@pytest.mark.parametrize('adapter',[DeviceIdealEndpoint,DeviceByteBridge])
def test_receive_into_retains_receiver_storage(adapter):
    network = IdealNetwork()
    endpoints = [adapter(network.endpoint(p),'cpu') for p in range(4)]
    buffers = [{(p-1)%4:torch.empty(4,dtype=torch.uint8), (p-2)%4:torch.empty(0,dtype=torch.uint8)} for p in range(4)]
    sources = [_payload(p,(p+1)%4) for p in range(4)]
    def work(p):
        outgoing = {(p+1)%4:sources[p],(p+2)%4:torch.empty(0,dtype=torch.uint8)}
        return endpoints[p].exchange_device_into(outgoing,{(p-1)%4:4,(p-2)%4:0},buffers[p])
    with concurrent.futures.ThreadPoolExecutor(4) as pool:
        outputs = list(pool.map(work,range(4)))
    for p,output in enumerate(outputs):
        assert output is buffers[p]
        source = sources[(p-1)%4]
        source.zero_()
        torch.testing.assert_close(output[(p-1)%4],_payload((p-1)%4,p))


@pytest.mark.parametrize('staging',['device','host'])
def test_replay_binds_receiver_buffers_and_preserves_checks(staging,monkeypatch):
    frames = [DeviceFrame('device',{1:b'abc'},{3:b'xyz'}),DeviceFrame('device',{2:b''},{0:b''})]
    buffers = [{3:torch.zeros(3,dtype=torch.uint8)},{0:torch.empty(0,dtype=torch.uint8)}]
    replay = DeviceReplay(frames,device='cpu',staging=staging,incoming_buffers=buffers)
    if staging == 'device':
        torch.testing.assert_close(buffers[0][3],torch.tensor(list(b'xyz'),dtype=torch.uint8))
        def unexpected(*args,**kwargs):
            raise AssertionError('device preload performed an in-exchange copy')
        monkeypatch.setattr(torch.Tensor,'copy_',unexpected)
    outgoing = {1:torch.tensor(list(b'abc'),dtype=torch.uint8)}
    assert replay.exchange_device_into(outgoing,{3:3},buffers[0]) is buffers[0]
    assert replay.exchange_device_into({2:torch.empty(0,dtype=torch.uint8)},{0:0},buffers[1]) is buffers[1]
    assert replay.round == 2
    monkeypatch.undo()
    wrong = DeviceReplay(frames,device='cpu',staging=staging,incoming_buffers=None)
    with pytest.raises(ValueError,match='outgoing mismatch'):
        wrong.exchange_device_into({1:torch.zeros(3,dtype=torch.uint8)},{3:3},buffers[0])


@pytest.mark.parametrize('incoming',[{}, {3:b'badsize'}, {3:torch.zeros(3,dtype=torch.uint8)}])
def test_bridge_receive_into_rejects_invalid_frame(incoming):
    class Static:
        def exchange(self,outgoing):
            return incoming
    bridge = DeviceByteBridge(Static(),'cpu')
    with pytest.raises((ValueError,TypeError)):
        bridge.exchange_device_into({}, {3:3}, {3:torch.empty(3,dtype=torch.uint8)})
