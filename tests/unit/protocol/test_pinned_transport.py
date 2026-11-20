import pytest
import torch

from rep4gpu.protocol.pinned_transport import PinnedDeviceByteBridge


class Echo:
    def exchange(self, outgoing):
        return outgoing

    def close(self):
        pass


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_staging_reuse_preserves_distinct_outputs_and_empty_frames():
    bridge = PinnedDeviceByteBridge(Echo())
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        outputs = []
        for count in (1024, 17, 2048, 0, 513):
            value = torch.arange(count, device='cuda').to(torch.uint8)
            received = bridge.exchange_device({1: value, 2: value + 1}, {1: count, 2: count})
            outputs.append((received, count))
        for result, count in outputs:
            expected = torch.arange(count, device='cuda').to(torch.uint8)
            assert torch.equal(result[1], expected)
            assert torch.equal(result[2], expected + 1)
    bridge.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_reject_wrong_public_size_before_upload():
    bridge = PinnedDeviceByteBridge(Echo())
    with pytest.raises(ValueError, match='expected'):
        bridge.exchange_device({1: torch.zeros(7, dtype=torch.uint8, device='cuda')}, {1: 8})
    bridge.close()


class BorrowingEcho(Echo):
    consumes_outgoing_before_return = True

    def __init__(self):
        self.alias_checks = []

    def exchange(self, outgoing):
        assert all(isinstance(value, memoryview) and value.readonly for value in outgoing.values())
        if 1 in outgoing and 2 in outgoing:
            self.alias_checks.append(outgoing[1] is outgoing[2])
        return outgoing


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_identical_slices_stage_once_and_borrowed_echo_survives_reuse():
    transport = BorrowingEcho()
    bridge = PinnedDeviceByteBridge(transport)
    stream = torch.cuda.Stream()
    saved = []
    with torch.cuda.stream(stream):
        for value in (3, 19, 255):
            source = torch.full((2048,), value, dtype=torch.uint8, device='cuda')
            # Distinct tensor objects with the same slice must also deduplicate.
            received = bridge.exchange_device({1: source[:1024], 2: source[:1024]}, {1: 1024, 2: 1024})
            saved.append((received, value))
            assert bridge._send_host.numel() == 1024
        for received, value in saved:
            assert torch.all(received[1] == value)
            assert torch.all(received[2] == value)
    assert transport.alias_checks == [True, True, True]
    bridge.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_overlapping_different_slices_are_not_deduplicated():
    transport = BorrowingEcho()
    bridge = PinnedDeviceByteBridge(transport)
    source = torch.arange(16, device='cuda').to(torch.uint8)
    received = bridge.exchange_device({1: source[:8], 2: source[4:12]}, {1: 8, 2: 8})
    assert bridge._send_host.numel() == 16
    assert transport.alias_checks == [False]
    assert torch.equal(received[1], source[:8])
    assert torch.equal(received[2], source[4:12])
    bridge.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_retaining_transport_keeps_owned_snapshots():
    class RetainingEcho(Echo):
        def __init__(self):
            self.history = []

        def exchange(self, outgoing):
            assert all(isinstance(value, bytes) for value in outgoing.values())
            self.history.append(dict(outgoing))
            return outgoing

    transport = RetainingEcho()
    bridge = PinnedDeviceByteBridge(transport)
    for value in (7, 23):
        source = torch.full((128,), value, dtype=torch.uint8, device='cuda')
        bridge.exchange_device({1: source, 2: source}, {1: 128, 2: 128})
    assert transport.history[0][1] == bytes([7]) * 128
    assert transport.history[1][1] == bytes([23]) * 128
    assert transport.history[0][1] is transport.history[0][2]
    bridge.close()
