"""Reusable pinned host staging for live CUDA protocol messages."""
import torch

from .device_transport import DeviceByteBridge, _buffers, _sizes, _validate_tensor
from .transport import SocketTransport


class PinnedDeviceByteBridge(DeviceByteBridge):
    """Batch outgoing copies behind one synchronization per exchange.

    Calls are sequential, as required by the party runtime. CUDA consumers use
    the calling stream. Receive staging stays alive until its upload completes;
    later exchanges wait for that upload before reusing the host buffer.
    Authentication and framing remain the underlying transport's responsibility.

    SocketTransport consumes borrowed send buffers before returning. Other
    transports receive immutable snapshots unless they explicitly declare
    ``consumes_outgoing_before_return = True``. Such transports may return
    aliases of the borrowed buffers, but must not retain them after the next
    exchange begins.
    """
    staging = 'pinned_host'

    def __init__(self, underlying, device='cuda'):
        super().__init__(underlying, device)
        if self.device.type != 'cuda':
            raise ValueError('pinned device bridge requires CUDA')
        self._send_host = torch.empty(0, dtype=torch.uint8, pin_memory=True)
        self._receive_host = torch.empty(0, dtype=torch.uint8, pin_memory=True)
        self._receive_ready = None
        self._borrow_send_buffers = isinstance(underlying, SocketTransport) or bool(
            getattr(underlying, 'consumes_outgoing_before_return', False))

    def exchange_device(self, outgoing, incoming_sizes):
        buffers = {p: torch.empty(size, dtype=torch.uint8, device=self.device)
                   for p, size in incoming_sizes.items()}
        return self.exchange_device_into(outgoing, incoming_sizes, buffers)

    def exchange_device_into(self, outgoing, incoming_sizes, buffers):
        _buffers(buffers, incoming_sizes, self.device)
        for value in outgoing.values():
            _validate_tensor(value, self.device)
        stream = torch.cuda.current_stream(self.device)
        if self._receive_ready is not None:
            stream.wait_event(self._receive_ready)
        # Only identical slices alias here. Deduplicating by secret contents
        # would make staging behavior depend on the payload's values.
        unique, peer_keys = {}, {}
        for peer, value in outgoing.items():
            key = (value.data_ptr(), value.numel())
            unique.setdefault(key, value)
            peer_keys[peer] = key
        size = sum(value.numel() for value in unique.values())
        if self._send_host.numel() < size:
            self._send_host = torch.empty(size, dtype=torch.uint8, pin_memory=True)
        staged, offset = {}, 0
        for key, value in unique.items():
            host = self._send_host[offset:offset + value.numel()]
            host.copy_(value, non_blocking=True)
            staged[key] = host
            offset += value.numel()
        # This also completes the previous receive upload before host reuse.
        stream.synchronize()
        messages = {key: (memoryview(value.numpy()).toreadonly() if self._borrow_send_buffers
                          else value.numpy().tobytes()) for key, value in staged.items()}
        incoming = self.underlying.exchange({peer: messages[key] for peer, key in peer_keys.items()})
        for peer, size in incoming_sizes.items():
            if size == 0:
                incoming.setdefault(peer, b'')
        _sizes(incoming, incoming_sizes)
        size = sum(incoming_sizes.values())
        if self._receive_host.numel() < size:
            self._receive_host = torch.empty(size, dtype=torch.uint8, pin_memory=True)
        offset = 0
        for peer, value in incoming.items():
            if not isinstance(value, (bytes, bytearray, memoryview)):
                raise TypeError('byte transport received a non-byte frame')
            host = self._receive_host[offset:offset + len(value)]
            if value:
                host.copy_(torch.frombuffer(bytearray(value), dtype=torch.uint8))
                buffers[peer].copy_(host, non_blocking=True)
            offset += len(value)
        self._receive_ready = torch.cuda.Event()
        self._receive_ready.record(stream)
        return buffers

    def close(self):
        if self._receive_ready is not None:
            self._receive_ready.synchronize()
        super().close()
