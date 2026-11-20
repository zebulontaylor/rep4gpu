"""Device-message adapters for the experimental GPU-resident protocol.

The ideal adapter is a colocated test harness, not party isolation. The byte
bridge carries the same device-generated messages over an existing transport,
including mutual TLS. Recording and replay are benchmark fixtures only.
"""
from dataclasses import dataclass
import torch

from .byte import ReplayTransport


def _device(value):
    device = torch.device(value)
    if device.type == 'cuda' and device.index is None:
        device = torch.device('cuda', torch.cuda.current_device())
    return device


def _validate_tensor(value, device=None):
    if not isinstance(value, torch.Tensor) or value.dtype != torch.uint8 or value.ndim != 1:
        raise TypeError('device messages must be flat uint8 tensors')
    if not value.is_contiguous():
        raise ValueError('device messages must be contiguous')
    if device is not None and value.device != torch.device(device):
        raise ValueError('message is on the wrong device')


def _bytes(value):
    _validate_tensor(value)
    return value.detach().cpu().numpy().tobytes()


def _tensor(value, device):
    if not value:
        return torch.empty(0, dtype=torch.uint8, device=device)
    return torch.frombuffer(bytearray(value), dtype=torch.uint8).to(device)


def _sizes(incoming, expected):
    if set(incoming) != set(expected):
        raise ValueError('device message peers differ from the public schedule')
    for peer, value in incoming.items():
        size = value.numel() if isinstance(value, torch.Tensor) else len(value)
        if size != expected[peer]:
            raise ValueError(f'device message from party {peer} has {size} bytes; expected {expected[peer]}')


def _buffers(buffers, sizes, device):
    _sizes(buffers, sizes)
    for value in buffers.values():
        _validate_tensor(value, device)


def _copy_bytes_into(incoming, sizes, buffers, device):
    _buffers(buffers, sizes, device)
    _sizes(incoming, sizes)
    for peer, value in incoming.items():
        if not isinstance(value, (bytes, bytearray, memoryview)):
            raise TypeError('byte transport received a non-byte frame')
        if value:
            buffers[peer].copy_(torch.frombuffer(bytearray(value), dtype=torch.uint8))
    return buffers


@dataclass
class _Envelope:
    payload: torch.Tensor
    ready: object = None

    def __len__(self):
        return self.payload.numel()


class DeviceIdealEndpoint:
    """Rendezvous between colocated roles, copying into receiver-owned storage."""
    staging = 'device'

    def __init__(self, underlying, device='cuda'):
        self.underlying = underlying
        self.device = _device(device)

    @property
    def bytes_sent(self):
        return self.underlying.bytes_sent

    @property
    def messages_sent(self):
        return self.underlying.messages_sent

    def exchange(self, outgoing):
        return self.underlying.exchange(outgoing)

    def exchange_device(self, outgoing, incoming_sizes):
        envelopes = {}
        for peer, payload in outgoing.items():
            _validate_tensor(payload, self.device)
            # Each destination owns a snapshot before the sender can return and
            # reuse its source buffer. The receive side adopts this allocation.
            snapshot = payload.clone()
            event = None
            if payload.is_cuda:
                event = torch.cuda.Event()
                event.record(torch.cuda.current_stream(payload.device))
            envelopes[peer] = _Envelope(snapshot, event)
        received = self.underlying.exchange(envelopes)
        if any(not isinstance(value, _Envelope) for value in received.values()):
            raise TypeError('device exchange received a byte-protocol frame')
        result = {}
        for peer, envelope in received.items():
            value = envelope.payload
            _validate_tensor(value, self.device)
            if value.is_cuda:
                stream = torch.cuda.current_stream(value.device)
                stream.wait_event(envelope.ready)
                value.record_stream(stream)
            result[peer] = value
        _sizes(result, incoming_sizes)
        return result


    def exchange_device_into(self, outgoing, incoming_sizes, buffers):
        _buffers(buffers, incoming_sizes, self.device)
        incoming = self.exchange_device(outgoing, incoming_sizes)
        _sizes(incoming, incoming_sizes)
        for value in incoming.values():
            _validate_tensor(value, self.device)
        for peer, value in incoming.items():
            buffers[peer].copy_(value)
        return buffers


class DeviceByteBridge:
    """Stage device messages through a byte transport; TLS policy stays there."""
    staging = 'host'

    def __init__(self, underlying, device='cuda'):
        self.underlying = underlying
        self.device = _device(device)

    @property
    def bytes_sent(self):
        return self.underlying.bytes_sent

    @property
    def messages_sent(self):
        return self.underlying.messages_sent

    @property
    def authenticated(self):
        return getattr(self.underlying, 'authenticated', False)

    def exchange(self, outgoing):
        return self.underlying.exchange(outgoing)

    def exchange_device(self, outgoing, incoming_sizes):
        for value in outgoing.values():
            _validate_tensor(value, self.device)
        incoming = self.underlying.exchange({peer: _bytes(value) for peer, value in outgoing.items()})
        # SocketTransport omits empty padding frames. Public zero-length
        # messages have the same representation and can be restored by shape.
        for peer, size in incoming_sizes.items():
            if size == 0:
                incoming.setdefault(peer, b'')
        _sizes(incoming, incoming_sizes)
        return {peer: _tensor(value, self.device) for peer, value in incoming.items()}

    def exchange_device_into(self, outgoing, incoming_sizes, buffers):
        _buffers(buffers, incoming_sizes, self.device)
        for value in outgoing.values():
            _validate_tensor(value, self.device)
        incoming = self.underlying.exchange({p: _bytes(v) for p, v in outgoing.items()})
        for peer, size in incoming_sizes.items():
            if size == 0:
                incoming.setdefault(peer, b'')
        return _copy_bytes_into(incoming, incoming_sizes, buffers, self.device)

    def close(self):
        return self.underlying.close()


@dataclass(frozen=True)
class DeviceFrame:
    kind: str
    outgoing: dict
    incoming: dict


class DeviceRecorder:
    """Capture a deterministic fixture on the host, outside benchmark timing."""
    def __init__(self, underlying):
        self.underlying = underlying
        self.transcript = []
        self.bytes_sent = self.messages_sent = 0
        self.staging = underlying.staging

    def _record(self, kind, outgoing, incoming):
        self.transcript.append(DeviceFrame(kind, outgoing, incoming))
        self.bytes_sent += sum(map(len, outgoing.values()))
        self.messages_sent += len(outgoing)

    def exchange(self, outgoing):
        incoming = self.underlying.exchange(outgoing)
        self._record('bytes', {p: bytes(v) for p, v in outgoing.items()},
                     {p: bytes(v) for p, v in incoming.items()})
        return incoming

    def exchange_device(self, outgoing, incoming_sizes):
        incoming = self.underlying.exchange_device(outgoing, incoming_sizes)
        self._record('device', {p: _bytes(v) for p, v in outgoing.items()},
                     {p: _bytes(v) for p, v in incoming.items()})
        return incoming


class DeviceReplay(ReplayTransport):
    """Replay fixed messages; fresh live requests must never reuse this state.

    ``staging='device'`` models incoming buffers already delivered to the GPU.
    ``'host'`` retains outgoing and incoming host/device copies. Neither mode
    includes network transfer time. PRG and checks execute in DeviceParty.
    ``incoming_buffers`` binds graph receive buffers in device-round order.
    Device fixtures preload them during construction, outside replay timing;
    host fixtures copy into them during each exchange. This preloading is only
    for recorded benchmark messages and never supplies masks for live requests.
    """
    def __init__(self, transcript, device='cuda', check_outgoing=True, staging='device', incoming_buffers=None):
        super().__init__([], check_outgoing)
        if staging not in ('device', 'host'):
            raise ValueError('staging must be device or host')
        self.transcript = list(transcript)
        self.device, self.staging = _device(device), staging
        supplied = None if incoming_buffers is None else list(incoming_buffers)
        if supplied is not None and len(supplied) != sum(f.kind == 'device' for f in self.transcript):
            raise ValueError('replay incoming buffer count differs from device rounds')
        self._bound_buffers = [None]*len(self.transcript)
        self._incoming_device = [None]*len(self.transcript)
        index = 0
        for round_index, frame in enumerate(self.transcript):
            if frame.kind != 'device':
                continue
            buffers = None if supplied is None else supplied[index]
            index += 1
            if buffers is not None:
                sizes = {p:len(v) for p,v in frame.incoming.items()}
                _buffers(buffers, sizes, self.device)
                self._bound_buffers[round_index] = buffers
                if staging == 'device':
                    _copy_bytes_into(frame.incoming, sizes, buffers, self.device)
                    self._incoming_device[round_index] = buffers
            elif staging == 'device':
                self._incoming_device[round_index] = {p:_tensor(v,self.device) for p,v in frame.incoming.items()}

    def _next(self, kind):
        if self.round >= len(self.transcript):
            raise ValueError('device replay transcript exhausted')
        frame = self.transcript[self.round]
        if frame.kind != kind:
            raise ValueError(f'replay frame {self.round} is {frame.kind}, expected {kind}')
        return frame

    def _advance(self, sizes):
        self.round += 1
        self.bytes_sent += sum(sizes)
        self.messages_sent += len(sizes)
        if self._phase_start is not None:
            self._phase_start = self._clock()

    def exchange(self, outgoing):
        frame = self._next('bytes')
        self._mark_phase()
        if self.check_outgoing and outgoing != frame.outgoing:
            raise ValueError(f'device replay outgoing mismatch at round {self.round}')
        self._advance([len(v) for v in outgoing.values()])
        return dict(frame.incoming)

    def exchange_device(self, outgoing, incoming_sizes):
        frame = self._next('device')
        for value in outgoing.values():
            _validate_tensor(value, self.device)
        host_outgoing = None
        if self.staging == 'host':
            host_outgoing = {p: _bytes(v) for p, v in outgoing.items()}
            result = {p: _tensor(v, self.device) for p, v in frame.incoming.items()}
        else:
            result = dict(self._incoming_device[self.round])
        self._mark_phase()
        if self.check_outgoing:
            if host_outgoing is None:
                host_outgoing = {p: _bytes(v) for p, v in outgoing.items()}
            if host_outgoing != frame.outgoing:
                raise ValueError(f'device replay outgoing mismatch at round {self.round}')
        _sizes(result, incoming_sizes)
        self._advance([value.numel() for value in outgoing.values()])
        return result


    def exchange_device_into(self, outgoing, incoming_sizes, buffers):
        frame = self._next('device')
        _buffers(buffers, incoming_sizes, self.device)
        _sizes(frame.incoming, incoming_sizes)
        for value in outgoing.values():
            _validate_tensor(value, self.device)
        bound = self._bound_buffers[self.round]
        if bound is not None and (set(bound) != set(buffers) or any(bound[p] is not buffers[p] for p in bound)):
            raise ValueError('replay receive buffers differ from the prepared binding')
        host_outgoing = None
        if self.staging == 'host':
            host_outgoing = {p:_bytes(v) for p,v in outgoing.items()}
            _copy_bytes_into(frame.incoming,incoming_sizes,buffers,self.device)
        elif bound is None:
            for peer,value in self._incoming_device[self.round].items():
                buffers[peer].copy_(value)
        self._mark_phase()
        if self.check_outgoing:
            if host_outgoing is None:
                host_outgoing = {p:_bytes(v) for p,v in outgoing.items()}
            if host_outgoing != frame.outgoing:
                raise ValueError(f'device replay outgoing mismatch at round {self.round}')
        self._advance([v.numel() for v in outgoing.values()])
        return buffers
