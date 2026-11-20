"""Live CUDA IPC messages between trusted processes on one Linux GPU."""
import ctypes
from functools import lru_cache
import struct

import torch

from .device import _buffers, _device, _validate_tensor


class _IpcHandle(ctypes.Structure):
    _fields_ = [('data', ctypes.c_ubyte * 64)]


@lru_cache(maxsize=1)
def _cudart():
    major = torch.version.cuda.split('.')[0]
    library = ctypes.CDLL(f'libcudart.so.{major}')
    ptr = ctypes.c_void_p
    signatures = {
        'cudaMalloc': (ctypes.POINTER(ptr), ctypes.c_size_t),
        'cudaFree': (ptr,),
        'cudaMemsetAsync': (ptr, ctypes.c_int, ctypes.c_size_t, ptr),
        'cudaMemcpyAsync': (ptr, ptr, ctypes.c_size_t, ctypes.c_int, ptr),
        'cudaIpcGetMemHandle': (ctypes.POINTER(_IpcHandle), ptr),
        'cudaIpcOpenMemHandle': (ctypes.POINTER(ptr), _IpcHandle, ctypes.c_uint),
        'cudaIpcCloseMemHandle': (ptr,),
        'cudaEventCreateWithFlags': (ctypes.POINTER(ptr), ctypes.c_uint),
        'cudaEventDestroy': (ptr,),
        'cudaEventRecord': (ptr, ptr),
        'cudaEventSynchronize': (ptr,),
        'cudaStreamWaitEvent': (ptr, ptr, ctypes.c_uint),
        'cudaIpcGetEventHandle': (ctypes.POINTER(_IpcHandle), ptr),
        'cudaIpcOpenEventHandle': (ctypes.POINTER(ptr), _IpcHandle),
    }
    for name, args in signatures.items():
        func = getattr(library, name)
        func.argtypes = args
        func.restype = ctypes.c_int
    library.cudaGetErrorString.argtypes = (ctypes.c_int,)
    library.cudaGetErrorString.restype = ctypes.c_char_p
    return library


def _call(name, *args):
    runtime = _cudart()
    code = getattr(runtime, name)(*args)
    if code:
        detail = runtime.cudaGetErrorString(code).decode(errors='replace')
        raise RuntimeError(f'{name} failed ({code}): {detail}')


# CUDA IPC exports the whole allocation. Dedicated buffers rounded to 2 MiB
# keep the allocation independent of PyTorch's slabs holding private shares.
_GRANULARITY = 2 << 20
_MEMCPY_DEVICE_TO_DEVICE = 3
_IPC_LAZY_ENABLE_PEER_ACCESS = 1
_EVENT_IPC_FLAGS = 0x02 | 0x04  # cudaEventDisableTiming | cudaEventInterprocess
_FRAME = struct.Struct('!16sQ64s64s')


class _CudaArrayView:
    """Zero-copy __cuda_array_interface__ view over a dedicated cudaMalloc buffer."""
    def __init__(self, ptr_val: int, size: int):
        self.__cuda_array_interface__ = {
            'shape': (size,),
            'typestr': '|u1',
            'data': (ptr_val, False),
            'version': 3,
        }


class CudaIpcTransport:
    """Send live device messages across local processes using receiver-owned copies.

    Each directed peer has two alternating export slots paired with interprocess
    CUDA events. Before sending a round's control descriptors, every rank
    synchronizes its previous round's ``_read_done`` event so all reads from the
    preceding parity slot are acknowledged before it can be overwritten, while
    allowing the current round's GPU writes to overlap with TLS control exchange
    via IPC event waits. A successful close unmaps every import before its owner
    frees any exports.

    Use this transport only with mutually trusted local processes. IPC grants
    write access to a peer's dedicated wire buffers. Use the underlying TLS
    mesh for seed setup before constructing this transport. Logical message
    counters exclude the IPC descriptors.
    """
    staging = 'cuda_ipc'

    def __init__(self, underlying, device='cuda'):
        self.underlying = underlying
        if not getattr(underlying, 'authenticated', False):
            raise ValueError('CUDA IPC descriptors require an authenticated control mesh')
        self.party = underlying.party
        self.device = _device(device)
        if self.device.type != 'cuda':
            raise ValueError('CUDA IPC transport requires CUDA')
        self._uuid = bytes(torch.cuda.get_device_properties(self.device).uuid.bytes)
        self.bytes_sent = self.messages_sent = self.rounds = self.device_rounds = 0
        self.device_payload_bytes_sent = self.host_payload_bytes_sent = self.control_bytes_sent = 0
        self._slots = {}
        self._exports = []
        self._export_events = []
        self._direct_exports = {}
        self._imports = {}
        self._import_events = {}
        self._read_done = None
        self._closed = False

    @property
    def authenticated(self):
        return getattr(self.underlying, 'authenticated', False)

    def _require_open(self):
        if self._closed:
            raise ConnectionError('CUDA IPC transport is closed')

    def exchange(self, outgoing):
        raise RuntimeError('use device messages for the CUDA IPC data channel')

    def _new_ipc_event(self):
        ev = ctypes.c_void_p()
        _call('cudaEventCreateWithFlags', ctypes.byref(ev), _EVENT_IPC_FLAGS)
        self._export_events.append(ev)
        ev_handle = _IpcHandle()
        _call('cudaIpcGetEventHandle', ctypes.byref(ev_handle), ev)
        return ev, bytes(ev_handle)

    def alloc_export_tensor(self, size: int) -> torch.Tensor:
        """Allocate a dedicated IPC-exportable uint8 CUDA wire buffer (zero sender copy)."""
        self._require_open()
        if size <= 0:
            return torch.empty(0, dtype=torch.uint8, device=self.device)
        capacity = ((size + _GRANULARITY - 1) // _GRANULARITY) * _GRANULARITY
        with torch.cuda.device(self.device):
            stream = torch.cuda.current_stream(self.device)
            ptr = ctypes.c_void_p()
            _call('cudaMalloc', ctypes.byref(ptr), capacity)
            self._exports.append(ptr)
            _call('cudaMemsetAsync', ptr, 0, capacity, stream.cuda_stream)
            mem_handle = _IpcHandle()
            _call('cudaIpcGetMemHandle', ctypes.byref(mem_handle), ptr)
            ev, ev_handle = self._new_ipc_event()
            tensor = torch.as_tensor(_CudaArrayView(ptr.value, size), device=self.device)
            # Track [capacity, mem_handle, ev, ev_handle, last_round, tensor_ref].
            self._direct_exports[ptr.value] = [capacity, bytes(mem_handle), ev, ev_handle, -2, tensor]
            return tensor

    def _export_slot(self, slot_key, size, stream):
        key = (slot_key, self.device_rounds % 2)
        slot = self._slots.get(key)
        if slot is None or size > slot[0]:
            capacity = ((size + _GRANULARITY - 1) // _GRANULARITY) * _GRANULARITY
            if slot is not None:
                # Retired exports remain mapped until collective close. Doubling
                # bounds their aggregate capacity by the live high-water mark.
                capacity = max(capacity, 2 * slot[0])
            ptr = ctypes.c_void_p()
            _call('cudaMalloc', ctypes.byref(ptr), capacity)
            self._exports.append(ptr)
            _call('cudaMemsetAsync', ptr, 0, capacity, stream.cuda_stream)
            handle = _IpcHandle()
            _call('cudaIpcGetMemHandle', ctypes.byref(handle), ptr)
            ev, ev_handle = self._new_ipc_event()
            slot = (capacity, ptr, bytes(handle), ev, ev_handle)
            self._slots[key] = slot
        return slot

    def exchange_device(self, outgoing, incoming_sizes):
        buffers = {p: torch.empty(n, dtype=torch.uint8, device=self.device)
                   for p, n in incoming_sizes.items()}
        return self.exchange_device_into(outgoing, incoming_sizes, buffers)

    def exchange_device_into(self, outgoing, incoming_sizes, buffers):
        self._require_open()
        _buffers(buffers, incoming_sizes, self.device)
        if any(p == self.party or p not in range(4) for p in (*outgoing, *incoming_sizes)):
            raise ValueError('invalid CUDA IPC peer')
        for value in outgoing.values():
            _validate_tensor(value, self.device)

        with torch.cuda.device(self.device):
            stream = torch.cuda.current_stream(self.device)
            # Acknowledge all reads from the preceding round before sending this
            # round's control frames so every peer's alternating slot is free on
            # the next round without blocking on this round's GPU compute.
            if self._read_done is not None:
                self._read_done.synchronize()
            control = {}
            staged = {}
            for p, value in outgoing.items():
                size = value.numel()
                if size:
                    ptr_val = value.data_ptr()
                    buf_key = (ptr_val, size)
                    frame = staged.get(buf_key)
                    if frame is None:
                        direct = self._direct_exports.get(ptr_val)
                        if direct is not None and size <= direct[0] and direct[4] < self.device_rounds - 1:
                            _, handle, ev, ev_handle, _, _ = direct
                            direct[4] = self.device_rounds
                            _call('cudaEventRecord', ev, stream.cuda_stream)
                        else:
                            _, target, handle, ev, ev_handle = self._export_slot(len(staged), size, stream)
                            _call('cudaMemcpyAsync', target, ptr_val, size,
                                  _MEMCPY_DEVICE_TO_DEVICE, stream.cuda_stream)
                            _call('cudaEventRecord', ev, stream.cuda_stream)
                        frame = _FRAME.pack(self._uuid, size, handle, ev_handle)
                        staged[buf_key] = frame
                    control[p] = frame
            frames = self.underlying.exchange(control)
            if set(frames) != {p for p, n in incoming_sizes.items() if n}:
                raise ValueError('CUDA IPC incoming peers differ from the public schedule')
            for p, frame in frames.items():
                if len(frame) != _FRAME.size:
                    raise ValueError('invalid CUDA IPC descriptor length')
                uuid, size, handle, ev_handle = _FRAME.unpack(frame)
                if uuid != self._uuid:
                    raise ValueError('CUDA IPC peers must share the same physical GPU')
                if size != incoming_sizes[p]:
                    raise ValueError('CUDA IPC message length differs from the public schedule')
                key = (p, handle)
                if key not in self._imports:
                    ptr = ctypes.c_void_p()
                    _call('cudaIpcOpenMemHandle', ctypes.byref(ptr), _IpcHandle.from_buffer_copy(handle),
                          _IPC_LAZY_ENABLE_PEER_ACCESS)
                    self._imports[key] = ptr
                ev_key = (p, ev_handle)
                if ev_key not in self._import_events:
                    ev = ctypes.c_void_p()
                    _call('cudaIpcOpenEventHandle', ctypes.byref(ev), _IpcHandle.from_buffer_copy(ev_handle))
                    self._import_events[ev_key] = ev
                _call('cudaStreamWaitEvent', stream.cuda_stream, self._import_events[ev_key], 0)
                _call('cudaMemcpyAsync', buffers[p].data_ptr(), self._imports[key],
                      size, _MEMCPY_DEVICE_TO_DEVICE, stream.cuda_stream)
                buffers[p].record_stream(stream)
            if self._read_done is None:
                self._read_done = torch.cuda.Event()
            self._read_done.record(stream)
        size = sum(value.numel() for value in outgoing.values())
        self.device_payload_bytes_sent += size
        self.bytes_sent += size
        self.control_bytes_sent += sum(map(len, control.values()))
        self.messages_sent += len(outgoing)
        self.rounds += 1
        self.device_rounds += 1
        return buffers

    def _unmap(self):
        for key in list(self._import_events):
            _call('cudaEventDestroy', self._import_events[key])
            del self._import_events[key]
        for key in list(self._imports):
            _call('cudaIpcCloseMemHandle', self._imports[key])
            del self._imports[key]

    def close(self):
        if self._closed:
            return
        try:
            with torch.cuda.device(self.device):
                torch.cuda.synchronize(self.device)
                self._unmap()
                self.underlying.exchange({})
                self._direct_exports.clear()
                for ev in self._export_events:
                    _call('cudaEventDestroy', ev)
                self._export_events.clear()
                for ptr in self._exports:
                    _call('cudaFree', ptr)
                self._exports.clear()
                self._slots.clear()
        finally:
            self._closed = True
            self.underlying.close()

    def abort(self, error=None):
        """Break the rendezvous. Failed worker processes release exports on exit."""
        if self._closed:
            return
        self._closed = True
        self.underlying.close()
        with torch.cuda.device(self.device):
            torch.cuda.synchronize(self.device)
            self._unmap()
