"""Fixed-shape inference graphs with checked communication boundaries."""
import math
from dataclasses import is_dataclass
import threading

import torch
import triton

from .device_inference import DeviceInferenceParty
from ..protocol.device_runtime import mask_domain
from ..gpu.crypto.gpu_prg_graph import _rows
from ..protocol.runtime import SCHEDULE, Share


def _map(value, fn):
    if isinstance(value, torch.Tensor):
        return fn(value)
    if isinstance(value, Share):
        return Share(tuple(fn(t) for t in value.parts), value.binary)
    if isinstance(value, tuple):
        return tuple(_map(v, fn) for v in value)
    if isinstance(value, list):
        return [_map(v, fn) for v in value]
    if isinstance(value, dict):
        return {k: _map(v, fn) for k, v in value.items()}
    if is_dataclass(value):
        return value
    raise TypeError('graph inputs and outputs must contain tensors, shares, or prepared weights')


class _OfflineTransport:
    def exchange(self, *args, **kwargs):
        raise RuntimeError('inference preparation cannot use a live transport')

    exchange_device = exchange


class _Builder:
    def __init__(self, rt, stream, check_at_send=False):
        self.rt, self.stream = rt, stream
        self.check_at_send = check_at_send
        self.pending = []
        self.capture = False
        self.actions, self.draws = [], []
        self.draw_index = 0
        self.graph = None
        self.pool = torch.cuda.graph_pool_handle()
        self.check_count = 0
        self.check_index = 0
        self.host_checks = None

    def stage_check(self, valid):
        # Capture the transfer as well as the comparison; each boundary reads
        # its retained host slice only after synchronizing this plan's stream.
        count = valid.numel()
        host = self.host_checks[self.check_index:self.check_index+count].view(valid.shape)
        self.check_index += count
        host.copy_(valid, non_blocking=True)
        return host

    def begin(self):
        self.graph = torch.cuda.CUDAGraph()
        self.graph.capture_begin(pool=self.pool)

    def boundary(self, action):
        if self.capture:
            self.graph.capture_end()
            self.actions.extend((('graph', self.graph), action))

    def end_segment(self):
        checks = None
        if len(self.pending) == 1:
            valid, message = self.pending[0]
            checks = ('verify', (self.stage_check(valid), message))
        elif self.pending:
            # Pack on the captured stream so one host transfer checks the entire
            # boundary, including the first failure's original diagnostic.
            checks = ('verify_batch',
                      (self.stage_check(torch.stack([valid for valid, _ in self.pending])),
                       tuple(message for _, message in self.pending)))
        self.pending.clear()
        self.graph.capture_end()
        self.actions.append(('graph', self.graph))
        if checks is not None:
            self.actions.append(checks)

    def exchange(self, outgoing, sizes):
        if self.capture:
            self.end_segment()
        incoming = {p: torch.zeros(n, dtype=torch.uint8, device=self.rt.device)
                    for p, n in sizes.items()}
        if self.capture:
            self.actions.append(('exchange', (outgoing, dict(sizes), incoming)))
            self.begin()
        return incoming

    def verify(self, actual, wanted, message):
        valid = (actual == wanted).all()
        if not self.capture:
            self.check_count += 1
        if self.check_at_send:
            if self.capture:
                self.pending.append((valid, message))
        else:
            if self.capture:
                self.boundary(('verify', (self.stage_check(valid), message)))
                self.begin()

    def masks(self, keys, excluded, shape, bits, *, packed=False):
        if not self.capture:
            nonces = torch.zeros(len(keys), dtype=torch.uint64, device=self.rt.device)
            host = torch.zeros(len(keys), dtype=torch.uint64, pin_memory=True)
            self.draws.append((nonces, host, excluded))
        else:
            nonces, _, saved = self.draws[self.draw_index]
            if saved != excluded:
                raise ValueError('inference draw schedule changed during capture')
        self.draw_index += 1
        dtype = torch.uint32 if packed and bits == 32 else torch.int64
        output = torch.empty((len(keys), math.prod(shape)), dtype=dtype, device=self.rt.device)
        count = math.prod(shape)
        if count:
            active = next(k for k in keys if k is not None)
            padded = tuple(k if k is not None else active for k in keys)
            padded += (active,)*(6-len(keys))
            valid = tuple(k is not None for k in keys)+(False,)*(6-len(keys))
            blocks = triton.cdiv(count, 16 if bits == 32 else 8)
            if blocks >= 2**32:
                raise ValueError('a draw must use fewer than 2**32 ChaCha blocks')
            block_size = 256 if packed and bits == 32 and count >= 1024*1024 else 128
            _rows[(triton.cdiv(blocks,block_size),len(keys))](*padded, nonces, output,
                    count, bits, valid, block_size, num_warps=4)
        return output


class InferencePlan:
    """Replay one public inference schedule; each run owns its returned tensors.

    Prepare plans before starting concurrent role threads. The callable must
    use fixed public control flow and the device protocol for every exchange.
    Captured weights must remain alive and unchanged for the plan's lifetime.
    With ``check_at_send=True``, local GPU computation can proceed after a tag
    comparison, but its host check finishes before the next exchange or return.
    This mode requires a callable with no external side effects beyond the
    intercepted protocol exchanges.
    """
    def __init__(self, rt, function, inputs, check_at_send=False):
        self.rt, self.lock, self.failed = rt, threading.Lock(), False
        self.inputs = _map(inputs, lambda t: t.clone())
        self.stream = torch.cuda.Stream(device=rt.device)
        self.ready = torch.cuda.Event()
        self.stream.wait_stream(torch.cuda.current_stream(rt.device))
        builder = _Builder(rt, self.stream, check_at_send)
        counters = (rt.joint_inputs, rt.multiplications)
        transport = rt.transport
        rt.transport = _OfflineTransport()
        rt._phase_builder = builder
        try:
            with torch.cuda.stream(self.stream):
                function(*self.inputs)
            self.stream.synchronize()
            # Warmup can leave large GEMM scratch blocks in the default pool.
            # Release those before graph-private pools retain their allocations.
            torch.cuda.empty_cache()
            size = sum(len(rows) for _, _, rows in builder.draws)
            with torch.cuda.stream(self.stream):
                self.nonces = torch.zeros(size, dtype=torch.uint64, device=rt.device)
            self.host_nonces = torch.zeros(size, dtype=torch.uint64, pin_memory=True)
            self.host_nonce_values = self.host_nonces.numpy()
            self.nonce_rows = tuple(e for _, _, rows in builder.draws for e in rows)
            self.nonce_excluded = tuple(e for e in self.nonce_rows if e is not None)
            offset = 0
            for i, (_, _, rows) in enumerate(builder.draws):
                end = offset+len(rows)
                builder.draws[i] = (self.nonces[offset:end], self.host_nonces[offset:end], rows)
                offset = end
            builder.capture = True
            builder.draw_index = 0
            builder.host_checks = torch.empty(builder.check_count, dtype=torch.bool, pin_memory=True)
            rt.joint_inputs, rt.multiplications = counters
            with torch.cuda.stream(self.stream):
                builder.begin()
                try:
                    self.output = function(*self.inputs)
                finally:
                    builder.end_segment()
            self.delta = (rt.joint_inputs-counters[0], rt.multiplications-counters[1])
            self.actions, self.draws = builder.actions, builder.draws
            self.receive_buffers = tuple(value[2] for kind,value in self.actions if kind == 'exchange')
            self.ready.record(self.stream)
        finally:
            rt._phase_builder = None
            rt.transport = transport
            rt.joint_inputs, rt.multiplications = counters

    def run(self, *inputs):
        if not self.lock.acquire(blocking=False):
            raise RuntimeError('an inference plan cannot run concurrently')
        try:
            if self.failed:
                raise RuntimeError('inference plan is poisoned after a failed run')
            pairs = []
            def collect(source, destination):
                if isinstance(destination, torch.Tensor):
                    if not isinstance(source, torch.Tensor) or (source.shape, source.dtype, source.device) != (destination.shape, destination.dtype, destination.device):
                        raise ValueError('inference input shape, dtype, or device changed')
                    pairs.append((destination, source))
                elif isinstance(destination, Share):
                    if not isinstance(source, Share) or source.binary != destination.binary:
                        raise ValueError('inference share kind changed')
                    collect(source.parts, destination.parts)
                elif is_dataclass(destination):
                    if source is not destination:
                        raise ValueError('captured prepared weights must retain their identity')
                else:
                    if type(source) is not type(destination) or len(source) != len(destination):
                        raise ValueError('inference input structure changed')
                    for key in destination if isinstance(destination, dict) else range(len(destination)):
                        collect(source[key], destination[key])
            collect(inputs, self.inputs)
            current = torch.cuda.current_stream(self.rt.device)
            self.stream.wait_stream(current)
            self.stream.wait_event(self.ready)
            with torch.cuda.stream(self.stream):
                # The previous upload must finish before its pinned source changes.
                self.ready.synchronize()
                reservations = iter(self.rt._reserve_nonces(self.nonce_excluded))
                try:
                    values = [0 if e is None else next(reservations)
                              for e in self.nonce_rows]
                    self.host_nonce_values[:] = values
                    self.nonces.copy_(self.host_nonces, non_blocking=True)
                    for destination, source in pairs:
                        destination.copy_(source)
                    for kind, value in self.actions:
                        if kind == 'graph':
                            value.replay()
                        elif kind == 'exchange':
                            outgoing, sizes, buffers = value
                            exchange_into = getattr(self.rt.transport, 'exchange_device_into', None)
                            incoming = (self.rt._exchange_device(outgoing, sizes) if exchange_into is None
                                        else exchange_into(outgoing, sizes, buffers))
                            if set(incoming) != set(buffers):
                                raise ValueError('invalid inference incoming peers')
                            for p, dst in buffers.items():
                                src = incoming[p]
                                if not isinstance(src, torch.Tensor) or src.dtype != dst.dtype or src.device != dst.device or src.shape != dst.shape or not src.is_contiguous():
                                    raise ValueError('invalid inference incoming frame')
                                if src is not dst:
                                    dst.copy_(src)
                        elif kind == 'verify_batch':
                            valid, messages = value
                            self.stream.synchronize()
                            for passed, message in zip(valid.tolist(), messages):
                                if not passed:
                                    raise ValueError(message)
                        else:
                            valid, message = value
                            self.stream.synchronize()
                            if not bool(valid.item()):
                                raise ValueError(message)
                    result = _map(self.output, lambda t: t.clone())
                    self.rt.joint_inputs += self.delta[0]
                    self.rt.multiplications += self.delta[1]
                except BaseException as error:
                    self.failed = True
                    abort = getattr(self.rt.transport, 'abort', None)
                    if abort is not None:
                        abort(error)
                    raise
                finally:
                    self.ready.record(self.stream)
            current.wait_event(self.ready)
            return result
        finally:
            self.lock.release()


class PhaseGraphParty(DeviceInferenceParty):
    """Opt in to whole-inference graph segments with live nonce reservations."""
    _phase_builder = None

    def prepare_inference(self, function, *inputs, check_at_send=False):
        lock = self.__dict__.setdefault('_phase_prepare_lock', threading.Lock())
        if not lock.acquire(blocking=False):
            raise RuntimeError('inference preparation cannot run recursively or concurrently')
        try:
            return InferencePlan(self, function, inputs, check_at_send)
        finally:
            lock.release()

    def _exchange_device(self, outgoing, sizes):
        if self._phase_builder is not None:
            return self._phase_builder.exchange(outgoing, sizes)
        return super()._exchange_device(outgoing, sizes)

    def _verify_tags(self, actual, wanted, message='device Rep4 joint input BLAKE3 hash mismatch'):
        if self._phase_builder is not None:
            return self._phase_builder.verify(actual, wanted, message)
        return super()._verify_tags(actual, wanted, message)

    def _random(self, excluded, shape, device, binary=False, *, packed=False):
        if self._phase_builder is None:
            return super()._random(excluded, shape, device, binary, packed=packed)
        key = self._device_keys[excluded].key(mask_domain(self.bits, self.binary_bits, binary, excluded))
        rows = self._phase_builder.masks((key,), (excluded,), shape, self._wire_bits(binary), packed=packed)
        return rows[0].reshape(shape)

    def _mask_batch(self, shape, binary, device, *, schedule=SCHEDULE):
        if self._phase_builder is None:
            return super()._mask_batch(shape, binary, device, schedule=schedule)
        excluded = tuple(None if self.party == r else r for _, _, r, _ in schedule)
        keys = tuple(None if e is None else self._device_keys[e].key(
            mask_domain(self.bits, self.binary_bits, binary, e)) for e in excluded)
        return self._phase_builder.masks(keys, excluded, shape, self._wire_bits(binary), packed=True).reshape(-1)
