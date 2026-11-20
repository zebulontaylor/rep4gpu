"""Commit integer layer states and replay selected boundaries in the prover.

Callbacks operate on plaintext prover tensors. A verified opening authenticates
bytes against an agreed root; it does not prove the layer's computation.
"""
from __future__ import annotations

import base64
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
import copy
import threading
from dataclasses import dataclass
import hashlib
import hmac
import json
import math
import struct

import numpy as np
import torch

from .merkle import MerkleTree, PackedMerkleLayout, hash_leaf, verify_merkle
from .state_digest import prepare_state_digest, prepare_state_digests, state_digest, finish_state_digest

State = Mapping[str, torch.Tensor]
_DTYPES = {torch.int32: '<i4', torch.int64: '<i8'}
_MAGIC = b'SIVSTATE\x01'


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode('ascii')


def _validate_state(state):
    if not isinstance(state, Mapping) or not state:
        raise ValueError('state must be a nonempty mapping of names to integer tensors')
    for name, value in state.items():
        if not isinstance(name, str) or not name:
            raise ValueError('state names must be nonempty strings')
        if (not isinstance(value, torch.Tensor) or value.dtype not in _DTYPES
                or value.layout != torch.strided or value.device.type not in ('cpu', 'cuda')):
            raise ValueError('state values must be dense CPU or CUDA int32/int64 tensors')


def _clone(state):
    _validate_state(state)
    storages = [(tensor.device, tensor.untyped_storage().data_ptr()) for tensor in state.values()]
    if len(set(storages)) != len(storages):
        return copy.deepcopy(dict(state))
    return {name: tensor.detach().clone() for name, tensor in state.items()}


def _aliases(state):
    names = sorted(state)
    relationships = []
    for index, name in enumerate(names):
        left = state[name]
        for other in names[index+1:]:
            right = state[other]
            if (left.numel() and right.numel() and left.device == right.device
                    and left.untyped_storage().data_ptr() == right.untyped_storage().data_ptr()):
                relationships.append((name, other, left.stride(), right.stride(),
                                      left.storage_offset()*left.element_size()-right.storage_offset()*right.element_size()))
    return tuple(relationships)


def encode_state(state: State) -> bytes:
    """Encode sorted names, shapes, and little-endian integer words without pickle.

    CUDA inputs are copied to the host. Strides and device are not committed;
    the logical tensor values, shape, dtype, and name are committed.
    """
    _validate_state(state)
    entries, chunks = [], []
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        dtype = _DTYPES[value.dtype]
        entries.append([name, dtype, list(value.shape)])
        chunks.append(value.numpy().astype(dtype, copy=False).tobytes())
    header = _json(entries)
    return _MAGIC + struct.pack('>I', len(header)) + header + b''.join(chunks)


def decode_state(payload: bytes) -> dict[str, torch.Tensor]:
    """Decode a canonical state into independent CPU tensors, rejecting extra bytes."""
    try:
        if not isinstance(payload, bytes) or not payload.startswith(_MAGIC):
            raise ValueError('invalid state version')
        offset = len(_MAGIC)
        size, = struct.unpack_from('>I', payload, offset)
        offset += 4
        header = payload[offset:offset+size]
        entries = json.loads(header)
        offset += size
        if not isinstance(entries, list) or not entries or _json(entries) != header:
            raise ValueError('noncanonical state header')
        result, previous = {}, None
        for entry in entries:
            if not isinstance(entry, list) or len(entry) != 3:
                raise ValueError('invalid tensor entry')
            name, dtype, shape = entry
            if (not isinstance(name, str) or not name or (previous is not None and name <= previous)
                    or dtype not in ('<i4', '<i8') or not isinstance(shape, list)
                    or len(shape) > 64 or any(type(n) is not int or n < 0 for n in shape)):
                raise ValueError('invalid tensor schema')
            previous = name
            count = math.prod(shape)
            nbytes = count * np.dtype(dtype).itemsize
            if nbytes > len(payload)-offset:
                raise ValueError('truncated state tensor')
            array = np.frombuffer(payload, dtype=dtype, count=count, offset=offset).reshape(shape)
            result[name] = torch.from_numpy(array.astype(np.dtype(dtype).newbyteorder('='), copy=True))
            offset += nbytes
        if offset != len(payload):
            raise ValueError('unexpected state bytes')
        return result
    except (TypeError, OverflowError, RecursionError, struct.error, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError('malformed state encoding') from exc


@dataclass(frozen=True)
class Layer:
    """A named deterministic transition over the complete replay state.

    Include residuals, caches, and explicit RNG state in the mapping if a later
    layer needs them. Callbacks must use the same parameters during replay.
    ``preserves_inputs=True`` promises not to mutate the input mapping, its
    tensor storage, or previously returned states, including aliased views.
    Prepared execution can then hash retained boundaries together without
    copying each one. Leave it false for callbacks that perform in-place work.
    """
    name: str
    forward: Callable[[State], State]
    preserves_inputs: bool = False


@dataclass(frozen=True)
class Commitment:
    root: bytes
    layer_names: tuple[str, ...]
    model_id: str
    run_id: str

    def _context(self):
        if (not isinstance(self.root, bytes) or len(self.root) != 32
                or not isinstance(self.layer_names, tuple)
                or any(not isinstance(n, str) or not n for n in self.layer_names)
                or len(set(self.layer_names)) != len(self.layer_names)
                or not isinstance(self.model_id, str) or not self.model_id
                or not isinstance(self.run_id, str) or not self.run_id):
            raise ValueError('invalid commitment metadata')
        return hashlib.sha256(b'rep4gpu.layers.v1\x00' + _json(
            [self.layer_names, self.model_id, self.run_id])).digest()

    def to_dict(self):
        self._context()
        return dict(version=1, root=self.root.hex(), layer_names=list(self.layer_names),
                    model_id=self.model_id, run_id=self.run_id)

    @classmethod
    def from_dict(cls, value):
        try:
            if (set(value) != {'version', 'root', 'layer_names', 'model_id', 'run_id'}
                    or type(value['version']) is not int or value['version'] != 1
                    or not isinstance(value['layer_names'], list)):
                raise ValueError('invalid commitment envelope')
            result = cls(bytes.fromhex(value['root']), tuple(value['layer_names']),
                         value['model_id'], value['run_id'])
            result._context()
            return result
        except (KeyError, TypeError, AttributeError) as exc:
            raise ValueError('malformed commitment envelope') from exc


@dataclass(frozen=True)
class LayerOpening:
    boundary: int
    state: bytes
    siblings: tuple[bytes, ...]

    def to_dict(self):
        return dict(version=1, boundary=self.boundary,
                    state=base64.b64encode(self.state).decode('ascii'),
                    siblings=[s.hex() for s in self.siblings])

    @classmethod
    def from_dict(cls, value):
        try:
            if (set(value) != {'version', 'boundary', 'state', 'siblings'}
                    or type(value['version']) is not int or value['version'] != 1
                    or type(value['boundary']) is not int or value['boundary'] < 0
                    or not isinstance(value['siblings'], list)):
                raise ValueError('invalid opening envelope')
            siblings = tuple(bytes.fromhex(s) for s in value['siblings'])
            if any(len(s) != 32 for s in siblings):
                raise ValueError('siblings must be SHA-256 digests')
            return cls(value['boundary'], base64.b64decode(value['state'], validate=True), siblings)
        except (KeyError, TypeError, AttributeError) as exc:
            raise ValueError('malformed opening envelope') from exc


def verify_layer(commitment: Commitment, opening: LayerOpening, *, expected_boundary: int):
    """Authenticate the requested boundary before returning its plaintext CPU state.

    Supply the commitment accepted before choosing the challenge, not a root
    supplied alongside the response. Apply transport message-size limits before
    decoding untrusted JSON. This function does not verify a transition or hide
    the opened values from the recipient.
    """
    context = commitment._context()
    count = len(commitment.layer_names)+1
    if (type(expected_boundary) is not int or not 0 <= expected_boundary < count
            or type(opening.boundary) is not int or opening.boundary != expected_boundary
            or not isinstance(opening.state, bytes)):
        raise ValueError('layer opening does not match the requested commitment')
    state = decode_state(opening.state)
    if not verify_merkle(commitment.root, context+state_digest(state),
                         opening.boundary, count, opening.siblings):
        raise ValueError('layer opening does not match the requested commitment')
    return state


def _stage(state):
    _validate_state(state)
    if any(value.device.type == 'cuda' for value in state.values()):
        prepared = prepare_state_digest(state)
        events = []
        for device in {value.device for value in state.values() if value.device.type == 'cuda'}:
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream(device))
            events.append(event)
        return prepared, events
    return _clone(state), []


def _hash_staged(index, context, staged):
    state, events = staged
    for event in events:
        event.synchronize()
    if events:
        devices = {value.device for value in state.digests if isinstance(value, torch.Tensor)}
        # Separate transfer streams keep digest copies off the inference stream.
        from contextlib import ExitStack
        with ExitStack() as stack:
            for device in devices:
                stack.enter_context(torch.cuda.stream(torch.cuda.Stream(device=device)))
            digest = state.finish(wait=False)
    else:
        digest = state_digest(state)
    return hash_leaf(index, context+digest)


class CommitmentRuntime:
    """Pipeline state snapshots and host hashes while executing plaintext layers.

    ``checkpoint_interval`` trades retained tensor memory for replay work.
    ``max_pending`` bounds snapshots awaiting hashes. CUDA tensor hashes run
    on the caller's current stream before later in-place mutations. Only tensor
    digests cross to the host; CPU state hashing overlaps later layers.
    Reuse the runtime to amortize worker startup, then call ``close``.
    """
    def __init__(self, layers: Sequence[Layer], *, checkpoint_interval=8,
                 hash_workers=2, max_pending=4):
        self.layers = tuple(layers)
        if any(not isinstance(layer, Layer) or not callable(layer.forward)
               or type(layer.preserves_inputs) is not bool for layer in self.layers):
            raise ValueError('layers must contain named Layer callbacks')
        Commitment(bytes(32), tuple(layer.name for layer in self.layers), 'model', 'run')._context()
        for value in (checkpoint_interval, hash_workers, max_pending):
            if type(value) is not int or value < 1:
                raise ValueError('checkpoint interval, workers and queue size must be positive integers')
        self.checkpoint_interval = checkpoint_interval
        self.hash_workers = hash_workers
        self.max_pending = max_pending
        self._pool = ThreadPoolExecutor(max_workers=hash_workers)
        self._lock = threading.Lock()

    def close(self):
        """Drain pending work and release the reusable hashing workers."""
        with self._lock:
            self._pool.shutdown(wait=True)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def prepare(self, inputs: State):
        """Capture fixed-shape CUDA inference and hashing for repeated commits."""
        return PreparedCommitmentRuntime(self, inputs)

    @torch.no_grad()
    def commit(self, inputs: State, *, model_id: str, run_id: str):
        """Execute once, returning only after the complete Merkle root is ready."""
        metadata = Commitment(bytes(32), tuple(layer.name for layer in self.layers), model_id, run_id)
        context = metadata._context()
        state = _clone(inputs)
        checkpoints, hashes, pending = {}, [], deque()
        with self._lock:
            pool = self._pool
            for boundary in range(len(self.layers)+1):
                if boundary:
                    state = self.layers[boundary-1].forward(state)
                _validate_state(state)
                if boundary % self.checkpoint_interval == 0:
                    checkpoints[boundary] = _clone(state)
                if len(pending) >= self.max_pending:
                    hashes.append(pending.popleft().result())
                pending.append(pool.submit(_hash_staged, boundary, context, _stage(state)))
            hashes.extend(future.result() for future in pending)
        tree = MerkleTree.from_leaf_hashes(hashes)
        commitment = Commitment(tree.root, metadata.layer_names, model_id, run_id)
        return CommittedExecution(self.layers, checkpoints, tuple(hashes), tree, commitment)


def _pack_checkpoints(checkpoints, device):
    """Pack storage once; layouts retain tensor strides and overlapping views."""
    chunks, offsets, layouts, size = [], {}, {}, 0
    for boundary, state in checkpoints.items():
        layout = []
        for name, value in state.items():
            storage = value.untyped_storage()
            key = (value.device, storage.data_ptr())
            if key not in offsets:
                padding = -size % 8
                if padding:
                    chunks.append(torch.zeros(padding, dtype=torch.uint8, device=device))
                    size += padding
                offsets[key] = size
                raw = torch.empty(0, dtype=torch.uint8, device=device).set_(
                    storage, 0, (storage.nbytes(),), (1,))
                chunks.append(raw)
                size += storage.nbytes()
            offset = offsets[key] + value.storage_offset()*value.element_size()
            layout.append((name, value.dtype, tuple(value.shape), value.stride(), offset))
        layouts[boundary] = tuple(layout)
    return torch.cat(chunks), layouts


def _unpack_checkpoint(blob, layout):
    return {name: torch.empty(0, dtype=dtype, device=blob.device).set_(
        blob.untyped_storage(), offset // (4 if dtype == torch.int32 else 8), shape, stride)
        for name, dtype, shape, stride, offset in layout}


class CommittedExecution:
    """Prover-local checkpoints and digests for one completed execution."""
    def __init__(self, layers, checkpoints, hashes, tree, commitment, checkpoint_pack=None):
        self._layers, self._checkpoints = layers, checkpoints
        self._checkpoint_pack = checkpoint_pack
        self._hashes, self._tree = hashes, tree
        self.commitment = commitment

    @torch.no_grad()
    def recompute(self, boundary: int) -> dict[str, torch.Tensor]:
        """Replay from the nearest checkpoint and reject any differing boundary.

        Returned tensors are independent of stored checkpoints. Replay callbacks
        must be deterministic, including device kernels and any random choices.
        """
        if type(boundary) is not int or not 0 <= boundary <= len(self._layers):
            raise ValueError('boundary must identify the input or a layer output')
        if self._checkpoint_pack is None:
            start = max(index for index in self._checkpoints if index <= boundary)
            state = _clone(self._checkpoints[start])
        else:
            blob, layouts = self._checkpoint_pack
            start = max(index for index in layouts if index <= boundary)
            state = _clone(_unpack_checkpoint(blob, layouts[start]))
        context = self.commitment._context()
        for index in range(start, boundary+1):
            if index > start:
                state = self._layers[index-1].forward(state)
            digest = hash_leaf(index, context+state_digest(state))
            if not hmac.compare_digest(digest, self._hashes[index]):
                raise ValueError(f'recomputed boundary {index} differs from its commitment')
        return _clone(state)

    def open(self, boundary: int) -> LayerOpening:
        """Return the recomputed state and its logarithmic-size authentication path."""
        state = encode_state(self.recompute(boundary))
        return LayerOpening(boundary, state, self._tree.proof(boundary))


class PreparedCommitmentRuntime:
    """A fixed-shape CUDA graph that hashes each boundary on the device.

    Preparation warms callbacks and captures inference, hashes, and sparse
    checkpoints. Each commit copies new inputs, replays the graph, and transfers
    all boundary digests in one batch. Captured parameters and callback behavior
    must remain consistent with the model identifier supplied by the caller.
    """
    @torch.no_grad()
    def __init__(self, runtime: CommitmentRuntime, inputs: State):
        _validate_state(inputs)
        devices = {tensor.device for tensor in inputs.values()}
        if len(devices) != 1 or next(iter(devices)).type != 'cuda':
            raise ValueError('prepared execution requires tensors on one CUDA device')
        self._device = next(iter(devices))
        self._layers = runtime.layers
        self._layer_names = tuple(layer.name for layer in self._layers)
        self._context_prefix = hashlib.sha256(b'rep4gpu.layers.v1\x00' + b'[' + _json(self._layer_names) + b',')
        self._model_context = None
        self._model_id = None
        self._preserve_states = all(layer.preserves_inputs for layer in self._layers)
        self._lock = threading.Lock()
        self._input_aliases = _aliases(inputs)
        self._inputs = _clone(inputs)
        self._input_names = set(inputs)
        self._input_schema = tuple((name, value.dtype, value.shape) for name, value in inputs.items())
        self._single_input = self._input_schema[0] if len(self._input_schema) == 1 else None
        self._graph = torch.cuda.CUDAGraph()
        self._ready = torch.cuda.Event(blocking=False)
        self._tree_layout = PackedMerkleLayout(len(self._layers)+1)
        self._blank_tree = bytes(self._tree_layout.byte_count)
        self._checkpoints = {}
        # Initialize lazy kernels and allocators before capture, on a side stream.
        stream = torch.cuda.Stream(device=self._device)
        stream.wait_stream(torch.cuda.current_stream(self._device))
        with torch.cuda.device(self._device), torch.cuda.stream(stream):
            state = _clone(inputs)
            warm_states, batch_bytes = [], 0
            for boundary in range(len(self._layers)+1):
                if boundary:
                    state = self._layers[boundary-1].forward(state)
                prepare_state_digest(state)
                batch_bytes += sum(t.numel()*t.element_size() for t in state.values())
                if batch_bytes <= 1 << 20:
                    warm_states.append(_clone(state))
                else:
                    warm_states.clear()
            self._batch_hashes = batch_bytes <= 1 << 20
            if self._batch_hashes:
                prepare_state_digests(warm_states)
        del warm_states
        stream.synchronize()
        prepared, snapshots = [], []
        hash_stream = torch.cuda.Stream(device=self._device)
        with torch.cuda.device(self._device), torch.cuda.graph(self._graph, stream=stream):
            state = self._inputs
            for boundary in range(len(self._layers)+1):
                if boundary:
                    state = self._layers[boundary-1].forward(state)
                _validate_state(state)
                if any(value.device != self._device for value in state.values()):
                    raise ValueError('prepared layers must keep state on the input device')
                if boundary % runtime.checkpoint_interval == 0:
                    self._checkpoints[boundary] = dict(state) if self._preserve_states else _clone(state)
                snapshot = (dict(state) if self._batch_hashes and self._preserve_states else
                            self._checkpoints[boundary] if boundary in self._checkpoints else _clone(state))
                if self._batch_hashes:
                    snapshots.append(snapshot)
                else:
                    hash_stream.wait_stream(stream)
                    with torch.cuda.stream(hash_stream):
                        prepared.append(prepare_state_digest(snapshot))
                        for value in snapshot.values():
                            value.record_stream(hash_stream)
                    if (boundary+1) % runtime.max_pending == 0:
                        stream.wait_stream(hash_stream)
            if self._batch_hashes:
                prepared = prepare_state_digests(snapshots)
            else:
                stream.wait_stream(hash_stream)
            self._checkpoint_buffer, self._checkpoint_layouts = _pack_checkpoints(self._checkpoints, self._device)
            self._packed = torch.cat([digest for state in prepared for digest in state.digests])
            self._packed_host = torch.empty(self._packed.shape, dtype=torch.uint8, pin_memory=True)
            self._packed_host.copy_(self._packed, non_blocking=True)
        self._packed_view = self._packed_host.numpy()
        self._schemas = tuple((state.metadata, len(state.digests)) for state in prepared)
        from .native_merkle import PreparedNativeMerkle
        self._native_finalizer = None
        self.native_unavailable = None
        try:
            self._native_finalizer = PreparedNativeMerkle(
                [schema for schema, _ in self._schemas], [count for _, count in self._schemas])
        except RuntimeError as exc:
            self.native_unavailable = str(exc)
        self._native_executor = None
        self.native_cuda_unavailable = None
        if (self._native_finalizer is not None and hasattr(self._graph, 'raw_cuda_graph_exec')
                and self._checkpoint_buffer.numel() and all(
                    value.is_contiguous() and value.numel() for value in self._inputs.values())):
            self._ready.record(torch.cuda.current_stream(self._device))
            self._ready.synchronize()
            try:
                self._native_executor = self._native_finalizer.prepare_cuda(
                    self._device.index, self._graph.raw_cuda_graph_exec(), self._ready.cuda_event,
                    self._packed_host.data_ptr(), self._checkpoint_buffer.data_ptr(),
                    self._checkpoint_buffer.numel(),
                    [(value.data_ptr(), value.numel()*value.element_size()) for value in self._inputs.values()])
            except RuntimeError as exc:
                self.native_cuda_unavailable = str(exc)

    def commit(self, inputs: State, *, model_id: str, run_id: str) -> CommittedExecution:
        """Return a completed root and checkpoints owned by this execution."""
        if self._single_input is not None and type(inputs) is dict:
            name, dtype, shape = self._single_input
            value = inputs.get(name)
            valid = (len(inputs) == 1 and isinstance(value, torch.Tensor) and value.shape == shape
                     and value.dtype == dtype and value.device == self._device and value.layout == torch.strided)
            values = (value,)
        else:
            valid = (isinstance(inputs, Mapping) and set(inputs) == self._input_names and all(
                isinstance(inputs[name], torch.Tensor) and inputs[name].shape == shape
                and inputs[name].dtype == dtype and inputs[name].device == self._device
                and inputs[name].layout == torch.strided for name, dtype, shape in self._input_schema)
                and _aliases(inputs) == self._input_aliases)
            values = tuple(inputs[name] for name, _, _ in self._input_schema) if valid else ()
        if not valid:
            raise ValueError('prepared inputs must match captured names, shapes, dtypes, device and aliases')
        if not isinstance(model_id, str) or not model_id or not isinstance(run_id, str) or not run_id:
            raise ValueError('model_id and run_id must be nonempty strings')
        with self._lock:
            def context_digest():
                if model_id != self._model_id:
                    self._model_context = self._context_prefix.copy()
                    self._model_context.update(_json(model_id) + b',')
                    self._model_id = model_id
                context_hash = self._model_context.copy()
                context_hash.update(_json(run_id) + b']')
                return context_hash.digest()

            execution = None

            def prepare_response():
                nonlocal execution
                # Allocate response objects while inference runs. None escape
                # until the completed hashes replace the placeholders.
                tree = self._tree_layout.bind(self._blank_tree)
                commitment = Commitment(bytes(32), self._layer_names, model_id, run_id)
                execution = CommittedExecution(self._layers, {}, tree.leaf_hashes, tree, commitment,
                                               (checkpoint_buffer, self._checkpoint_layouts))
                return context_digest()

            if self._native_executor is not None and all(value.is_contiguous() for value in values):
                checkpoint_buffer = torch.empty_like(self._checkpoint_buffer)
                packed_tree = self._native_executor.execute_with_prepare(
                    [value.data_ptr() for value in values],
                    checkpoint_buffer.data_ptr(), torch.cuda.current_stream(self._device).cuda_stream,
                    prepare_response)
                execution._tree._packed = packed_tree
                object.__setattr__(execution.commitment, 'root', packed_tree[-32:])
                return execution
            else:
                with torch.cuda.device(self._device):
                    for name, value in inputs.items():
                        self._inputs[name].copy_(value)
                    self._graph.replay()
                    checkpoint_buffer = self._checkpoint_buffer.clone()
                    context = prepare_response() if self._native_finalizer is not None else context_digest()
                    self._ready.record(torch.cuda.current_stream(self._device))
                    self._ready.synchronize()
                    packed = self._packed_view.tobytes()
                    if self._native_finalizer is not None:
                        packed_tree = self._native_finalizer.compute_packed(packed, context)
                        execution._tree._packed = packed_tree
                        object.__setattr__(execution.commitment, 'root', packed_tree[-32:])
                        return execution
                    else:
                        hashes, offset = [], 0
                        for boundary, (schema, count) in enumerate(self._schemas):
                            digests = [packed[offset+i*32:offset+(i+1)*32] for i in range(count)]
                            fingerprint = finish_state_digest(schema, digests)
                            hashes.append(hash_leaf(boundary, context+fingerprint))
                            offset += count*32
                        tree = MerkleTree.from_leaf_hashes(hashes)
        commitment = Commitment(tree.root, self._layer_names, model_id, run_id)
        return CommittedExecution(self._layers, {}, hashes, tree, commitment,
                                  (checkpoint_buffer, self._checkpoint_layouts))
