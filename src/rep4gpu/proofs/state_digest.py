"""Canonical integer-state fingerprints with optional on-device BLAKE3 hashing.

Tensor digests use official unkeyed BLAKE3-256. SHA-256 binds their sorted names,
shapes, and little-endian dtypes to a versioned state fingerprint.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import struct
import sys

from blake3 import blake3
import torch

from ..gpu.crypto.gpu_hash import digest as gpu_digest, digest_batch as gpu_digest_batch

_DOMAIN = b'rep4gpu.state-digest.v1\x00'
_DTYPES = {torch.int32: '<i4', torch.int64: '<i8'}


def finish_state_digest(metadata: bytes, digests: Sequence[bytes]) -> bytes:
    """Bind canonical metadata and ordered 32-byte tensor digests with SHA-256."""
    if not isinstance(metadata, bytes):
        raise TypeError('metadata must be bytes')
    result = hashlib.sha256(_DOMAIN)
    result.update(struct.pack('>Q', len(metadata)))
    result.update(metadata)
    for digest in digests:
        if not isinstance(digest, bytes) or len(digest) != 32:
            raise ValueError('tensor digests must be 32-byte strings')
        result.update(digest)
    return result.digest()


@dataclass(frozen=True)
class PreparedStateDigest:
    """Metadata and ordered CPU bytes or CUDA uint8 digest tensors.

    CUDA work is enqueued on each device's current stream. Warm preparation
    before graph capture to compile the hash kernels. Call ``finish`` after
    replay; it transfers only tensor digests and synchronizes their devices.
    Captured outputs remain live and reflect each subsequent graph replay.
    """
    metadata: bytes
    digests: tuple[bytes | torch.Tensor, ...]

    def finish(self, *, wait: bool = True) -> bytes:
        """Finalize, optionally relying on caller-managed producer ordering.

        With ``wait=False``, the caller must synchronize producer events or
        wait on their streams before calling. Transfers use the caller's
        current stream on each device, allowing a worker transfer stream.
        """
        groups = {}
        resolved = list(self.digests)
        for index, value in enumerate(self.digests):
            if isinstance(value, torch.Tensor):
                groups.setdefault(value.device, []).append((index, value))
        for device, entries in groups.items():
            # Preparation or graph replay may use a different stream than the
            # caller. Finish is the explicit synchronization boundary.
            if wait:
                torch.cuda.synchronize(device)
            packed = torch.cat([value for _, value in entries]).cpu().numpy().tobytes()
            for position, (index, _) in enumerate(entries):
                resolved[index] = packed[position * 32:(position + 1) * 32]
        return finish_state_digest(self.metadata, resolved)


def prepare_state_digest(state: Mapping[str, torch.Tensor]) -> PreparedStateDigest:
    """Enqueue fingerprints without CUDA synchronization or payload transfers.

    Supports dense CPU/CUDA int32 and int64 tensors, including views, scalars,
    and empty tensors. CUDA input mutations must be ordered after preparation
    on the same stream, or coordinated explicitly by the caller.
    """
    metadata, values = _state_values(state)
    digests = []
    for value in values:
        if value.is_cuda:
            digests.append(gpu_digest(value.view(torch.uint8), block_tree=True))
        else:
            array = value.numpy().astype(_DTYPES[value.dtype], copy=False)
            digests.append(blake3(memoryview(array).cast('B')).digest())
    return PreparedStateDigest(metadata, tuple(digests))


def _state_values(state: Mapping[str, torch.Tensor]):
    if not isinstance(state, Mapping) or not state:
        raise ValueError('state must be a nonempty mapping')
    if any(not isinstance(name, str) or not name for name in state):
        raise ValueError('state names must be nonempty strings')
    entries, values = [], []
    for name in sorted(state):
        value = state[name]
        if (not isinstance(value, torch.Tensor) or value.dtype not in _DTYPES
                or value.layout != torch.strided or value.device.type not in ('cpu', 'cuda')):
            raise ValueError('state values must be dense CPU or CUDA int32/int64 tensors')
        dtype = _DTYPES[value.dtype]
        entries.append([name, dtype, list(value.shape)])
        value = value.detach().contiguous().reshape(-1)
        if value.is_cuda and sys.byteorder != 'little':
            raise RuntimeError('CUDA state fingerprints require a little-endian host')
        values.append(value)
    metadata = json.dumps(entries, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode('ascii')
    return metadata, values


def prepare_state_digests(
    states: Sequence[Mapping[str, torch.Tensor]],
) -> tuple[PreparedStateDigest, ...]:
    """Prepare states together, batching CUDA tensors by device and byte length.

    Fingerprints and ordering match separate ``prepare_state_digest`` calls.
    Inputs are copied into contiguous batches on each device's current stream;
    order later input mutations on that stream. Warm preparation before CUDA
    graph capture, and finish outputs after replay as for a single state.
    """
    metadata = []
    outputs = []
    groups = {}
    for state_index, state in enumerate(states):
        encoded, values = _state_values(state)
        metadata.append(encoded)
        outputs.append([None] * len(values))
        for tensor_index, value in enumerate(values):
            if value.is_cuda:
                raw = value.view(torch.uint8)
                groups.setdefault((value.device, raw.numel()), []).append(
                    (state_index, tensor_index, raw))
            else:
                array = value.numpy().astype(_DTYPES[value.dtype], copy=False)
                outputs[state_index][tensor_index] = blake3(memoryview(array).cast('B')).digest()
    for entries in groups.values():
        packed = torch.stack([value for _, _, value in entries])
        digests = gpu_digest_batch(packed, block_tree=True)
        for row, (state_index, tensor_index, _) in enumerate(entries):
            outputs[state_index][tensor_index] = digests[row]
    return tuple(PreparedStateDigest(encoded, tuple(digests))
                 for encoded, digests in zip(metadata, outputs))


def state_digest(state: Mapping[str, torch.Tensor]) -> bytes:
    """Return a canonical state fingerprint, waiting for CUDA hashes if needed."""
    return prepare_state_digest(state).finish()
