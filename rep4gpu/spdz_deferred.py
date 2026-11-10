"""Deferred authentication for fixed-circuit masked SPDZ openings.

Only the private masked-opening hook queues checks. Public ``open`` and
``open_many`` authenticate every pending opening before sending any output
value share, then check that output normally. See ``docs/deferred-mac-checks.md``
for the preprocessing assumptions and the checked-output boundary.
"""

from dataclasses import dataclass
import json
import struct

import torch

from .spdz import PRIME
from .spdz_compression import (CHACHA_BACKEND, DeviceCompressedParty,
                               check_residual_batches_device)
from .spdz_distributed import ProtocolError, _hash
from .spdz_device import DeviceDistributedParty


@dataclass(frozen=True)
class _Pending:
    residual: torch.Tensor
    schema: bytes


class DeviceDeferredCompressedParty(DeviceCompressedParty):
    """Queue fresh masked differences, with an automatic checked-output boundary.

    Use this variant only for a fixed public arithmetic circuit. Tentative
    masked candidates stay inside matrix, elementwise, and square operations;
    they must not control application branches, logs, or external effects.
    ``max_pending_values`` is a public memory limit shared by every role. A
    full queue is checked before the next masked broadcast; an opening larger
    than the limit is checked immediately after flushing the current queue.
    Public outputs with at most ``output_check_threshold`` values use the
    full-vector MAC check; larger outputs use independent-key compression.
    Set the threshold to zero to force compressed output checks.
    """

    def __init__(self, *args, max_pending_values=2**25, output_check_threshold=4096, **kwargs):
        if isinstance(max_pending_values, bool) or not isinstance(max_pending_values, int) or not 1 <= max_pending_values < 2**31:
            raise ValueError('max_pending_values must be an integer in [1,2**31)')
        if isinstance(output_check_threshold, bool) or not isinstance(output_check_threshold, int) or not 0 <= output_check_threshold < 2**31:
            raise ValueError('output_check_threshold must be an integer in [0,2**31)')
        self.max_pending_values = max_pending_values
        self.output_check_threshold = output_check_threshold
        self.last_output_check_mode = None
        self._pending = []
        self._pending_values = 0
        self._flush_index = 0
        super().__init__(*args, **kwargs)
        self.crypto_variant = 'SPDZ31G1/BLAKE3/deferred-independent-ChaCha-MAC/hybrid-output'

    @property
    def pending_values(self):
        return self._pending_values

    def abort(self, error):
        super().abort(error)
        # Poisoning retires these checks rather than making them reusable.
        self._pending.clear()
        self._pending_values = 0

    def flush_pending(self):
        """Authenticate the complete frozen queue, without opening an output."""
        with self.operation():
            if not self._pending:
                return
            if not 0 <= self._flush_index < 2**64:
                raise ProtocolError('deferred MAC batch counter exhausted')
            layout = json.dumps({'field': PRIME, 'keys': self.mac_keys,
                                 'backend': CHACHA_BACKEND.hex(),
                                 'max_pending_values': self.max_pending_values,
                                 'output_check_threshold': self.output_check_threshold,
                                 'segments': [entry.schema.hex() for entry in self._pending]},
                                sort_keys=True, separators=(',', ':')).encode()
            context = (b'SPDZ31G-deferred-check-v1' + self._session_hash
                       + struct.pack('<QQ', self.round, self._flush_index) + _hash(layout))
            check_residual_batches_device(self, (entry.residual for entry in self._pending), context=context)
            # Keep the queue intact through the entire authenticated check.
            self._pending.clear()
            self._pending_values = 0
            self._flush_index += 1

    def open_many(self, shares, *, label=b''):
        with self.operation():
            self.flush_pending()
            shares = tuple(shares)
            if not shares:
                raise ValueError('open_many requires at least one matrix')
            count = sum(share.values.numel() for share in shares)
            full_vector = self.output_check_threshold > 0 and count <= self.output_check_threshold
            mode = b'full-vector' if full_vector else b'compressed'
            public_label = (b'deferred-output/' + struct.pack('<QQ', self.max_pending_values,
                            self.output_check_threshold) + mode + b'\0' + label)
            self.last_output_check_mode = mode.decode('ascii')
            if full_vector:
                return DeviceDistributedParty.open_many(self, shares, label=public_label)
            return super().open_many(shares, label=public_label)

    def _masked_open_many(self, shares, *, label=b''):
        from .spdz_device import _sum_values
        import triton
        with self.operation():
            shares = tuple(shares)
            if not shares:
                raise ValueError('masked opening requires at least one matrix')
            self._ensure_canonical(shares)
            count = sum(share.values.numel() for share in shares)
            if count >= 2**31:
                raise ValueError('device masked openings support fewer than 2**31 values')
            if count > self.max_pending_values:
                self.flush_pending()
                oversized_label = (b'deferred-oversized/' + struct.pack('<QQ', self.max_pending_values,
                                   self.output_check_threshold) + label)
                return super().open_many(shares, label=oversized_label)
            if self._pending_values + count > self.max_pending_values:
                self.flush_pending()
            value_round = self.round
            schema = {'index': len(self._pending), 'round': value_round, 'count': count,
                      'shapes': [list(share.values.shape) for share in shares],
                      'label': label.hex(), 'max_pending_values': self.max_pending_values,
                      'output_check_threshold': self.output_check_threshold}
            encoded = json.dumps(schema, sort_keys=True, separators=(',', ':')).encode()
            context = (b'SPDZ31G-deferred-masked-v1' + self._session_hash
                       + struct.pack('<Q', value_round) + _hash(encoded) + _hash(CHACHA_BACKEND))
            values = self.broadcast_device(self._packed_values(shares), label=context + b'/values',
                                           canonical_byte_offset=0)
            opened = torch.empty(count, dtype=torch.int64, device=self.device)
            if count:
                _sum_values[(triton.cdiv(count, 256),)](
                    *(values[p] for p in range(4)), opened, count, 256)
            # This allocation freezes the exact MAC residual of the completed
            # broadcast. It does not alias inputs, candidate values, or scratch.
            residual = self._residual(shares, opened).contiguous()
            schema['context'] = context.hex()
            frozen_schema = json.dumps(schema, sort_keys=True, separators=(',', ':')).encode()
            self._pending.append(_Pending(residual, frozen_schema))
            self._pending_values += count
            results, offset = [], 0
            for share in shares:
                size = share.values.numel()
                results.append(opened[offset:offset + size].reshape(share.values.shape))
                offset += size
            return results
