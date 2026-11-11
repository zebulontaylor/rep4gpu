"""Single-use dense private lookup preprocessing using the existing Rep4 roles.

This is a truth-table reference, not compact FSS key generation. Preprocessing
stores one secretly shifted table per input coordinate. Lookup opens only a
fresh masked index modulo the table size and selects output share components
locally. It moves substantial communication and memory into preprocessing.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch


@dataclass
class _Lease:
    consumed: bool = False


class PreparedLUT:
    """Runtime-bound, single-use masked table for a fixed public input shape.

    The index is interpreted modulo ``size``. The caller must enforce any
    activation domain and quantize the input before lookup. The table and mask
    must never be reused, including when a previous lookup failed.
    """

    def __init__(self, runtime, values, mask, shape, size):
        self._runtime = runtime
        self._values = values
        self._mask = mask
        self.shape = tuple(shape)
        self.size = size
        self._lease = _Lease()

    def __copy__(self):
        raise TypeError('private lookup preprocessing cannot be copied')

    def __deepcopy__(self, memo):
        raise TypeError('private lookup preprocessing cannot be copied')

    @property
    def consumed(self):
        return self._lease.consumed

    @property
    def local_storage_bytes(self):
        if self._values is None:
            return 0
        return sum(p.numel()*p.element_size() for p in self._values.parts) + sum(
            p.numel()*p.element_size() for p in self._mask.parts)

    def lookup(self, index, *, runtime=None):
        """Return table[index modulo size] as shares, consuming this material.

        Reducing each share component before opening is essential. Opening a
        full-width index plus a small mask would reveal the input's high bits.
        The checked opening reveals only component residues whose missing
        value is determined by the uniformly masked modular index.
        """
        rt = self._runtime
        if runtime is not None and runtime is not rt:
            raise ValueError('private lookup material belongs to another runtime')
        if self._lease.consumed:
            raise RuntimeError('private lookup material has already been consumed')
        if getattr(index, 'binary', False) or tuple(index.shape) != self.shape or index.device != rt.device:
            raise ValueError('lookup requires arithmetic shares with the prepared shape and device')
        self._lease.consumed = True
        masked = rt.add(index, self._mask)
        residues = rt._share(part & (self.size-1) for part in masked.parts)
        opened = rt.open(residues) & (self.size-1)
        output = rt._share(torch.gather(part, -1, opened.unsqueeze(-1)).squeeze(-1)
                           for part in self._values.parts)
        self._values = None
        self._mask = None
        return output


def prepare_lut(runtime, public_values, shape):
    """Prepare fresh dense LUT material without an additional trusted dealer.

    ``public_values`` is a public int64 vector with power-of-two length 2..4096.
    Each scalar input coordinate needs its own mask and shifted table. The
    shape and public values must agree at all four roles.

    For component i, draw r_i uniformly modulo table size using the existing
    PRG shared by all roles except i. Start T in component zero, locally rotated
    by r_0. At stage i=1,2,3, components j<i are the only nonzero components.
    The two roles knowing both component j and r_i jointly input its rotation,
    with receiver i and outsider j. Each result has support on components i,j.
    The resulting sum is T[j-sum(r_i)] with ordinary Rep4 consistency checks.
    Thus preprocessing sends six table-sized corrections, rather than running
    a multiplexer on every table entry for every index bit.
    """
    if getattr(runtime, 'bits', None) not in (32, 64):
        raise ValueError('private lookup requires a ring32 or ring64 runtime')
    values = torch.as_tensor(public_values, device=runtime.device)
    if values.dtype != torch.int64 or values.ndim != 1:
        raise ValueError('public LUT values must be a one-dimensional int64 tensor')
    size = values.numel()
    if not 2 <= size <= 4096 or size & (size-1):
        raise ValueError('private LUT size must be a power of two in [2,4096]')
    shape = tuple(shape)
    if any(type(n) is not int or n < 1 for n in shape) or math.prod(shape) < 1:
        raise ValueError('private LUT input shape must be nonempty in every dimension')
    if runtime.bits == 32 and not bool(((values >= -(1 << 31)) & (values < 1 << 31)).all()):
        raise ValueError('LUT outputs must fit the signed runtime ring')
    rt = runtime
    components = {}
    for excluded in range(4):
        if excluded != rt.party:
            components[excluded] = rt._random(excluded, shape, rt.device) & (size-1)
    mask = rt._share(components[(rt.party+slot+1) % 4] for slot in range(3))
    table_shape = (*shape, size)
    columns = torch.arange(size, dtype=torch.int64, device=rt.device)

    def rotate(part, amount):
        indices = (columns-amount.unsqueeze(-1)) & (size-1)
        return torch.gather(part, -1, indices)

    parts = [torch.zeros(table_shape, dtype=torch.int64, device=rt.device) for _ in range(3)]
    if rt.party != 0:
        parts[(0-rt.party-1) % 4] = rotate(values.expand(table_shape), components[0])
    table = rt._share(parts)
    for receiver in (1, 2, 3):
        specs = []
        for outsider in range(receiver):
            sender, backup = [role for role in range(4) if role not in (receiver, outsider)]
            value = None
            if rt.party in (sender, backup):
                part = table.parts[(outsider-rt.party-1) % 4]
                value = rotate(part, components[receiver])
            specs.append((sender, backup, receiver, outsider, value))
        terms = rt.joint_input_batch(specs, table_shape, device=rt.device)
        table = terms[0]
        for term in terms[1:]:
            table = rt.add(table, term)
    return PreparedLUT(rt, table, mask, shape, size)
