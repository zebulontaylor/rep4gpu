"""Private input sharing with GPU-resident BLAKE3 consistency checks."""
import math

import torch

from ..gpu.crypto.gpu_hash import digest


class DeviceInputMixin:
    """Replace byte-staged input hashing in a device party.

    Mask placement and the three input rounds match ``Party.input``. Input
    tags use BLAKE3 rather than BLAKE2b, so every party must select this variant.
    The two recipients verify their corrections before the input is returned.
    """

    def input(self, owner, value=None, shape=None, binary=False):
        if owner not in range(4):
            raise ValueError('input owner must be in [0,3]')
        if value is not None:
            if self.party != owner:
                raise ValueError('only the input owner supplies plaintext')
            value = torch.as_tensor(value, dtype=torch.int64, device=self.device)
            shape = value.shape
        if shape is None or (self.party == owner and value is None):
            raise ValueError('owner supplies its input and other parties supply its shape')
        shape = tuple(shape)
        width = self._wire_bits(binary)
        dtype = {8: torch.uint8, 16: torch.uint16, 32: torch.uint32, 64: torch.int64}[width]
        storage = self._storage_dtype(binary)
        packed = storage in (torch.uint8, torch.uint16)
        parts = [torch.zeros(shape, dtype=storage, device=self.device) for _ in range(3)]
        for excluded in ((owner + 1) % 4, (owner + 3) % 4):
            if self.party != excluded:
                parts[(excluded - self.party - 1) % 4] = self._random(
                    excluded, shape, self.device, binary, packed=packed)
        recipients = ((owner + 1) % 4, (owner + 3) % 4)
        outgoing = {}
        if self.party == owner:
            if packed:
                # Canonical narrow operands keep this temporary subtraction in int32 range.
                correction = value.to(storage).to(torch.int32) - parts[0].to(torch.int32) - parts[2].to(torch.int32)
            else:
                correction = value ^ parts[0] ^ parts[2] if binary else value - parts[0] - parts[2]
            parts[1] = self._normalize(correction, binary).to(storage)
            wire = parts[1].contiguous().reshape(-1).to(dtype).view(torch.uint8)
            outgoing = {p: wire for p in recipients}
        sizes = {owner: math.prod(shape) * (width // 8)} if self.party in recipients else {}
        incoming = self._exchange_device(outgoing, sizes)
        if self.party in recipients:
            wire = incoming[owner]
            if (wire.numel() != sizes[owner] or wire.dtype != torch.uint8
                    or wire.ndim != 1 or not wire.is_contiguous() or wire.device != self.device):
                raise ValueError('invalid private input frame')
            if wire.storage_offset() % (width // 8):
                wire = wire.clone()
            parts[((owner + 2) % 4 - self.party - 1) % 4] = wire.view(dtype).to(storage).reshape(shape)
            tag = digest(wire, block_tree=self.hash_block_tree)
            empty = digest(torch.empty(0, dtype=torch.uint8, device=self.device), block_tree=self.hash_block_tree)
            pair = (tag, empty) if self.party == (owner - 1) % 4 else (empty, tag)
        checks = []
        for index in range(2):
            peer = (self.party + 2) % 4
            outgoing = {peer: pair[index]} if self.party in recipients else {}
            sizes = {peer: 32} if self.party in recipients else {}
            received = self._exchange_device(outgoing, sizes)
            if self.party in recipients:
                checks.append(received[peer])
        if self.party in recipients:
            self._verify_tags(torch.stack(checks), torch.stack(pair[::-1]),
                              'device private input BLAKE3 hash mismatch')
        return self._share(parts, binary)
