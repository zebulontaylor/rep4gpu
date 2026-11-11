"""Batch checked product claims with independent ring challenges."""
import hashlib
import secrets
import struct

import torch

from .runtime import Share

_CHALLENGE_ROWS = 128


class ProductWitnessBatch:
    """Freeze arithmetic claims and check them without opening residuals.

    For any fixed nonzero residual vector over Z/(2**64), an independent
    uniform ring coefficient vector annihilates it with probability at most
    1/2. The 128 independent rows therefore miss an invalid batch with
    probability at most 2**-128, including residuals divisible by 2**63.
    This bound assumes ideal independent coefficient rows. SHAKE256 expands
    the fresh 256-bit public seed into those rows, so the implementation also
    relies on its pseudorandomness. A single row does not provide field-size
    soundness.
    """

    def __init__(self, runtime, *, chunk_size=1024):
        if runtime.bits != 64:
            raise ValueError('product witness batches require ring64')
        if type(chunk_size) is not int or chunk_size <= 0:
            raise ValueError('product witness chunk size must be positive')
        self.runtime = runtime
        self.chunk_size = chunk_size
        self._claims = []
        self._sealed = False

    def add(self, left, right, claimed):
        """Clone a product claim and return the frozen claimed output share."""
        if self._sealed:
            raise RuntimeError('product witness batch is sealed')
        if any(value.binary for value in (left, right, claimed)):
            raise ValueError('product witness claims require arithmetic shares')
        if any(value.shape != left.shape or value.device != left.device
               for value in (right, claimed)) or not left.parts[0].numel():
            raise ValueError('product witness claims require matching nonempty tensors')
        frozen = tuple(Share(tuple(part.clone() for part in value.parts))
                       for value in (left, right, claimed))
        self._claims.append(frozen)
        return frozen[2]

    def _public_seed(self):
        rt = self.runtime
        coin = None
        # Every private contribution is shared before any contribution sum
        # opens. An honest role supplies uniform entropy unknown to an attacker
        # while all four claims and contributions become fixed.
        for owner in range(4):
            local = (torch.tensor(struct.unpack('<4q', secrets.token_bytes(32)),
                                  dtype=torch.int64, device=rt.device)
                     if rt.party == owner else None)
            shared = rt.input(owner, local, (4,))
            coin = shared if coin is None else rt.add(coin, shared)
        opened = rt.open(coin).detach().cpu().tolist()
        return struct.pack('<4q', *opened)

    def verify(self, session):
        """Seal claims, sample challenges, and register one private batch check."""
        session._active()
        if self._sealed:
            raise RuntimeError('product witness batch is sealed')
        if session.runtime is not self.runtime:
            raise ValueError('product witness session uses another runtime')
        if not self._claims:
            raise ValueError('cannot verify an empty product witness batch')
        self._sealed = True
        rt = self.runtime
        seed = self._public_seed()
        device = self._claims[0][0].device
        local = torch.zeros((3, _CHALLENGE_ROWS), dtype=torch.int64, device=device)
        linear = torch.zeros_like(local)
        for claim_index, (left, right, claimed) in enumerate(self._claims):
            xs, ys, zs = (tuple(part.reshape(-1) for part in value.parts)
                          for value in (left, right, claimed))
            for start in range(0, xs[0].numel(), self.chunk_size):
                end = min(start+self.chunk_size, xs[0].numel())
                count = end-start
                domain = b'Rep4/product-witness/ring64/v1'+seed+struct.pack('<QQ', claim_index, start)
                raw = hashlib.shake_256(domain).digest(_CHALLENGE_ROWS*count*8)
                weights = torch.frombuffer(bytearray(raw), dtype=torch.int64).reshape(_CHALLENGE_ROWS, count).to(device)
                x0,x1,x2 = (part[start:end] for part in xs)
                y0,y1,y2 = (part[start:end] for part in ys)
                terms = ((x0+x1)*y0+x0*y1,
                         (x1+x2)*y1+x1*y2,
                         x0*y2+x2*y0)
                for slot in range(3):
                    local[slot] += (weights*terms[slot]).sum(dim=1)
                    linear[slot] += (weights*zs[slot][start:end]).sum(dim=1)
        contributions = [None]*5
        contributions[(rt.party-1)%4], contributions[rt.party], contributions[4] = local.unbind(0)
        product = rt._reshare_product(contributions, False, device)
        session.residual(rt.sub(product, Share(tuple(linear.unbind(0)))), 0)
        self._claims.clear()
