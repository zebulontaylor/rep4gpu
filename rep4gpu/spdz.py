"""Trusted-preprocessing simulator for authenticated prime-field matrix products.

This module keeps all parties and the dealer in one process. It exercises Beaver
algebra and detects inconsistent simulated openings; it is not a distributed
malicious-majority protocol or an implementation of SPDZ2k.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
import numpy as np
import torch

PRIME = 2147483647


class AuthenticationError(RuntimeError):
    """An opened value does not match its authentication shares."""


def _random(shape, device):
    """Sample field elements by rejection from OS randomness."""
    count = int(np.prod(shape))
    result = np.empty(count, dtype=np.int64)
    used = 0
    while used < count:
        words = np.frombuffer(os.urandom(4 * (2 * (count - used) + 32)), dtype=np.uint32)
        accepted = words[words < PRIME][: count - used]
        result[used:used + len(accepted)] = accepted
        used += len(accepted)
    return torch.from_numpy(result.reshape(shape)).to(device)


def field_matmul(a, b):
    """Exact product over F_(2**31-1) for canonical int64 matrices.

    CUDA uses centered signed-byte tensor products and Mersenne reduction.
    CPU splits each element into 16-bit limbs; three integer products recover
    the low, cross and high terms without overflowing int64 accumulators.
    """
    if a.dtype != torch.int64 or b.dtype != torch.int64:
        raise TypeError("field matrices must have dtype int64")
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError("expected compatible matrices")
    if a.device != b.device:
        raise ValueError("matrices must use the same device")
    if a.shape[1] * (2**17 - 2) ** 2 >= 2**63:
        raise ValueError("inner dimension exceeds the exact accumulation bound")
    if a.is_cuda:
        from .field_kernels import field_matmul as cuda_field_matmul
        return cuda_field_matmul(a, b)
    a, b = a.remainder(PRIME), b.remainder(PRIME)
    multiply = torch.matmul
    al, ah = a & 65535, a >> 16
    bl, bh = b & 65535, b >> 16
    low = multiply(al, bl)
    high = multiply(ah, bh)
    cross = multiply(al + ah, bl + bh) - low - high
    result = (low.remainder(PRIME) + (cross.remainder(PRIME) << 16)).remainder(PRIME)
    return (result + (high.remainder(PRIME) << 32)).remainder(PRIME)


@dataclass
class AuthShare:
    """Value shares [party, row, col] and MAC shares [key, party, row, col]."""
    values: torch.Tensor
    macs: torch.Tensor


@dataclass
class MatrixTriple:
    a: AuthShare
    b: AuthShare
    c: AuthShare
    used: bool = False


class TrustedDealer:
    """Generate correlated shares using an ideal trusted dealer.

    Every secret and every MAC key is visible to this process. Generating these
    objects is preprocessing and must be timed separately from online execution.
    """
    def __init__(self, parties=4, mac_keys=5, device="cpu"):
        if parties < 2 or mac_keys < 1:
            raise ValueError("require at least two parties and one MAC key")
        self.parties = parties
        self.mac_keys = mac_keys
        self.device = torch.device(device)
        self.alpha_shares = _random((mac_keys, parties), self.device)
        self.alpha = self.alpha_shares.sum(1).remainder(PRIME)

    def share(self, value):
        value = value.to(device=self.device, dtype=torch.int64).remainder(PRIME)
        if value.ndim != 2:
            raise ValueError("authenticated values must be matrices")
        shares = _random((self.parties, *value.shape), self.device)
        shares[-1] = (value - shares[:-1].sum(0)).remainder(PRIME)
        macs = _random((self.mac_keys, self.parties, *value.shape), self.device)
        target = self.alpha[:, None, None] * value
        macs[:, -1] = (target - macs[:, :-1].sum(1)).remainder(PRIME)
        return AuthShare(shares, macs)

    def triple(self, m, k, n):
        a = _random((m, k), self.device)
        b = _random((k, n), self.device)
        return MatrixTriple(self.share(a), self.share(b), self.share(field_matmul(a, b)))


class AuthenticatedOnline:
    """Evaluate authenticated Beaver products in the centralized simulator.

    Openings are checked before their values are used. The simulator atomically
    sums all parties' residuals; a real implementation needs authenticated
    broadcast and a commitment/coin-toss protocol to prevent rushing attacks.
    """
    def __init__(self, dealer):
        self.alpha_shares = dealer.alpha_shares
        self.parties = dealer.parties
        self.opened_elements = 0
        self.mac_residual_elements = 0

    def open(self, x):
        value = x.values.sum(0).remainder(PRIME)
        residuals = (x.macs - self.alpha_shares[:, :, None, None] * value).remainder(PRIME)
        self.opened_elements += value.numel()
        self.mac_residual_elements += residuals.numel()
        if torch.any(residuals.sum(1).remainder(PRIME) != 0).item():
            raise AuthenticationError("opening failed the field MAC check")
        return value

    @staticmethod
    def subtract(x, y):
        return AuthShare((x.values - y.values).remainder(PRIME),
                         (x.macs - y.macs).remainder(PRIME))

    def matmul(self, x, y, triple):
        if triple.used:
            raise ValueError("matrix triples are single-use")
        # Consume before opening so failed attempts cannot reuse these masks.
        triple.used = True
        d = self.open(self.subtract(x, triple.a))
        e = self.open(self.subtract(y, triple.b))
        public = field_matmul(d, e)
        values = triple.c.values.clone()
        macs = triple.c.macs.clone()
        for party in range(self.parties):
            values[party] += field_matmul(d, triple.b.values[party])
            values[party] += field_matmul(triple.a.values[party], e)
            for key in range(self.alpha_shares.shape[0]):
                macs[key, party] += field_matmul(d, triple.b.macs[key, party])
                macs[key, party] += field_matmul(triple.a.macs[key, party], e)
                macs[key, party] += self.alpha_shares[key, party] * public
        values[0] += public
        return AuthShare(values.remainder(PRIME), macs.remainder(PRIME))
