"""Rep4 masked probabilistic truncation with the MP-SPDZ small-gap correction."""

import hashlib
import secrets

import torch

from ..protocol.runtime import bytes_tensor, tensor_bytes


def trunc_pr(rt, x, shift):
    """Round x/2**shift to floor or ceil without opening x.

    Requires arithmetic shares and the public range contract
    ``abs(signed(x)) < 2**(rt.bits-2)``. The range is not checked secretly.
    The probability of rounding up is ``(x mod 2**shift)/2**shift``.
    """
    k = rt.bits - 1
    if x.binary:
        raise ValueError('probabilistic truncation requires arithmetic shares')
    if not isinstance(shift, int) or not 0 <= shift < k:
        raise ValueError('shift must be an integer in [0, ring_bits-2]')
    if shift == 0:
        return x

    parts = []
    for offset in range(1, 4):
        excluded = (rt.party + offset) % 4
        parts.append(rt._random(excluded, x.shape, x.device)
                     if excluded in (2, 3) else torch.zeros_like(x.parts[0]))
    mask = rt._share(parts)
    masked = rt.add(rt.add(x, mask), rt.public(1 << (k-1), like=x))

    # Parties 0 and 1 know identical masked partial sums, checked crosswise.
    outgoing = {}
    if rt.party < 2:
        partial = masked.parts[1] + masked.parts[2 if rt.party == 0 else 0]
        data = tensor_bytes(rt._normalize(partial), rt.bits)
        outgoing[2 + rt.party] = data
        outgoing[3 - rt.party] = hashlib.blake2b(data, digest_size=32).digest()
    incoming = rt.transport.exchange(outgoing)
    opened_masked = None
    if rt.party >= 2:
        data = incoming.get(rt.party-2, b'')
        digest = incoming.get(3-rt.party, b'')
        if not secrets.compare_digest(hashlib.blake2b(data, digest_size=32).digest(), digest):
            raise ValueError('hash mismatch in probabilistic truncation masked opening')
        if len(data) != x.parts[0].numel() * (rt.bits // 8):
            raise ValueError('invalid probabilistic truncation masked payload length')
        partial = bytes_tensor(data, x.shape, x.device, rt.bits)
        opened_masked = rt._normalize(partial + masked.parts[1] + masked.parts[2 if rt.party == 2 else 0])

    def split(value):
        msb = (value >> k) & 1
        upper = (value & ((1 << k)-1)) >> shift
        return torch.stack((msb, upper))

    mask_inputs = split(sum(mask.parts)) if rt.party < 2 else None
    masked_inputs = split(opened_masked) if rt.party >= 2 else None
    r, c = rt.joint_input_batch(
        [(0, 1, 3, 2, mask_inputs), (2, 3, 1, 0, masked_inputs)],
        (2, *x.shape), False, x.device)
    both = rt.mul(r[0], c[0])
    xor = rt.sub(rt.add(r[0], c[0]), rt.mul_public(both, 2))
    result = rt.add(rt.sub(c[1], r[1]), rt.lshift(xor, k-shift))
    return rt.sub(result, rt.public(1 << (k-shift-1), like=x))
