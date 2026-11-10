"""Exact arithmetic/binary conversion using Rep4's replicated components.

Binary words pack the 64 bits of each value into one int64. Rep4 multiplication
then evaluates 64 independent AND gates per word. No value is opened during
conversion, signed truncation, or comparison.
"""
import torch

from .runtime import Share


def _stack(values):
    return Share(tuple(torch.stack([x.parts[j] for x in values]) for j in range(3)), values[0].binary)


def _map(x, fn):
    return Share(tuple(fn(t) for t in x.parts), x.binary)


def _and_pairs(rt, pairs):
    result = rt.mul(_stack([a for a, _ in pairs]), _stack([b for _, b in pairs]))
    return [result[i] for i in range(len(pairs))]


def _carry_save(rt, a, b, c):
    propagate = rt.add(a, b)
    ab, pc = _and_pairs(rt, [(a, b), (propagate, c)])
    return rt.add(propagate, c), rt.lshift(rt.add(ab, pc), 1)


def binary_add(rt, a, b):
    """Add two binary-shared words modulo the runtime ring width."""
    if not a.binary or not b.binary:
        raise ValueError('binary_add requires binary shares')
    original = propagate = rt.add(a, b)
    generate = rt.mul(a, b)
    for distance in (1 << i for i in range(rt.bits.bit_length() - 1)):
        gp, pp = _and_pairs(rt, [(propagate, rt.lshift(generate, distance)),
                                  (propagate, rt.lshift(propagate, distance))])
        generate = rt.add(generate, gp)
        propagate = pp
    result = rt.add(original, rt.lshift(generate, 1))
    return _map(result, lambda t: t & 0xffffffff) if rt.bits == 32 else result


def _components(rt, x, binary):
    """Embed each known component in the same replicated ownership slots."""
    values = []
    for excluded in range(4):
        values.append(Share(tuple(t if (rt.party + j + 1) % 4 == excluded else torch.zeros_like(t)
                                  for j, t in enumerate(x.parts)), binary))
    return values


def a2b(rt, x):
    """Convert an arithmetic ring word to binary sharing without opening it.

    The initial split is MP-SPDZ's local four-component split. Two carry-save
    stages reduce the four addends to two, followed by a parallel-prefix sum.
    """
    if x.binary:
        return x
    a, b, c, d = _components(rt, x, True)
    s, carry = _carry_save(rt, a, b, c)
    s, carry = _carry_save(rt, s, carry, d)
    result = binary_add(rt, s, carry)
    return _map(result, lambda t: t & 0xffffffff) if rt.bits == 32 else result


def b2a(rt, x, *, bits=None, signed=True):
    """Convert low ``bits`` binary bits to one arithmetic ring value.

    Each bit's four binary components become arithmetic bits. Two batched
    multiplication rounds evaluate their XOR, then a local weighted sum
    combines them. Set ``signed`` to sign-extend the highest selected bit.
    """
    if not x.binary:
        raise ValueError('b2a requires binary shares')
    bits = rt.bits if bits is None else bits
    if not 1 <= bits <= rt.bits:
        raise ValueError(f'bits must be in [1, {rt.bits}]')
    positions = torch.arange(bits, dtype=torch.int64, device=x.device)
    positions = positions.reshape((bits,) + (1,) * len(x.shape))
    planes = Share(tuple((part.unsqueeze(0) >> positions) & 1 for part in x.parts), True)
    a, b, c, d = _components(rt, planes, False)
    ab, cd = _and_pairs(rt, [(a, b), (c, d)])
    left = rt.sub(rt.add(a, b), rt.mul_public(ab, 2))
    right = rt.sub(rt.add(c, d), rt.mul_public(cd, 2))
    merged = rt.sub(rt.add(left, right), rt.mul_public(rt.mul(left, right), 2))
    weights = (torch.ones(bits, dtype=torch.int64, device=x.device) << torch.arange(bits, device=x.device))
    if signed and bits < 64:
        weights[-1] = -(1 << (bits - 1))
    weights = weights.reshape((bits,) + (1,) * len(x.shape))
    return rt._share((part * weights).sum(dim=0) for part in merged.parts)


def trunc(rt, x, bits):
    """Signed floor division by 2**bits, including int64 boundary values."""
    if not isinstance(bits, int) or not 0 <= bits < rt.bits:
        raise ValueError(f'shift must be an integer in [0, {rt.bits - 1}]')
    if x.binary:
        raise ValueError('trunc requires arithmetic shares')
    if bits == 0:
        return x
    word = a2b(rt, x)
    shifted = _map(word, lambda t: (t >> bits) & ((1 << (rt.bits - bits)) - 1))
    return b2a(rt, shifted, bits=rt.bits - bits, signed=True)


def sign_bit(rt, x):
    if x.device.type == 'cuda':
        from .exact_truncation import sign_bit_packed
        return sign_bit_packed(rt, x)
    from .compact_conversion import b2a as compact_b2a
    word = a2b(rt, x)
    return compact_b2a(rt, _map(word, lambda t: (t >> (rt.bits - 1)) & 1), bits=1, signed=False)


def trunc_relu(rt, x, bits):
    """Fuse signed truncation and ReLU before converting back to arithmetic."""
    if not isinstance(bits, int) or not 0 <= bits < rt.bits:
        raise ValueError(f'shift must be an integer in [0, {rt.bits - 1}]')
    word = a2b(rt, x)
    # Replicate each binary sign share to all bit positions, then complement
    # the reconstructed sign without revealing it.
    sign_mask = _map(word, lambda t: -((t >> (rt.bits - 1)) & 1))
    keep = rt.add(sign_mask, rt.public(-1, like=word))
    clipped = rt.mul(word, keep)
    if bits:
        clipped = _map(clipped, lambda t: (t >> bits) & ((1 << (rt.bits - bits)) - 1))
    return b2a(rt, clipped, bits=rt.bits - bits, signed=False)


def signed_lt(rt, x, y):
    """Compare signed ring values without subtraction-overflow errors."""
    if not isinstance(y, Share):
        y = rt.public(y, like=x)
    packed = _stack([x, y, rt.sub(x, y)])
    signs = sign_bit(rt, packed)
    sx, sy, sd = signs[0], signs[1], signs[2]
    different = rt.sub(rt.add(sx, sy), rt.mul_public(rt.mul(sx, sy), 2))
    return rt.add(sd, rt.mul(different, rt.sub(sx, sd)))
