"""Exact packed aggregation of private rejection bits."""

import torch

from ..protocol.runtime import Share
from .upper_zero import _pack_bits, _unpack_bits


def binary_sign(rt, value):
    """Extract a full-ring sign in XOR sharing without opening the word."""
    if value.binary:
        return Share(tuple((part >> (rt.bits-1))&1 for part in value.parts), True)
    pair = rt._normalize(value.parts[1]+value.parts[2 if rt.party % 2 == 0 else 0])
    a, b = rt.joint_input_batch(
        [(0,1,3,2,pair if rt.party < 2 else None),
         (2,3,1,0,pair if rt.party >= 2 else None)],
        value.shape, True, value.device)
    if value.device.type == 'cuda':
        from ..gpu.bits.carry_tree_swar import carry_bits
        carry = carry_bits(rt, a, b, [rt.bits-1])[0]
    else:
        from .exact_truncation import _binary_add_carries
        _, generate = _binary_add_carries(rt, a, b)
        carry = Share(tuple((part >> (rt.bits-2))&1 for part in generate.parts), True)
    return Share(tuple(((left >> (rt.bits-1))&1)^((right >> (rt.bits-1))&1)^bit
                       for left,right,bit in zip(a.parts,b.parts,carry.parts)), True)


def binary_and(rt, left, right):
    """Evaluate bitwise AND with one packed binary multiplication."""
    if not left.binary or not right.binary or left.shape != right.shape or left.device != right.device:
        raise ValueError('binary bit operands require binary shares with matching shapes and devices')
    left = Share(tuple(part&1 for part in left.parts), True)
    right = Share(tuple(part&1 for part in right.parts), True)
    packed_left, count = _pack_bits(rt, left)
    packed_right, _ = _pack_bits(rt, right)
    return _unpack_bits(rt, rt.mul(packed_left, packed_right), count, left.shape)


def binary_any(rt, values):
    """Return a scalar XOR-shared bit indicating any supplied rejection.

    Inputs contain secret bits in their low positions. Every OR layer packs
    independent gates into binary words; no input or intermediate is opened.
    """
    values = tuple(values)
    if not values:
        raise ValueError('binary aggregation requires at least one input')
    device = values[0].device
    if any(not value.binary or value.device != device for value in values):
        raise ValueError('binary aggregation requires binary shares on one device')
    level = Share(tuple(torch.cat([value.parts[slot].reshape(-1)&1 for value in values])
                        for slot in range(3)), True)
    count = level.parts[0].numel()
    if not count:
        raise ValueError('binary aggregation requires nonempty inputs')
    while count > 1:
        pairs = count // 2
        left = Share(tuple(part[:2*pairs:2] for part in level.parts), True)
        right = Share(tuple(part[1:2*pairs:2] for part in level.parts), True)
        packed_left, packed_count = _pack_bits(rt, left)
        packed_right, _ = _pack_bits(rt, right)
        product = _unpack_bits(rt, rt.mul(packed_left, packed_right), packed_count, (pairs,))
        combined = rt.add(rt.add(left, right), product)
        if count % 2:
            level = Share(tuple(torch.cat((part, previous[-1:]))
                                for part, previous in zip(combined.parts, level.parts)), True)
        else:
            level = combined
        count = (count+1)//2
    return Share(tuple(part.reshape(()) for part in level.parts), True)
