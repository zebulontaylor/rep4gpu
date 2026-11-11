"""Private unsigned ranges from the carry into a public bit position."""

import torch

from .compact_conversion import b2a
from .conversion import _and_pairs, _map
from ..protocol.runtime import Share


def _low_carry(rt, a, b, width):
    """Compute only the prefix needed for carry into ``width`` on CPU."""
    propagate = rt.add(a, b)
    generate = rt.mul(a, b)
    distance = 1
    while distance < width:
        gp, pp = _and_pairs(rt, [
            (propagate, rt.lshift(generate, distance)),
            (propagate, rt.lshift(propagate, distance))])
        generate = rt.add(generate, gp)
        propagate = pp
        distance *= 2
    return _map(generate, lambda part: (part >> (width-1)) & 1)


def upper_zero_residual(rt, value, width, *, carry_impl=None, convert_impl=None):
    """Return zero exactly when the ring64 word lies in ``[0, 2**width)``.

    The residual is the upper part of the reconstructed word shifted back to
    its original position. It must be checked for exact zero. Random linear
    aggregation in ring64 has only a worst-case 1/2 detection probability per
    independent challenge: an invalid residual can equal ``2**63``.

    Two jointly known pair sums u,v reconstruct the input modulo 2**64. The
    high sum is u_hi+v_hi+carry_width. Shifting it left by width discards the
    possible ring carry and leaves exactly the input's upper bits. Only the
    low width-bit carry tree and one B2A bit are needed for this residual.
    Joint inputs use the runtime's checked input protocol.
    """
    if rt.bits != 64 or value.binary:
        raise ValueError('upper-zero requires ring64 arithmetic shares')
    if type(width) is not int or not 1 <= width < 64:
        raise ValueError('width must be an integer in [1, 63]')
    pair = rt._normalize(value.parts[1] + value.parts[2 if rt.party % 2 == 0 else 0])
    roles = ((0, 1, 3, 2), (2, 3, 1, 0))
    a, b = rt.joint_input_batch(
        [(*roles[0], pair if rt.party < 2 else None),
         (*roles[1], pair if rt.party >= 2 else None)],
        value.shape, True, value.device)
    mask = (1 << (64-width))-1
    upper = (pair >> width) & mask
    high_a, high_b = rt.joint_input_batch(
        [(*roles[0], upper if rt.party < 2 else None),
         (*roles[1], upper if rt.party >= 2 else None)],
        value.shape, False, value.device)
    if carry_impl is not None:
        carry = carry_impl(rt, a, b, [width])[0]
    elif value.device.type == 'cuda':
        from ..gpu.bits.carry_tree_gpu import carry_bits
        carry = carry_bits(rt, a, b, [width])[0]
    else:
        carry = _low_carry(rt, a, b, width)
    if convert_impl is not None:
        arithmetic = convert_impl(rt, _map(carry, lambda part: part.unsqueeze(0)), [1])
    else:
        arithmetic = b2a(rt, carry, bits=1, signed=False)
    return rt.lshift(rt.add(rt.add(high_a, high_b), arithmetic), width)


def _pack_bits(rt, value):
    count = value.parts[0].numel()
    width = rt.binary_bits
    padding = (-count) % width
    shifts = torch.arange(width, device=value.device)
    def pack(part):
        flat = part.reshape(-1)
        if padding:
            flat = torch.nn.functional.pad(flat, (0, padding))
        return (flat.reshape(-1, width) << shifts).sum(dim=1)
    return _map(value, pack), count


def _unpack_bits(rt, value, count, shape):
    shifts = torch.arange(rt.binary_bits, device=value.device)
    return _map(value, lambda part: ((part[:, None] >> shifts) & 1).reshape(-1)[:count].reshape(shape))


def _all_bits(rt, words, length):
    positions = torch.arange(length, device=words.device)
    leaves = _map(words, lambda part: (part.unsqueeze(-1) >> positions) & 1)
    while length > 1:
        pairs = length // 2
        left = _map(leaves, lambda part: part[..., :2*pairs:2])
        right = _map(leaves, lambda part: part[..., 1:2*pairs:2])
        packed_left, count = _pack_bits(rt, left)
        packed_right, _ = _pack_bits(rt, right)
        reduced = _unpack_bits(rt, rt.mul(packed_left, packed_right), count, left.shape)
        if length % 2:
            leaves = Share(tuple(torch.cat((part, tail[..., -1:]), dim=-1)
                                 for part, tail in zip(reduced.parts, leaves.parts)), True)
        else:
            leaves = reduced
        length = (length+1)//2
    return _map(leaves, lambda part: part.squeeze(-1))


def upper_zero_violation(rt, value, width, *, carry_impl=None, convert_impl=None, binary=False):
    """Return an exact arithmetic rejection bit for any ring64 input.

    Let U,V be the unsigned high pair sums and c the carry from their low
    width bits. The range holds exactly when V equals -U-c modulo
    2**(64-width). Two packed binary equality trees test -U and -U-1;
    the low carry selects the required equality. Only the rejection bit is
    converted to arithmetic unless ``binary`` keeps it in XOR sharing for
    aggregation with other rejection bits. No private value is opened.
    """
    if rt.bits != 64 or value.binary:
        raise ValueError('upper-zero requires ring64 arithmetic shares')
    if type(width) is not int or not 1 <= width < 64:
        raise ValueError('width must be an integer in [1, 63]')
    pair = rt._normalize(value.parts[1] + value.parts[2 if rt.party % 2 == 0 else 0])
    mask = (1 << (64-width))-1
    high = (pair >> width) & mask
    a, b, target0 = rt.joint_input_batch(
        [(0, 1, 3, 2, pair if rt.party < 2 else None),
         (2, 3, 1, 0, pair if rt.party >= 2 else None),
         (0, 1, 3, 2, (-high) & mask if rt.party < 2 else None)],
        value.shape, True, value.device)
    # The second target is -U-1 == ~U in the high-bit ring, so the
    # already-shared first pair supplies it without another joint input.
    target1 = rt.add(_map(a, lambda part: (part >> width) & mask),
                     rt.public(mask, like=a, binary=True))
    if carry_impl is not None:
        carry = carry_impl(rt, a, b, [width])[0]
    elif value.device.type == 'cuda':
        from ..gpu.bits.carry_tree_swar import carry_bits
        carry = carry_bits(rt, a, b, [width])[0]
    else:
        carry = _low_carry(rt, a, b, width)
    high_b = _map(b, lambda part: (part >> width) & mask)
    equal_words = Share(tuple(torch.stack((first ^ other, second ^ other))
                              for first, second, other in zip(target0.parts, target1.parts, high_b.parts)), True)
    equal_words = rt.add(equal_words, rt.public(mask, like=equal_words, binary=True))
    equal = _all_bits(rt, equal_words, 64-width)
    delta = rt.add(equal[0], equal[1])
    packed_carry, count = _pack_bits(rt, carry)
    packed_delta, _ = _pack_bits(rt, delta)
    chosen = _unpack_bits(rt, rt.mul(packed_carry, packed_delta), count, value.shape)
    valid = rt.add(equal[0], chosen)
    invalid = rt.add(valid, rt.public(1, like=valid, binary=True))
    if binary:
        return invalid
    if convert_impl is not None:
        return convert_impl(rt, _map(invalid, lambda part: part.unsqueeze(0)), [1])
    if value.device.type == 'cuda':
        from ..gpu.bits.compact_conversion_gpu import weighted_b2a_gpu
        return weighted_b2a_gpu(rt, _map(invalid, lambda part: part.unsqueeze(0)), [1])
    return b2a(rt, invalid, bits=1, signed=False)
