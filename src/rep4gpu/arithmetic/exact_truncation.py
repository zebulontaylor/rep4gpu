"""Exact truncation by converting addition carries rather than every output bit."""

import torch

from .conversion import _and_pairs, _components, _map, _stack, b2a, sign_bit
from .compact_conversion import weighted_b2a, b2a as compact_b2a
from ..protocol.runtime import Share


def sign_bit_packed(rt, x, *, carry_impl=None, convert_impl=None):
    """Extract the signed-ring sign from arithmetic or binary sharing."""
    from ..gpu.bits.carry_tree_gpu import carry_bits
    if x.binary:
        sign = Share(tuple((part >> (rt.bits-1)) & 1 for part in x.parts), True)
        if convert_impl is None:
            return compact_b2a(rt, sign, bits=1, signed=False)
        planes = Share(tuple(part.unsqueeze(0) for part in sign.parts), True)
        return convert_impl(rt, planes, [1])
    pair = rt._normalize(x.parts[1] + x.parts[2 if rt.party % 2 == 0 else 0])
    a, b = rt.joint_input_batch(
        [(0, 1, 3, 2, pair if rt.party < 2 else None),
         (2, 3, 1, 0, pair if rt.party >= 2 else None)],
        x.shape, True, x.device)
    carry = (carry_impl or carry_bits)(rt, a, b, [rt.bits-1])[0]
    if convert_impl is not None:
        import triton
        from ..gpu.bits.carry_tree_gpu import _sign
        result = torch.empty((3, *x.shape), dtype=torch.int64, device=x.device)
        count = x.parts[0].numel()
        if count:
            _sign[(triton.cdiv(count, 256), 3)](*a.parts, *b.parts, *carry.parts,
                                              result, count, rt.bits, 256)
        planes = Share(tuple(part.unsqueeze(0) for part in result.unbind(0)), True)
        return convert_impl(rt, planes, [1])
    sign = Share(tuple(((left >> (rt.bits-1)) & 1) ^ ((right >> (rt.bits-1)) & 1) ^ bit
                       for left, right, bit in zip(a.parts, b.parts, carry.parts)), True)
    return compact_b2a(rt, sign, bits=1, signed=False)


def _binary_add_carries(rt, a, b):
    """Return a+b modulo the ring and the carry-out bit at every position."""
    original = propagate = rt.add(a, b)
    generate = rt.mul(a, b)
    for distance in (1 << i for i in range(rt.bits.bit_length()-1)):
        gp, pp = _and_pairs(rt, [(propagate, rt.lshift(generate, distance)),
                                 (propagate, rt.lshift(propagate, distance))])
        generate = rt.add(generate, gp)
        propagate = pp
    result = rt.add(original, rt.lshift(generate, 1))
    if rt.bits < 64:
        result = _map(result, lambda t: t & ((1 << rt.bits) - 1))
    return result, generate


def _validate(rt, x, shift):
    if x.binary:
        raise ValueError('exact truncation requires arithmetic shares')
    if not isinstance(shift, int) or not 0 <= shift < rt.bits:
        raise ValueError(f'shift must be an integer in [0, {rt.bits-1}]')


def _four_component_trunc_and_sign(rt, x, shift):
    a, b, c, d = _components(rt, x, True)
    pairs, pair_generate = _binary_add_carries(rt, _stack([a, c]), _stack([b, d]))
    word, final_generate = _binary_add_carries(rt, pairs[0], pairs[1])
    def bit(value, position):
        return _map(value, lambda t: (t >> position) & 1)
    generates = [pair_generate[0], pair_generate[1], final_generate]
    # Carries into shift affect the quotient by one; ring carries remove a
    # whole unsigned ring. The final sign supplies two's-complement extension.
    binary_bits = _stack([bit(g, shift-1) for g in generates] +
                         [bit(g, rt.bits-1) for g in generates] +
                         [bit(word, rt.bits-1)])
    arithmetic_bits = b2a(rt, binary_bits, bits=1, signed=False)
    high_mask = (1 << (rt.bits-shift))-1
    base = rt._share(tuple((part >> shift) & high_mask for part in x.parts))
    carry_low = rt.add(rt.add(arithmetic_bits[0], arithmetic_bits[1]), arithmetic_bits[2])
    carry_high = rt.add(rt.add(arithmetic_bits[3], arithmetic_bits[4]), arithmetic_bits[5])
    carry_high = rt.add(carry_high, arithmetic_bits[6])
    result = rt.sub(rt.add(base, carry_low), rt.lshift(carry_high, rt.bits-shift))
    return result, arithmetic_bits[6]


def _trunc_and_sign(rt, x, shift, with_sign=True, carry_impl=None, convert_impl=None):
    # Each pair is already jointly known by two parties in the input sharing.
    pair = rt._normalize(x.parts[1] + x.parts[2 if rt.party % 2 == 0 else 0])
    first = pair if rt.party < 2 else None
    second = pair if rt.party >= 2 else None
    tuples = ((0, 1, 3, 2), (2, 3, 1, 0))
    a, b = rt.joint_input_batch(
        [(*tuples[0], first), (*tuples[1], second)], x.shape, True, x.device)
    high_mask = (1 << (rt.bits-shift))-1
    upper = (pair >> shift) & high_mask
    high_a, high_b = rt.joint_input_batch(
        [(*tuples[0], upper if rt.party < 2 else None),
         (*tuples[1], upper if rt.party >= 2 else None)],
        x.shape, False, x.device)
    if x.device.type == 'cuda':
        from ..gpu.bits.carry_tree_gpu import truncation_bits
        selected = _stack((carry_impl or truncation_bits)(rt, a, b, shift))
    else:
        word, generate = _binary_add_carries(rt, a, b)
        def bit(value, position):
            return _map(value, lambda t: (t >> position) & 1)
        selected = _stack([bit(generate, shift-1), bit(generate, rt.bits-1),
                           bit(word, rt.bits-1)])
    correction_weights = [1, -(1 << (rt.bits-shift)), -(1 << (rt.bits-shift))]
    convert = convert_impl or weighted_b2a
    if with_sign:
        converted = convert(rt, selected, [correction_weights, [0, 0, 1]])
        correction, sign = converted[0], converted[1]
    else:
        correction, sign = convert(rt, selected, correction_weights), None
    return rt.add(rt.add(high_a, high_b), correction), sign


def trunc_four_component(rt, x, shift):
    """Reference exact truncation with three adds and seven converted bits."""
    _validate(rt, x, shift)
    return x if shift == 0 else _four_component_trunc_and_sign(rt, x, shift)[0]


def trunc(rt, x, shift, *, carry_impl=None, convert_impl=None):
    """Exact signed floor division over the full 32-bit or 64-bit ring."""
    _validate(rt, x, shift)
    return x if shift == 0 else _trunc_and_sign(
        rt, x, shift, with_sign=False, carry_impl=carry_impl, convert_impl=convert_impl)[0]


def trunc_bounded(rt, x, shift):
    """Exact floor under the caller-established strict quarter-ring bound.

    The reconstructed signed input must satisfy ``abs(x) < 2**(bits-2)``.
    This function does not establish that bound or accept arbitrary ring words;
    use ``trunc`` when no public proof or prior private check supplies it.
    Shares remain in the original ring, and no intermediate value is opened.

    Adding 2**(bits-2) puts the reconstructed value below the signed midpoint.
    If its two unsigned pair sums are u,v, their ring wrap is then exactly
    msb(u) OR msb(v). Absorb both individual MSBs into the arithmetic upper
    inputs; only the low carry and the product of those MSBs need conversion.
    """
    _validate(rt, x, shift)
    if shift > rt.bits-2:
        raise ValueError('bounded truncation shift must not exceed bits-2')
    if shift == 0 or x.parts[0].numel() == 0:
        return x
    offset = 1 << (rt.bits-2)
    shifted = rt.add(x, rt.public(offset, like=x))
    pair = rt._normalize(shifted.parts[1] + shifted.parts[2 if rt.party % 2 == 0 else 0])
    high_mask = (1 << (rt.bits-shift))-1
    upper = ((pair >> shift) & high_mask) - (((pair >> (rt.bits-1)) & 1) << (rt.bits-shift))
    high_a, high_b = rt.joint_input_batch(
        [(0,1,3,2,upper if rt.party<2 else None),
         (2,3,1,0,upper if rt.party>=2 else None)],
        x.shape, False, x.device)
    if x.device.type == 'cuda':
        from ..gpu.bits.bounded_exact_truncation_gpu import low_carry_and_top_product
        from ..gpu.bits.compact_conversion_gpu import weighted_b2a_gpu
        low, top = low_carry_and_top_product(rt, pair, x.shape, shift)
        convert = weighted_b2a_gpu
    else:
        # The CPU path is an integer-protocol reference; the GPU path packs
        # only the required bits before sharing and prunes the carry tree.
        a,b = rt.joint_input_batch(
            [(0,1,3,2,pair if rt.party<2 else None),
             (2,3,1,0,pair if rt.party>=2 else None)],
            x.shape, True, x.device)
        _, generate = _binary_add_carries(rt, a, b)
        low = _map(generate, lambda part: (part >> (shift-1)) & 1)
        top = rt.mul(_map(a, lambda part: (part >> (rt.bits-1)) & 1),
                     _map(b, lambda part: (part >> (rt.bits-1)) & 1))
        convert = weighted_b2a
    weight = 1 << (rt.bits-shift)
    if weight == 1 << 63:
        weight = -(1 << 63)
    correction = convert(rt, _stack([low, top]), [1, weight])
    result = rt.add(rt.add(high_a, high_b), correction)
    return rt.sub(result, rt.public(offset >> shift, like=result))


def trunc_relu(rt, x, shift, *, carry_impl=None, convert_impl=None):
    """Exact signed floor division followed by ReLU, with three B2A bits."""
    _validate(rt, x, shift)
    result, sign = ((x, sign_bit(rt, x)) if shift == 0 else
                    _trunc_and_sign(rt, x, shift, carry_impl=carry_impl, convert_impl=convert_impl))
    return rt.mul(result, rt.sub(rt.public(1, like=x), sign))
