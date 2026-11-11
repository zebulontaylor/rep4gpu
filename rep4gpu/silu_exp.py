"""Bounded exp(-abs(x)) used by the direct-output SiLU witness relation.

The caller establishes -16*S <= negative_abs <= 0, where S=2**f. Argument
reduction and quadratic Horner steps use exact floor; the four squares round
to nearest with ties upward. This rounding is part of the verifier/prover contract.
"""

from __future__ import annotations


# Ascending coefficients from Remez exchange on [-1, 0]. The reproducible fit
# and the Q24 SiLU accuracy checks are in benchmarks/silu_minimax_quality.py.
EXP_REMEZ6 = (
    0.99999998518003464347477931596879097,
    0.99999852097101613406699590732047738,
    0.49997587268881779391991995849975992,
    0.16651849451065545397226961064595529,
    0.04122972494751453183612013702504348,
    0.00766001339840198240238012498521745,
    0.00085090205511427933157068565072565,
)
# Lower C1 by three Q24 units to compensate for the quadratic schedule's
# floor bias. The exhaustive benchmark checks all signed Q24 inputs.
EXP_REMEZ6_Q24 = (16777216, 16777188, 8388203, 2793717, 691720, 128514, 14276)


def encoded_coefficients(fractional_bits):
    """Return degree-six coefficients; the 6.41e-7 accuracy target is Q24 only."""
    if type(fractional_bits) is not int or not 20 <= fractional_bits <= 24:
        raise ValueError('SiLU exponential fractional_bits must be in [20,24]')
    if fractional_bits == 24:
        return EXP_REMEZ6_Q24
    scale = 1 << fractional_bits
    return tuple(round(c * scale) for c in EXP_REMEZ6)


def silu_exp_negative(fp, negative_abs, *, bounded_floor=False, multiply=None):
    """Compute encoded exp(negative_abs/S) for a caller-bounded input.

    Requires ring64 arithmetic and exact-floor truncation. Write T for the
    reduced input and U=floor(T*T/S). Then 0<=U<=-T<=S. The coefficients
    give 0<=Q2<=C4+C6 and 0<=Q1<=C2+C4+C6<=C1, so
    -C1*S<=C1*T+U*Q1<=0. With C0=S, the final polynomial
    lies in [0,S]. Each nearest square preserves this interval.
    All floor numerators have magnitude <=S*S+S/2<2**62.

    bounded_floor selects the quarter-ring floor protocol for Rep4 runtimes.
    Plaintext prover runtimes use their ordinary exact shift in either mode.
    The fixed schedule has seven secret products and nine truncations,
    including argument reduction by 16 and the public coefficient products.
    A custom ``multiply`` must return products bound to its two operands; an
    opt-in witness verifier can establish those identities before acceptance.
    """
    rt = fp.runtime
    multiply = rt.mul if multiply is None else multiply
    if getattr(rt, 'bits', None) != 64:
        raise ValueError('SiLU exponential requires a ring64 runtime')
    if getattr(rt, 'rounding', 'floor') != 'floor':
        raise ValueError('SiLU exponential requires exact-floor truncation')
    if getattr(negative_abs, 'binary', False):
        raise ValueError('SiLU exponential requires arithmetic shares')
    f = fp.fractional_bits
    coefficients = encoded_coefficients(f)
    scale = 1 << f

    if bounded_floor and hasattr(rt, 'joint_input_batch'):
        from .exact_truncation import trunc_bounded

        def floor(value, shift):
            return trunc_bounded(rt, value, shift)
    else:
        floor = rt.trunc

    reduced = floor(negative_abs, 4)
    c0, c1, c2, c3, c4, c5, c6 = coefficients
    square = floor(multiply(reduced, reduced), f)
    numerator = rt.add(rt.mul_public(reduced, c5), rt.mul_public(square, c6))
    q2 = rt.add(rt.public(c4, like=reduced), floor(numerator, f))
    numerator = rt.add(rt.mul_public(reduced, c3), multiply(square, q2))
    q1 = rt.add(rt.public(c2, like=reduced), floor(numerator, f))
    numerator = rt.add(rt.mul_public(reduced, c1), multiply(square, q1))
    result = rt.add(rt.public(c0, like=reduced), floor(numerator, f))
    for _ in range(4):
        product = multiply(result, result)
        rounded = rt.add(product, rt.public(scale // 2, like=product))
        result = floor(rounded, f)
    return result
