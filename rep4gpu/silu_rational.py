"""Opt-in Q24 rational SiLU with an exhaustively checked numerical schedule.

The approximation covers inclusive [-16,16]. Its six product claims use the
challenge assumptions of ProductWitnessBatch. It changes the numerical
contract from the default exponential-based SiLU verifier.
"""
from __future__ import annotations

from .exact_truncation import trunc_bounded
from .witness_activations import WitnessProgram, _same_shape
from .witness_prover import _input, _nearest_quotient

SCALE=1 << 24
POWER_SCALE=1 << 30
POLYNOMIAL_SCALE=1 << 21
A_COEFFICIENTS=(0,134217728,1108819968,1612275712,395427840)
B_COEFFICIENTS=(2097152,62064384,203923712,129811904,8445632)
PRODUCT_NAMES=tuple(f'product_{i}' for i in range(6))


def rational_silu_plain(encoded):
    """Evaluate nearest-even Q24 outputs with integer NumPy or Torch arrays.

    The caller supplies int64 inputs in inclusive [-16,16]. Every power and
    polynomial reduction rounds to nearest with ties upward. Only the final
    rational division rounds to nearest even.
    """
    z=(encoded*encoded+(1 << 25)) >> 26
    powers=[encoded*0+POWER_SCALE,z]
    for _ in range(3): powers.append((powers[-1]*z+POWER_SCALE//2) >> 30)
    a=(sum(c*p for c,p in zip(A_COEFFICIENTS,powers))+POWER_SCALE//2) >> 30
    b=(sum(c*p for c,p in zip(B_COEFFICIENTS,powers))+POWER_SCALE//2) >> 30
    numerator=encoded*b+2*SCALE*a
    denominator=2*b
    quotient=numerator//denominator
    remainder=numerator-quotient*denominator
    return quotient+((2*remainder>denominator)|((2*remainder==denominator)&((quotient&1)!=0)))


def rational_silu_witnesses(inputs):
    """Build the output and six raw products from plaintext encoded Q24 x."""
    if set(inputs) != {'x'}:
        raise ValueError('rational SiLU requires only x')
    x=_input(inputs['x'],SCALE,-16,16)
    products=[]
    def multiply(a,b):
        result=a*b
        products.append(result.clone())
        return result
    z=(multiply(x,x)+(1 << 25)) >> 26
    powers=[x*0+POWER_SCALE,z]
    for _ in range(3): powers.append((multiply(powers[-1],z)+POWER_SCALE//2) >> 30)
    a=(sum(c*p for c,p in zip(A_COEFFICIENTS,powers))+POWER_SCALE//2) >> 30
    b=(sum(c*p for c,p in zip(B_COEFFICIENTS,powers))+POWER_SCALE//2) >> 30
    numerator=multiply(x,b)+2*SCALE*a
    y=_nearest_quotient(numerator,2*b)
    multiply(y,2*b)
    return {'output':y,**dict(zip(PRODUCT_NAMES,products))}


def rational_silu_program():
    """Verify the rational schedule with inclusive input bounds and Q24 output.

    On valid inputs, 0<=zi<=2**30. Positive coefficients give polynomial
    accumulator bounds below 2**62, including the rounding offsets. B lies
    in [2**21,406342784], so D=2*B is positive and below 2**30.
    The output relation accepts either nearest neighbor at an exact tie; the
    plaintext builder chooses the even neighbor.
    """
    def prepare(session,witnesses):
        y=witnesses['output']
        session._arithmetic(y)
        for name in PRODUCT_NAMES:
            session._arithmetic(witnesses[name])
            _same_shape(witnesses[name],y.shape,name)
        session.upper_zero(session.runtime.add(y,session.runtime.public(SCALE,like=y)),29)
        return witnesses

    def recompute(session,inputs,witnesses):
        from .product_witness import ProductWitnessBatch
        if session.fp.fractional_bits != 24:
            raise ValueError('rational SiLU requires Q24 arithmetic')
        if set(inputs) != {'x'}:
            raise ValueError('rational SiLU requires only x')
        rt=session.runtime
        if getattr(rt,'rounding','floor') != 'floor':
            raise ValueError('rational SiLU requires exact-floor arithmetic')
        x=inputs['x']
        # The first range gives -16*S<=x<48*S. Therefore 16*S-x
        # lies in (-32*S,32*S], inside the signed 31-bit range.
        session.upper_zero(rt.add(x,rt.public(16*SCALE,like=x)),30)
        session.bounded_nonnegative(rt.sub(rt.public(16*SCALE,like=x),x),31)
        y=witnesses['output']
        _same_shape(y,x.shape,'output')
        batch=ProductWitnessBatch(rt)
        claims=iter(witnesses[name] for name in PRODUCT_NAMES)
        def multiply(a,b): return batch.add(a,b,next(claims))
        def nearest(value,shift):
            return trunc_bounded(rt,rt.add(value,rt.public(1 << (shift-1),like=value)),shift)
        z=nearest(multiply(x,x),26)
        powers=[rt.public(POWER_SCALE,like=x),z]
        for _ in range(3): powers.append(nearest(multiply(powers[-1],z),30))
        def polynomial(coefficients):
            total=rt.public(0,like=x)
            for coefficient,power in zip(coefficients,powers):
                total=rt.add(total,rt.mul_public(power,coefficient))
            return nearest(total,30)
        a,b=polynomial(A_COEFFICIENTS),polynomial(B_COEFFICIENTS)
        denominator=rt.mul_public(b,2)
        numerator=rt.add(multiply(x,b),rt.mul_public(a,2*SCALE))
        error=rt.sub(multiply(y,denominator),numerator)
        doubled=rt.mul_public(error,2)
        # Y+S in [0,32*S) prevents wrap in R. Passing the lower check
        # gives U=R+D in [0,2**31). Since 2*D<2**31, 2*D-U lies
        # strictly between -2**31 and 2**31, so its sign needs 32 bits.
        session.upper_zero(rt.add(doubled,denominator),31)
        session.bounded_nonnegative(rt.sub(denominator,doubled),32)
        batch.verify(session)
        return y
    bounds={'output':(-16,16),**{name:(-(1 << 37),1 << 37) for name in PRODUCT_NAMES}}
    return WitnessProgram('silu_rational34',bounds,recompute,prepare)
