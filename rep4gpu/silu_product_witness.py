"""Opt-in batched product claims for the exact direct-output SiLU contract.

The numerical schedule and accepted input/output domains match the default
SiLU verifier. ProductWitnessBatch checks products under its challenge PRG
assumption; input and output ranges retain their ordinary exact checks.
"""
from __future__ import annotations

from .numerical import NumericalFixedPoint
from .silu_exp import silu_exp_negative
from .witness_activations import WitnessProgram, _same_shape
from .witness_prover import _PlainFloor, _input, _nearest_quotient

PRODUCT_NAMES = tuple(f'product_{index}' for index in range(10))


def silu_product_witness_program():
    """Verify ten supplied raw products without changing the SiLU output.

    All claims enter through checked party inputs before evaluation. Every
    product is bound to its operands by the batch verifier before outputs are
    released; hence successful verification establishes the same intermediate
    bounds used by the default exact-floor exponential evaluator.
    """
    def prepare(session,witnesses):
        y=witnesses['output']
        session._arithmetic(y)
        for name in PRODUCT_NAMES:
            session._arithmetic(witnesses[name])
            _same_shape(witnesses[name],y.shape,name)
        session.upper_zero(session.runtime.add(y,session.runtime.public(session.scale,like=y)),
                           session.fp.fractional_bits+5)
        return witnesses

    def recompute(session,inputs,witnesses):
        from .product_witness import ProductWitnessBatch
        if set(inputs) != {'x'}:
            raise ValueError('SiLU product program requires only x')
        rt,fp=session.runtime,session.fp
        x,y=inputs['x'],witnesses['output']
        _same_shape(y,x.shape,'output')
        batch=ProductWitnessBatch(rt)
        claims=iter(witnesses[name] for name in PRODUCT_NAMES)
        def multiply(a,b):
            return batch.add(a,b,next(claims))
        sign,absolute=session.signed_absolute_input(x,16*session.scale)
        exponential=silu_exp_negative(fp,fp._negative(absolute),bounded_floor=True,multiply=multiply)
        denominator=fp.add(fp.constant(1,exponential),exponential)
        # select(sign,-E,S) = S + sign*(-E-S).
        delta=rt.sub(fp._negative(exponential),fp.constant(1,exponential))
        multiplier=rt.add(fp.constant(1,exponential),multiply(sign,delta))
        numerator=multiply(absolute,multiplier)
        error=rt.sub(multiply(y,denominator),numerator)
        doubled=rt.mul_public(error,2)
        session.upper_zero(rt.add(doubled,denominator),fp.fractional_bits+3)
        session.bounded_nonnegative(rt.sub(denominator,doubled),fp.fractional_bits+4)
        batch.verify(session)
        return y

    # Product claims are arbitrary ring words until the batch identities pass.
    # The custom preparation deliberately leaves them unclamped.
    bounds={'output':(-16,16),**{name:(-(1 << 30),1 << 30) for name in PRODUCT_NAMES}}
    return WitnessProgram('silu_product_witness',bounds,recompute,prepare)


def silu_product_witnesses(inputs, fractional_bits=24):
    """Build exact raw product claims and the nearest-even output in plaintext."""
    if set(inputs) != {'x'}:
        raise ValueError('SiLU product builder requires only x')
    class Trace(_PlainFloor):
        def __init__(self): self.products=[]
        def mul(self,a,b):
            result=super().mul(a,b)
            self.products.append(result.clone())
            return result
        def mul_public(self,a,value):
            return a*self.public(value,like=a)
    rt=Trace()
    fp=NumericalFixedPoint(rt,fractional_bits)
    x=_input(inputs['x'],fp.scale,-16,16)
    exponential=silu_exp_negative(fp,-x.abs())
    sign=(x<0).long()
    multiplier=fp.scale+rt.mul(sign,-exponential-fp.scale)
    numerator=rt.mul(x.abs(),multiplier)
    denominator=fp.scale+exponential
    y=_nearest_quotient(numerator,denominator)
    rt.mul(y,denominator)
    assert len(rt.products)==len(PRODUCT_NAMES)
    return {'output':y,**dict(zip(PRODUCT_NAMES,rt.products))}
