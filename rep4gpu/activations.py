"""Fixed-point tensor operations composed from Rep4 ring protocols."""

from __future__ import annotations

import math
from dataclasses import dataclass
from collections.abc import Sequence

import torch


# Coefficients use t=x/4; this avoids losing small high-degree coefficients.
_GELU = (0., 2., 6.14346484935911, 0., -13.515904886692134,
         0., 20.612324834637704, 0., -16.487688254545997, 0., 5.252040339124279)
_SILU = (0., 2., 3.961774507163562, 0., -4.773731192674444,
         0., 5.2706358068456, 0., -3.571599205913506, 0., 1.0417182672910728)


_QUADRATIC_OFFSETS = {'relu': 1/4, 'gelu': 951/4096, 'silu': 427/4096}
_LOCAL_QUADRATICS = {'gelu': (3,3), 'silu': (1,2)}


@dataclass(frozen=True)
class QuadraticPlan:
    """Public integer coefficients for a bounded quadratic approximation.

    For scale S=2**fractional_bits, u is the input rescaled to S/2**prescale.
    The output is trunc((quadratic_coeff*u*u + linear_coeff*u + constant_numerator)/2**shift).
    Input and numerator bounds are caller contracts, not secret range checks.
    """
    kind: str
    bound: float
    fractional_bits: int
    ring_bits: int
    prescale: int
    shift: int
    linear_coeff: int
    constant_numerator: int
    input_bound: int
    normalized_bound: int
    numerator_abs_bound: int
    profile: str = 'uniform'
    quadratic_coeff: int = 1
    denominator_power2: int = 3

    @property
    def coefficients(self):
        """Return ascending real coefficients before fixed-point rounding."""
        return (0. if self.profile=='local' else _QUADRATIC_OFFSETS[self.kind],
                1/2,self.quadratic_coeff/(1<<self.denominator_power2))

    def encoded_numerator(self, normalized):
        """Evaluate public plaintext u after checking its declared integer bound."""
        if not isinstance(normalized, torch.Tensor) or normalized.dtype != torch.int64:
            raise TypeError('normalized values must be a plaintext int64 tensor')
        if not bool(((normalized >= -self.normalized_bound) &
                     (normalized <= self.normalized_bound)).all()):
            raise ValueError('normalized input exceeds the quadratic plan bound')
        return self.quadratic_coeff*normalized*normalized + self.linear_coeff*normalized + self.constant_numerator


def quadratic_plan(kind='relu', bound=4., *, fractional_bits=14, bits=32, profile='uniform'):
    """Plan a quadratic on a declared interval contained in [-4,4].

    Offsets c are 1/4 (ReLU), 951/4096 (GELU), and 427/4096 (SiLU).
    ReLU's real polynomial has maximum error 1/4 on [-4,4]. The other
    offsets fit the same interval; these low-degree substitutes need model
    accuracy evaluation. They can produce negative outputs and nonzero f(0).

    The uniform profile uses the coefficients above. The local profile is
    zero-preserving: GELU uses x/2+3*x*x/8 and SiLU uses x/2+x*x/4.
    These fixed local approximations have much larger tail errors: about 4.001
    and 2.072 respectively at |x|=4. They do not provide uniform accuracy.

    The smallest prescale keeps every numerator strictly inside the signed
    quarter ring, including either rounding outcome of the input truncation.
    This also prevents square overflow. The fixed-point input contract is
    abs(signed(X)) <= floor(bound*2**fractional_bits).
    """
    if kind not in _QUADRATIC_OFFSETS:
        raise ValueError('quadratic kind must be relu, gelu, or silu')
    if profile not in ('uniform','local'):
        raise ValueError('quadratic profile must be uniform or local')
    if profile=='local' and kind not in _LOCAL_QUADRATICS:
        raise ValueError('local quadratic profile supports gelu and silu')
    quadratic_coeff,denominator = _LOCAL_QUADRATICS[kind] if profile=='local' else (1,3)
    if not isinstance(fractional_bits, int) or not 2 <= fractional_bits <= 62:
        raise ValueError('bounded quadratics require fractional_bits in [2,62]')
    if bits not in (32,64):
        raise ValueError('quadratic ring bits must be 32 or 64')
    bound = float(bound)
    if not math.isfinite(bound) or not 0 < bound <= 4:
        raise ValueError('quadratic bound must be finite and in (0,4]')
    scale = 1 << fractional_bits
    input_bound = math.floor(bound*scale)
    quarter = 1 << (bits-2)
    if input_bound >= quarter:
        raise ValueError('quadratic input bound must fit strictly inside the quarter ring')
    constant = 0 if profile=='local' else round(_QUADRATIC_OFFSETS[kind]*scale)
    for prescale in range((fractional_bits+denominator-1)//2+1):
        shift = fractional_bits+denominator-2*prescale
        if shift < 1:
            break
        normalized_bound = (input_bound+(1<<prescale)-1)>>prescale
        linear = 1 << (prescale+shift-1)
        constant_numerator = constant << shift
        numerator_bound = quadratic_coeff*normalized_bound**2+normalized_bound*linear+abs(constant_numerator)
        if numerator_bound < quarter and max(shift,fractional_bits+prescale) < bits-1:
            return QuadraticPlan(kind,bound,fractional_bits,bits,prescale,shift,
                                 linear,constant_numerator,input_bound,
                                 normalized_bound,numerator_bound,profile,quadratic_coeff,denominator)
    raise ValueError('quadratic precision and bound cannot fit this ring')


class FixedPoint:
    """Interpret ring shares as signed integers scaled by ``2**fractional_bits``.

    Multiplication follows the runtime's truncation policy: exact floor by
    default, or bounded probabilistic floor/ceil when selected. Intermediate
    signed products must fit in the ring; overflow wraps before truncation.
    """

    def __init__(self, runtime, fractional_bits: int = 16):
        if not isinstance(fractional_bits, int) or not 0 <= fractional_bits < 63:
            raise ValueError("fractional_bits must be an integer in [0, 62]")
        self.runtime = runtime
        self.fractional_bits = fractional_bits
        self.scale = 1 << fractional_bits

    def encode(self, value, *, device=None):
        """Encode plaintext with nearest-even rounding before secret input."""
        values = torch.as_tensor(value, dtype=torch.float64, device=device)
        scaled = torch.round(values * self.scale)
        if not bool(torch.all(torch.isfinite(scaled) & (scaled >= -(2.**63)) & (scaled < 2.**63))):
            raise ValueError("encoded value must fit in signed int64")
        return scaled.to(torch.int64)

    def decode(self, opened):
        """Decode explicitly opened plaintext; this method accepts no shares."""
        if not isinstance(opened, torch.Tensor):
            raise TypeError("decode expects an opened torch.Tensor")
        return opened.to(torch.float64) / self.scale

    def constant(self, value, like):
        return self.runtime.public(self.encode(value, device=like.device), like=like)

    def add(self, x, y):
        return self.runtime.add(x, y)

    def sub(self, x, y):
        return self.runtime.sub(x, y)

    def mul(self, x, y):
        return self.runtime.trunc(self.runtime.mul(x, y), self.fractional_bits)

    def matmul(self, x, y):
        """Accumulate unscaled products, then truncate the sum once."""
        return self.runtime.trunc(self.runtime.matmul(x, y), self.fractional_bits)

    def matmul_relu(self, x, y, bias=None):
        """Add a fixed-point shared bias before one fused truncation and ReLU."""
        accumulated = self.runtime.matmul(x, y)
        if bias is not None:
            accumulated = self.runtime.add(
                accumulated, self.runtime.mul_public(bias, self.scale))
        return self.runtime.trunc_relu(accumulated, self.fractional_bits)

    def quadratic_plan(self, kind='relu', bound=4., *, profile='uniform'):
        """Return public coefficients and integer bounds for bounded_quadratic."""
        return quadratic_plan(kind,bound,fractional_bits=self.fractional_bits,
                              bits=self.runtime.bits,profile=profile)

    def _quadratic_normalized(self, normalized, plan):
        rt = self.runtime
        square = rt.mul(normalized,normalized)
        if plan.quadratic_coeff!=1:
            square = rt.mul_public(square,plan.quadratic_coeff)
        numerator = rt.add(square,rt.mul_public(normalized,plan.linear_coeff))
        if plan.constant_numerator:
            numerator = rt.add(numerator,rt.public(plan.constant_numerator,like=normalized))
        return rt.trunc(numerator,plan.shift)

    def bounded_quadratic(self, x, *, kind='relu', bound=4., profile='uniform'):
        """Approximate an activation with one secret square and no comparisons.

        The caller establishes the encoded input bound from quadratic_plan.
        This method does not clip or privately check inputs. Input rescaling
        reduces precision by prescale bits; runtime.trunc chooses floor or probabilistic
        rounding for both rescalings. Existing relu/gelu/silu methods are unchanged.
        """
        if getattr(x,'binary',False):
            raise ValueError('bounded quadratics require arithmetic shares')
        plan = self.quadratic_plan(kind,bound,profile=profile)
        normalized = self.runtime.trunc(x,plan.prescale) if plan.prescale else x
        return self._quadratic_normalized(normalized,plan)

    def matmul_quadratic(self, x, weight, bias=None, *, kind='relu', bound=4., profile='uniform'):
        """Accumulate a matrix product and evaluate a bounded quadratic.

        Bias uses the same scale as x. The signed untruncated accumulator must
        fit the ring and satisfy abs(accumulator) <= plan.input_bound*scale.
        Probabilistic rounding additionally requires the strict quarter ring. One input truncation includes plan.prescale,
        avoiding a separate rescaling before the secret square.
        """
        plan = self.quadratic_plan(kind,bound,profile=profile)
        accumulated = self.runtime.matmul(x,weight)
        if bias is not None:
            accumulated = self.runtime.add(accumulated,self.runtime.mul_public(bias,self.scale))
        normalized = self.runtime.trunc(accumulated,self.fractional_bits+plan.prescale)
        return self._quadratic_normalized(normalized,plan)

    def polynomial(self, x, coefficients: Sequence[float]):
        """Evaluate ascending-order coefficients using fixed-point Horner steps."""
        if len(coefficients) == 0:
            raise ValueError("polynomial requires at least one coefficient")
        if not all(math.isfinite(float(c)) for c in coefficients):
            raise ValueError("polynomial coefficients must be finite")
        if len(coefficients) == 1:
            return self.constant(coefficients[0], x)
        # The first Horner product has a public operand and needs no resharing.
        leading = self.encode(coefficients[-1], device=x.device)
        result = self.runtime.trunc(
            self.runtime.mul_public(x, leading), self.fractional_bits)
        result = self.add(result, self.constant(coefficients[-2], x))
        for coefficient in reversed(coefficients[:-2]):
            result = self.add(self.mul(result, x), self.constant(coefficient, x))
        return result

    def relu(self, x):
        """Return exact max(x,0), including the minimum signed ring value."""
        keep = self.runtime.sub(self.runtime.public(1, like=x), self.runtime.sign_bit(x))
        return self.runtime.mul(x, keep)

    def _select(self, bit, when_true, when_false):
        return self.runtime.add(when_false, self.runtime.mul(bit, self.runtime.sub(when_true, when_false)))

    def clamp(self, x, lower: float, upper: float):
        """Clamp with secret comparisons and public fixed-point endpoints."""
        if not math.isfinite(lower) or not math.isfinite(upper) or lower > upper:
            raise ValueError("clamp requires finite lower <= upper")
        lo, hi = self.constant(lower, x), self.constant(upper, x)
        below = self.runtime.lt_public(x, int(round(lower * self.scale)))
        above = self.runtime.lt_public(x, int(round(upper * self.scale)))
        return self._select(below, lo, self._select(above, x, hi))

    def _smooth(self, x, coefficients):
        below = self.runtime.lt_public(x, -4 * self.scale)
        above = self.runtime.lt_public(x, 4 * self.scale)
        lo, hi = self.constant(-4., x), self.constant(4., x)
        clipped = self._select(below, lo, self._select(above, x, hi))
        normalized = self.runtime.trunc(clipped, 2)
        square = self.mul(normalized, normalized)
        even = self.polynomial(square, coefficients[2::2])
        middle = self.add(self.runtime.trunc(clipped, 1), self.mul(square, even))
        zero = self.runtime.public(0, like=x)
        return self._select(below, zero, self._select(above, middle, x))

    def gelu(self, x):
        """Degree-ten GELU approximation on [-4,4], with zero/identity tails."""
        return self._smooth(x, _GELU)

    def silu(self, x):
        """Degree-ten SiLU approximation on [-4,4], with zero/identity tails."""
        return self._smooth(x, _SILU)
