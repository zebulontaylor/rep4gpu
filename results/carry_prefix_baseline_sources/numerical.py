"""Bounded fixed-point evaluation of exp, reciprocal, erf, and activations.

All domains describe decoded inputs and are caller obligations. No method opens
shares, clips inputs, or converts their ring. Only public configuration controls
iteration counts. Each multiplication uses the runtime's truncation policy.

Range argument for Q20..Q24 in ring64: with S=2**f, exp reduces |x|<=9
by 16, so |z|<=0.562501. Its Horner accumulators have magnitude below 2,
and its four squares finish below 8105. Thus every untruncated product is
below 8105*S*S < 2**62, even allowing floor/ceil rounding at every step.
Reciprocal's initial y=1/2 and six Newton steps keep |y|<1.001 and
|x*y|<2.003. The erf polynomial has Horner intermediates below 5 and its
argument t is below 1.001; abs(x)**2 is below 9.001. Activation products
have magnitude below 17. These bounds also satisfy strict quarter-ring
probabilistic truncation requirements. They concern reconstructed integers;
individual random shares can occupy the entire ring.

The erf evaluator uses Abramowitz and Stegun 7.1.26, whose real-arithmetic
absolute approximation error is <=1.5e-7. Fixed-point rounding adds error;
that reference bound is not a bound for the integer implementation.
Source: https://digital.library.unt.edu/ark:/67531/metadc40302/m1/315/ocr/
"""

from __future__ import annotations

import math

from .activations import FixedPoint

ERF_P = 0.3275911
ERF_COEFFICIENTS = (0., 0.254829592, -0.284496736, 1.421413741,
                    -1.453152027, 1.061405429)
EXP_COEFFICIENTS = tuple(1 / math.factorial(i) for i in range(13))


class NumericalFixedPoint(FixedPoint):
    """Evaluate numerical formulas in a declared ring64 Q20..Q24 domain.

    Encodings must already use this scale and ring. The class cannot discover
    a secret input's range or scale. Binary shares are rejected. Public domain
    constants are fixed so unsupported configurations fail before execution.
    """

    def __init__(self, runtime, fractional_bits=24):
        if getattr(runtime, 'bits', None) != 64:
            raise ValueError('numerical evaluation requires a ring64 runtime')
        if type(fractional_bits) is not int or not 20 <= fractional_bits <= 24:
            raise ValueError('numerical fractional_bits must be in [20,24]')
        super().__init__(runtime, fractional_bits)

    @staticmethod
    def _arithmetic(x):
        if getattr(x, 'binary', False):
            raise ValueError('numerical evaluation requires arithmetic shares')

    def _negative(self, x):
        return self.runtime.mul_public(x, -1)

    def _abs_sign(self, x):
        self._arithmetic(x)
        sign = self.runtime.sign_bit(x)
        return self._select(sign, self._negative(x), x), sign

    def constant(self, value, like):
        """Create a public scalar without copying a host tensor during capture."""
        if not math.isfinite(float(value)):
            raise ValueError('numerical constants must be finite')
        return self.runtime.public(round(float(value)*self.scale), like=like)

    def _public_mul(self, x, coefficient):
        encoded = round(float(coefficient)*self.scale)
        return self.runtime.trunc(self.runtime.mul_public(x, encoded), self.fractional_bits)

    def polynomial(self, x, coefficients):
        """Evaluate quantized coefficients using capture-safe Horner steps.

        Leading coefficients that encode as zero contribute exact zero under
        either truncation policy, so their Horner stages need no communication.
        """
        if not coefficients or not all(math.isfinite(float(c)) for c in coefficients):
            raise ValueError('polynomial requires finite nonempty coefficients')
        coefficients = tuple(coefficients)
        while len(coefficients) > 1 and round(float(coefficients[-1])*self.scale) == 0:
            coefficients = coefficients[:-1]
        if len(coefficients) == 1:
            return self.constant(coefficients[0], x)
        result = self.add(self._public_mul(x, coefficients[-1]),
                          self.constant(coefficients[-2], x))
        for coefficient in reversed(coefficients[:-2]):
            result = self.add(self.mul(result, x), self.constant(coefficient, x))
        return result

    def exp(self, x):
        """Approximate exp(x) for -9 <= x <= 9 with a fixed schedule.

        Degree-12 Taylor coefficients at x/16 precede four squarings. At Q20
        only degrees through nine encode nonzero; Q21..Q24 retain degree ten.
        Before coefficient quantization, the Taylor remainder is below 2e-13
        on the reduced interval. Quantization and repeated rounding dominate
        error, especially for large positive x; this is not correctly rounded exp.
        """
        self._arithmetic(x)
        reduced = self.runtime.trunc(x, 4)
        result = self.polynomial(reduced, EXP_COEFFICIENTS)
        for _ in range(4):
            result = self.mul(result, result)
        return result

    def reciprocal(self, x):
        """Approximate 1/x for 1 <= x <= 2 using six Newton steps.

        Starting with y=1/2 gives |1-x*y|<=1/2. Exact arithmetic squares
        that residual each iteration, so the final residual is <=2**-64.
        Integer truncation determines the remaining fixed-point error.
        """
        self._arithmetic(x)
        y = self.constant(0.5, x)
        two = self.constant(2., x)
        for _ in range(6):
            y = self.mul(y, self.sub(two, self.mul(x, y)))
        return y

    def erf(self, x):
        """Approximate erf(x) for -3 <= x <= 3 with A&S 7.1.26.

        For a=abs(x), t=1/(1+p*a), evaluate sign(x)*(1-P(t)*exp(-a*a)).
        The rational evaluation uses secret exp and reciprocal primitives;
        the polynomial approximates erf itself rather than GELU.
        """
        absolute, sign = self._abs_sign(x)
        denominator = self.add(self.constant(1., x), self._public_mul(absolute, ERF_P))
        t = self.reciprocal(denominator)
        tail = self.mul(self.polynomial(t, ERF_COEFFICIENTS),
                        self.exp(self._negative(self.mul(absolute, absolute))))
        positive = self.sub(self.constant(1., x), tail)
        return self._select(sign, self._negative(positive), positive)

    def silu(self, x):
        """Evaluate x/(1+exp(-x)) stably for -4 <= x <= 4."""
        absolute, sign = self._abs_sign(x)
        exponential = self.exp(self._negative(absolute))
        inverse = self.reciprocal(self.add(self.constant(1., x), exponential))
        sigmoid = self._select(sign, self.mul(exponential, inverse), inverse)
        return self.mul(x, sigmoid)

    def gelu(self, x):
        """Evaluate x*(1+erf(x/sqrt(2)))/2 for -4 <= x <= 4."""
        self._arithmetic(x)
        error_function = self.erf(self._public_mul(x, 1 / math.sqrt(2)))
        factor = self.runtime.trunc(self.add(self.constant(1., x), error_function), 1)
        return self.mul(x, factor)

    def swiglu(self, gate, value):
        """Return silu(gate)*value; each decoded input lies in [-4,4]."""
        self._arithmetic(value)
        return self.mul(self.silu(gate), value)
