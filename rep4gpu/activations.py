"""Fixed-point tensor operations composed from Rep4 ring protocols."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch


# Coefficients use t=x/4; this avoids losing small high-degree coefficients.
_GELU = (0., 2., 6.14346484935911, 0., -13.515904886692134,
         0., 20.612324834637704, 0., -16.487688254545997, 0., 5.252040339124279)
_SILU = (0., 2., 3.961774507163562, 0., -4.773731192674444,
         0., 5.2706358068456, 0., -3.571599205913506, 0., 1.0417182672910728)


class FixedPoint:
    """Interpret ring shares as signed integers scaled by ``2**fractional_bits``.

    Multiplication truncates toward negative infinity. Intermediate signed
    products must fit in the runtime's ring; overflow wraps before truncation.
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
