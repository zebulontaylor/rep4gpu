"""Check prover-supplied activation witnesses using private Rep4 constraints.

A session keeps recomputed outputs private until one aggregate acceptance bit
is opened. Range checks clamp invalid operands before subsequent arithmetic,
so even arbitrary ring words cannot exploit wraparound in built-in checks.
Program callbacks are trusted protocol code, not code supplied by the prover.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from collections.abc import Callable, Mapping

import torch

from .numerical import NumericalFixedPoint


class InvalidWitness(ValueError):
    """The batch contained an out-of-domain input or an invalid witness."""


@dataclass(frozen=True)
class WitnessProgram:
    """Named witness schema and a trusted recomputation callback.

    Bounds are inclusive decoded fixed-point bounds. The callback receives
    ``(session, inputs, bounded_witnesses)``, registers at least one residual
    constraint, and returns the recomputed share. It must bound inputs and
    intermediate products itself; schema validation alone proves no relation
    between the witnesses and inputs. Witness shapes belong to the callback.
    """
    name: str
    witness_bounds: Mapping[str, tuple[float, float]]
    recompute: Callable

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name or not self.witness_bounds or not callable(self.recompute):
            raise ValueError('a witness program needs a name, schema, and callback')
        bounds = {name: tuple(interval) for name, interval in self.witness_bounds.items()}
        for name, interval in bounds.items():
            if not isinstance(name, str) or not name or len(interval) != 2:
                raise ValueError('witness schema requires named lower/upper bounds')
            lo, hi = interval
            if not math.isfinite(lo) or not math.isfinite(hi) or lo > hi:
                raise ValueError('witness bounds must be finite and ordered')
        object.__setattr__(self, 'witness_bounds', MappingProxyType(bounds))


class WitnessSession:
    """Accumulate checks and release shared results only after acceptance.

    Use the same public call order, schemas, and shapes at every Rep4 party.
    ``finish`` reveals only whether all checks passed, including on rejection.
    It uses an exact bounded count of violation bits, not a random combination
    over the ring. The session is single-use and cannot be reused after failure.
    """

    def __init__(self, runtime, fractional_bits=24):
        self.fp = NumericalFixedPoint(runtime, fractional_bits)
        self.runtime = runtime
        self.scale = self.fp.scale
        self._count = None
        self._count_bound = 0
        self._residuals = 0
        self._outputs = []
        self._closed = False

    def _active(self):
        if self._closed:
            raise RuntimeError('witness session is closed')

    @staticmethod
    def _arithmetic(value):
        if getattr(value, 'binary', False):
            raise ValueError('witness evaluation requires arithmetic shares')
        if math.prod(value.shape) == 0:
            raise ValueError('empty witness tensors are not supported')

    def sum(self, value, dim=None, keepdim=False):
        """Sum arithmetic share components locally; callers bound the sum."""
        self._arithmetic(value)
        return self.runtime._share(p.sum(dim=dim, keepdim=keepdim) for p in value.parts)

    def _violation(self, bit):
        size = math.prod(bit.shape)
        if self._count_bound + size >= 1 << 62:
            raise ValueError('too many witness constraints for a bounded count')
        self._count_bound += size
        total = self.sum(bit)
        self._count = total if self._count is None else self.runtime.add(self._count, total)

    def _interval(self, value, lower, upper):
        self._active()
        self._arithmetic(value)
        if not -(1 << 63) <= lower <= upper < (1 << 63)-1:
            raise ValueError('constraint endpoints must fit signed ring64')
        rt = self.runtime
        below = rt.lt_public(value, lower)
        above = rt.sub(rt.public(1, like=value), rt.lt_public(value, upper+1))
        self._violation(below)
        self._violation(above)
        return below, above

    def bounded(self, value, lower, upper):
        """Register an inclusive decoded range and return a safely clamped share.

        Clamp selections use ring arithmetic and full signed comparisons; their
        outputs equal the endpoints even for the minimum signed ring word.
        The sticky rejection flag prevents a clamped invalid value from passing.
        """
        if not math.isfinite(lower) or not math.isfinite(upper) or lower > upper:
            raise ValueError('bounds must be finite and ordered')
        lo, hi = math.ceil(lower*self.scale), math.floor(upper*self.scale)
        below, above = self._interval(value, lo, hi)
        rt = self.runtime
        return self.fp._select(below, rt.public(lo, like=value),
                               self.fp._select(above, rt.public(hi, like=value), value))

    def residual(self, error, tolerance_lsb=0):
        """Require every encoded residual to lie in [-tolerance,+tolerance].

        The callback must establish a non-wrapping range for the expression
        producing ``error``. This check does not repair an overflowed identity.
        """
        if type(tolerance_lsb) is not int or tolerance_lsb < 0:
            raise ValueError('residual tolerance must be a nonnegative integer')
        self._interval(error, -tolerance_lsb, tolerance_lsb)
        self._residuals += 1

    def nonnegative(self, value):
        """Require a bounded signed expression to be nonnegative.

        As with ``residual``, the callback establishes that the expression did
        not wrap. This supports secret-dependent bounds without opening them.
        """
        self._active()
        self._arithmetic(value)
        self._violation(self.runtime.lt_public(value, 0))
        self._residuals += 1

    def evaluate(self, program, inputs, witnesses):
        """Queue an output; no result is returned before ``finish`` succeeds."""
        self._active()
        try:
            if set(witnesses) != set(program.witness_bounds):
                raise ValueError('witness names must exactly match the program schema')
            safe = {name: self.bounded(witnesses[name], *bounds)
                    for name, bounds in program.witness_bounds.items()}
            before = self._residuals
            result = program.recompute(self, inputs, safe)
            if self._residuals == before:
                raise ValueError('program must register a residual constraint')
            self._arithmetic(result)
            self._outputs.append(result)
        except BaseException:
            self._closed = True
            self._outputs.clear()
            raise

    def finish(self):
        """Open one acceptance bit and return queued shares, or reject the batch."""
        self._active()
        self._closed = True
        if not self._outputs:
            raise ValueError('cannot finish an empty witness session')
        accepted = self.runtime.lt_public(self._count, 1)
        outputs, self._outputs = tuple(self._outputs), []
        if int(self.runtime.open(accepted).item()) != 1:
            raise InvalidWitness('activation witness batch rejected')
        return outputs


def input_witnesses(runtime, program, owner, shapes, values=None):
    """Ingest encoded prover values through the existing checked party input.

    ``owner`` is the Rep4 role receiving prover values. Only that role supplies
    ``values``; all roles provide the same name-to-shape mapping. This helper
    does not implement an external prover transport or authenticate that link.
    """
    names = sorted(program.witness_bounds)
    if set(shapes) != set(names) or (values is not None and set(values) != set(names)):
        raise ValueError('input names must exactly match the witness schema')
    if owner not in range(4):
        raise ValueError('witness input owner must be a Rep4 role')
    if (runtime.party == owner) != (values is not None):
        raise ValueError('only the witness input owner supplies encoded values')
    if values is not None:
        values = {name: torch.as_tensor(values[name], device=runtime.device) for name in names}
        for name, value in values.items():
            if value.dtype != torch.int64 or tuple(value.shape) != tuple(shapes[name]):
                raise ValueError('witness input must be int64 with its declared public shape')
    return {name: runtime.input(owner, None if values is None else values[name], shapes[name])
            for name in names}


def _same_shape(value, expected, name):
    if tuple(value.shape) != tuple(expected):
        raise ValueError(f'{name} has the wrong public shape')


def _checked_reciprocal(session, denominator, witness, maximum=2):
    _same_shape(witness, denominator.shape, 'reciprocal')
    fp, rt, scale = session.fp, session.runtime, session.scale
    denominator = session.bounded(denominator, 1, maximum)
    if maximum*scale*scale >= 1 << 62:
        raise ValueError('reciprocal domain exceeds the raw product bound')
    raw = rt.mul(denominator, witness)
    error = rt.sub(raw, rt.public(scale*scale, like=raw))
    # This proves W is a nearest integer to S²/D, with either tie accepted.
    # D is secret: compare 2*error against +/-D without opening either.
    # maximum<=4096 gives |2*error +/- D|<2**62 for every bounded W.
    doubled = rt.mul_public(error, 2)
    session.nonnegative(rt.add(doubled, denominator))
    session.nonnegative(rt.sub(denominator, doubled))
    return witness


def numerical_program(kind):
    """Replace Newton reciprocal iterations in the good numerical formulas.

    Inputs use ``x`` and, for SwiGLU, ``value``. One ``reciprocal`` witness has
    the same shape as x and is the nearest encoding of the reciprocal of the
    *recomputed fixed-point denominator*. Exponential and erf polynomials are
    still evaluated privately; a witness is not an unconstrained output claim.
    """
    if kind not in ('reciprocal', 'silu', 'gelu', 'swiglu'):
        raise ValueError('unsupported numerical witness activation')

    def recompute(session, inputs, witnesses):
        expected = {'x', 'value'} if kind == 'swiglu' else {'x'}
        if set(inputs) != expected:
            raise ValueError('numerical program input names do not match')
        x = session.bounded(inputs['x'], 1, 2) if kind == 'reciprocal' else session.bounded(inputs['x'], -4, 4)
        _same_shape(witnesses['reciprocal'], x.shape, 'reciprocal')

        class WithWitness(NumericalFixedPoint):
            def reciprocal(self, denominator):
                return _checked_reciprocal(session, denominator, witnesses['reciprocal'])

        fp = WithWitness(session.runtime, session.fp.fractional_bits)
        if kind == 'swiglu':
            _same_shape(inputs['value'], x.shape, 'value')
            value = session.bounded(inputs['value'], -4, 4)
            return fp.swiglu(x, value)
        return getattr(fp, kind)(x)

    return WitnessProgram(kind, {'reciprocal': (0., 1.)}, recompute)


def rmsnorm_program(epsilon=1e-5):
    """Normalize rows using one inverse-square-root witness per row.

    Inputs ``x`` lie in [-4,4], and the last dimension is reduced. Witness
    ``inverse_sqrt`` has shape ``x.shape[:-1]+(1,)`` and lies in [1/8,512].
    Epsilon is in [1e-5,1], keeping the encoded variance positive at Q20..Q24.
    The root is checked through ``s*(V*s)`` to bound intermediate products.
    This operation excludes the optional learned RMSNorm weight.
    """
    if not math.isfinite(epsilon) or not 1e-5 <= epsilon <= 1:
        raise ValueError('RMSNorm epsilon must lie in [1e-5,1]')

    def recompute(session, inputs, witnesses):
        if set(inputs) != {'x'} or not inputs['x'].shape:
            raise ValueError('RMSNorm requires a non-scalar x input')
        fp, rt, scale = session.fp, session.runtime, session.scale
        x = session.bounded(inputs['x'], -4, 4)
        width = x.shape[-1]
        if width > 1 << 20:
            raise ValueError('RMSNorm width exceeds the accumulation bound')
        s = witnesses['inverse_sqrt']
        _same_shape(s, (*x.shape[:-1], 1), 'inverse_sqrt')
        squares = fp.mul(x, x)
        total = session.sum(squares, dim=-1, keepdim=True)
        # Power-of-two dimensions avoid quantizing the public averaging factor.
        if width & (width-1):
            if width > 4096:
                raise ValueError('non-power-of-two RMSNorm widths must be <=4096')
            variance = fp._public_mul(total, 1/width)
        else:
            variance = rt.trunc(total, width.bit_length()-1)
        variance = fp.add(variance, fp.constant(epsilon, variance))
        variance = session.bounded(variance, round(epsilon*scale)/scale, 18)
        product = session.bounded(fp.mul(variance, s), 0, 5)
        raw = rt.mul(product, s)
        error = rt.sub(raw, rt.public(scale*scale, like=raw))
        # For the nearest encoded root, V*s differs from sqrt(V) by
        # <=V/(2*S). Root quantization contributes <5/S to the
        # identity; rounding V*s contributes <=512/S (floor or ceil).
        # Raw products are <=18*512*S² and <=5*512*S², both <2**62.
        session.residual(error, 518*scale)
        return fp.mul(x, s)

    return WitnessProgram('rmsnorm', {'inverse_sqrt': (1/8, 512.)}, recompute)


def softmax_program():
    """Check a row maximum and reciprocal while recomputing private exp.

    Input ``x`` lies in [-4,4]. Row witnesses ``shift`` and ``reciprocal`` have
    shape ``x.shape[:-1]+(1,)``. Shift is checked as an exact encoded maximum:
    all differences are nonpositive and at least one is zero in each row.
    This avoids assuming shift invariance of a finite-precision exponential.
    """
    def recompute(session, inputs, witnesses):
        if set(inputs) != {'x'} or not inputs['x'].shape:
            raise ValueError('softmax requires a non-scalar x input')
        fp, rt, scale = session.fp, session.runtime, session.scale
        x = session.bounded(inputs['x'], -4, 4)
        width = x.shape[-1]
        if width > 4096:
            raise ValueError('softmax width must not exceed 4096')
        shape = (*x.shape[:-1], 1)
        _same_shape(witnesses['shift'], shape, 'shift')
        _same_shape(witnesses['reciprocal'], shape, 'reciprocal')
        shifted = session.bounded(fp.sub(x, witnesses['shift']), -8, 0)
        at_zero = rt.sub(rt.public(1, like=shifted), rt.lt_public(shifted, 0))
        count = session.sum(at_zero, dim=-1, keepdim=True)
        session.residual(rt.lt_public(count, 1))
        exponential = fp.exp(shifted)
        denominator = session.sum(exponential, dim=-1, keepdim=True)
        inverse = _checked_reciprocal(session, denominator, witnesses['reciprocal'], width)
        return fp.mul(exponential, inverse)

    return WitnessProgram('softmax', {'shift': (-4., 4.), 'reciprocal': (0., 1.)}, recompute)
