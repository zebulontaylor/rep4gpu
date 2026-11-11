"""Check prover-supplied activation witnesses using private Rep4 constraints.

A session keeps recomputed outputs private until one aggregate acceptance bit
is opened. Range checks either clamp operands or register a sticky violation
before later arithmetic relies on their bounds.
Program callbacks are trusted protocol code, not code supplied by the prover.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from collections.abc import Callable, Mapping

import torch

from ..arithmetic.numerical import NumericalFixedPoint


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
    Optional ``prepare_witnesses(session, witnesses)`` replaces the default
    schema clamps with trusted range checks and returns the checked mapping.
    It must establish every bound needed by the recomputation on accepted paths.
    """
    name: str
    witness_bounds: Mapping[str, tuple[float, float]]
    recompute: Callable
    prepare_witnesses: Callable | None = None

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
        if self.prepare_witnesses is not None and not callable(self.prepare_witnesses):
            raise ValueError('witness preparation must be callable')


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
        self._binary_violations = []
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

    def _interval(self, value, lower, upper, *, binary=False):
        self._active()
        self._arithmetic(value)
        if not -(1 << 63) <= lower <= upper < (1 << 63)-1:
            raise ValueError('constraint endpoints must fit signed ring64')
        rt = self.runtime
        thresholds = (lower, upper+1)
        operands = [value]
        for bound in thresholds:
            if bound:
                operands.append(rt.sub(value, rt.public(bound, like=value)))
        packed = rt._share(torch.stack([v.parts[j] for v in operands]) for j in range(3))
        if binary:
            from ..arithmetic.binary_aggregation import binary_sign, binary_and
            signs = binary_sign(rt, packed)
        else:
            signs = rt.sign_bit(packed)
        sx, position = signs[0], 1
        comparisons = []
        for bound in thresholds:
            if bound == 0:
                comparisons.append(sx)
                continue
            sd = signs[position]
            position += 1
            both = binary_and(rt, sx, sd) if binary else rt.mul(sx, sd)
            # A negative public bound uses sx AND sd; a positive bound uses
            # sx OR sd. The original sign handles subtraction wraparound.
            comparisons.append(both if bound < 0 else
                               (rt.add(rt.add(sx, sd), both) if binary else
                                rt.sub(rt.add(sx, sd), both)))
        below = comparisons[0]
        one = rt.public(1, like=comparisons[1], binary=binary)
        above = rt.add(one, comparisons[1]) if binary else rt.sub(one, comparisons[1])
        if binary:
            self._binary_violations.extend((below, above))
        else:
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
        self._interval(error, -tolerance_lsb, tolerance_lsb, binary=True)
        self._residuals += 1

    def nonnegative(self, value):
        """Require a bounded signed expression to be nonnegative.

        As with ``residual``, the callback establishes that the expression did
        not wrap. This supports secret-dependent bounds without opening them.
        """
        self._active()
        self._arithmetic(value)
        from ..arithmetic.binary_aggregation import binary_sign
        self._binary_violations.append(binary_sign(self.runtime, value))
        self._residuals += 1

    def upper_zero(self, value, width):
        """Require an unsigned width-bit value without clamping it.

        Later bounds may depend on this check passing. A violation remains
        sticky until finish, even if later arithmetic wraps on invalid inputs.
        """
        self._active()
        self._arithmetic(value)
        from ..arithmetic.upper_zero import upper_zero_violation
        self._binary_violations.append(upper_zero_violation(self.runtime, value, width, binary=True))
        self._residuals += 1

    def signed_absolute_input(self, value, maximum):
        """Check |value|<=maximum and return its sign and clamped magnitude.

        maximum is an encoded positive bound below 2**62. For INT64_MIN,
        negation leaves the word unchanged, but maximum-abs wraps to a
        negative word; the same upper-bound bit rejects and clamps it.
        """
        self._active()
        self._arithmetic(value)
        if type(maximum) is not int or not 0 < maximum < 1 << 62:
            raise ValueError('absolute input bound must lie in (0,2**62)')
        rt = self.runtime
        sign = rt.sign_bit(value)
        absolute = self.fp._select(sign, self.fp._negative(value), value)
        limit = rt.public(maximum, like=value)
        above = rt.sign_bit(rt.sub(limit, absolute))
        self._violation(above)
        return sign, self.fp._select(above, limit, absolute)

    def bounded_nonnegative(self, value, width):
        """Check sign under a preceding signed width-bit range constraint."""
        self._active()
        self._arithmetic(value)
        if type(width) is not int or not 2 <= width < 64:
            raise ValueError('signed comparison width must be in [2,63]')
        rt = self.runtime
        if value.device.type == 'cuda' and hasattr(rt, 'joint_input_batch'):
            from ..gpu.bits.bounded_activation import signed_width_sign
            bit = signed_width_sign(rt, value, width, binary=True)
            self._binary_violations.append(bit)
        else:
            bit = rt.sign_bit(value)
            self._violation(bit)
        self._residuals += 1

    def evaluate(self, program, inputs, witnesses):
        """Queue an output; no result is returned before ``finish`` succeeds."""
        self._active()
        try:
            if set(witnesses) != set(program.witness_bounds):
                raise ValueError('witness names must exactly match the program schema')
            safe = ({name: self.bounded(witnesses[name], *bounds)
                     for name, bounds in program.witness_bounds.items()}
                    if program.prepare_witnesses is None
                    else program.prepare_witnesses(self, witnesses))
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
        if self._binary_violations:
            from ..arithmetic.binary_aggregation import binary_any
            from ..arithmetic.compact_conversion import b2a
            rejected = binary_any(self.runtime, self._binary_violations)
            self._binary_violations.clear()
            self._violation(b2a(self.runtime, rejected, bits=1, signed=False))
        # The accumulated count lies in [0,2**62), so count-1 cannot wrap.
        accepted = self.runtime.sign_bit(
            self.runtime.sub(self._count, self.runtime.public(1, like=self._count)))
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


def _checked_reciprocal(session, denominator, witness, maximum=2, *, check_domain=True):
    _same_shape(witness, denominator.shape, 'reciprocal')
    fp, rt, scale = session.fp, session.runtime, session.scale
    if check_domain:
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


def silu_output_program(*, gated=False):
    """Check a nearest-rounded SiLU output against the recomputed exponential.

    The shared ``output`` witness is the SiLU gate, including for SwiGLU.
    Gate inputs lie in [-16,16]. The numerator and identity stay at Q(2*f),
    avoiding the two sequential output rescalings of the reciprocal recipe.
    """
    def prepare(session, witnesses):
        y = witnesses['output']
        session._arithmetic(y)
        shifted = session.runtime.add(y, session.runtime.public(session.scale, like=y))
        session.upper_zero(shifted, session.fp.fractional_bits+5)
        return witnesses

    def recompute(session, inputs, witnesses):
        expected = {'x', 'value'} if gated else {'x'}
        if set(inputs) != expected:
            raise ValueError('SiLU output program input names do not match')
        fp, rt = session.fp, session.runtime
        from ..arithmetic.silu_exp import silu_exp_negative

        x = inputs['x']
        y = witnesses['output']
        _same_shape(y, x.shape, 'output')
        sign, absolute = session.signed_absolute_input(x, 16*session.scale)
        exponential = silu_exp_negative(fp, fp._negative(absolute), bounded_floor=True)
        denominator = fp.add(fp.constant(1, exponential), exponential)
        multiplier = fp._select(sign, fp._negative(exponential), fp.constant(1, exponential))
        numerator = rt.mul(absolute, multiplier)
        error = rt.sub(rt.mul(y, denominator), numerator)
        doubled = rt.mul_public(error, 2)
        # Passing the witness range gives -S<=Y<31*S, hence |R|<2**56
        # at Q24. If U1=R+D is in [0,8*S), U2=2*D-U1 lies in
        # (-8*S,4*S], so bit f+3 is its signed comparison bit.
        session.upper_zero(rt.add(doubled, denominator), fp.fractional_bits+3)
        session.bounded_nonnegative(rt.sub(denominator, doubled), fp.fractional_bits+4)
        if gated:
            _same_shape(inputs['value'], x.shape, 'value')
            from ..arithmetic.exact_truncation import trunc_bounded
            value = session.bounded(inputs['value'], -4, 4)
            # Passing the range gives |y*value|<124*S*S<2**62.
            return trunc_bounded(rt, rt.mul(y, value), fp.fractional_bits)
        return y

    return WitnessProgram('swiglu' if gated else 'silu', {'output': (-16., 16.)}, recompute, prepare)


def numerical_program(kind, *, direct_output=True):
    """Build a checked numerical activation with one named shared witness.

    SiLU and SwiGLU use an ``output`` witness for the SiLU gate by default.
    Other recipes, or ``direct_output=False``, use a ``reciprocal`` witness
    for the recomputed fixed-point denominator. Inputs use ``x`` and, for
    SwiGLU, ``value``. Witnesses have the same public shape as x.
    """
    if kind not in ('reciprocal', 'silu', 'gelu', 'swiglu'):
        raise ValueError('unsupported numerical witness activation')
    if direct_output and kind in ('silu', 'swiglu'):
        return silu_output_program(gated=kind == 'swiglu')

    def recompute(session, inputs, witnesses):
        expected = {'x', 'value'} if kind == 'swiglu' else {'x'}
        if set(inputs) != expected:
            raise ValueError('numerical program input names do not match')
        x = session.bounded(inputs['x'], 1, 2) if kind == 'reciprocal' else session.bounded(inputs['x'], -4, 4)
        _same_shape(witnesses['reciprocal'], x.shape, 'reciprocal')

        class WithWitness(NumericalFixedPoint):
            def reciprocal(self, denominator):
                # For bounded SiLU inputs, the ordered nonnegative Taylor
                # coefficients keep exp(-abs(x)) in [0,1] after every floor.
                return _checked_reciprocal(session, denominator, witnesses['reciprocal'],
                                           check_domain=kind not in ('silu', 'swiglu'))

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
    Squares are summed in tiles of at most 512 coordinates before resharing
    and truncation, so the raw tile sum stays below 2**62 even at Q24.
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
        tile_width = min(width, 512)
        tiles = (width + tile_width - 1) // tile_width
        padding = tiles * tile_width - width
        tiled = rt._share(
            torch.nn.functional.pad(part, (0, padding)).reshape(
                *x.shape[:-1], tiles, tile_width)
            for part in x.parts)
        # This is the same bilinear Rep4 product as mul, but local terms
        # reduce before resharing. At |x|<=4, each tile is <=8192*S².
        raw_tiles = rt._product(tiled, tiled, lambda a, b: (a*b).sum(dim=-1))
        squares = rt.trunc(raw_tiles, fp.fractional_bits)
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
        if shifted.device.type == 'cuda' and hasattr(rt, 'joint_input_batch'):
            from ..gpu.bits.bounded_activation import posttrunc_sign
            # The preceding private clamp establishes |shifted|<=8*S,
            # including the negative endpoint, for this shorter sign circuit.
            negative = posttrunc_sign(rt, shifted, 64-fp.fractional_bits-5)
        else:
            negative = rt.sign_bit(shifted)
        at_zero = rt.sub(rt.public(1, like=shifted), negative)
        count = session.sum(at_zero, dim=-1, keepdim=True)
        session.nonnegative(rt.sub(count, rt.public(1, like=count)))
        exponential = fp.exp(shifted)
        denominator = session.sum(exponential, dim=-1, keepdim=True)
        inverse = _checked_reciprocal(session, denominator, witnesses['reciprocal'], width)
        return fp.mul(exponential, inverse)

    return WitnessProgram('softmax', {'shift': (-4., 4.), 'reciprocal': (0., 1.)}, recompute)
