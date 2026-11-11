"""Fixtures, signed interval proofs, and private numerical MLP evaluation."""
import numpy as np
import torch

from benchmarks.numerical_primitives import IntegerOracle, Interval
from rep4gpu.numerical import NumericalFixedPoint


def fixture(kind, batch=128, width=2048, hidden=8192, out=2048, fractional_bits=24):
    """Reproduce inference.fixture's RNG stream without computing its ReLU output.

    SwiGLU appends an independent value projection from seed 20261110. Its
    three matrices contain more parameters than the two-matrix SiLU/GELU MLP.
    """
    if kind not in ('silu', 'gelu', 'swiglu'):
        raise ValueError('kind must be silu, gelu or swiglu')
    if min(batch, width, hidden, out) <= 0:
        raise ValueError('fixture dimensions must be positive')
    if type(fractional_bits) is not int or not 20 <= fractional_bits <= 24:
        raise ValueError('fractional_bits must be in [20,24]')
    rng = np.random.default_rng(20261109)
    arrays = [rng.normal(0, .5, (batch, width)),
              rng.normal(0, 1/np.sqrt(width), (width, hidden)),
              rng.normal(0, .05, hidden),
              rng.normal(0, 1/np.sqrt(hidden), (hidden, out)),
              rng.normal(0, .05, out)]
    if kind == 'swiglu':
        value_rng = np.random.default_rng(20261110)
        arrays += [value_rng.normal(0, 1/np.sqrt(width), (width, hidden)),
                   value_rng.normal(0, .05, hidden)]
    return ([torch.from_numpy(a.copy()) for a in arrays],
            [torch.from_numpy(np.rint(a*(1 << fractional_bits)).astype(np.int64)) for a in arrays])


class ForwardOracle(IntegerOracle):
    """Extend the numerical oracle with checked dot products against point weights."""
    def _check_bound(self, bound):
        if not np.isfinite(bound) or bound >= 2**62:
            raise ValueError(f'raw product or accumulator bound {bound} exceeds strict 2**62 limit')
        self.maximum = max(self.maximum, int(np.ceil(bound)))

    def add(self, a, b):
        # Triangle bounds also protect additions before signed arithmetic runs.
        bound = max(float(a.lo.double().abs().max()), float(a.hi.double().abs().max())) + max(float(b.lo.double().abs().max()), float(b.hi.double().abs().max()))
        self._check_bound(bound)
        return super().add(a, b)

    def matmul(self, a, b):
        if not torch.equal(b.lo, b.hi):
            raise ValueError('matrix weights must be point intervals')
        weights = b.lo
        # Each column uses its absolute weight sum, weighted by the largest
        # endpoint magnitude at each reduction index. This covers every partial
        # accumulator, including cancellation, without a max*width estimate.
        magnitude = torch.maximum(a.lo.double().abs(), a.hi.double().abs()).amax(dim=0)
        bound = float((magnitude @ weights.double().abs()).max())
        # Float64's dot-product error is bounded by gamma_n; inflate upward
        # before accepting any int64 operation, including split dot products.
        n = weights.shape[0]
        self._check_bound(bound/(1-n*np.finfo(np.float64).eps)+1)
        if torch.equal(a.lo, a.hi):
            result = a.lo @ weights
            return Interval(result, result)
        positive, negative = weights.clamp_min(0), weights.clamp_max(0)
        return Interval(a.lo @ positive + a.hi @ negative,
                        a.hi @ positive + a.lo @ negative)


def _forward(runtime, values, fractional_bits, kind, validate=False):
    fixed = NumericalFixedPoint(runtime, fractional_bits)
    x, gate, gate_bias, down, down_bias, *value = values
    def projection(weights, bias):
        raw = runtime.add(runtime.matmul(x, weights), runtime.mul_public(bias, 1 << fractional_bits))
        result = runtime.trunc(raw, fractional_bits)
        if validate and (bool((result.lo < -(4 << fractional_bits)).any()) or bool((result.hi > (4 << fractional_bits)).any())):
            raise ValueError('fixture activation input exceeds [-4,4]')
        return result
    hidden = projection(gate, gate_bias)
    hidden = fixed.swiglu(hidden, projection(*value)) if kind == 'swiglu' else getattr(fixed, kind)(hidden)
    return runtime.add(runtime.trunc(runtime.matmul(hidden, down), fractional_bits), down_bias)


def infer(party, values, fractional_bits=24, kind='silu'):
    """Evaluate private matrices and activations, opening only the final output."""
    return party.open(_forward(party, values, fractional_bits, kind))


def oracle(kind, integers, fractional_bits=24, probabilistic=False):
    runtime = ForwardOracle(probabilistic)
    interval = _forward(runtime, [runtime.public(v) for v in integers], fractional_bits, kind, validate=True)
    return interval, dict(maximum_pretruncation_abs_bound=runtime.maximum,
                          strict_probabilistic_accumulator_limit=2**62,
                          maximum_rounding_interval_width=int((interval.hi-interval.lo).max()),
                          activation_input_bound=4,
                          range_validation='public fixture only; not a private range check')


def reference(values, kind, *, preserve_dtype=False):
    """Use float64 for error measurement or retain dtype for baseline timing."""
    x, gate, gate_bias, down, down_bias, *value = values if preserve_dtype else [v.double() for v in values]
    hidden = x @ gate + gate_bias
    if kind == 'swiglu':
        hidden = torch.nn.functional.silu(hidden)*(x @ value[0]+value[1])
    elif kind == 'gelu':
        hidden = torch.nn.functional.gelu(hidden)
    else:
        hidden = torch.nn.functional.silu(hidden)
    return hidden @ down + down_bias


def metrics(output, truth, interval, fractional_bits=24):
    scale = 1 << fractional_bits
    error = output.double()/scale-truth
    conservative = torch.maximum((interval.lo.double()/scale-truth).abs(), (interval.hi.double()/scale-truth).abs())
    return dict(max_error=float(error.abs().max()), rmse=float(error.square().mean().sqrt()),
                max_relative_error=float((error.abs()/truth.abs().clamp_min(1e-12)).max()),
                conservative_rounding_max_error=float(conservative.max()),
                top1_agreement=float((output.argmax(-1) == truth.argmax(-1)).double().mean()),
                max_logprob_error=float((torch.log_softmax(output.double()/scale, dim=-1)-
                                         torch.log_softmax(truth, dim=-1)).abs().max()))
