"""Plaintext witness builders for the verifier's exact-floor fixed-point formulas.

Run these in the prover process on encoded inputs. They are not MPC operations
and must never receive private verifier shares. Probabilistic verifier rounding
can change denominators; these builders target the exact-floor runtime.
"""
from __future__ import annotations

import math

import torch

from .numerical import NumericalFixedPoint


class _PlainFloor:
    bits = 64

    def public(self, value, like=None):
        return torch.as_tensor(value, dtype=torch.int64,
                               device=None if like is None else like.device)

    def add(self, x, y):
        return x + y

    def sub(self, x, y):
        return x - y

    def mul(self, x, y):
        bound = float(x.double().abs().max()) * float(y.double().abs().max())
        if bound >= 2**62:
            raise ValueError('plaintext witness computation exceeds its product bound')
        return x * y

    def mul_public(self, x, value):
        return self.mul(x, self.public(value, like=x))

    def trunc(self, x, bits):
        return x >> bits

    def sign_bit(self, x):
        return (x < 0).long()


def _input(value, scale, lower, upper):
    if not isinstance(value, torch.Tensor) or value.dtype != torch.int64 or not value.numel():
        raise ValueError('prover inputs must be nonempty encoded int64 tensors')
    if not bool(((value >= math.ceil(lower*scale)) &
                 (value <= math.floor(upper*scale))).all()):
        raise ValueError('prover input exceeds the activation domain')
    return value


def numerical_witnesses(kind, inputs, fractional_bits=24):
    """Return encoded reciprocal claims for a built-in numerical program.

    Inputs are a mapping with ``x`` and, for SwiGLU, ``value``. The builder
    reproduces the quantized denominator, rather than using an ideal floating
    activation's intermediate. The verifier still checks every resulting claim.
    """
    if kind not in ('reciprocal', 'silu', 'gelu', 'swiglu'):
        raise ValueError('unsupported numerical witness activation')
    expected = {'x', 'value'} if kind == 'swiglu' else {'x'}
    if set(inputs) != expected:
        raise ValueError('prover input names do not match the activation')
    runtime = _PlainFloor()
    claims = {}

    class Prover(NumericalFixedPoint):
        def reciprocal(self, denominator):
            claim = torch.round(self.scale**2 / denominator.double()).long()
            claims['reciprocal'] = claim
            return claim

    fp = Prover(runtime, fractional_bits)
    bounds = (1, 2) if kind == 'reciprocal' else (-4, 4)
    x = _input(inputs['x'], fp.scale, *bounds)
    if kind == 'swiglu':
        value = _input(inputs['value'], fp.scale, -4, 4)
        if value.shape != x.shape:
            raise ValueError('SwiGLU inputs must have matching shapes')
        fp.swiglu(x, value)
    else:
        getattr(fp, kind)(x)
    return claims


def rmsnorm_witnesses(x, fractional_bits=24, epsilon=1e-5):
    """Return one inverse-root claim per row for ``rmsnorm_program(epsilon)``."""
    from .witness_activations import rmsnorm_program
    rmsnorm_program(epsilon)
    rt = _PlainFloor()
    fp = NumericalFixedPoint(rt, fractional_bits)
    x = _input(x, fp.scale, -4, 4)
    if not x.ndim:
        raise ValueError('RMSNorm requires rows')
    width = x.shape[-1]
    if width > 1 << 20 or (width & (width-1) and width > 4096):
        raise ValueError('RMSNorm width exceeds the verifier domain')
    total = fp.mul(x, x).sum(dim=-1, keepdim=True)
    variance = (fp._public_mul(total, 1/width) if width & (width-1)
                else rt.trunc(total, width.bit_length()-1))
    variance = variance + round(epsilon*fp.scale)
    root = torch.round(torch.rsqrt(variance.double()/fp.scale)*fp.scale).long()
    return {'inverse_sqrt': root}


def softmax_witnesses(x, fractional_bits=24):
    """Return an encoded maximum and reciprocal of the recomputed exp sum."""
    fp = NumericalFixedPoint(_PlainFloor(), fractional_bits)
    x = _input(x, fp.scale, -4, 4)
    if not x.ndim or x.shape[-1] > 4096:
        raise ValueError('softmax requires rows of at most 4096 elements')
    shift = x.max(dim=-1, keepdim=True).values
    denominator = fp.exp(x-shift).sum(dim=-1, keepdim=True)
    inverse = torch.round(fp.scale**2 / denominator.double()).long()
    return {'shift': shift, 'reciprocal': inverse}
