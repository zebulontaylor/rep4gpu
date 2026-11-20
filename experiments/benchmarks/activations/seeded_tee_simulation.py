"""Ring64 seeded activation helper simulation, colocated with Rep4 role 0.

Ordinary CUDA code reconstructs activations; there is no enclave or protected
channel. The factory retains every helper seed. This measures a proposed TEE
boundary's communication, not activation privacy or malicious helper security.
"""
import math

import torch

from rep4gpu.gpu.crypto.gpu_prg import DomainKeys, key_bytes
from rep4gpu.gpu.crypto.gpu_prg_batch import generate_rows
from rep4gpu.models.phase_graph import PhaseGraphParty
from rep4gpu.protocol.runtime import Share

_DOMAIN = b'PUBLIC-FIXTURE-SEEDED-SWIGLU-v1'
_STREAMS = (4, 5, 6)


def clear_activation(kind, gate_raw, value_raw=None, fractional_bits=24):
    """Floor-rescale Q(2f), evaluate in float64, and round to nearest-even Qf.

    Supported functions are ReLU, SiLU, exact-formula GELU, and SwiGLU. The
    caller supplies signed int64 accumulators whose decoded ranges are valid.
    """
    if kind not in ('relu', 'silu', 'gelu', 'swiglu'):
        raise ValueError('unsupported helper activation')
    if type(fractional_bits) is not int or not 20 <= fractional_bits <= 24:
        raise ValueError('fractional_bits must be in [20,24]')
    if (kind == 'swiglu') != (value_raw is not None):
        raise ValueError('only SwiGLU requires a value input')
    if value_raw is not None and gate_raw.shape != value_raw.shape:
        raise ValueError('gate and value must have matching shape')
    scale = 1 << fractional_bits
    gate = (gate_raw >> fractional_bits).double()/scale
    if kind == 'relu':
        result = torch.relu(gate)
    elif kind == 'gelu':
        result = torch.nn.functional.gelu(gate, approximate='none')
    else:
        result = torch.nn.functional.silu(gate)
        if value_raw is not None:
            result = result*((value_raw >> fractional_bits).double()/scale)
    return torch.round(result*scale).to(torch.int64)


def clear_swiglu(gate, value, fractional_bits=24):
    """Floor-rescale signed raw accumulators, then return round-to-even Qf."""
    return clear_activation('swiglu', gate, value, fractional_bits)


def seeded_tee_party(helper_seed):
    """Build simulation roles with independent output-mask streams.

    The simulated enclave at role 0 holds all three output keys. Other roles
    hold only keys for their replicated components. Fresh counters are separate
    from ordinary MPC mask streams and are reserved on every graph execution.
    """
    helper_seed = bytes(helper_seed)
    key_bytes(helper_seed, _DOMAIN)

    class SeededTEEParty(PhaseGraphParty):
        activation_backend = 'public_fixture_seeded_tee_swiglu'

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if self.bits != 64:
                raise ValueError('seeded TEE simulation requires ring64')
            domains = DomainKeys(helper_seed, self.device)
            self._tee_keys = tuple(domains.key(_DOMAIN+bytes((i,)))
                                   if self.party == 0 or self.party != i else None
                                   for i in range(3))
            self.counters.update({i: 0 for i in _STREAMS})

        def _tee_masks(self, shape):
            if self._phase_builder is not None:
                result = self._phase_builder.masks(self._tee_keys, _STREAMS, shape, 64)
            else:
                reserved = self._reserve_nonces(_STREAMS)
                nonces = tuple(n if k is not None else None
                               for k, n in zip(self._tee_keys, reserved))
                result = generate_rows(self._tee_keys, nonces, math.prod(shape), bits=64)
            return result.reshape(3, *shape)

        def tee_activation(self, kind, gate_raw, value_raw=None, fractional_bits=24):
            """Return shared Qf activations from Q(2f) inputs in two exchanges.

            For N coordinates, global payload is 24*N+32 bytes for unary
            functions or 32*N+32 for SwiGLU. The caller establishes valid
            accumulator and output ranges. This is an ordinary CUDA helper,
            not an enclave or a protected activation implementation.
            """
            if kind not in ('relu', 'silu', 'gelu', 'swiglu'):
                raise ValueError('unsupported helper activation')
            if type(fractional_bits) is not int or not 20 <= fractional_bits <= 24:
                raise ValueError('fractional_bits must be in [20,24]')
            if (kind == 'swiglu') != (value_raw is not None):
                raise ValueError('only SwiGLU requires a value input')
            inputs = (gate_raw,) if value_raw is None else (gate_raw, value_raw)
            if any(x.binary or x.device != self.device for x in inputs):
                raise ValueError('TEE simulation requires matching arithmetic device shares')
            if value_raw is not None and gate_raw.shape != value_raw.shape:
                raise ValueError('gate and value must have matching shape')
            return self._tee_apply(inputs, lambda *values: clear_activation(
                kind, values[0], values[1] if len(values) == 2 else None, fractional_bits))

        def tee_swiglu(self, gate_raw, value_raw, fractional_bits=24):
            """Consume raw SwiGLU projections through the seeded helper boundary."""
            return self.tee_activation('swiglu', gate_raw, value_raw, fractional_bits)

        def tee_trunc(self, raw, fractional_bits=24):
            """Floor-rescale inside the helper for 24*N+32 global bytes."""
            if type(fractional_bits) is not int or not 20 <= fractional_bits <= 24:
                raise ValueError('fractional_bits must be in [20,24]')
            if raw.binary or raw.device != self.device:
                raise ValueError('TEE simulation requires matching arithmetic device shares')
            return self._tee_apply((raw,), lambda a: a >> fractional_bits)

        def _tee_apply(self, inputs, function):
            shape = inputs[0].shape
            if not math.prod(shape):
                return inputs[0]
            missing = None
            if self.party in (1, 2):
                slot = (0-self.party-1) % 4
                missing = torch.stack(tuple(x.parts[slot] for x in inputs))
                missing = missing.contiguous().view(torch.uint8).reshape(-1)
            builder = self._phase_builder
            previous = None if builder is None else builder.check_at_send
            if builder is not None:
                builder.check_at_send = False
            try:
                incoming = self._exchange_joint_wires([(1, 2, 0, missing)], (len(inputs), *shape), False)
            finally:
                if builder is not None:
                    builder.check_at_send = previous
            masks = self._tee_masks(shape)
            residual = None
            if self.party == 0:
                absent = incoming[0].view(torch.int64).reshape(len(inputs), *shape)
                clear = function(*(sum(x.parts)+a for x, a in zip(inputs, absent)))
                residual = clear-masks[0]-masks[1]-masks[2]
            wire = residual.contiguous().view(torch.uint8).reshape(-1) if residual is not None else None
            received = self._exchange_device({1: wire, 2: wire} if self.party == 0 else {},
                                            {0: 8*math.prod(shape)} if self.party in (1, 2) else {})
            if self.party in (1, 2):
                residual = received[0].view(torch.int64).reshape(shape)
            components = (*masks.unbind(0), residual)
            return Share(tuple(components[(self.party+j+1) % 4] for j in range(3)), False)

    return SeededTEEParty


def infer(party, values, fractional_bits=24, *, tee_down=False):
    """Keep projections in MPC and floor-rescale gate/value inside the helper."""
    x, gate, gate_bias, down, down_bias, value, value_bias = values
    def raw_projection(weight, bias):
        return party.add(party.matmul(x, weight), party.mul_public(bias, 1 << fractional_bits))
    hidden = party.tee_swiglu(raw_projection(gate, gate_bias),
                              raw_projection(value, value_bias), fractional_bits)
    raw = party.matmul(hidden, down)
    scaled = (party.tee_trunc(raw, fractional_bits) if tee_down
              else party.trunc(raw, fractional_bits))
    output = party.add(scaled, down_bias)
    return party.open(output)
