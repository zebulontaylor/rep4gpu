"""Ring32 seeded ReLU helper simulation, colocated with Rep4 role 0.

This ordinary CUDA benchmark reconstructs activations without an enclave,
attestation, or protected channels. Its factory retains the helper seed. It
measures proposed boundary payloads, not activation privacy or malicious-helper
security. Independent-holder checks cover the remote missing component; a real
backend must also authenticate the colocated host's three input components and
the enclave's output delivery.
"""
import math

import torch

from rep4gpu.gpu_prg import DomainKeys, key_bytes
from rep4gpu.gpu_prg_batch import generate_rows
from rep4gpu.phase_graph import PhaseGraphParty
from rep4gpu.runtime import Share


_DOMAIN = b'PUBLIC-FIXTURE-SEEDED-RELU-RING32-v1'
_STREAMS = (4, 5, 6)
_MASK = 0xffffffff


def seeded_relu_party(helper_seed):
    """Build simulation roles with separate, fresh output-mask streams.

    The helper at role 0 holds all three output keys. Other roles hold keys
    only for their replicated components. These counters are separate from
    ordinary MPC draws and advance during every live graph execution.
    """
    helper_seed = bytes(helper_seed)
    key_bytes(helper_seed, _DOMAIN)

    class SeededReLUParty(PhaseGraphParty):
        activation_backend = 'public_fixture_seeded_tee_relu_ring32'

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if self.bits != 32:
                raise ValueError('seeded ReLU simulation requires ring32')
            domains = DomainKeys(helper_seed, self.device)
            self._tee_keys = tuple(
                domains.key(_DOMAIN + bytes((i,)))
                if self.party == 0 or self.party != i else None
                for i in range(3))
            self.counters.update({i: 0 for i in _STREAMS})

        def _tee_masks(self, shape):
            if self._phase_builder is not None:
                result = self._phase_builder.masks(self._tee_keys, _STREAMS, shape, 32)
            else:
                reserved = self._reserve_nonces(_STREAMS)
                nonces = tuple(n if key is not None else None
                               for key, n in zip(self._tee_keys, reserved))
                result = generate_rows(self._tee_keys, nonces, math.prod(shape), bits=32)
            return result.reshape(3, *shape)

        def _validate_helper_input(self, raw, fractional_bits):
            if type(fractional_bits) is not int or not 0 <= fractional_bits < 32:
                raise ValueError('fractional_bits must be an integer in [0,31]')
            if raw.binary or raw.device != self.device or any(
                    part.dtype != torch.int64 or part.shape != raw.shape
                    for part in raw.parts):
                raise ValueError('helper requires matching arithmetic int64 device shares')

        def tee_relu(self, raw, fractional_bits=14):
            """Floor-rescale a signed ring32 accumulator and apply ReLU.

            Every nonempty call uses two exchanges and 12*N+32 global payload
            bytes, including shift zero. Outputs are freshly reshared.
            """
            self._validate_helper_input(raw, fractional_bits)
            return self._tee_apply(raw, lambda value: (value >> fractional_bits).clamp_min(0))

        def tee_trunc(self, raw, fractional_bits=14):
            """Floor-rescale inside the helper for 12*N+32 global bytes."""
            self._validate_helper_input(raw, fractional_bits)
            return self._tee_apply(raw, lambda value: value >> fractional_bits)

        def _tee_apply(self, raw, function):
            count = math.prod(raw.shape)
            if not count:
                return raw
            missing = None
            if self.party in (1, 2):
                slot = (0 - self.party - 1) % 4
                missing = raw.parts[slot].to(torch.int32).contiguous().view(torch.uint8).reshape(-1)

            builder = self._phase_builder
            previous = None if builder is None else builder.check_at_send
            if builder is not None:
                builder.check_at_send = False
            try:
                incoming = self._exchange_joint_wires([(1, 2, 0, missing)], raw.shape, False)
            finally:
                if builder is not None:
                    builder.check_at_send = previous

            masks = self._tee_masks(raw.shape)
            residual = None
            if self.party == 0:
                absent = incoming[0].view(torch.int32).to(torch.int64).reshape(raw.shape)
                word = (sum(raw.parts) + absent) & _MASK
                signed = (word ^ 0x80000000) - 0x80000000
                clear = function(signed)
                residual = (clear - masks[0] - masks[1] - masks[2]) & _MASK
            wire = (residual.to(torch.int32).contiguous().view(torch.uint8).reshape(-1)
                    if residual is not None else None)
            received = self._exchange_device(
                {1: wire, 2: wire} if self.party == 0 else {},
                {0: 4 * count} if self.party in (1, 2) else {})
            if self.party in (1, 2):
                residual = received[0].view(torch.int32).to(torch.int64).reshape(raw.shape) & _MASK
            components = (*masks.unbind(0), residual)
            return Share(tuple(components[(self.party + j + 1) % 4] for j in range(3)), False)

    return SeededReLUParty


def infer(party, values, fractional_bits=14, *, tee_down=False):
    """Keep both linear products in MPC and apply helper floor-rescaling."""
    x, w1, b1, w2, b2 = values
    raw1 = party.add(party.matmul(x, w1), party.mul_public(b1, 1 << fractional_bits))
    hidden = party.tee_relu(raw1, fractional_bits)
    raw2 = party.matmul(hidden, w2)
    scaled = (party.tee_trunc(raw2, fractional_bits) if tee_down
              else party.trunc(raw2, fractional_bits))
    return party.open(party.add(scaled, b2))
