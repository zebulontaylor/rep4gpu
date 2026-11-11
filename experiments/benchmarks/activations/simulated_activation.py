"""Public-fixture simulation of two trusted helpers performing exact ReLU.

This benchmark reconstructs activations in ordinary Python/CUDA processes. It
provides no enclave, host isolation, or activation privacy. The helper seed is
shared by a Python factory closure, not protected from the benchmark process.
Weights continue through the existing Rep4 matrix protocol.
"""
import math

import torch

from rep4gpu.gpu.crypto.gpu_prg import DomainKeys, key_bytes
from rep4gpu.models.phase_graph import PhaseGraphParty
from rep4gpu.protocol.runtime import Share


_HELPER_DOMAIN = b'PUBLIC-FIXTURE-SIMULATED-ACTIVATION-v1'


def simulated_activation_party(helper_seed):
    """Return a ring32 benchmark role with a simulated trusted ReLU backend.

    Helpers 0/1 reconstruct the input and independently prepare identical full
    four-component random output shares. Hosts 2/3 receive their three-component
    bundles from helper 0, checked against helper 1's hash. Each nonempty call
    executes two checked exchanges and sends 8*N ring words plus 128 tag bytes
    globally. This is a simulation cost, not a secure deployment protocol.
    """
    helper_seed = bytes(helper_seed)
    key_bytes(helper_seed, _HELPER_DOMAIN)

    class SimulatedActivationParty(PhaseGraphParty):
        activation_backend = 'public_fixture_simulated_trusted_relu'

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if self.bits != 32:
                raise ValueError('the simulated activation benchmark requires ring32')
            self._fixture_helper_key = (DomainKeys(helper_seed, self.device).key(_HELPER_DOMAIN)
                                        if self.party in (0, 1) else None)

        def _fixture_checked_exchange(self, entries, shape):
            # Unlike ordinary local arithmetic, reconstruction must wait for its
            # input verdict even when the surrounding benchmark defers checks.
            builder = self._phase_builder
            previous = None if builder is None else builder.check_at_send
            if builder is not None:
                builder.check_at_send = False
            try:
                return self._exchange_joint_wires(entries, shape, False)
            finally:
                if builder is not None:
                    builder.check_at_send = previous

        def _fixture_masks(self, shape):
            if self.party == 2:
                return None
            if self.party == 3:
                # All three holders of excluded-party stream 2 advance together.
                self._random(2, (0,), self.device, packed=True)
                return None
            key = self._fixture_helper_key
            if self._phase_builder is not None:
                return self._phase_builder.masks((key,), (2,), (3, *shape), 32,
                                                 packed=True)[0].reshape(3, *shape)
            from rep4gpu.gpu.crypto.gpu_prg_batch import generate_rows
            nonce = self._reserve_nonces((2,))[0]
            return generate_rows((key,), (nonce,), 3*math.prod(shape), bits=32,
                                 packed=True)[0].reshape(3, *shape)

        def trunc_relu(self, x, shift):
            if x.binary or x.device != self.device or any(
                    part.dtype != torch.int64 or part.shape != x.shape for part in x.parts):
                raise ValueError('simulated activation requires matching arithmetic int64 shares')
            if not isinstance(shift, int) or not 0 <= shift < 32:
                raise ValueError('simulated activation shift must be in [0,31]')
            if not math.prod(x.shape):
                return x

            def component(index):
                return x.parts[(index-self.party-1) % 4]

            missing_zero = (component(0).to(torch.int32).contiguous().view(torch.uint8).reshape(-1)
                            if self.party in (1, 2) else None)
            missing_one = (component(1).to(torch.int32).contiguous().view(torch.uint8).reshape(-1)
                           if self.party in (0, 3) else None)
            incoming = self._fixture_checked_exchange(
                [(1, 2, 0, missing_zero), (0, 3, 1, missing_one)], x.shape)
            masks = self._fixture_masks(x.shape)
            bundles = None
            if self.party in (0, 1):
                missing = incoming[self.party].view(torch.int32).to(torch.int64).reshape(x.shape)
                word = (x.parts[0]+x.parts[1]+x.parts[2]+missing) & 0xffffffff
                signed = (word ^ 0x80000000)-0x80000000
                clear = (signed >> shift).clamp_min(0)
                random = masks.to(torch.int64)
                fourth = (clear-random[0]-random[1]-random[2]) & 0xffffffff
                components = (*random.unbind(0), fourth)
                bundles = {p: torch.stack([components[(p+j+1) % 4] for j in range(3)])
                           for p in range(4)}
            outgoing = {p: bundles[p].to(torch.int32).contiguous().view(torch.uint8).reshape(-1)
                        for p in (2, 3)} if bundles is not None else {}
            received = self._fixture_checked_exchange(
                [(0, 1, 2, outgoing.get(2)), (0, 1, 3, outgoing.get(3))], (3, *x.shape))
            self.joint_inputs += 4
            if self.party in (0, 1):
                return Share(tuple(bundles[self.party].unbind(0)), False)
            bundle = received[self.party-2].view(torch.int32).to(torch.int64) & 0xffffffff
            return Share(tuple(bundle.reshape(3, *x.shape).unbind(0)), False)

    return SimulatedActivationParty
