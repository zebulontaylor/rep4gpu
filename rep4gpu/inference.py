"""GPU inference runtime with compact exact or masked probabilistic rescaling."""

from .fused_runtime import FusedParty


class InferenceParty(FusedParty):
    """Use GPU share kernels and carry-based fixed-point rescaling.

    ``rounding='floor'`` is exact over the selected ring. ``'probabilistic'``
    requires every rescaled signed accumulator to satisfy
    ``abs(x) < 2**(bits-2)`` and rounds to floor or ceiling with an unbiased
    fresh-mask choice. The caller must establish this range before execution.
    Channels and correlated seeds have the same contracts as ``Party``.
    """

    def __init__(self, party, transport, seeds, device='cuda', *, bits=64,
                 binary_bits=None, prg='aes', rounding='floor'):
        if rounding not in ('floor', 'probabilistic'):
            raise ValueError("rounding must be 'floor' or 'probabilistic'")
        self.rounding = rounding
        super().__init__(party, transport, seeds, device=device, bits=bits,
                         binary_bits=bits if binary_bits is None else binary_bits,
                         prg=prg)

    def trunc(self, x, shift):
        if self.rounding == 'probabilistic':
            from .probabilistic import trunc_pr
            return trunc_pr(self, x, shift)
        from .exact_truncation import trunc
        return trunc(self, x, shift)

    def trunc_relu(self, x, shift):
        if self.rounding == 'probabilistic':
            result = self.trunc(x, shift)
            keep = self.sub(self.public(1, like=result), self.sign_bit(result))
            return self.mul(result, keep)
        from .exact_truncation import trunc_relu
        return trunc_relu(self, x, shift)
