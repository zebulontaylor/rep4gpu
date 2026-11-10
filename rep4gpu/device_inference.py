"""Experimental GPU-crypto inference with exact fixed-point rescaling."""
from .device_runtime import DeviceParty


class DeviceInferenceParty(DeviceParty):
    """Use GPU masks and checks with fixed-point rescaling.

    Optimized mode batches mask generation and fuses packed carry and conversion
    work. Both modes execute the same logical messages and produce the same
    local shares for a fixed seed and operation sequence.
    Floor rescaling is exact over the ring. Probabilistic rescaling requires
    ``abs(signed(x)) < 2**(bits-2)`` for each accumulator and rounds to its floor
    or ceiling. The caller must establish that range before execution.
    """
    def __init__(self, party, transport, seeds, device='cuda', *, bits=64,
                 binary_bits=None, optimized=True, gpu_open=False, rounding='floor'):
        if rounding not in ('floor', 'probabilistic'):
            raise ValueError("rounding must be 'floor' or 'probabilistic'")
        self.optimized = optimized
        self.gpu_open = gpu_open
        self.rounding = rounding
        super().__init__(party, transport, seeds, device=device, bits=bits,
                         binary_bits=bits if binary_bits is None else binary_bits,
                         prg_batch=optimized)

    def open(self, x):
        if not self.gpu_open:
            return super().open(x)
        from .device_open import open_device
        return open_device(self, x)

    def _conversion_options(self):
        if not self.optimized:
            return {}
        from .carry_tree_swar import truncation_bits
        from .compact_conversion_gpu import weighted_b2a_gpu
        return dict(carry_impl=truncation_bits, convert_impl=weighted_b2a_gpu)

    def trunc(self, x, shift):
        if self.rounding == 'probabilistic':
            from .device_probabilistic import trunc_pr
            return trunc_pr(self, x, shift)
        from .exact_truncation import trunc
        return trunc(self, x, shift, **self._conversion_options())

    def trunc_relu(self, x, shift):
        if self.rounding == 'probabilistic':
            result = self.trunc(x, shift)
            return self.mul(result, self.sub(self.public(1, like=result), self.sign_bit(result)))
        from .exact_truncation import trunc_relu
        return trunc_relu(self, x, shift, **self._conversion_options())

    def sign_bit(self, x):
        if not self.optimized:
            return super().sign_bit(x)
        from .exact_truncation import sign_bit_packed
        from .carry_tree_swar import carry_bits
        from .compact_conversion_gpu import weighted_b2a_gpu
        return sign_bit_packed(self, x, carry_impl=carry_bits, convert_impl=weighted_b2a_gpu)

    def prepare_weight_share(self, weight, *, storage='limbs'):
        if storage != 'five32':
            return super().prepare_weight_share(weight, storage=storage)
        if self.bits != 32 or weight.binary:
            raise ValueError('five32 weights require arithmetic ring32 shares')
        from .five_product import prepare_five_weights
        return prepare_five_weights(*weight.parts)

    def matmul(self, x, y):
        from .five_product import FiveWeights, local_products_five
        if isinstance(y, FiveWeights):
            if self.bits != 32 or x.binary:
                raise ValueError('five32 multiplication requires arithmetic ring32 shares')
            if self.optimized:
                from .tuned_kernels import local_products_five_tuned
                local = local_products_five_tuned(x.parts, y)
            else:
                local = local_products_five(x.parts, y)
        elif self.optimized:
            if x.binary or y.binary:
                raise ValueError('binary matrix products are not implemented')
            from .tuned_kernels import local_products_tuned
            local = local_products_tuned(*x.parts, *y.parts, bits=self.bits)
        else:
            return super().matmul(x, y)
        contributions = [None]*5
        contributions[(self.party-1)%4], contributions[self.party], contributions[4] = local
        return self._reshare_product(contributions, False, x.device)
