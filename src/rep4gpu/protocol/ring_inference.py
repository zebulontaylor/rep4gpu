"""GPU-crypto ring Rep4 inference with exact floor or probabilistic rescaling."""
from .device_runtime import DeviceParty


class DeviceInferenceParty(DeviceParty):
    """Use GPU ChaCha20 masks and BLAKE3 checks with fixed-point rescaling.

    Optimized mode batches mask generation and fuses packed carry and conversion
    work. Both modes execute the same logical messages and produce the same
    local shares for a fixed seed and operation sequence.
    Floor rescaling is exact over the ring. Probabilistic rescaling requires
    ``abs(signed(x)) < 2**(bits-2)`` for each accumulator and rounds to its floor
    or ceiling. The caller must establish that range before execution.
    Packed ring16 arithmetic keeps matrix and elementwise operations narrow.
    Sign and truncation temporarily widen shares for the existing conversion
    kernels and repack arithmetic results, without opening intermediates.
    """

    def __init__(self, party, transport, seeds, device='cuda', *, bits=64,
                 binary_bits=None, optimized=True, gpu_open=False, rounding='floor',
                 bounded_sign=False, sparse_products=False, narrow_products=False,
                 packed_arithmetic=False):
        if bits in (8, 16) and rounding == 'probabilistic':
            raise ValueError('ring8/ring16 device inference requires exact floor rounding')
        if rounding not in ('floor', 'probabilistic'):
            raise ValueError("rounding must be 'floor' or 'probabilistic'")
        self.optimized = optimized
        self.gpu_open = gpu_open
        self.rounding = rounding
        self.bounded_sign = bool(bounded_sign)
        self.sparse_products = bool(sparse_products)
        self.narrow_products = bool(narrow_products)
        super().__init__(
            party, transport, seeds, device=device, bits=bits,
            binary_bits=(32 if bits in (8, 16) else bits) if binary_bits is None else binary_bits,
            prg_batch=optimized, packed_arithmetic=packed_arithmetic,
        )

    def _wide_conversion(self, method, x, *args):
        """Keep existing bit conversions wide, then restore arithmetic storage.

        Party operations execute sequentially; graph capture records this
        fixed storage boundary without changing the live nonce schedule.
        """
        import torch
        from .runtime import Share
        self.packed_arithmetic = False
        try:
            result = method(Share(tuple(t.to(torch.int64) for t in x.parts), x.binary), *args)
        finally:
            self.packed_arithmetic = True
        return self._share(result.parts, result.binary)

    def open(self, x):
        self._validate_share_storage(x)
        if not self.gpu_open:
            return super().open(x)
        from .device_open import open_device
        return open_device(self, x)

    def _conversion_options(self):
        if not self.optimized:
            return {}
        from ..gpu.bits.carry_tree_swar import truncation_bits
        from ..gpu.bits.compact_conversion_gpu import weighted_b2a_gpu
        return dict(carry_impl=truncation_bits, convert_impl=weighted_b2a_gpu)

    def trunc(self, x, shift):
        self._validate_share_storage(x)
        if self.packed_arithmetic:
            return self._wide_conversion(self.trunc, x, shift)
        if self.rounding == 'probabilistic':
            from ..gpu.bits.device_probabilistic import trunc_pr
            return trunc_pr(self, x, shift)
        from ..arithmetic.exact_truncation import trunc
        return trunc(self, x, shift, **self._conversion_options())

    def trunc_relu(self, x, shift):
        self._validate_share_storage(x)
        if self.packed_arithmetic:
            return self._wide_conversion(self.trunc_relu, x, shift)
        if self.rounding == 'probabilistic':
            result = self.trunc(x, shift)
            if self.bounded_sign and shift > 0:
                from ..gpu.bits.bounded_activation import posttrunc_sign
                sign = posttrunc_sign(self, result, shift)
            else:
                sign = self.sign_bit(result)
            return self.mul(result, self.sub(self.public(1, like=result), sign))
        from ..arithmetic.exact_truncation import trunc_relu
        return trunc_relu(self, x, shift, **self._conversion_options())

    def sign_bit(self, x):
        self._validate_share_storage(x)
        if self.packed_arithmetic:
            return self._wide_conversion(self.sign_bit, x)
        if not self.optimized:
            return super().sign_bit(x)
        from ..arithmetic.exact_truncation import sign_bit_packed
        from ..gpu.bits.carry_tree_swar import carry_bits
        from ..gpu.bits.compact_conversion_gpu import weighted_b2a_gpu
        return sign_bit_packed(self, x, carry_impl=carry_bits, convert_impl=weighted_b2a_gpu)

    def prepare_weight_share(self, weight, *, storage='limbs'):
        self._validate_share_storage(weight)
        if storage == 'strassen32':
            if self.bits != 32 or weight.binary:
                raise ValueError('strassen32 weights require arithmetic ring32 shares')
            from .runtime import PreparedMatrix
            from ..gpu.matmul.strassen_kernels import prepare_strassen_weight
            packed = super().prepare_weight_share(weight, storage='words32')
            return PreparedMatrix(tuple(prepare_strassen_weight(part, levels=1) for part in packed.parts))
        return super().prepare_weight_share(weight, storage=storage)

    def matmul(self, x, y):
        self._validate_share_storage(x, y)
        from ..gpu.matmul.strassen_kernels import StrassenWeight, local_products_strassen
        if self.bits in (8, 16):
            if x.binary or y.binary:
                raise ValueError('binary matrix products are not implemented')
            from ..gpu.matmul.kernels import local_products
            local = local_products(*x.parts, *y.parts, bits=self.bits)
        elif hasattr(y, 'parts') and isinstance(y.parts[0], StrassenWeight):
            if self.bits != 32 or x.binary or y.binary:
                raise ValueError('strassen32 multiplication requires arithmetic ring32 shares')
            local = local_products_strassen(x.parts, y.parts, opaque=self.optimized)
        elif self.optimized:
            if x.binary or y.binary:
                raise ValueError('binary matrix products are not implemented')
            from ..gpu.matmul.tuned_kernels import local_products_tuned
            local = local_products_tuned(*x.parts, *y.parts, bits=self.bits)
        else:
            return super().matmul(x, y)
        contributions = [None] * 5
        contributions[(self.party - 1) % 4], contributions[self.party], contributions[4] = local
        return self._reshare_product(contributions, False, x.device)
