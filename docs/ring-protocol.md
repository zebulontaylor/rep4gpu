# Ring Rep4 Protocol ($\mathbb{Z}/2^k$)

## Share layout and multiplication

A secret value $x \in \mathbb{Z}/2^k$ ($k \in \{8, 16, 32, 64\}$) is split into
four additive components $x \equiv x_0 + x_1 + x_2 + x_3 \pmod{2^k}$ (or XOR for
binary shares). Party $p \in \{0, 1, 2, 3\}$ holds the three components
$(x_{(p+1)\bmod 4}, x_{(p+2)\bmod 4}, x_{(p+3)\bmod 4})$ and three 256-bit PRG
seeds, one shared by each triple excluding party $e \ne p$.

Multiplication (elementwise or matrix) evaluates local cross-component products
grouped into six joint-input routes `(sender, backup, receiver, outsider)`
(`SCHEDULE` in `protocol/runtime.py`):
- `sender` and `backup` both know the local contribution $v$ and mask $m$.
- `sender` sends $v - m$ to `receiver`; `backup` sends a 32-byte digest
  `H(v - m)` to `receiver`.
- `receiver` validates the frame size and rejects with an exception if
  `H(received) != tag`.

## GPU-resident cryptography (`DeviceParty`)

`DeviceParty` replaces host PRG/BLAKE2b with CUDA kernels while preserving the
exact six-joint Rep4 route schedule and 32-byte tag length:

- **Masks** (`gpu/crypto/gpu_prg*.py`): Domain-separated 256-bit keys are derived
  via SHA-256 from each triple seed and public metadata `(bits, binary_bits,
  is_binary, wire_width, excluded)`. ChaCha20 runs on CUDA with a 64-bit block
  counter (starting at 0) and a monotonically incremented 64-bit draw nonce
  reserved atomically before launch.
- **Integrity tags** (`gpu/crypto/gpu_hash.py`): Official unkeyed BLAKE3-256 on
  CUDA (chunk counters, tree parents, `ROOT` finalization), verified against CPU
  `blake3`.
- **Checked opening** (`protocol/device_open.py`): When `gpu_open=True`, each
  party sends its missing component clockwise and a BLAKE3 backup tag
  counterclockwise; reconstruction happens only after tag verification succeeds.

## Exact ring matrix multiplication (`gpu/matmul/`)

Secret ring products never cast values to floating point:
- **64-bit & 32-bit limbs** (`kernels.py`): Exact balanced `int8` limb
  decomposition executed on INT8 Tensor Cores via Triton.
- **Compact `words32` & SM89 modular-add** (`ring32_opaque.py`, `tuned_kernels.py`):
  Stores each ring32 weight component in 4 bytes and uses explicit modular
  addition (`32/128/32` tile on SM 8.9) to prevent saturating MMA overflow.
- **Exact 1-level Strassen (`strassen32`)** (`strassen_kernels.py`): Precomputes
  static weight combinations (1.75x compact storage) and runs batched Strassen
  local products for even batch/matrix dimensions without changing wire messages
  or ring outputs.
- **Ring8 & Ring16** (`kernels.py`, `packed_arithmetic=True`): Exact narrow ring
  products and optional `uint8`/`uint16` share storage.

## Fixed-point truncation and activations

`FixedPoint` (`arithmetic/activations.py`) provides nearest-even plaintext
encoding (`encode`), explicit decoding (`decode`), `matmul`, `matmul_relu`,
`relu`, `gelu`, `silu`, and arbitrary `polynomial` evaluation.

Two rescaling modes are supported in `DeviceInferenceParty`:

1. **`rounding='floor'` (exact)**: Carry-based truncation
   (`arithmetic/exact_truncation.py`, `gpu/bits/carry_tree_swar.py`,
   `gpu/bits/compact_conversion_gpu.py`) computes only the required carry/sign
   bits via a bit-packed SWAR carry tree and converts them back with compact
   weighted B2A, matching signed arithmetic right-shift `x >> s` exactly.
2. **`rounding='probabilistic'`**: Small-gap masked truncation
   (`gpu/bits/device_probabilistic.py`) requires every rescaled signed
   accumulator to satisfy $|x| < 2^{k-2}$. Each result equals $\lfloor x / 2^s \rfloor$
   or $\lceil x / 2^s \rceil$ with unbiased rounding under uniform fresh masks.
   Optional optimizations preserve this exact rounding law:
   - `bounded_sign=True` (`gpu/bits/bounded_activation.py`): Extracts the ReLU
     sign bit from the reduced post-truncation width $k - s$.
   - `sparse_products=True`: Uses the balanced 4-joint route when operands have
     public complementary support `{0,1}` vs `{2,3}` (saves 1/3 payload on
     selected truncation/sign/B2A products).
   - `narrow_products=True` (`gpu/matmul/narrow_product_gpu.py`): Uses 16-bit
     checked products for truncation corrections multiplied by $-2^{k-s}$ when
     $1 \le s \le 16$, where any lift error divisible by $2^{16}$ vanishes mod $2^k$.

## Captured CUDA phase graphs (`PhaseGraphParty`)

`prepare_inference(fn, *sample_inputs, check_at_send=...)` traces one dry run
(consuming zero live counters and sending no messages) and captures CUDA graph
segments between transport exchanges:
- Each `plan.run(*inputs)` reserves fresh nonces atomically, uploads them to
  persistent device nonce vectors, and replays the captured segments.
- With `check_at_send=True`, GPU tag comparisons are packed into one pinned host
  verdict transfer before every transport send or output release.
- Any failed check aborts immediately and poisons the plan; reserved nonces stay
  consumed.
