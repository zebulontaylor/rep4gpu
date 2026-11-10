# Exact large-prime GPU products

`rep4gpu.field_q256_bytes` multiplies canonical matrices modulo
`q = 2**256 - 189` using integer tensor cores. It is an arithmetic primitive;
the complete authenticated q256 runtime still uses CPU big integers.

Inputs have shape `[M,K,16]` and `[K,N,16]`, with little-endian radix-65536
limbs stored as CUDA `int32` or `uint16`. Every limb must lie in `[0,65535]`
and every represented value must be below `q`. The caller establishes these
canonicality conditions; the kernel checks shapes, devices, and storage types.
The implementation supports `K <= 4096` and returns the same limb layout.

The byte implementation stores each unsigned byte `u` as the signed byte
`u-128`. It reconstructs every exact unsigned dot product from the signed
tensor-core result and the two byte sums:

```text
sum(u*v) = sum((u-128)*(v-128))
           + 128*(sum(u-128) + sum(v-128)) + 16384*K
```

The resulting byte diagonals are accumulated in int64. High coefficients fold
using `2**256 = 189 mod q`; carry propagation and a final conditional reduction
produce canonical limbs. The `K` bound keeps signed tensor accumulators and
the larger reconstructed coefficients within their integer storage ranges.

For a fixed local weight share, `prepare_q256_bytes(B)` creates an independent
snapshot containing 32 signed bytes per field element and the required column
sums. Passing this snapshot to `field_matmul_q256_bytes(A, prepared)` avoids
repacking `B`. Preparation is real work: count it once per retained snapshot,
and charge it again when weights or their shares change. A raw `B` argument
includes packing in the multiplication call.

The positive seven-bit implementation in `field_q256_tensor` and the SIMT
implementation in `field_q256` remain independent comparison paths. The tests
use Python big-integer products, full-field inputs, carry boundaries, tails,
zero inner dimensions, and prepared-snapshot ownership.

From the repository root:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m pytest -q \
  tests/test_field_q256.py tests/test_field_q256_tensor.py tests/test_field_q256_bytes.py
.venv/bin/python -m benchmarks.field_q256_tensor_bench \
  --output artifacts/field_q256_tensor_bench.json
```

The benchmark separates prepared multiplication, raw-input multiplication,
eager execution, and CUDA-graph execution. These timings exclude secret-share
authentication, cryptographic preprocessing, communication, and model input.
