# Exact GPU ring matrix products

[`ring_matmul`](../rep4gpu/kernels.py) and `local_products` multiply integer matrices modulo `2^32` or `2^64`. They accept `torch.int64` activation components. Ring32 outputs are canonical unsigned values stored in int64 tensors; ring64 outputs retain their signed int64 bit patterns. Signed interpretation belongs to the surrounding [Rep4 protocol](protocol.md).

The kernels use integer tensor-core products or integer SIMT reductions. Floating-point matrix multiplication cannot replace them: replicated components span the entire ring even when the reconstructed model values are small.

## Balanced signed-byte convolution

For `L = bits/8`, let `C = sum(128 * 256^i, i=0..L-1)`. The tensor-core path extracts signed digits

```text
d_i(x) = (((x + C) >> (8*i)) & 255) - 128.
```

The addition wraps in the ring. Each digit lies in `[-128,127]`, and `x = sum(d_i(x) * 256^i)` modulo the ring. Carries from adding `C` matter: the balanced digits of 255 begin `[-1,1]`, whose weighted sum is `-1 + 256 = 255`.

A product needs only diagonals `d = i+j < L`; higher diagonals vanish modulo `2^bits`. Each diagonal sums exact signed-int8 dot products, then shifts that integer sum by `8*d`. Ring32 uses ten limb-pair products per scalar matrix product; ring64 uses thirty-six. Static [`PreparedWeight`](../rep4gpu/kernels.py) tensors cache the balanced weight digits.

For a 64-element K tile, diagonal `d` has absolute accumulator bound `(d+1) * 64 * 16384`. The largest bounds are 4,194,304 for ring32 and 8,388,608 for ring64, both below signed int32 limits. `local_products` resets this accumulator for each diagonal, K tile, and local product term. A combined activation such as `x0+x1` is re-encoded into balanced digits, so its individual digits obey the same bound.

The completed int32 diagonal is sign-extended to int64, interpreted as uint64, shifted, and added to the uint64 reconstructed result. This preserves negative diagonals and makes shifts through bit 56 valid. The final mask selects low 32 bits for ring32; uint64 wrap already supplies ring64 arithmetic.

## Keep ring reconstruction out of saturating MMA accumulators

The generated integer MMA instruction uses `.satfinite.s32`: an overflowing signed int32 accumulator saturates instead of wrapping. Its accumulator must contain a bounded diagonal sum. An arbitrary reconstructed uint32 ring value does not satisfy that bound.

In particular, writing `out += dot(a0,b0,0)` with a uint32 `out` can let the compiler fold `out` into the dot's MMA accumulator. The baseline uint64 reconstruction prevents that unsafe fusion. Masking or truncating `out` back to uint32 inside the K loop can reintroduce it.

[`test_saturating_fusion_regression`](../tests/test_tuned_kernels.py) uses seed `3280909` and dimensions `128×71×39`. In the rejected uint32-reconstruction variant, one accumulator follows:

```text
2,147,444,789 + 16,800 + 54,187 = 2,147,515,776
signed saturation:                         2,147,483,647
lost contribution:                              32,129
```

That loss changes output 0 at `(102,18)` from `3,609,944,954` to `3,609,912,825`. The regression checks all local outputs against integer CPU arithmetic. Small cleartext bounds cannot rule out this failure because the local replicated components remain full-ring values.

A different design can retain separate int32 diagonal accumulators through the full K reduction and pack only afterward, provided every intermediate sum has a proved bound. For two local terms in ring32, the conservative bound is `8*K*16384 < 2^31`, requiring `K < 16384`. That condition does not describe or expand the maintained baseline's interface.

## Storage and dispatch

Static weight formats in [`kernels.py`](../rep4gpu/kernels.py) have these costs per component matrix entry:

| Format | Ring32 | Ring64 | Behavior |
|---|---:|---:|---|
| Raw int64 | 8 bytes | 8 bytes | Extract balanced digits during tensor-core execution. |
| Balanced int8 limbs | 4 bytes | 8 bytes | Cache four or eight digit planes; rebuild after changing weights. |
| Compact int32 words | 4 bytes | — | Preserve ring32 words and avoid digit-plane reconstruction in decode. |

`keep_original=True` retains another 8 bytes per entry with limb storage. Compact words retain no original int64 tensor. One-row `local_products` uses a SIMT decode reduction: raw or compact weights load words directly; limb-only weights reconstruct each ring word from signed digit planes. The direct `ring_matmul` auto selector uses decode for one-row, unprepared products with `K >= 2048`; its other auto cases use the tensor-core path.

| Local-product selector | M/N/K tile | Warps | Pipeline |
|---|---|---:|---|
| Baseline tensor core | `16/32/64` | 4 | Triton default |
| One-row SIMT decode | `1/32/32` | 4 | Integer reduction |
| Optimized compact ring32 local products | `32/128/32` | 4 | One stage |
| Tuned compact ring32 single product | `32/128/64` | 4 | One stage |

[`local_products_tuned`](../rep4gpu/tuned_kernels.py) selects the optimized local-product row only for compact ring32 weights, at least sixteen activation rows, and SM 8.9. It uses the explicit modular-add kernel described below. Other architectures, rings, formats, and smaller batches fall back to `local_products`. `ring_matmul_tuned` retains the uint64 reconstruction and its `32/128/64` tile for single products.

## Explicit modular accumulation for ring32

[`ring32_opaque.py`](../rep4gpu/ring32_opaque.py) reconstructs each ring32
output with an explicit CUDA `add.u32` inline-assembly operation. The compiler
sees this addition as an opaque operation, so it cannot fold the arbitrary
ring accumulator into a saturating MMA accumulator. Each diagonal still starts
at zero and accumulates only its bounded signed-byte dot products. Casting the
completed diagonal to uint32, shifting it, and adding modulo `2**32` gives the
same low word as the baseline uint64 reconstruction.

The selected K tile is 32. Its largest diagonal has magnitude at most
`4*32*128**2 = 2,097,152`, well below the signed int32 limit. Input components
are reduced to uint32 before balanced-byte extraction; full-width int64 input
bit patterns therefore retain their exact ring32 interpretation. Outputs are
canonical unsigned ring32 values stored in int64, as in the baseline.

This is a local arithmetic optimization. It changes no masks, messages,
rounding, recipients, or protocol checks. The original saturation regression,
worst-case balanced digits, partial tiles, and full-width random inputs are
covered by [exact integer tests](../tests/test_ring32_opaque.py).

## Optional matrix Strassen preparation

`DeviceInferenceParty.prepare_weight_share(weight, storage='strassen32')`
prepares one level of exact matrix Strassen for ring32 prefill. Each weight
component is divided into four blocks. The seven stored combinations are
`B00+B11`, `B00`, `B01-B11`, `B10-B00`, `B11`, `B00+B01`, and `B10+B11`,
all modulo `2**32`. No query-dependent work is moved to preparation.

At inference time, a fused kernel forms the seven corresponding combinations
of each activation component. A batched grid computes all seven leaf products
and all three Rep4 local contributions, then another kernel reconstructs the
three original contributions. Strassen's identities use only addition and
multiplication, so they hold in this ring for every share bit pattern. The
existing resharing protocol receives exactly the same three contributions.
Optimized mode uses the explicit modular-add reconstruction for leaf products.

One level performs seven half-size matrix products instead of eight, reducing
scalar product count by 12.5%. The batched grid also changes kernel occupancy;
the measured speedup need not equal the arithmetic reduction. Every activation
sum, leaf product, and output combination remains inside the online timer.

The format retains 1.75 times the storage of the three compact components:
21 bytes per original matrix entry per party instead of 12. The full two-layer
2048 → 8192 → 2048 fixture retains 704,643,072 weight bytes per party. This is
static weight caching; it requires no one-use preprocessing correlations.
Matrix dimensions and activation row counts must be divisible by two. Use
`words32` for single-row decode. The lower-level experimental functions also
allow two levels, but the inference storage option selects only one because
additional subdivision regressed the down projection.

The implementation is in [strassen_kernels.py](../rep4gpu/strassen_kernels.py).
[Kernel tests](../tests/test_strassen_kernels.py) compare against full-width
integer arithmetic, and [inference tests](../tests/test_device_inference.py)
check the same local shares, wire transcript, output, and counters as `words32`.

## Optional five-GEMM local algebra

The standard three local contributions expand into six matrix products. [`five_product.py`](../rep4gpu/five_product.py) instead prepares these ring32 products:

```text
p0 = (x0+x1) @ (y0+y1)       p1 = x1 @ y1
p2 = (x1+x2) @ (y1+y2)       p3 = x2 @ y2
p4 = (x0+x1+x2) @ (y0+y1+y2)

v0 = p0-p1
v1 = p2-p3
v4 = p4-p0-p2+p1
```

Expansion gives the same `v0`, `v1`, and `v4` as the six-product kernel, modulo `2^32`. The surrounding joint-input schedule is unchanged.

`FiveWeights` stores five compact matrices: `y0+y1`, `y1`, `y1+y2`, `y2`, and `y0+y1+y2`. This costs 20 bytes per entry versus 12 bytes for the original three compact components. Two stored matrices alias original components. Retaining both formats therefore costs 24 bytes per entry, excluding source int64 tensors and temporary workspace. The option also adds activation sums and product-combination kernels; five GEMMs alone do not guarantee lower latency. Use [`five_product_bench.py`](../benchmarks/five_product_bench.py) to compare the selected shapes and storage costs.
