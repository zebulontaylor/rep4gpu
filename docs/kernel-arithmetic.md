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

In particular, writing `out += dot(a0,b0,0)` with a uint32 `out` can let the compiler fold `out` into the dot's MMA accumulator. The maintained uint64 reconstruction prevents that unsafe fusion. Masking or truncating `out` back to uint32 inside the K loop can reintroduce it.

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
| Optional tuned compact ring32 | `32/128/64` | 4 | One stage |

[`local_products_tuned`](../rep4gpu/tuned_kernels.py) selects the last row only for compact ring32 weights, at least sixteen activation rows, and SM 8.9. It calls the existing uint64-reconstruction kernel with different tile settings. Other architectures, rings, formats, and smaller batches fall back to `local_products`.

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
