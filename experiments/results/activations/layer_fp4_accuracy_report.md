# MXFP4 and NVFP4 activation accuracy

On the dense fixture, both formats retain identical 4-bit element codes for all four activations and all three backends. MXFP4 scales also match. NVFP4 frequently selects a different FP32 tensor scale, so identical element codes do not imply an identical complete representation. The activation stress fixture exposes element-code differences as well.

## Method

This experiment reuses the preserved integer outputs from the [one-layer Q24 benchmark](layer_bit_accuracy_report.md): 4,096 dense-layer outputs or 8,357 activation stress inputs, three runs each, for ReLU, SiLU, GELU, and SwiGLU. The backends are the ordinary CUDA helper simulation, fully private floor truncation, and fully private probabilistic truncation. No MPC computation was rerun or replaced.

The following deterministic software output quantizers use FP32 inputs and scale arithmetic, nearest-even E2M1 rounding, and saturation. They are not native GPU kernel emulators.

- MXFP4: 32 consecutive elements per row share an E8M0 scale. The selected exponent is `floor(log2(block_amax)) - 2`, bounded by the format range. Zero blocks use exponent −127. This is the conversion rule in [OCP MX v1.0, section 6.3](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf).
- NVFP4: 16 consecutive elements share an E4M3 block scale, with a tensor-wide FP32 scale `tensor_amax / (448 * 6)`. The block scale is the E4M3-rounded value of `(block_amax / 6) / tensor_scale`. Zero blocks use local scale zero; all-zero tensors use global scale one. This follows the [documented two-level scaling formula](https://docs.nvidia.com/deeplearning/transformer-engine/features/low_precision_training/nvfp4/nvfp4.html). No stochastic cast, Hadamard transform, or 2D weight scaling is included.

Blocks run along the last dimension without crossing row boundaries. Final partial blocks are zero-padded; padded payload bits do not contribute to mismatch counts. FP4 element codes preserve signed zero. Tests cover rounding ties, saturation, exponent boundaries, zero scales, partial blocks, and identical element codes with differing scale metadata.

Dense results compare to the original float64 layer, then apply the same declared quantizer to both outputs. Stress results use the common floor-rescaled activation inputs to isolate activation error. “Dynamic” recomputes scales independently. “Frozen” uses the exact reference block and tensor scales for both outputs, isolating value rounding.

## Dense-layer element codes and scales

All rows have 100% matching element codes and zero differing payload bits for both formats, in every repeat. MXFP4 block scales also match in every dense run. The table below exposes NVFP4 metadata differences. The tensor scale is one stored FP32 word, so its differing bits are counted once, not once per output.

| Activation | Backend | NVFP4 block-scale bits different, max | NVFP4 tensor-scale bits different, max | Complete NVFP4 element encoding match % | Frozen-scale NVFP4 match % |
| --- | --- | ---: | ---: | ---: | ---: |
| relu | TEE simulation | 0 | 2 | 0.0000 | 100.0000 |
| relu | Private floor | 0 | 2 | 0.0000 | 100.0000 |
| relu | Private probabilistic | 0 | 2 | 0.0000 | 100.0000 |
| silu | TEE simulation | 0 | 0 | 100.0000 | 100.0000 |
| silu | Private floor | 0 | 2 | 0.0000 | 100.0000 |
| silu | Private probabilistic | 0 | 0 | 100.0000 | 100.0000 |
| gelu | TEE simulation | 0 | 1 | 0.0000 | 100.0000 |
| gelu | Private floor | 0 | 7 | 0.0000 | 100.0000 |
| gelu | Private probabilistic | 0 | 7 | 0.0000 | 100.0000 |
| swiglu | TEE simulation | 0 | 1 | 0.0000 | 100.0000 |
| swiglu | Private floor | 0 | 6 | 0.0000 | 100.0000 |
| swiglu | Private probabilistic | 0 | 8 | 0.0000–100.0000 | 100.0000 |

A 0% complete-element match above means every element depends on a tensor scale whose bits differ; it does not mean every 4-bit payload is wrong. Zero-valued elements can still decode to equal numbers. The JSON reports decoded equality and numerical error separately.

## Activation stress results

The percentages below compare the four payload bits only, using independently selected scales. Bit totals are summed over the 8,357 logical element codes and shown as the largest value across repeats. Scale differences are recorded separately in the JSON.

| Activation | Backend | MXFP4 code match % | MXFP4 differing payload bits, max | NVFP4 code match % | NVFP4 differing payload bits, max |
| --- | --- | ---: | ---: | ---: | ---: |
| relu | TEE simulation | 100.0000 | 0 | 100.0000 | 0 |
| relu | Private floor | 100.0000 | 0 | 100.0000 | 0 |
| relu | Private probabilistic | 100.0000 | 0 | 100.0000 | 0 |
| silu | TEE simulation | 99.8564 | 17 | 99.9880 | 1 |
| silu | Private floor | 99.8085 | 27 | 100.0000 | 0 |
| silu | Private probabilistic | 99.8205–99.8325 | 24 | 99.9880–100.0000 | 1 |
| gelu | TEE simulation | 99.8564 | 17 | 99.9880 | 1 |
| gelu | Private floor | 99.8085 | 27 | 100.0000 | 0 |
| gelu | Private probabilistic | 99.7846–99.8205 | 28 | 99.9880–100.0000 | 1 |
| swiglu | TEE simulation | 99.9402 | 9 | 99.9761 | 2 |
| swiglu | Private floor | 99.8803 | 11 | 99.9521 | 4 |
| swiglu | Private probabilistic | 99.8205–99.9043 | 22 | 99.9521–99.9761 | 4 |

Some NVFP4 payload mismatches are signed zero only. For deterministic TEE/private-floor stress runs, all NVFP4 payload mismatches are signed-zero differences. This does not remove tensor-scale differences affecting nonzero decoded values. Hamming distance counts differing encoding bits, not lost numerical precision.

## Quantizing the layer operands

The output-cast experiment does not execute the layer in FP4. A separate software experiment quantizes inputs and weights along their reduction dimension, decodes them, and evaluates matmul, bias addition, and activation in FP32 on CPU. Both inputs and weights use 1D blocks, including NVFP4 weights. This excludes native Tensor Core accumulation behavior, weight calibration, and vendor training recipes.

The following payload match rates compare the TEE output with that quantized-operand result, with common scales taken from the quantized-operand result. They differ substantially from simply quantizing the completed high-precision output.

| Activation | MXFP4 operand emulation, code match % | NVFP4 operand emulation, code match % |
| --- | ---: | ---: |
| relu | 79.5898 | 80.9814 |
| silu | 62.6465 | 63.6230 |
| gelu | 67.2119 | 67.4805 |
| swiglu | 55.4199 | 54.9072 |

## Reproduce

From the repository root, with the previous JSON and NPZ available:

```sh
.venv/bin/python -m experiments.benchmarks.activations.layer_fp4_accuracy
.venv/bin/python -m pytest tests/benchmarks/test_block_fp4.py \
  tests/benchmarks/test_layer_fp4_accuracy.py -q
```

Twelve tests passed. The comparison evaluated all 24 saved cases and all three repeats. JSON records hashes of the input artifacts and source files, per-bit mismatch counts, payload and scale differences, full representation equality, decoded errors, dynamic/frozen comparisons, and the separate operand emulation.

- [Detailed results](layer_fp4_accuracy.json)
- [Quantizer source](../../benchmarks/activations/block_fp4.py)
- [Comparison source](../../benchmarks/activations/layer_fp4_accuracy.py)

Exact matching depends on the quantization recipe, block layout, scales, and arithmetic. A 100% match on this finite dense fixture is not a universal guarantee or proof of a prover’s execution precision.
