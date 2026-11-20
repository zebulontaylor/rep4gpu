# One-layer activation bit accuracy

The Q24 implementation is not generally bit-exact with floating-point layers. The helper and deterministic private ReLU agree exactly with each other, but smooth activations use different numerical algorithms and rounding. Coarse output formats hide many differences on this fixture; near-zero and rounding-boundary inputs expose more of them. Native FP16/BF16 execution has substantially different rounding from a float64 layer followed by one output cast.

## Workload and references

- Dense fixture: one `32 × 64` input batch, `64 × 128` shared weight matrix, shared bias, and an activation; 4,096 outputs per function. SwiGLU uses two affine projections followed by `SiLU(gate) * value`, without a down projection.
- Activation stress fixture: 8,357 values comprising a uniform grid over `[-4,4]`, that grid offset by one Q24 unit, signed values within 64 Q24 units of zero, and values within 16 Q24 units of both endpoints. This suite tests activation inputs directly, without a dense matrix multiplication.
- ReLU, SiLU, exact-formula GELU (`approximate="none"`), and SwiGLU run in three modes: simulated helper, fully private deterministic floor, and fully private probabilistic truncation. Every mode uses ring64 Q24 and three fresh executions. All four roles agree; every output satisfies its integer oracle.
- Dense inputs, weights, and biases enter through Rep4 private input. The helper reconstructs activation inputs in ordinary CUDA; no enclave or hardware attestation is present.
- The original reference evaluates the unquantized fixture in CPU float64. The quantized reference decodes Q24 inputs and weights before float64 affine computation. The floored reference applies float64 activations to the exact floor-rescaled Q24 preactivations, isolating activation error.
- Format comparisons cast completed outputs through PyTorch to FP32, FP16, BF16, FP8 E4M3FN, or FP8 E5M2. There is no FP8 scaling or native FP8 matrix multiplication. Float64 transcendental functions are numerical references, not proofs of correctly rounded real functions.
- Separate native FP16/BF16 GPU baselines cast original inputs, weights, and biases first, then run PyTorch matmul, bias addition, and activation in that dtype. TF32 and reduced-precision FP16/BF16 reduction are disabled.

The GPU was NVIDIA GeForce RTX 4060 Laptop GPU with PyTorch 2.13.0+cu130. The random fixture seed is fixed; protocol masks are fresh. Rates below range across the three repeats. Repeating deterministic cases does not increase the number of unique test inputs.

## Dense layer versus the original float64 layer

Cells are percentages of outputs with identical encoded bits after the indicated output cast.

| Activation | Backend | FP16 | BF16 | FP8 E4M3FN | FP8 E5M2 | FP32 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| relu | TEE simulation | 99.9756 | 100.0000 | 100.0000 | 100.0000 | 58.7158 |
| relu | Private floor | 99.9756 | 100.0000 | 100.0000 | 100.0000 | 58.7158 |
| relu | Private probabilistic | 99.9512–99.9756 | 99.9756–100.0000 | 100.0000 | 100.0000 | 59.2773–59.6191 |
| silu | TEE simulation | 99.8779 | 100.0000 | 100.0000 | 100.0000 | 15.1367 |
| silu | Private floor | 99.8291 | 100.0000 | 100.0000 | 100.0000 | 3.2471 |
| silu | Private probabilistic | 99.7803–99.8535 | 99.9756–100.0000 | 100.0000 | 100.0000 | 7.2998–7.9102 |
| gelu | TEE simulation | 99.8047 | 100.0000 | 100.0000 | 100.0000 | 13.7695 |
| gelu | Private floor | 99.6826 | 99.9756 | 100.0000 | 100.0000 | 2.1729 |
| gelu | Private probabilistic | 99.7803–99.8291 | 99.9268–99.9756 | 100.0000 | 100.0000 | 6.1523–6.3721 |
| swiglu | TEE simulation | 99.4873 | 99.8779 | 100.0000 | 99.9756 | 6.3232 |
| swiglu | Private floor | 99.1699 | 99.8291 | 100.0000 | 99.9756 | 2.4902 |
| swiglu | Private probabilistic | 99.2920–99.3896 | 99.8535–99.9023 | 100.0000 | 99.9512–99.9756 | 3.6621–4.1992 |

## Differing bits and ULPs in dense outputs

Hamming distance is the number of differing encoded bits, not lost numerical precision. Adjacent values can flip several bits when a mantissa carries. Mean counts include exactly matching outputs as zero. Each cell reports the largest per-run mean, maximum Hamming distance, and maximum finite-pair ULP distance as `mean / max bits / max ULP`.

| Activation | Backend | FP16 | BF16 | FP8 E4M3FN | FP8 E5M2 |
| --- | --- | ---: | ---: | ---: | ---: |
| relu | TEE simulation | 0.001221 / 5 / 1 | 0.000000 / 0 / 0 | 0.000000 / 0 / 0 | 0.000000 / 0 / 0 |
| relu | Private floor | 0.001221 / 5 / 1 | 0.000000 / 0 / 0 | 0.000000 / 0 / 0 | 0.000000 / 0 / 0 |
| relu | Private probabilistic | 0.001465 / 5 / 1 | 0.000244 / 1 / 1 | 0.000000 / 0 / 0 | 0.000000 / 0 / 0 |
| silu | TEE simulation | 0.002197 / 3 / 1 | 0.000000 / 0 / 0 | 0.000000 / 0 / 0 | 0.000000 / 0 / 0 |
| silu | Private floor | 0.004150 / 6 / 1 | 0.000000 / 0 / 0 | 0.000000 / 0 / 0 | 0.000000 / 0 / 0 |
| silu | Private probabilistic | 0.003906 / 4 / 1 | 0.000244 / 1 / 2 | 0.000000 / 0 / 0 | 0.000000 / 0 / 0 |
| gelu | TEE simulation | 0.003906 / 6 / 1 | 0.000000 / 0 / 0 | 0.000000 / 0 / 0 | 0.000000 / 0 / 0 |
| gelu | Private floor | 0.006104 / 6 / 1 | 0.000732 / 3 / 1 | 0.000000 / 0 / 0 | 0.000000 / 0 / 0 |
| gelu | Private probabilistic | 0.004883 / 7 / 1 | 0.001221 / 2 / 2 | 0.000000 / 0 / 0 | 0.000000 / 0 / 0 |
| swiglu | TEE simulation | 0.010742 / 7 / 1 | 0.001709 / 3 / 12 | 0.000000 / 0 / 0 | 0.000488 / 2 / 1 |
| swiglu | Private floor | 0.018311 / 5 / 1 | 0.002686 / 4 / 20 | 0.000000 / 0 / 0 | 0.000488 / 2 / 1 |
| swiglu | Private probabilistic | 0.016846 / 7 / 1 | 0.002197 / 3 / 20 | 0.000000 / 0 / 0 | 0.000977 / 2 / 1 |

## Activation-only stress accuracy

These comparisons use the floored reference, so they isolate activation error from input and matrix quantization. Exact percentages count a signed-zero difference as a bit mismatch. The final column counts signed-zero-only mismatches in FP8 E4M3FN; those cases have no numerical difference after casting.

| Activation | Backend | FP16 exact % | BF16 exact % | E4M3FN exact % | E5M2 exact % | E4M3FN signed-zero mismatches, max |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| relu | TEE simulation | 100.0000 | 100.0000 | 100.0000 | 100.0000 | 0 |
| relu | Private floor | 100.0000 | 100.0000 | 100.0000 | 100.0000 | 0 |
| relu | Private probabilistic | 100.0000 | 100.0000 | 100.0000 | 100.0000 | 0 |
| silu | TEE simulation | 99.9043 | 99.2102 | 99.9880 | 99.9880 | 1 |
| silu | Private floor | 99.0068 | 99.1743 | 100.0000 | 100.0000 | 0 |
| silu | Private probabilistic | 99.4974–99.6051 | 99.1983–99.2102 | 99.9880–100.0000 | 99.9880–100.0000 | 1 |
| gelu | TEE simulation | 98.1213 | 99.0068 | 99.9880 | 99.9880 | 1 |
| gelu | Private floor | 95.9316 | 98.6957 | 99.9761 | 100.0000 | 0 |
| gelu | Private probabilistic | 94.7589–95.1777 | 98.4205–98.4564 | 99.9641–99.9761 | 99.9880–100.0000 | 1 |
| swiglu | TEE simulation | 99.9043 | 99.1384 | 99.9761 | 99.9761 | 2 |
| swiglu | Private floor | 98.7077 | 98.7077 | 99.9521 | 99.9521 | 4 |
| swiglu | Private probabilistic | 98.8632–98.8991 | 98.7436–98.7555 | 99.9521–99.9761 | 99.9521–99.9761 | 4 |

## Private outputs versus the simulated helper

Integer equality refers to complete Q24 words. `Max Q24 units` is an integer difference, not a floating ULP. Rates in the output formats can be higher because those casts erase differences.

| Suite | Activation | Private rounding | Equal Q24 words, minimum | Total words per run | Max Q24 units | FP16 exact % | BF16 exact % |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| dense | relu | Private floor | 4096 | 4096 | 0 | 100.0000 | 100.0000 |
| dense | relu | Private probabilistic | 3063 | 4096 | 1 | 99.9756–100.0000 | 99.9756–100.0000 |
| dense | silu | Private floor | 1013 | 4096 | 11 | 99.9023 | 100.0000 |
| dense | silu | Private probabilistic | 1593 | 4096 | 7 | 99.8779–99.9268 | 99.9756–100.0000 |
| dense | gelu | Private floor | 601 | 4096 | 9 | 99.8291 | 99.9756 |
| dense | gelu | Private probabilistic | 1307 | 4096 | 7 | 99.8291–99.9268 | 99.9268–99.9756 |
| dense | swiglu | Private floor | 1519 | 4096 | 16 | 99.2920 | 99.9268 |
| dense | swiglu | Private probabilistic | 2351 | 4096 | 9 | 99.6094–99.7070 | 99.9023–99.9512 |
| stress | relu | Private floor | 8357 | 8357 | 0 | 100.0000 | 100.0000 |
| stress | relu | Private probabilistic | 8357 | 8357 | 0 | 100.0000 | 100.0000 |
| stress | silu | Private floor | 570 | 8357 | 12 | 99.0427 | 99.1863 |
| stress | silu | Private probabilistic | 2040 | 8357 | 8 | 99.5214–99.6650 | 99.5453–99.6769 |
| stress | gelu | Private floor | 1033 | 8357 | 8 | 96.5059 | 98.6718 |
| stress | gelu | Private probabilistic | 1606 | 8357 | 7 | 95.0102–95.3811 | 98.6957–98.7675 |
| stress | swiglu | Private floor | 637 | 8357 | 39 | 98.7077 | 98.9470 |
| stress | swiglu | Private probabilistic | 1564 | 8357 | 29 | 98.9470–98.9590 | 99.0308–99.0547 |

## Native low-precision GPU baselines

These rates compare against actual native FP16/BF16 layer outputs, not the float64-then-cast reference. The lower match rates show why specifying only the output dtype is insufficient for bit-exact verification: input precision, intermediate rounding, activation kernels, and operation order also matter.

| Activation | Backend | Native FP16 exact % | Native BF16 exact % |
| --- | --- | ---: | ---: |
| relu | TEE simulation | 73.1201 | 73.3398 |
| relu | Private floor | 73.1201 | 73.3398 |
| relu | Private probabilistic | 73.1201 | 73.3398 |
| silu | TEE simulation | 43.1641 | 42.3340 |
| silu | Private floor | 43.1641 | 42.3340 |
| silu | Private probabilistic | 43.1641–43.1885 | 42.3340 |
| gelu | TEE simulation | 45.6299 | 45.3613 |
| gelu | Private floor | 45.6787 | 45.3613 |
| gelu | Private probabilistic | 45.6055–45.6299 | 45.3369–45.3857 |
| swiglu | TEE simulation | 26.4893 | 26.1230 |
| swiglu | Private floor | 26.4648 | 26.1230 |
| swiglu | Private probabilistic | 26.4404–26.4648 | 26.1230 |

## Reproduce and inspect

From the repository root, run:

```sh
.venv/bin/python -m experiments.benchmarks.activations.layer_bit_accuracy \
  --output experiments/results/activations/layer_bit_accuracy_q24.json
```

The JSON includes per-run error magnitudes, Hamming histograms, mismatch counts for every bit position, ULP statistics, signed-zero counts, nonfinite counts, native baselines, and source hashes. The NPZ preserves every opened integer output. No nonfinite outputs occurred in these fixtures. Protocol execution times include setup and are not TEE or distributed-network latency measurements.

- [Detailed measurements](layer_bit_accuracy_q24.json)
- [Raw integer outputs](layer_bit_accuracy_q24.npz)
- [Execution log](layer_bit_accuracy_q24.log)
- [MXFP4 and NVFP4 comparison](layer_fp4_accuracy_report.md)
- [Benchmark source](../../benchmarks/activations/layer_bit_accuracy.py)
- [Helper simulation](../../../docs/experiments/seeded-tee-simulation.md)

All cases passed party agreement and integer-oracle checks. The helper tests cover captured and uncaptured execution for all four activations; bit-metric tests cover adjacent-value carries, signed zeros, and nonfinite handling. These finite fixtures measure accuracy, not a universal bit-exactness or quantization-substitution guarantee.
