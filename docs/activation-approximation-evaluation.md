# Bounded GELU and SiLU approximation evaluation

Mixed-weight degree-eight SiLU gives the lowest fixed-point FFN error among the tested uniform and mixed SiLU fits. Mixed-weight degree-ten GELU improves on degree eight, with another multiplication and rescaling stage. Both cost more private execution time than the existing local quadratics. These candidates are benchmark implementations; the default inference activation is unchanged.

## Approximation quality

The quality benchmark uses the original synthetic 128 × 2048 → 8192 → 2048 FFN weights and inputs. It quantizes activation inputs and evaluation to Q14 with floor truncation, while keeping projections and biases in float64 to isolate activation error. Output RMSE and top-1 agreement compare with the corresponding exact floating GELU or SiLU. These are synthetic output comparisons, not trained-model quality measurements.

| Activation | Approximation | FFN output RMSE | Top-1 agreement | Maximum activation error on [-4,4] |
| --- | --- | ---: | ---: | ---: |
| GELU | local quadratic | 0.023250 | 89.84% | 4.000127 |
| GELU | mixed degree8 | 0.002676 | 99.22% | 0.045467 |
| GELU | mixed degree10 | 0.002170 | 99.22% | 0.015203 |
| GELU | uniform degree10 | 0.003230 | 99.22% | 0.008856 |
| SiLU | local quadratic | 0.010992 | 91.41% | 2.071951 |
| SiLU | mixed degree8 | 0.000626 | 95.31% | 0.009164 |
| SiLU | mixed degree10 | 0.000722 | 95.31% | 0.002604 |
| SiLU | uniform degree10 | 0.000874 | 94.53% | 0.001964 |

Mixed fits use least-squares weights `0.01 + 0.99*exp(-0.5*(x/0.5)**2)` over 16,001 points in `[-4,4]`. Uniform fits weight every point equally. All fitted candidates preserve zero and the exact `x/2` term, approximating only the even component in powers of `x/4`. Pure Gaussian-weighted fits are also recorded, but their large interval-tail errors make them unsuitable for a general `[-4,4]` contract.

Q14 inputs and outputs do not imply Q14 intermediates. The benchmark chooses the largest internal fractional precision satisfying a conservative coefficient-sum product bound below `2**30`. Mixed degree-eight GELU uses Q12 internally, mixed degree-ten GELU Q11, and mixed degree-eight SiLU Q13. Increasing degree can therefore increase rounding error: mixed degree-ten SiLU has lower maximum interval error but worse fixture RMSE than degree eight.

The existing degree-ten coefficient sets are also evaluated using this bounded, reduced-precision schedule. Those rows do not measure the existing `_smooth` implementation, which clips inputs and selects tails with private comparisons.

## Private activation cost

The selected GPU run uses seven repetitions on an RTX 4060 Laptop GPU, ring32, probabilistic truncation, CUDA phase graphs, sparse products, narrow corrections, and checked output opening. Inputs are already shared. Each live execution uses fresh masks. The replay estimate sums isolated-party phase maxima and excludes network latency and bandwidth; four colocated roles share one GPU.

Each row measures only activation evaluation plus checked opening on 8,192 or 1,048,576 elements. It excludes projections and is not a full-FFN latency. The extra standalone opening and lack of fusion also prevent adding these times directly to an existing FFN result. The quality table above uses deterministic floor rounding; these timings use probabilistic rounding, with actual opened grid errors retained in the JSON.

| Activation | Approximation | Batch 1 estimate | Batch 128 estimate | Batch 128 four roles on one GPU | Rounds |
| --- | --- | ---: | ---: | ---: | ---: |
| GELU | local quadratic | 0.680 ms | 6.827 ms | 26.602 ms | 9 |
| GELU | mixed degree 8 | 2.051 ms | 22.408 ms | 88.163 ms | 27 |
| GELU | mixed degree 10 | 2.350 ms | 25.893 ms | 99.829 ms | 31 |
| SiLU | local quadratic | 0.602 ms | 6.570 ms | 25.588 ms | 9 |
| SiLU | mixed degree 8 | 1.764 ms | 22.362 ms | 86.166 ms | 27 |
| SiLU | mixed degree 10 | 2.042 ms | 25.785 ms | 99.895 ms | 31 |

The caller must establish `abs(x) <= 4`; these bounded candidates perform no private clipping or range check. Integer interval propagation covers every floor/ceiling truncation outcome for each benchmark input, and every party agrees on the opened result. The full sweep also includes degrees four and six and uniform fits.

## SwiGLU scope

The synthetic SwiGLU quality check multiplies the approximated SiLU gate by an independent value branch and applies a floating random projection. Its output RMSE falls from 0.005518 for the local quadratic to 0.000313 for mixed degree eight. It isolates the activation approximation; the gate multiplication and projections are not fixed-point secure operations in this check. Full secure SwiGLU inference and trained-model accuracy remain unmeasured.

## Reproduce and inspect

From the repository root with the documented environment activated:

```sh
python -m benchmarks.activation_approximation_quality
python -m benchmarks.activation_polynomial_protocol --profile uniform mixed --repeats 3
python -m benchmarks.activation_polynomial_protocol --profile mixed --degree 2 8 10 --repeats 7 \
  --output results/activation_polynomial_protocol_selected.json
python -m pytest tests/test_activation_approximation_quality.py tests/test_activation_polynomial_protocol.py tests/test_activations.py -q
```

The quality benchmark runs on CPU; protocol timing requires CUDA. The focused checks cover independent scalar integer evaluation, all 131,073 Q14 encodings in `[-4,4]` for the polynomial floor schedule, probabilistic interval endpoints, and input-range rejection.

- [Quality results and coefficients](../results/activation_approximation_quality.json)
- [Full protocol sweep](../results/activation_polynomial_protocol.json)
- [Selected seven-repeat protocol results](../results/activation_polynomial_protocol_selected.json)
- [Quality benchmark](../benchmarks/activation_approximation_quality.py)
- [Protocol benchmark](../benchmarks/activation_polynomial_protocol.py)

Each protocol report retains source hashes and explicit coefficient arrays. The selected run's hashes match the benchmark sources accompanying this report.
