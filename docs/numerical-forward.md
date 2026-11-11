# Full numerical FFN forward passes

Complete secret-weight SiLU, GELU, and SwiGLU forward passes pass the four-party checks at batches 1 and 8. Each model has input width 2048, hidden width 8192, and output width 2048. Arithmetic uses ring64 Q24 and the numerical exponential, reciprocal, and erf primitives.

## Forward computation

- SiLU and GELU: `activation(x @ W_up + b_up) @ W_down + b_down`.
- SwiGLU: `(silu(x @ W_gate + b_gate) * (x @ W_value + b_value)) @ W_down + b_down`.

Party 0 supplies the private input; party 1 supplies the private matrices and biases. Only the final output is explicitly opened. Static weights remain resident across queries, and CUDA graphs copy only the dynamic input shares. SwiGLU has three matrices rather than two; its hidden width is not reduced to equalize parameter counts. Each FP16 comparison evaluates that case's own architecture.

The fixtures use the original inference RNG stream, with an independent value-projection seed for SwiGLU. References evaluate the original floating formulas in float64. These are synthetic FFNs, not a trained model or a complete transformer.

## Measured performance and error

The RTX 4060 Laptop GPU run uses three repetitions, fresh masks, probabilistic rounding, checked communication boundaries, and checked final output opening. Timing includes both dense projections, biases, and activation evaluation; SwiGLU also includes its value projection and secret gate multiplication. Input/model sharing and graph preparation are outside the timed forward pass.

| Activation | Batch | Zero-network estimate | Four roles on one GPU | FP16 graph | Estimate / FP16 | Maximum output error | Output RMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| silu | 1 | 15.385 ms | 73.366 ms | 0.2848 ms | 54.0× | 1.561e-06 | 4.841e-07 |
| gelu | 1 | 18.015 ms | 86.973 ms | 0.2842 ms | 63.4× | 1.728e-06 | 5.177e-07 |
| swiglu | 1 | 17.707 ms | 87.894 ms | 0.4191 ms | 42.2× | 8.980e-07 | 2.742e-07 |
| silu | 8 | 22.321 ms | 100.667 ms | 0.2903 ms | 76.9× | 1.828e-06 | 4.812e-07 |
| gelu | 8 | 24.999 ms | 113.599 ms | 0.2875 ms | 86.9× | 2.074e-06 | 5.175e-07 |
| swiglu | 8 | 26.776 ms | 118.076 ms | 0.4321 ms | 62.0× | 9.829e-07 | 2.630e-07 |

The zero-network estimate sums isolated-party phase maxima and excludes network latency and bandwidth. The colocated column measures the complete four-role execution on one GPU. Neither is a physical multi-GPU network measurement. Output error is measured on the recorded fresh execution; all live timing executions pass party-agreement and propagated integer-interval checks.

All six cases have 100% top-1 agreement with their floating references. The full numerical paths are slower than the earlier low-degree ring32/Q14 substitutes. Both precision and activation algorithms differ, so this experiment does not isolate either change alone.

| Activation | Communication rounds | Batch 1 payload | Batch 8 payload |
| --- | ---: | ---: | ---: |
| silu | 144 | 35.59 MB | 284.61 MB |
| gelu | 176 | 43.46 MB | 347.52 MB |
| swiglu | 152 | 37.69 MB | 301.38 MB |

Payload counts sum outgoing bytes across all four roles and include consistency tags. They exclude initial input and model distribution. The largest recorded CUDA allocation peak is 5.25 GiB. Batch 128 was not tested.

## Correctness checks

The public integer oracle checks absolute-sum dot-product bounds before running int64 multiplication, including signed weights and every partial accumulator. It propagates all permitted floor/ceiling outcomes through the numerical activations and final projection. Both SwiGLU input projections must satisfy `[-4,4]`; out-of-range fixtures are rejected. Every accepted accumulator remains below the strict `2**62` truncation limit. This validates the public fixture and does not privately check arbitrary deployment inputs.

All parties agree on the final opened output. Isolated-party replay reproduces both the recorded output and outgoing messages exactly. GPU tests additionally compare floor-rounded full forwards with the integer oracle, then change the private input and rerun the same graphs to detect stale input capture. The focused validation passes 37 CPU checks and three GPU checks.

## Reproduce

Run from the repository root with the CUDA Python environment activated:

```sh
python -m benchmarks.numerical_inference --width 2048 --batch 1 --repeats 3 \
  --output results/numerical_inference_b1.json
python -m benchmarks.numerical_inference --width 2048 --batch 8 --repeats 3 \
  --output results/numerical_inference_b8.json
python -m pytest tests/test_numerical_inference_common.py tests/test_numerical_inference.py \
  tests/test_numerical.py tests/test_activations.py -q
```

- [Batch 1 results](../results/numerical_inference_b1.json)
- [Batch 8 results](../results/numerical_inference_b8.json)
- [Forward benchmark](../benchmarks/numerical_inference.py)
- [Fixtures and integer oracle](../benchmarks/numerical_inference_common.py)
- [Primitive API and domains](numerical-primitives.md)
