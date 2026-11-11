# Experiments

Run these commands from the repository root after installing `rep4gpu` with
`python -m pip install -e '.[test,crypto]'`. The benchmarks use generated public
fixtures; see the [results report](../docs/results.md) for measured behavior and
benchmark assumptions.

## Benchmarks

| Directory | Measures |
| --- | --- |
| [`benchmarks/inference/`](benchmarks/inference/) | Complete networks, GPU graph execution, socket transport, and helper simulations |
| [`benchmarks/activations/`](benchmarks/activations/) | Approximation quality, nonlinear protocols, private lookup tables, and witness activations |
| [`benchmarks/kernels/`](benchmarks/kernels/) | Exact ring products, compact weights, Strassen products, and GPU cryptography |
| [`benchmarks/proofs/`](benchmarks/proofs/) | SiLU proof generation, verification, bounds, and communication accounting |
| [`benchmarks/common/`](benchmarks/common/) | Shared timing utilities |

For a small CPU inference measurement:

```sh
python -m experiments.benchmarks.inference.inference --device cpu --quick \
  --output /tmp/rep4gpu-inference.json
```

Use `--help` on an individual benchmark for its arguments. The
[next-token example](examples/README.md) trains a synthetic model and checks its
private outputs against integer and floating-point references.

## Saved results

[`results/`](results/) groups retained evidence into `inference/`, `activations/`,
`kernels/`, `proofs/`, `transport/`, `examples/`, and `validation/`. Transport
runs retain per-party log directories alongside their JSON summaries; validation
logs are under `results/validation/logs/`.

The [results report](../docs/results.md) links the measurements behind its
comparisons. The [socket report](results/transport/socket_inference_report.md)
links separate-process runs. Recorded JSON and logs retain the source paths and
fingerprints from their original runs; those paths describe the recorded
checkout. New runs fingerprint the current source layout.
