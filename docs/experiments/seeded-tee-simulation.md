# Seeded ring64 helper simulation

[Documentation index](../README.md)

The seeded helper benchmark keeps the three SwiGLU projections in Rep4 over
`Z/(2**64)` and moves the activation into a helper colocated with role 0.
Gate and value projections enter the helper as raw Q48 accumulator shares,
including their scaled biases. The helper floor-rescales them to Q24, evaluates
SiLU times value in float64, and rounds its output to Q24 with ties to even.
The down projection uses the existing MPC probabilistic truncation by default.
`--tee-down` moves that final floor-rescaling into a second helper invocation.

This is an ordinary CUDA simulation. It has no enclave, attestation, protected
channels, or activation privacy. The Python factory retains the helper seeds.
A deployable backend must authenticate the colocated host's input components
against independent holders and deliver output shares over enclave-authenticated
channels. The existing independent-holder input hash covers the remote missing
component; it does not authenticate components supplied locally by host 0.

## Seeded output sharing

Three independent, domain-separated PRG streams generate components 0, 1, and 2.
Each ordinary role generates only the components it holds. The simulated
helper additionally generates component 0, which its colocated role does not
hold. Component 3 is the output minus the three generated components modulo
`2**64`. Role 0 retains its copy locally, roles 1 and 2 receive the residual,
and role 3 generates all its components locally. No role receives its excluded
component as an output share.

Separate counters 4, 5, and 6 reserve fresh output-mask draws for every live or
captured execution. Preparing a graph does not advance these counters. This
output sharing does not alter the existing MPC mask streams.

For `N` SwiGLU coordinates, the two helper exchanges send these global payloads:

| Step | Bytes |
| --- | ---: |
| Missing component 0 for gate and value, with independent-holder hash | `16*N + 32` |
| Residual component 3 to roles 1 and 2 | `16*N` |
| Total | `32*N + 32` |

A single-input helper truncation instead costs `24*N + 32` bytes. Counts include
each outgoing application payload once and exclude transport framing and
future enclave-channel authentication overhead.

## FFN communication model

For model width `d` and intermediate width `h`, the leading global bytes per
token per layer are:

- Existing matrix protocols and helper SwiGLU: `128*(h+d)`.
- Also moving down-projection truncation into the helper: `128*h + 72*d`.

These formulas exclude fixed tags, input/model distribution, and explicit output
opening. The standalone benchmark opens its final output, adding `32*d` bytes
per token plus tags. A transformer retaining shared intermediate states does
not need an output opening after every FFN.

For 64 layers with `d=4096` and `h=14336`, the leading FFN totals are 151.0 MB
and 136.3 MB per token respectively, globally across all four roles. They are
architecture extrapolations, not measurements of a trained model. Dividing by
four gives mean outgoing traffic; individual roles send different amounts.

The matrix protocol already uses sender payloads with independent backup
hashes. Shifting random local product shares does not implement truncation of
the reconstructed product: carries and ring wrap must be handled. This
experiment leaves the matrix protocol intact rather than assuming a cheaper
multiply-and-truncate protocol.

## Measured communication

The ring64 Q24 SwiGLU fixture uses width 2048, intermediate width 8192,
and output width 2048. All totals below sum outgoing payloads over four roles.

| Path | Decode bytes/token | Batch-8 bytes/token | Rounds |
| --- | ---: | ---: | ---: |
| Private numerical activation | 37,688,320 | 37,673,088 | 152 |
| Seeded helper SwiGLU | 1,377,248 | 1,376,380 | 10 |
| Also helper down rescaling | 1,262,336 | 1,261,664 | 9 |

The optimized paths reduce recorded application traffic by 96.35% and 96.65%
respectively. Maximum output error against the floating reference stays below
`1.02e-6` on these synthetic fixtures. This does not measure trained-model
accuracy or establish a private TEE implementation.

Including the final opening and fixed tags, the exact byte formulas for batch
`B` are `B*(128*h + 160*d) + 992` and
`B*(128*h + 104*d) + 768`, respectively.

## Run

From the repository root, with the CUDA environment installed:

```sh
.venv/bin/python -m experiments.benchmarks.inference.seeded_tee_inference --batch 1 8 --repeats 3 \
  --output experiments/results/activations/seeded_tee_inference.json
.venv/bin/python -m experiments.benchmarks.inference.seeded_tee_inference --batch 1 8 --repeats 3 \
  --tee-down --output experiments/results/activations/seeded_tee_inference_down.json
.venv/bin/python -m pytest tests/benchmarks/test_seeded_tee_simulation.py -q
```

The benchmark checks party agreement and integer bounds on every fresh forward.
Its public fixture oracle validates accumulator ranges and activation inputs;
it does not privately validate deployment inputs. Colocated elapsed times
exclude physical network transfers and TEE transitions.

- [Helper implementation](../../experiments/benchmarks/activations/seeded_tee_simulation.py)
- [Forward benchmark](../../experiments/benchmarks/inference/seeded_tee_inference.py)
- [Measured FFN results](../../experiments/results/activations/seeded_tee_inference.json)
- [Results with helper down rescaling](../../experiments/results/activations/seeded_tee_inference_down.json)

## One-layer bit accuracy

The helper also exposes `tee_activation` for ReLU, SiLU, GELU, and SwiGLU.
Each call floor-rescales Q48 inputs to Q24, evaluates the activation in float64,
and returns nearest-even Q24 output shares. The generic entry point also accepts
Q20 through Q23 with the corresponding doubled input scale. `tee_swiglu`
retains its existing interface and delegates to this implementation.

The [one-layer bit-accuracy report](../../experiments/results/activations/layer_bit_accuracy_report.md)
compares these helper outputs with fully private numerical activations, using
both deterministic floor and probabilistic truncation. It includes dense-layer
and activation stress fixtures, encoded-bit and ULP differences in FP32, FP16,
BF16, and two FP8 formats, plus native FP16/BF16 GPU layer baselines. Output
casting and native low-precision execution are measured separately.
