# Activation helper simulation

[Documentation index](../README.md)

The `simulated_relu` benchmark measures a two-helper replacement for the first
fixed-point truncation and ReLU in the resident feed-forward fixture. It runs
real CUDA reconstruction, activation, fresh output sharing, integrity checks,
and communication through the benchmark transport. Matrix weights remain in
the existing Rep4 representation and use the existing matrix protocol.

This is a public-fixture simulation. Helpers run in ordinary Python/CUDA
processes and reconstruct the activations. The shared Python factory retains
the helper seed. There is no enclave, host isolation, attestation, or activation
privacy. The benchmark does not establish a deployable trusted-helper protocol
or preserve the original Rep4 trust model for activations.

## Run the fixture

From the repository root, with the project's CUDA Python environment active:

```sh
python -m experiments.benchmarks.inference.activation_inference \
  --activation simulated_relu --batch 1 128 --width 2048 --repeats 9 \
  --output experiments/results/activations/activation_helper_simulation.json
```

The runner compares the simulated route with floating-point ReLU baselines and
writes timing, accuracy, per-phase payloads, and simulation metadata to the
output JSON. It uses ring32 with 14 fractional bits, resident compact weights,
segmented CUDA graphs, and checked device output opening. Setup and input
sharing are outside the online timing boundary.

## Two checked exchanges

Each role stores the three global share components other than its own index.
For `N` activation elements, the simulation performs these exchanges:

| Exchange | Payload sender | Independent hash sender | Receiver | Payload |
| --- | --- | --- | --- | --- |
| Reconstruct helper 0 | Role 1 | Role 2 | Role 0 | Missing component 0, `N` ring words |
| Reconstruct helper 1 | Role 0 | Role 3 | Role 1 | Missing component 1, `N` ring words |
| Return output shares | Helper 0 | Helper 1 | Role 2 | Components 3, 0, 1, `3N` ring words |
| Return output shares | Helper 0 | Helper 1 | Role 3 | Components 0, 1, 2, `3N` ring words |

The two reconstruction rows share one exchange. The two output rows share the
next exchange. Every row includes a 32-byte BLAKE3 tag from its independent
hash sender. At ring32, the global outgoing cost is therefore:

- Reconstruction: `2N` ring words plus 64 tag bytes, or `8N + 64` bytes.
- Output sharing: `6N` ring words plus 64 tag bytes, or `24N + 64` bytes.
- Total: **`8N` ring words plus 128 tag bytes**, or **`32N + 128` bytes**.

These totals count each outgoing payload once across all four roles. They
exclude transport framing, matrix resharing, the later output truncation, and
checked output opening.

## Reconstruction and fresh output sharing

Helpers 0 and 1 check their missing input component before reconstruction. Each
reconstructs the same signed ring32 integer, shifts it right arithmetically by
the requested fractional-bit count, and clamps the result at zero. This first
stage uses exact signed floor truncation. The final feed-forward output
rescaling still uses the benchmark's selected truncation mode.

The helpers use an independent fixture PRF key and a fresh draw nonce to
generate three random output components. The fourth component is the clear
activation minus their sum modulo `2**32`. Every ordinary output is a full
four-component sharing; it does not use a public-zero component pair. Each
helper retains its own three-component bundle. Roles 2 and 3 check their
received bundles against helper 1's tags before later computation uses them.
These checks establish agreement on the bundles sent to roles 2 and 3. They
do not immediately cross-check the helpers’ locally retained bundles or prove
that a helper computed the intended activation. Identical errors in both
helpers can pass the comparison.

The helper PRF consumes one virtual reservation from excluded-party stream 2.
Roles 0, 1, and 3 advance that reservation together; role 3 consumes an empty
draw without receiving the helper key. This keeps the ordinary Rep4 mask
counters aligned for subsequent matrix, truncation, and opening operations.
The factory creates a helper CUDA key only for roles 0 and 1. That allocation
choice supplies no isolation inside the shared benchmark process.

CUDA capture preserves the two checked communication boundaries. Input checks
finish before reconstruction, and output checks finish before the returned
shares are used. Each live graph replay reserves new nonces. Fixed-transcript
replay restores the recording's counters only to reproduce that benchmark
fixture; it is not a live request path.

## Interpret the comparison

The runner uses the original floating-point ReLU network for its baseline and
accuracy reference. The simulated first stage rounds down deterministically,
while the ordinary probabilistic path can round to either adjacent integer.
The existing propagated probabilistic intervals contain the simulated first
stage's floor result and remain conservative for the final output.

GPU-delivered replay assumes peer payloads are already on the GPU. It excludes
network latency and bandwidth, enclave transitions, protected-memory costs,
and attestation. Host-staged replay includes its host/device copies but still
excludes network transfer. Neither mode measures a TEE. A lower simulated
latency shows the cost of this executed public-fixture replacement under that
timing boundary; it does not show that the private Rep4 path has that latency.
