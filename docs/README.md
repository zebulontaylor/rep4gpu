# Documentation

Use the [project README](../README.md) for installation and first runs. The
[performance and validation report](results.md) explains the measured results;
retained experiment outputs are grouped under [`experiments/results/`](../experiments/results/).

## Protocol

- [Rep4 protocol mapping](protocol/protocol.md): share ownership, messages, checks, and MP-SPDZ validation.
- [CUDA masks and checks](protocol/device-protocol.md): device cryptography, transport, and graph execution.
- [MAMA preprocessing findings](protocol/native-mama-sacrifice.md): native reference behavior and independent-sacrifice changes.

- [Layer commitments and replay](protocol/layer-commitments.md): Merkle roots, checkpoint replay, verifier openings, and CUDA preparation.
- [Prover/verifier setup](protocol/prover-verifier.md): separate TLS services, model binding, challenges, and full or sampled transition checks.
- [Private Rep4 audit](protocol/private-rep4-audit.md): external prover, four verifier servers, share commitments, and private transition checks.

## Arithmetic

- [Exact GPU ring matrix products](arithmetic/kernel-arithmetic.md): limb products, weight formats, and accumulator bounds.
- [Fixed-point activations](arithmetic/activations.md): encoding, rounding, truncation, and polynomial approximations.
- [Private numerical primitives](arithmetic/numerical-primitives.md): integer reference operations and their protocol implementations.
- [Numerical FFN forward passes](arithmetic/numerical-forward.md): composed forward evaluation and numerical comparisons.

## Experiments

- [Private text classification](experiments/text-classifier.md): train a quantized SMS classifier and measure four-process checked inference.
- [Private Text CNN inference](experiments/text-cnn.md): 100%-dense SMS Text CNN with odd-prime CRT 1-product polynomial MPC at ~18–19× native BF16 on one GPU.
- [Activation approximation evaluation](experiments/activation-approximation-evaluation.md): approximation quality and protocol costs.
- [Activation helper simulation](experiments/activation-helper-simulation.md): public-fixture helper behavior and its privacy limits.
- [Seeded ring64 helper simulation](experiments/seeded-tee-simulation.md): seeded activation and down-projection experiments.
- [Witness activations](experiments/witness-activations.md): shared prover intermediates and verification recipes.
- [Batched product witnesses](experiments/product-witnesses.md): product checks and communication comparisons.
- [Committed-input SiLU proof](experiments/proof-activations.md): the native proof helper and MPC bridge.

Runnable [benchmarks](../experiments/benchmarks/) and the
[next-token example](../experiments/examples/README.md) live alongside the retained results.
