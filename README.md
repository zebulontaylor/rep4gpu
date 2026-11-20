# Rep4 GPU inference

This prototype evaluates secret-weight feed-forward networks with four-party
replicated sharing. It provides exact 16-bit, 32-bit and 64-bit ring matmuls, fixed-point
arithmetic, ReLU, and polynomial GELU/SiLU approximations. Inputs, weights, and
intermediate activations remain shared until an explicit output opening.

The [performance and validation report](docs/results.md) records a private prefill
estimate of 16.43 ms, or 46.39× FP16, using optional exact Strassen products
and 1.75 times the compact weight storage. The best earlier decode estimate is
3.007 ms, or 10.75× FP16. These use prepared inference graphs, bounded
probabilistic Q14 rounding, and zero-network GPU delivery. Complete inference
has not reached 10× FP16. A checked matrix-vector product reaches 8.29× FP16.
Actual colocated timings are reported separately. The optional
[CUDA cryptography variant](docs/protocol/device-protocol.md) generates masks and
consistency digests on the GPU. The classifier supports
[CUDA IPC on one GPU](docs/experiments/text-classifier.md#keep-classifier-messages-on-the-gpu)
and [NCCL across four GPUs](docs/experiments/nccl-classifier.md). The single-GPU
measurements do not establish physical GPUDirect RDMA performance.

[Activation experiments](docs/results.md#sparse-products-and-activation-experiments)
include optional private quadratics and a public-fixture helper simulation.
No TEE is required or measured; the simulated helper path reconstructs
activations and does not preserve Rep4 activation privacy. The
[seeded ring64 helper](docs/experiments/seeded-tee-simulation.md) also supports SwiGLU and
optional helper-based down-projection rescaling.

Rep4 tolerates one malicious party among four. Two colluding parties can
reconstruct the secrets. This differs from the all-but-one privacy target in
the [project design](https://docs.google.com/document/d/192TcAOZ43OQI1E3Q01Mxw7hRUGCXEL74rriaW2aI4ko/edit).

## Repository layout

```text
src/
  rep4gpu/
    protocol/          Party runtimes, transports, seed setup, and PRG
    arithmetic/        Fixed-point operations, conversions, and truncation
    gpu/               Matmul, bit-operation, and cryptography kernels
    models/            Model sharing and inference graph execution
    proofs/            Witness checks and activation proofs
  native/              Rust proof backend and MP-SPDZ patches
experiments/
  benchmarks/          Inference, activation, kernel, and proof experiments
  examples/            Runnable synthetic next-token example
  results/             Saved measurements and logs grouped by subject
tests/
  unit/                Protocol, arithmetic, GPU, inference, and proof tests
  integration/         Multi-party, model, and golden-transcript tests
  benchmarks/          Benchmark correctness tests
  fixtures/            MP-SPDZ reference programs and recorded evidence
docs/                  Protocol guides, arithmetic contracts, and results report
tools/                 Golden-fixture generators and plotting utilities
```

Start with the [documentation index](docs/README.md) for implementation details
or the [experiment guide](experiments/README.md) for runnable measurements and
saved results. Run Python module commands from the repository root after the
editable installation below.

## Run locally

Use Linux with a CUDA-capable NVIDIA GPU and a CUDA-enabled PyTorch installation.
The CPU protocol tests also run without a GPU. From this directory:

```sh
python -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install -e '.[test,crypto]'
python -m pytest -q
```

The tensor-core kernels use Triton. The working environment has an RTX 4060 Laptop GPU
with 8 GB of VRAM; benchmark JSON files record package versions and source
fingerprints.
The [kernel arithmetic guide](docs/arithmetic/kernel-arithmetic.md) explains exact limb
products, compact weights, and the accumulator bounds required by tensor cores.

Run a complete toy network in four separate processes with mutual TLS and
distributed seed setup:

```sh
python -m experiments.benchmarks.inference.socket_inference --help
python -m experiments.benchmarks.inference.socket_inference --device cpu --bits 64 --batch 1
```

The launcher creates temporary per-party certificates and private input files,
checks every party's output, and retains summaries, logs, and exit codes. It
removes the temporary keys and private input files after the processes finish.
See the [socket inference report](experiments/results/transport/socket_inference_report.md) for the
retained CPU and CUDA runs.

For GPU performance measurements with secret weights already resident:

```sh
python -m experiments.benchmarks.inference.inference --device cuda --bits 32 --fused --prg aes \
  --truncation compact --batch 16 --width 512 --repeats 5 \
  --output experiments/results/inference/inference-local.json

python -m experiments.benchmarks.inference.device_inference --bits 32 --binary-bits 32 \
  --optimized --gpu-open --rounding probabilistic --fractional-bits 14 \
  --prepared --weight-storage words32 --batch 1 128 --width 2048 --repeats 5 \
  --output experiments/results/inference/device-local.json
```

`compact` uses exact carry-based truncation. `probabilistic` selects MP-SPDZ's
masked rounding protocol and validates its input range on the public test
fixture. `reference` retains the original full-bit conversion for comparison.
Use `--bits 64 --binary-bits 64` when the 32-bit accumulator range is insufficient. The benchmark
rejects fixtures that violate the selected ring and truncation bounds.

The benchmark reports fresh-mask four-party execution on one machine and an
ideal-network estimate from isolated-party replay. Replay is only a timing
fixture: it reuses a recorded transcript and must never handle live inputs.
The reference estimate includes local PRG, hashing, host staging, and computation.
The device benchmark separately reports GPU-delivered and host-staged estimates.
Both exclude network latency and bandwidth. CPU NumPy, eager GPU PyTorch, and CUDA-graph
PyTorch baselines appear separately. Model sharing and setup are reported
outside the resident-model inference timer.

## Use the operations

Each `Party` holds three replicated components and only its three correlated
randomness seeds. All parties must execute the same public sequence of calls.
Only the input owner supplies the plaintext tensor:

```python
from rep4gpu import FixedPoint
from rep4gpu.models.inference import InferenceParty

party = InferenceParty(party_id, transport, local_seeds, bits=64)
fp = FixedPoint(party, fractional_bits=12)
x = party.input(0, encoded_input if party.party == 0 else None, input_shape)
w = party.input(1, encoded_weights if party.party == 1 else None, weight_shape)
y = fp.matmul(x, w)
y = fp.relu(y)
opened = party.open(y)
```

`encoded_input` and `encoded_weights` are integer tensors scaled by `2**12`.
`InferenceParty` defaults to exact floor rescaling and AES-256-CTR masks. Set
`rounding='probabilistic'` only after establishing its stronger signed range
contract. The lower-level `Party` retains the full-bit conversion reference.
Keep the party instance and its counters for the lifetime of a session. A new
instance needs newly generated correlated seeds; restarting counters with old
seeds reuses masks and can expose relationships between private inputs.
`FixedPoint.encode` provides nearest-even plaintext encoding; `decode` accepts
an explicitly opened tensor. Matmul accumulates ring products before rescaling.
The caller must choose a precision and ring width that contain the signed
accumulators. Ring arithmetic wraps; it does not saturate or privately validate
input ranges.

Use `SocketTransport` with distinct verified mutual-TLS identities and
`setup.correlated_seeds` for separate parties. Plain TCP and `IdealNetwork` are
local test transports. `make_seeds` is a trusted test fixture, not distributed
seed provisioning. The [protocol mapping](docs/protocol/protocol.md) describes component
ownership, masked messages, and checks. The [activation guide](docs/arithmetic/activations.md)
defines rounding and approximation behavior.
The [layer commitment runtime](docs/protocol/layer-commitments.md) commits prover
hidden states to a Merkle root and reconstructs requested layers from sparse
checkpoints, with authenticated plaintext openings for a verifier.
The [prover/verifier setup](docs/protocol/prover-verifier.md) runs separate TLS
processes with a public integer model, fresh challenges, and full or sampled
transition checks. It includes a local launcher and commands for two servers.
For verification without plaintext intermediate openings, use the
[private Rep4 audit](docs/protocol/private-rep4-audit.md). An external prover
sends authenticated replicated trace shares to four verifier processes. Each
verifier holds secret-shared weights and biases provisioned by the model owner.
They check the selected transitions in Rep4 and open only acceptance and the
final output; the input remains public. The demo reports online application
traffic separately from model-share provisioning sizes.
The [witness activation guide](docs/experiments/witness-activations.md) describes checked
prover intermediates, custom recomputation recipes, and communication comparisons.

The [trained next-token example](experiments/examples/README.md) fits a small synthetic MLP,
then checks private inference against both floating-point and integer outputs.
Its CPU and CUDA logits also match the [whole-network MP-SPDZ golden](tests/fixtures/mpspdz/toy_ffn_report.md).
The same example guide shows how to share a supported `torch.nn.Sequential`
model once with `share_sequential` and run private inputs through it.

## Validate against MP-SPDZ

```sh
python tools/validation/golden_rep4.py --output tests/fixtures/mpspdz/rep4_transcripts.json
python tools/validation/golden_mpspdz.py --help
python tools/validation/golden_mpspdz.py
python -m pytest tests/integration/test_rep4_oracle.py tests/integration/test_runtime_golden.py -q
```

The golden harness uses the installed MP-SPDZ tree by default and runs its
compiler and four-party executable in an isolated working directory. The
[golden report](tests/fixtures/mpspdz/report.md) records actual executions, source and
binary hashes, and the native 32-bit build. Deterministic masks also let the
tests compare each runtime message and digest byte-for-byte against an
independent oracle.

Core multiplication, input, and opening follow MP-SPDZ's message layout.
The socket transport uses its own length-prefixed frames; it is not a binary
interoperability layer for MP-SPDZ. Golden comparisons cover application
payloads, recipients, and protocol phases, excluding transport framing.
Activation circuits include explicit protocol changes, so matching arithmetic
does not imply identical whole-program communication. The carry-based exact
truncation converts three carry/sign bits instead of every output bit. The
32-bit golden uses generic per-bit conversion because the installed snapshot's
optimized binary-to-arithmetic path fails the retained 32-bit boundary fixture.

## Scope

The inference fixtures are generated dense feed-forward networks, not trained
language models. They measure arithmetic and communication costs and compare
toy logits and log-probabilities. Attention, tokenization, full transformer
execution, authenticated datacenter records, model binding for trained networks, and a validated
model-substitution detection threshold are not implemented.

The implementation is a research prototype with checked transcripts and
fault-injection tests, not a formally verified or independently audited MPC
deployment. Its privacy assumptions require independent party isolation,
authenticated confidential channels and fresh randomness. Pending internal
checks complete before unmasked output shares are broadcast; the final output
check completes before acceptance.
