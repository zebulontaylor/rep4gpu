# Rep4 GPU inference

This prototype evaluates secret-weight feed-forward networks with four-party
replicated sharing. It provides exact 32-bit and 64-bit ring matmuls, fixed-point
arithmetic, ReLU, and polynomial GELU/SiLU approximations. Inputs, weights, and
intermediate activations remain shared until an explicit output opening.

The [performance and validation report](RESULTS.md) records 31.8× FP16 decode
and 85.1× FP16 prefill estimates with bounded probabilistic Q14 rounding and
zero-network GPU delivery. A checked matrix-vector product reaches 8.29× FP16.
Actual colocated timings are reported separately. The optional
[CUDA cryptography variant](docs/device-protocol.md) generates masks and
consistency digests on the GPU; physical GPUDirect networking is not implemented.

Rep4 tolerates one malicious party among four. Two colluding parties can
reconstruct the secrets. This differs from the all-but-one privacy target in
the [project design](https://docs.google.com/document/d/192TcAOZ43OQI1E3Q01Mxw7hRUGCXEL74rriaW2aI4ko/edit).
The [malicious-majority experiment](docs/malicious-majority.md) has separate
trust assumptions and a separate large-prime MASCOT preprocessing bridge.

## Run locally

Use Linux with a CUDA-capable NVIDIA GPU and a CUDA-enabled PyTorch installation.
The CPU protocol tests also run without a GPU. From this directory:

```sh
python -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install -e '.[test,crypto]'
python -m pytest -q
```

The tensor-core kernels use Triton. The optional SIMT experiment uses the CUDA
driver and NVRTC libraries. The working environment has an RTX 4060 Laptop GPU
with 8 GB of VRAM; benchmark JSON files record package versions and source
fingerprints.
The [kernel arithmetic guide](docs/kernel-arithmetic.md) explains exact limb
products, compact weights, and the accumulator bounds required by tensor cores.

Run a complete toy network in four separate processes with mutual TLS and
distributed seed setup:

```sh
python -m benchmarks.socket_inference --help
python -m benchmarks.socket_inference --device cpu --bits 64 --batch 1
```

The launcher creates temporary per-party certificates and private input files,
checks every party's output, and retains summaries, logs, and exit codes. It
removes the temporary keys and private input files after the processes finish.
See the [socket inference report](results/socket_inference_report.md) for the
retained CPU and CUDA runs.

For GPU performance measurements with secret weights already resident:

```sh
python -m benchmarks.inference --device cuda --bits 32 --fused --prg aes \
  --truncation compact --batch 16 --width 512 --repeats 5 \
  --output results/inference-local.json

python -m benchmarks.device_inference --bits 32 --binary-bits 32 \
  --optimized --gpu-open --rounding probabilistic --fractional-bits 14 \
  --prepared --weight-storage words32 --batch 1 128 --width 2048 --repeats 5 \
  --output results/device-local.json
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
from rep4gpu.inference import InferenceParty

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
seed provisioning. The [protocol mapping](docs/protocol.md) describes component
ownership, masked messages, and checks. The [activation guide](docs/activations.md)
defines rounding and approximation behavior.

The [trained next-token example](examples/README.md) fits a small synthetic MLP,
then checks private inference against both floating-point and integer outputs.
Its CPU and CUDA logits also match the [whole-network MP-SPDZ golden](tests/golden/toy_ffn_report.md).
The same example guide shows how to share a supported `torch.nn.Sequential`
model once with `share_sequential` and run private inputs through it.

## Validate against MP-SPDZ

```sh
python scripts/golden_rep4.py --output tests/golden/rep4_transcripts.json
python scripts/golden_mpspdz.py --help
python scripts/golden_mpspdz.py
python -m pytest tests/golden -q
```

The golden harness uses the installed MP-SPDZ tree by default and runs its
compiler and four-party executable in an isolated working directory. The
[golden report](tests/golden/report.md) records actual executions, source and
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
execution, authenticated datacenter records, model binding, and a validated
model-substitution detection threshold are not implemented.

The implementation is a research prototype with checked transcripts and
fault-injection tests, not a formally verified or independently audited MPC
deployment. Its privacy assumptions require independent party isolation,
authenticated confidential channels and fresh randomness. Pending internal
checks complete before unmasked output shares are broadcast; the final output
check completes before acceptance.

## Malicious-majority GPU prototype

The separate prime-field implementation supports authenticated matrix products,
elementwise products, and square activations with up to three malicious roles
out of four, under its [preprocessing and channel assumptions](docs/malicious-majority.md).
Each online worker loads only its own material. The original small-field MAMA
producer has a confirmed [mask-privacy limitation](docs/native-mama-sacrifice.md#mask-privacy-limitation):
its successful correctness checks do not establish 128-bit-private preprocessing.
Its bundles remain functional test material. The replacement large-prime
[MASCOT bridge](docs/mascot-preprocessing.md), with a fresh sacrifice challenge
after authentication, has passed native correlation and malformed-input
checks and completed one fresh trained-model inference. Its field conversion
is too expensive for the performance target. The [direct native-field
path](docs/direct-field-preprocessing.md) removes that conversion and cuts
fresh toy preprocessing from 22.35 minutes to 0.416 seconds; its complete
online runtime uses CPU big-integer checks with optional GPU products for
fixed-model evaluation.

Separate [exact q256 tensor-core kernels](docs/large-prime-kernels.md) support
prepared local weights. The [matrix-correlation expansion helper](docs/matrix-pcg-linear.md)
combines raw local terms before applying their public linear map. Both are
isolated arithmetic components; neither supplies complete preprocessing.
The [F127 kernels](docs/field-fp127.md) also prepare fixed left operands for
the public linear maps used by experimental matrix-correlation generators.

The malicious-majority target includes amortized fresh cryptographic
preprocessing and is **not met**. Online-only timing improvements do not
establish that target. Both native paths remove the central randomness dealer. The direct path is
much faster, but fresh preprocessing still dominates its measured cost.

[Model-dependent preparation](docs/prepared-model.md) moves the weight-mask
opening and part of each matrix product before the query. Every query consumes
fresh correlations. Deferred internal checks finish before output shares are
sent; small outputs use full-vector checks and larger outputs use independent
per-key random projections.

A prepared matrix batch can be partitioned into disjoint one-use query handles.
This shares the fixed-model preparation across the batch while preserving a
fresh input mask for each query; it does not reuse a query's correlations.

The trained examples use bounded integer square MLPs with public scale decoding.
They do not implement general fixed-point transformer inference. See
[the measured timings](RESULTS.md) for the clear-model comparators, network
assumptions, and separate preprocessing costs, and the
[direct-field execution guide](docs/direct-field-preprocessing.md) to run without a dealer.

```sh
python -m examples.spdz_square_toy
python -m scripts.run_spdz_distributed --device cuda --m 128 --k 256 --n 256 \
  --output results/spdz_distributed.json
```

The commands above use explicit trusted-dealer test fixtures. The native guide
uses distributed preprocessing and consumes each role's bundle once.
