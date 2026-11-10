# CUDA masks and checks for Rep4

[`DeviceParty`](../rep4gpu/device_runtime.py) is an optional Rep4 variant that
generates ChaCha20 masks and BLAKE3-256 joint-input tags on CUDA. It uses the
[Rep4 share layout and six joint-input routes](protocol.md#joint-inputs-and-multiplication-messages),
with the same sender, backup, receiver, outsider, packed ring values, and
32-byte tag size. Its masks and tags differ from the reference runtime and
MP-SPDZ transcripts; all participating roles must select the same variant and
public parameters.

The threat model remains at most one malicious party among four, with abort.
The consistency argument still relies on an honest sender or backup checking
the same masked value, private authenticated channels, and collision-resistant
hashing. This implementation has compatibility and fault tests, rather than a
formal security audit or dishonest-majority security.

## Masks and tags

[`gpu_prg.py`](../rep4gpu/gpu_prg.py) derives a 256-bit key from each triple-shared
seed using SHA-256 and a length-framed public domain. The domain identifies the
arithmetic ring width, binary width, arithmetic/binary mode, wire width, and
excluded party. Domain keys are cached on the party's CUDA device.

Each draw starts ChaCha20 at block counter zero and uses that excluded party's
64-bit draw counter as its nonce. The runtime reserves and advances the counter
atomically before generation, including when generation fails, and rejects
exhaustion. A key/nonce pair must never serve distinct draws.
Use fresh session seeds, keep counters advancing across requests, and do not
recreate a party with old seeds and counters reset to zero. Capturing and
replaying a live mask-generation call would repeat its nonce; fixed-fixture
benchmark replay is the only intended use of that repetition.
Each party must execute one ordered protocol operation at a time, in the same
public sequence as its peers. Atomic nonce reservation does not order concurrent
network calls across parties.

The ChaCha state uses the original 64-bit block-counter/64-bit nonce layout.
Draws use fewer than `2**32` blocks, preserving compatibility with the CPU
reference's equivalent initial state. This is not an IETF 96-bit nonce API.

[`gpu_hash.py`](../rep4gpu/gpu_hash.py) implements official unkeyed BLAKE3-256,
including chunk counters, tree parents, and final `ROOT` compression. It matches
the [official BLAKE3 reference](https://github.com/BLAKE3-team/BLAKE3/blob/master/reference_impl/reference_impl.rs)
and CPU `blake3` package, including empty inputs and partial chunks. It replaces
BLAKE2b-256 for device joint inputs; retaining 32 bytes does not make these tags
interchangeable. `hash_block_tree=True` reduces chunk groups within each CUDA
block and produces the same digest.

The receiver validates public frame sizes and compares every received backup
tag with a GPU digest of the corresponding payload before assembling the
result shares. The Boolean comparison synchronizes with the host. A mismatch
raises an error. The inherited private-input correction checks and explicit
output opening use the checked BLAKE2b byte protocol by default; private-input
masks use the overridden ChaCha20 generator. `DeviceInferenceParty(gpu_open=True)`
instead sends the opening component and its BLAKE3 check through the device
transport. It retains the clockwise component round followed by the
counterclockwise digest round, and reconstructs only after a successful check.

## Transport boundaries

| Adapter | What it provides |
|---|---|
| `DeviceIdealEndpoint` | Copies messages into receiver-owned CUDA storage and orders streams with events. Roles share one process and device; this is a colocated test harness without party isolation. |
| `DeviceByteBridge` | Copies outgoing device messages to host bytes, uses an existing byte transport, then copies received bytes to CUDA. Confidentiality and authentication come from that underlying transport. |
| `DeviceReplay` | Replays recorded fixture messages. Device mode assumes peer messages are already delivered to CUDA; host mode retains host/device staging. Both omit network transfer time. |

These adapters do not implement GPUDirect or a distributed GPU transport. For
real connections, wrap a `SocketTransport` configured with mutual TLS and
distinct party identities in `DeviceByteBridge`. Obtain fresh seeds through
`correlated_seeds()` on those authenticated confidential connections before
constructing the party. Each party receives only its three seeds. The
`make_seeds()` helper used by local benchmarks is trusted fixture setup.

`DeviceRecorder` copies transcripts to host memory. Recordings and replay state
belong to the timing fixture and must not supply randomness or messages for
fresh requests.

## Fixed-point inference

[`DeviceInferenceParty`](../rep4gpu/device_inference.py) combines this transport
and cryptography with exact floor rescaling. It accepts the same shared inputs
and `FixedPoint` operations as `InferenceParty`:

```python
from rep4gpu.device_inference import DeviceInferenceParty
from rep4gpu.device_transport import DeviceByteBridge

party = DeviceInferenceParty(
    party_id, DeviceByteBridge(tls_transport, 'cuda'), local_seeds, bits=64)
```

Its optimized path batches the six mask rows in one GPU launch, packs carry
nodes with whole-word bit operations, and fuses local weighted conversion.
Ring32 masks and internal carry state use four-byte storage. With compact
ring32 weights and at least 16 rows, SM 8.9 uses the validated larger matmul
tile; other cases retain the baseline kernel. The [kernel guide](kernel-arithmetic.md)
describes the arithmetic and storage choices.
Set `optimized=False` to use the separate-kernel reference path. For the same
seeds and call sequence, both paths produce identical messages, local shares,
and counters. These optimizations change local computation, not the cryptographic
variant or protocol rounds. Ring32 still requires signed accumulator bounds;
neither path privately checks those bounds.

Set `rounding='probabilistic'` to select the small-gap masked truncation protocol.
Every rescaled signed accumulator must satisfy `abs(x) < 2**(bits-2)`.
Each result is its exact floor or ceiling, with unbiased rounding under uniform
fresh masks. Integer multiples of the scale are exact. The GPU port retains
three checked protocol rounds and uses BLAKE3 for its masked opening; no secret
sign is opened. ReLU still computes a secure sign and selection after rescaling.
This option changes the rounding contract and complete activation traffic.
The benchmark propagates rounding intervals through both layers and checks
every output against those public bounds.

## Run the optional benchmark

From the repository root, in a CUDA environment with PyTorch, Triton, and NVRTC:

```sh
python -m benchmarks.device_inference --quick --bits 32 --binary-bits 32 \
  --optimized --repeats 3 --output results/device_quick.json
```

The JSON checks exact fixed-point outputs, reports setup separately, and records
fresh live samples plus isolated-party replay estimates. Replay estimates sum
per-phase maxima of party medians and exclude network latency and bandwidth.
The `--staging host` option adds host copies over the same local ideal network;
it does not turn this benchmark into a TLS or distributed run. Both device and
host replay estimates are reported in either staging mode.

Add `--gpu-open` to benchmark device-side output checks, or
`--rounding probabilistic` to measure the bounded rounding variant. The latter
reports interval agreement rather than claiming equality with floor-rounded
inference. Every replay must still reproduce its particular recorded output
exactly.

`GraphDeviceParty` is a separate opt-in experiment for prepared local product
graphs. Call `prepare_product` sequentially outside protocol execution; it
consumes no counters. Each live replay uploads freshly reserved nonces, and
received-message checks remain outside capture and precede reconstruction.
Its [microbenchmark](../benchmarks/graph_product.py) reports a benefit for small
products and a regression for large ones. It is not selected by the inference
benchmark or by `DeviceInferenceParty`.

For independent CPU-crypto compatibility, tree-boundary cases, message ownership,
and tamper rejection, use the project's `crypto` and `test` extras and run:

```sh
python -m pytest -q tests/test_gpu_prg.py tests/test_gpu_hash.py \
  tests/test_device_transport.py tests/test_device_runtime.py
```
