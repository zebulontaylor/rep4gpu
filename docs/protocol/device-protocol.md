# CUDA masks and checks for Rep4

[Documentation index](../README.md)

[`DeviceParty`](../../src/rep4gpu/protocol/device_runtime.py) is an optional Rep4 variant that
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

[`gpu_prg.py`](../../src/rep4gpu/gpu/crypto/gpu_prg.py) derives a 256-bit key from each triple-shared
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

[`gpu_hash.py`](../../src/rep4gpu/gpu/crypto/gpu_hash.py) implements official unkeyed BLAKE3-256,
including chunk counters, tree parents, and final `ROOT` compression. It matches
the [official BLAKE3 reference](https://github.com/BLAKE3-team/BLAKE3/blob/master/reference_impl/reference_impl.rs)
and CPU `blake3` package, including empty inputs and partial chunks. It replaces
BLAKE2b-256 for device joint inputs; retaining 32 bytes does not make these tags
interchangeable. `hash_block_tree=True` reduces chunk groups within each CUDA
block and produces the same digest. Aligned full-chunk messages up to 32 KiB
use coalesced loads into shared memory and a warp-local parent tree. Other
message sizes and alignments use the general chunk kernel.

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
| `CudaIpcTransport` | Copies CUDA frames across four trusted local processes on one GPU. TLS carries CUDA IPC descriptors. Only dedicated per-peer wire buffers are exported. |
| `NcclDeviceTransport` | Sends CUDA buffers between four ranks on distinct GPUs. The NCCL data channel requires a protected fabric. |
| `DeviceReplay` | Replays recorded fixture messages. Device mode assumes peer messages are already delivered to CUDA; host mode retains host/device staging. Both omit network transfer time. |

For TLS payloads, wrap a `SocketTransport` configured with mutual TLS and
distinct party identities in `DeviceByteBridge`. The [classifier GPU transports](../experiments/text-classifier.md#keep-classifier-messages-on-the-gpu)
separate CUDA payloads from the TLS control channel. Obtain fresh seeds through
`correlated_seeds()` on those authenticated confidential connections before
constructing the party. Each party receives only its three seeds. The
`make_seeds()` helper used by local benchmarks is trusted fixture setup.

`DeviceRecorder` copies transcripts to host memory. Recordings and replay state
belong to the timing fixture and must not supply randomness or messages for
fresh requests.

## Fixed-point inference

[`DeviceInferenceParty`](../../src/rep4gpu/models/device_inference.py) combines this transport
and cryptography with exact floor rescaling. It accepts the same shared inputs
and `FixedPoint` operations as `InferenceParty`:

```python
from rep4gpu.models.device_inference import DeviceInferenceParty
from rep4gpu.protocol.device_transport import DeviceByteBridge

party = DeviceInferenceParty(
    party_id, DeviceByteBridge(tls_transport, 'cuda'), local_seeds, bits=64)
```

Its optimized path batches the six mask rows in one GPU launch, packs carry
nodes with whole-word bit operations, and fuses local weighted conversion.
ChaCha20 writes each block contiguously, and aligned BLAKE3 blocks use vector
loads. Their generated masks and digests are unchanged. Ring32 multiplication and joint-input
masks, including truncation masks, use four-byte storage. Internal ring32
carry state also uses four-byte storage. With compact ring32 weights and at
least 16 rows, SM 8.9 uses the validated larger matmul
tile; other cases retain the baseline kernel. The [kernel guide](../arithmetic/kernel-arithmetic.md)
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

With probabilistic rounding, `bounded_sign=True` extracts the ReLU sign from
the shorter signed range after rescaling. For a positive shift `s`, the rounded
value lies in `[-2**(bits-s-2), 2**(bits-s-2)]`, including both endpoints. Its
sign is therefore bit `bits-s-1`. Ring32 Q14 uses a 16-bit carry subtree and
one final bit instead of padding the prefix to 32 bits. The final step computes
only the carry with one packed AND; it does not compute an unused propagate
bit. Sign extraction and binary-to-arithmetic pair preparation share a kernel.
The joint inputs pack only the consumed low bits into binary words. For the
power-of-two-plus-one carry prefix, these packed shares feed the carry and
sign-conversion kernels directly. Ring32 Q14 carries 18 bits per activation
through those initial exchanges instead of one 32-bit word; every Boolean
AND still uses checked multiplication with fresh masks.
This preserves the rounding and ReLU behavior but changes the internal Boolean messages; every
role must select the same option. Shift zero retains the full-ring sign path.

## Products with public zero components

`DeviceInferenceParty(sparse_products=True, ...)` omits two identically zero
contributions when the caller knows that one operand has support only on
components `{0,1}` and the other only on `{2,3}`. This applies to the joint
inputs used by probabilistic truncation, the initial bounded-sign carry
products, and the final compact bit-to-arithmetic conversion. It does not
inspect secrets or apply sparsity to general matrix products.

The four remaining contribution indices are `1,3,4,5`. Each retains its two
original knowledgeable holders. Their `(sender, backup, receiver, outsider)`
routes are `(2,1,0,3)`, `(3,0,1,2)`, `(0,2,3,1)`, and `(1,3,2,0)` respectively.
Every party sends and receives one payload and one 32-byte tag. Each party
consumes three correlated mask draws; zero contributions consume none.
For N ring elements, the global payload is `4*N*(bits/8) + 128` bytes instead
of `6*N*(bits/8) + 192`, in the same single exchange.

This option changes mask counters and internal transcripts. Every role must
select it consistently. The default remains disabled. The internal selected
product helpers require the stated public support invariant; they do not
privately verify zero components. Subsequent carry-tree products use the
ordinary six-contribution protocol because their supports are no longer
complementary.

## Narrow truncation correction products

`DeviceInferenceParty(narrow_products=True, ...)` uses 16-bit arithmetic for
the sign-correction product inside probabilistic truncation when
`1 <= shift <= 16`. Other truncation shifts retain the ordinary product path,
as do products outside this correction. The option is independent of
`sparse_products` and defaults to disabled. All roles must select the same mode.

For ring width `w`, the final correction uses the product `b` only through
`-b * 2**(w-shift)`. Replacing `b` by any value congruent modulo `2**16` changes
that term by a multiple of `2**(w-shift+16)`. This is zero in the full ring
when `shift <= 16`. The implementation projects the existing complementary
arithmetic shares into the 16-bit ring, evaluates the four balanced checked
joints, and lifts each resulting component to an `int64` value in `[0,65535]`.
The coefficient removes any carries introduced by lifting those components.
The truncation output remains in the original 32-bit or 64-bit ring.

The internal `product16` helper does **not** return a general full-ring product.
Its result can enter a full-ring expression only with a coefficient that
annihilates `2**16`. In particular, using this correction at `shift=17` would
be incorrect. The existing strict quarter-ring input contract and exact
probabilistic rounding law remain unchanged; no approximation or statistical
failure mode is introduced.

Masks use purpose-specific keys derived from the domain
`rep4/trunc-pr/product16/v1` plus the ordinary arithmetic domain metadata. Each
holder reserves fresh counters through the party runtime. A mask row generates
`ceil(N/2)` 32-bit ChaCha20 words and treats their halfwords as independent
uniform 16-bit masks. For odd `N`, the final unused halfword is padding within
that row; it is not sent or reused for another row.

Each retained joint sends exactly `2*N` little-endian payload bytes and a
full 32-byte BLAKE3 backup tag. The global exchange therefore carries
`8*N + 128` bytes, with the same checked routes and single communication round
as the four-joint full-width product. Frame validation and every tag check
still run, including under segmented inference capture. This changes private
intermediate shares and transcripts while preserving the rounding rule.

## Prepared inference graphs

[`PhaseGraphParty`](../../src/rep4gpu/models/phase_graph.py) captures the local CUDA work for
a fixed public inference schedule. Prepare each role sequentially, after
sharing the resident model and before starting concurrent inference:

```python
from rep4gpu import FixedPoint
from rep4gpu.models.phase_graph import PhaseGraphParty

party = PhaseGraphParty(
    party_id, transport, local_seeds, bits=32, gpu_open=True,
    rounding='probabilistic', bounded_sign=True)
fp = FixedPoint(party, fractional_bits=14)

# After sharing x, w1, b1, w2, and b2 through this party:
w1 = party.prepare_weight_share(w1, storage='words32')
w2 = party.prepare_weight_share(w2, storage='words32')
def evaluate(x):
    hidden = fp.matmul_relu(x, w1, b1)
    return party.open(fp.add(fp.matmul(hidden, w2), b2))

plan = party.prepare_inference(evaluate, x)
opened = plan.run(next_x)
```

`next_x` must have the same public shape, dtype, device, and sharing mode as
`x`. Captured weights and biases stay unchanged for the plan's lifetime.
Preparation sends no messages and consumes no live mask counters. Each run
reserves fresh nonces, uploads them once, and regenerates every mask. Returned
tensors own their storage, so a later run does not overwrite an earlier result.
Graph preparation and its memory allocations belong to model setup.

By default, each received tag is checked before the next dependent graph
segment. `prepare_inference(evaluate, x, check_at_send=True)` instead permits
local GPU work on private scratch after computing the comparison. Every pending
comparison must pass before the next transport call or returned output,
including before sending final output shares. This option requires fixed
public control flow with no external side effects beyond the intercepted
protocol exchanges. A failed check or malformed frame aborts the run and
poisons the plan; its reserved nonces remain consumed.
When a boundary has several pending comparisons, the graph packs their
verdicts for one host transfer and reports the first failed check.
Verdict copies into retained pinned host buffers are captured in the graph.
The host synchronizes the plan's stream before reading a verdict, so a later
run cannot reuse an earlier successful check.

The graph and eager paths produce identical messages, shares, and counters
for the same seeds, inputs, and protocol options. They use the same transport
security assumptions. Graph capture does not provide GPU networking.

Graphs keep fixed receive buffers. The byte bridge copies received host bytes
directly into those buffers. Device replay preloads recorded messages into
the same buffers before timing, matching its assumption that incoming data
has already arrived on the GPU. Host replay includes those transfers in each
timed exchange. Neither replay mode measures network transfer time.

## Run the optional benchmark

From the repository root, in a CUDA environment with PyTorch, Triton, and NVRTC:

```sh
python -m experiments.benchmarks.inference.device_inference --quick --bits 32 --binary-bits 32 \
  --optimized --repeats 3 --output experiments/results/inference/device_quick.json
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

Use `--phase-graphs --gpu-open` to include prepared inference graphs in both
fresh live execution and isolated replay. Add `--bounded-sign` with
probabilistic rounding, and `--graph-check-at-send` to measure the optional
check boundary described above. For the resident 2048 → 8192 → 2048 block:

```sh
python -m experiments.benchmarks.inference.device_inference --bits 32 --binary-bits 32 \
  --fractional-bits 14 --prepared --weight-storage words32 --optimized \
  --gpu-open --rounding probabilistic --bounded-sign --phase-graphs \
  --graph-check-at-send --batch 1 128 --width 2048 --repeats 7 \
  --output experiments/results/inference/device-graphs.json
```

`GraphDeviceParty` is a separate opt-in experiment for prepared local product
graphs. Call `prepare_product` sequentially outside protocol execution; it
consumes no counters. Each live replay uploads freshly reserved nonces, and
received-message checks remain outside capture and precede reconstruction.
Its [microbenchmark](../../experiments/benchmarks/kernels/graph_product.py) reports a benefit for small
products and a regression for large ones. It is not selected by the inference
benchmark or by `DeviceInferenceParty`.

For independent CPU-crypto compatibility, tree-boundary cases, message ownership,
and tamper rejection, use the project's `crypto` and `test` extras and run:

```sh
python -m pytest -q tests/unit/gpu/test_gpu_prg.py tests/unit/gpu/test_gpu_hash.py \
  tests/unit/inference/test_device_transport.py tests/unit/inference/test_device_runtime.py
```
