# Text classifier throughput

The trained classifier runs with private features and weights in four Rep4
processes, but the measured throughput does **not** meet the less-than-20-times
native slowdown target.

The best measured resident slowdown is **192×**, using a private 512→4
ternary encoder and a fixed signed sum of squared activations at batch 65,536.
Checked inference takes 25.54 ms versus 0.13291 ms for the matched sparse native
model. All learned encoder weights, sparsity locations, and biases remain private. Preprocessing is excluded
from both timers, as required by the throughput target.

The [run guide](../../../docs/experiments/text-classifier.md) describes training,
benchmark commands, timing boundaries, and privacy assumptions. Each linked
JSON has a sibling `_logs` directory with all four worker results and logs.

| Device and batch | Checked resident inference | Including input sharing | Resident slowdown |
| --- | ---: | ---: | ---: |
| [CPU, 128](text_classifier_cpu128.json) | 11.03 ms | 13.45 ms | 1,671× |
| [CUDA graphs, 1,024](text_classifier_cuda1024_graph.json) | 12.87 ms | 23.89 ms | 2,263× |
| [CUDA graphs, 8,192](text_classifier_cuda8192_graph.json) | 21.02 ms | 135.37 ms | 1,015× |

The batch-8,192 run processes about 389,695 texts/s with inputs already shared,
or 60,518 texts/s including sharing. Its fastest measured native dense baseline
takes 0.02071 ms per batch. Four processes share one RTX 4060 Laptop GPU and
communicate through loopback TLS. These are actual checked executions with fresh
randomness, not estimates from transcript replay. They are not measurements of
four independently provisioned GPUs.

All measured outputs agree exactly with the quantized integer reference. All 40
focused tests pass, covering CPU/CUDA signed boundaries, full-ring local-product
agreement, fresh graph inputs and counters, and a four-process TLS launch.

## Classification and precision

The [training metadata](../examples/text_classifier/metadata.json) records 4,105
training messages and 1,025 held-out messages after normalized deduplication.
Held-out accuracy is 97.46%, spam precision is 97.14%, and spam recall is 81.6%.
Floating-point and integer predictions agree on every held-out message. SMS spam
quality does not establish suitability for model-control classification.

The [precision sweep](text_classifier_precision.json) evaluates the same held-out
set at several weight scales. Scale 32 preserves every floating-point prediction
with an absolute accumulator bound of 6,027, which fits a signed 14-bit ring.
Scale 128 fits a signed 16-bit ring and also preserves every prediction. These
are arithmetic bounds; those linear runs used 32-bit shares. A separate trained
16-bit model and protocol comparison appear below.

## Remaining performance gap

The specialized one-column kernel removes unused tensor-core output columns,
and live CUDA phase graphs reduce launch overhead. The complete classifier
still spends much more time on input sharing, protocol exchanges, and checks
than native inference spends on this single small matrix-vector product.

At batch 8,192, private input sharing sends 33.62 MB of application payload
globally; inference sends another 0.972 MB. TLS/TCP framing is additional.
Reducing the ring width alone cannot close the observed thousand-fold resident
gap. Reaching the target needs a different cost balance: a more substantial
classifier, a lower-overhead protocol implementation, and measurements on the
intended separate-party hardware and network. Enlarging a model merely to slow
the native baseline would not establish a useful classification result.

## Packed input transfer

The [packed-input run](text_classifier_cuda8192_packed_input.json) narrows ring32
shares on the GPU before transferring them to the host, and widens received
shares on the GPU. This preserves the wire format while avoiding doubled PCIe
traffic and CPU conversion work. Online time at batch 8,192 falls from
135.37 ms to 97.36 ms, increasing throughput from 60,518 to 84,139 texts/s.
The measured resident slowdown remains 943× native.

The [transport profile](text_classifier_transport_profile.json) attributes about
85% of resident execution time to the device/host bridge and socket exchanges.
The profile is instrumented; the packed-input comparison above uses an
uninstrumented run. Peer waiting is included in socket time, so per-party phase
maxima cannot be added to obtain a critical-path breakdown.

## Compiled ring16 comparison

The [HPMPC experiment](hpmpc_classifier/classifier_summary.json) evaluates the
same linear checkpoint at scale 32 using AVX512, Quad with `CV_FIX=1`, and
preprocessing included. After enabling TCP_NODELAY, median maximum-party
execution is 22.44 ms for 8,192 messages. The exact native int16 AVX512 baseline
is 0.0872 ms, giving a 257× slowdown. All labels and consistency checks pass
across seven runs. The patch, native baseline source, and party logs accompany
the summary.

This comparison uses public deterministic mask seeds and plaintext loopback
TCP, as identified by the upstream benchmark setup. It establishes arithmetic
and performance evidence only; it is not a private deployment. The pinned
upstream revision is recorded in the summary. The patched consistency-check
configuration follows the fixes discussed in
[Brüggemann and Schneider](https://encrypto.de/papers/BS26.pdf).

## Private neural encoder and polynomial scaling

The neural model learns a 512→128 encoder, squares its hidden activations, and
applies a learned 128→1 classifier head. Training uses separate training,
validation, and held-out partitions; validation selects the epoch and threshold.
The [training metadata](../examples/neural_text_classifier/metadata.json) records
97.66% quantized held-out accuracy, 96.33% precision, and 84.0% spam recall.

Absorbing positive scales into the polynomial coefficients removes all three
intermediate truncations. The
[export metadata](../examples/unscaled_text_classifier/metadata.json) records
the bound-selected scales and the 1,267,197,386 global signed output bound.
This representation has the same measured accuracy, precision, and recall as
Q6, with 99.90% agreement with the float model. It computes a different integer
approximation; it is not claimed to reproduce every Q6 intermediate.

| Batch 8,192, pinned staging | Resident inference | Including input sharing | Inference payload | Resident slowdown |
| --- | ---: | ---: | ---: | ---: |
| [Linear](text_classifier_cuda8192_pinned_staging.json) | 17.51 ms | 84.28 ms | 0.972 MB | 851× |
| [Neural with exact truncation](neural_text_classifier_cuda8192_graph_pinned.json) | 346.22 ms | 409.30 ms | 359.74 MB | 4,054× |
| [Neural without truncation](unscaled_text_classifier_cuda8192_graph_pinned.json) | 57.18 ms | 121.53 ms | 51.30 MB | 668× |

All runs use live four-process TLS exchanges and fresh protocol randomness.
Every measured label matches its model's integer reference. GPU tests also
compare full private scores with the integer reference for both neural
representations, including bias broadcasting. Scores are opened only in these
correctness tests; the benchmark opens labels.

The neural comparisons retain the same float architecture in the native
denominator. The unscaled representation improves measured resident execution
by about 6× and reduces inference payload by about 7×. The remaining gap is
still too large to claim the throughput goal.

## True ring16 execution

CPU and CUDA runtimes now support two-byte arithmetic messages, with
domain-separated masks, exact modular products, signed opening, and checked
private inputs. CUDA phase graphs and prepared two-limb weights also support
ring16. Binary carry words remain 32 or 64 bits; probabilistic CUDA truncation
is explicitly unsupported for ring16.

The combined ring16, device-input, wire, sparse-native, model, and benchmark
validation suite passes 67 tests. This includes full-width modular operands,
prepared and unprepared kernels, signed boundaries, captured execution, and
rejection of corrupted protocol messages.

The [bounded model](../examples/ring16_text_classifier/metadata.json) uses binary
hashed features, a learned ternary encoder, a square activation, and an integer
head. Its certified global maximum is 2,184. Removing zero head terms reduces
the hidden width to 328 without changing scores. Held-out accuracy is 96.78%,
spam precision is 88.98%, and spam recall is 84.0%. All learned weights, biases,
and sparsity locations remain private.

| Same bounded model, batch 8,192 | Resident inference | Including input sharing | Global payload | Resident slowdown |
| --- | ---: | ---: | ---: | ---: |
| [Ring32](ring16_model_ring32_cuda8192.json) | 158.08 ms | 206.26 ms | 163.50 MB | 3,579× |
| [Ring16](ring16_classifier_cuda8192.json) | 116.24 ms | 144.46 ms | 81.94 MB | 2,644× |

Both runs use GPU input checks, pinned staging, live TLS, and fresh masks.
Every measured label matches the exact integer reference. The ring16 run has
five measured iterations; the ring32 control has three. These are local
measurements on the same GPU, not distributed deployment estimates.

The denominator includes a validated sparse native kernel, which takes about
0.044 ms per batch in these runs. Dense FP16 alone takes about 0.151 ms and
would give an unfairly favorable slowdown. The
[separate native report](sparse_classifier_native.json) includes exact full-score
and label validation and CPU comparisons. This model's 16-bit representation
improves its own execution, but does not beat the earlier smaller neural model
or establish the 20× goal.

## Input checking and concurrent sessions

The [GPU input-check run](unscaled_text_classifier_cuda8192_device_input.json)
keeps the earlier 128-wide neural model and moves its input consistency hash
from CPU BLAKE2b to GPU BLAKE3. Online time is 110.56 ms, compared with 121.53 ms
for the preceding pinned-staging run. The input protocol retains the same mask
placement and three rounds, and tests check exact share equivalence and
rejection of inconsistent corrections.

The concurrent-session harness measures complete batches within the common
steady-state window using host monotonic timestamps. Native baselines run before
MPC workers start. The two-session attempts at
[batch 4,096](unscaled_classifier_concurrent2_b4096.json) and
[batch 2,048](unscaled_classifier_concurrent2_b2048.json) both exhausted the
8 GB GPU while eight party processes were alive. Their sibling session
directories preserve nonzero exit codes and allocation-failure logs. They
provide no valid aggregate throughput measurement.

## Private encoder with a fixed square head

The fixed-head model computes `H = counts @ w1 + b1`, then sums the first half
of `H²`, subtracts the second half, and adds a learned output bias. The signs
are fixed architecture constants. Every learned parameter remains private.
Reducing the local square contributions before resharing avoids transmitting
the full hidden tensor for this operation; it retains the product protocol's
masking and consistency checks.

The [32-wide checkpoint metrics](../examples/fixed_head_text_classifier/model_h32.metrics.json)
record 97.66% held-out accuracy, 95.50% precision, 84.8% recall, and complete
float/integer label agreement. Its global output bound is 626,636,672, which
fits the signed 32-bit ring. Width 32 was selected as a speed/quality candidate
from validation results before evaluating its held-out predictions; the
original maximum-validation-score selection was width 128.

| Fixed head | Resident inference | Including input sharing | Native inference | Resident slowdown |
| --- | ---: | ---: | ---: | ---: |
| [Batch 8,192](fixed_head_h32_cuda8192.json) | 27.46 ms | 85.90 ms | 0.04448 ms | 617× |
| [Batch 32,768](fixed_head_h32_cuda32768.json) | 69.66 ms | 368.36 ms | 0.20020 ms | 348× |

Every measured label agrees with the integer reference, and all four workers
exit successfully. The larger batch reaches 470,424 messages/s with shared
inputs. It also includes narrower CUDA tile dispatch for the 32-column encoder.
The combined fixed-head, fused square-sum, tile-dispatch, phase-graph, and
multiplex validation suite passes 61 tests, including corrupted-message checks
and graph input lifetime regression coverage. Its global inference payload is
29.05 MB. Input sharing adds 134.22 MB and is
reported separately. Both runs use four processes on the same GPU and loopback
TLS; no separate-host speedup is assumed.

The [two-session run](fixed_head_h32_multiplex2_b4096_no_input_cycle.json)
uses four processes, with two independent protocol sessions and CUDA streams in
each process. At batch 4,096 per session it reaches 120,199 messages/s including
fresh input sharing, or 1,540× slower than native processing of the combined
batch. Sessions retain independent seeds, channels, and checks. A recursive
input-collection closure in `InferencePlan.run` previously retained input
tensors until cyclic garbage collection; releasing that closure permits this
run and the larger single-session batch to fit in GPU memory.

## Packed arithmetic storage experiment

The [kernel benchmark](../kernels/packed_ring16.json) compares persistent
`uint16` storage against the current `int64` arithmetic storage for ring16.
Prepared matrix products at batch 8,192 improve from 0.974 to 0.374 ms for
32 output columns, and from 9.420 to 4.825 ms for 328 columns. Generating
4,194,304 ring16 masks improves from 0.134 to 0.0294 ms for one output row.
Full-ring packed/wide outputs and PRG streams agree exactly; sampled matrix
rows also agree with an independent CPU reference.

These are isolated kernel timings, not a packed end-to-end MPC result. The
network already sends two-byte ring16 elements, so packed device storage does
not reduce wire payload. The benchmark script and timing method are recorded
in the JSON.

## Packed storage in checked four-party inference

Opt-in CUDA ring16 arithmetic storage now uses `uint16` shares, with binary
shares unchanged. Tests compare packed and wide local shares and complete wire
transcripts under identical seeds, exercise eager and captured execution, and
reject corrupted input and product messages. Packed shares are rejected by
incompatible runtimes before communication or nonce consumption.

| Model, batch 8,192 | Wide storage | Packed storage | Packed including input sharing |
| --- | ---: | ---: | ---: |
| [328-wide encoder](ring16_packed_storage_packed.json) | 112.07 ms | 81.28 ms | 103.62 ms |
| [4-wide fixed head](tiny_ring16_packed_storage_packed.json) | 19.20 ms | 12.68 ms | 42.20 ms |

The corresponding wide controls are
[328-wide](ring16_packed_storage_wide.json) and
[4-wide](tiny_ring16_packed_storage_wide.json). All four workers exit successfully
and every measured label matches the integer reference. Wire bytes are identical
between each model's packed and wide runs. Native timing varies between runs,
so the storage comparison uses absolute MPC time rather than ratios from
separately timed native baselines.

The [compact model metadata](../examples/tiny_ring16_fixed_head/metadata.json)
records validation selection of four hidden units with 64 ternary coefficients
each. Held-out accuracy is 97.07%, precision is 92.79%, and recall is 82.4%.
Its global output bound is 15,927. All learned encoder coefficients, sparsity
locations, and biases remain private; output signs are fixed public constants.
The sparse native baseline takes 0.02026 ms per batch, yielding a resident
slowdown of 626× despite the packed model's 646,000 messages/s. This result does
not meet the 20× target.

Larger packed batches amortize fixed overhead:

| Compact ring16 model | Resident inference | Including input sharing | Sparse native | Resident slowdown |
| --- | ---: | ---: | ---: | ---: |
| [Batch 32,768](tiny_ring16_packed_storage_b32768.json) | 23.38 ms | 177.06 ms | 0.07757 ms | 301× |
| [Batch 65,536](tiny_ring16_packed_storage_b65536.json) | 38.22 ms | 337.64 ms | 0.13278 ms | 288× |

The largest batch reaches 1.715 million messages/s with shared inputs, or about
194,100 messages/s including input sharing. It has five measured repetitions;
all labels match and all workers exit successfully. Tokenization remains outside
both timers. The complete preprocessed-input service remains substantially slower than
resident inference.

The pinned bridge also stages identical outgoing tensor slices once and passes
read-only host views to the synchronous socket transport. Unknown transports
retain immutable snapshots unless they opt into the borrowed-buffer contract.
Five focused tests cover aliasing and buffer reuse; an
[actual four-party TLS smoke run](pinned_borrowed_tls_smoke.json) passes all
checks. This change preserves payload bytes and has no isolated throughput
claim in this report.

## Narrow four-column matrix products

The compact classifier now dispatches its ring16, 512-input, four-output matrix
products to the existing exact SIMT decode kernel with four output columns per
tile. This avoids unused tensor-core columns. Full-ring tests cover packed and
wide inputs, prepared and raw weights, and exact CPU-reference agreement.

The [combined narrow-kernel and staging run](tiny_ring16_packed_n4_b65536.json)
at batch 65,536 takes **25.54 ms** for checked resident inference, versus
**0.13291 ms** for the sparse native baseline: **192.13×** slowdown and
**2.566 million messages/s**. Including fresh input sharing takes **219.97 ms**,
or about **297,900 messages/s** and **1,655×** native. All four workers exit
successfully and every measured label agrees exactly with the integer reference.

The [local-kernel measurements](../kernels/ring16_n4_tiles.json)
compare the same ring16 product at 4.202 ms with the previous tile and 0.936 ms
with the selected decode tile. Eight independent full-ring oracle tests pass.
The end-to-end run includes both the narrow kernel and host-staging changes;
it does not isolate their individual contributions. The private model and
classification quality are unchanged. The throughput target remains unmet.

## Validation limits

The final combined classifier suite passes **86 tests**: packed arithmetic,
narrow matrix products, staging-buffer ownership, square contraction, phase
graph checks and lifetimes, model export, native-baseline selection, and TLS
launching. This suite includes the fixes for empty packed multiplication, shift
fallbacks, and packed prepared-graph outputs.

The focused packed-storage and existing inference regression run passes 49 tests
and fails two older optimized-versus-reference transcript checks at shift 14
for rings 32 and 64. Both failures reproduce on pristine commit `0e84244`:
the optimized carry circuit has a different gate layout, while those tests
require identical wire transcripts. The
[isolated baseline log](../kernels/packed_storage_pristine_head_regression.log)
records two passes and the same two failures without workspace changes.
These failures are not attributed to packed storage, but the broader regression
suite is not fully green.
