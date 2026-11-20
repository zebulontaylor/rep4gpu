# Layer commitment performance

Prepared CUDA decode is below **3% median overhead** in three 201-sample trials:
2.70%, 2.58%, and 2.46%. Each inference commits all 17 boundaries, retains
checkpoints, and returns a completed Merkle root. The
[combined results](layer_commitments_cuda_decode_summary.json) retain the
threshold checks and links to every trial.

## Decode measurement

The fixture executes 16 exact ring64 residual matrix layers, `x @ weight + x`,
with batch 1, width 2048, checkpoint interval 8, and resident weights. It uses
the project's exact `ring_matmul` on an RTX 4060 Laptop GPU. Layer callbacks
preserve their inputs, so the runtime can batch the small boundary states
without copying each one. This is synthetic plaintext prover inference, not a
trained transformer or distributed Rep4 run.

Each trial prepares fresh graphs, warms both paths 101 times, and measures 201
pairs with alternating baseline/committed order. Every committed inference has
a fresh run ID. The baseline is an inference-only CUDA graph with input copies
and a completion wait. The committed interval includes input copies, inference,
checkpoint retention, all hashes, digest transfer, and the completed root.
Preparation and challenge replay are outside the interval.

| Trial | Plain median | Committed median | Overhead | Paired median overhead | Raw samples |
| --- | ---: | ---: | ---: | ---: | --- |
| 1 | 2.259 ms | 2.320 ms | 2.703% | 2.628% | [JSON](layer_commitments_cuda_decode.json) |
| 2 | 2.259 ms | 2.317 ms | 2.579% | 2.606% | [JSON](layer_commitments_cuda_decode_trial2.json) |
| 3 | 2.256 ms | 2.312 ms | 2.456% | 2.518% | [JSON](layer_commitments_cuda_decode_trial3.json) |

The pooled median of the 603 paired overhead measurements is 2.585%. Individual
samples can exceed 3%; this is a median-latency result, not a tail-latency bound.
All final states matched plain inference and authenticated against their roots.
The JSON files include samples, source fingerprints, backend selection, CPU
affinity, and Torch/CUDA/GPU metadata. CPU affinity was unrestricted.

The [earlier decode result](layer_commitments_cuda_decode_before_optimization.json)
reported 8.99%. Its timing harness added a device-wide synchronization after
`commit()` had already completed. The current harness stops when `commit()`
returns; readiness tests confirm that the graph and checkpoint copy are complete
at that point. Plain inference still waits for completion. The before/after
figures therefore include both implementation improvements and removal of that
redundant timing cost.

## What changed

- Small traces batch BLAKE3 tensor hashing by byte length, reducing per-boundary
  kernel launches. Declared input-preserving callbacks avoid snapshot copies.
- Sparse checkpoints share one packed allocation, with an independent copy for
  each returned execution. Replay still preserves overlapping tensor views.
- Digest transfer runs inside the CUDA graph into pinned host memory.
- A native helper combines launch and completion handling with OpenSSL state,
  leaf, node, and root hashing. Context encoding and response allocation overlap
  inference. It does not defer root computation until a challenge arrives.
- Completed tree hashes stay in one immutable buffer. Membership paths are
  sliced only when requested.

The fast path requires a C compiler, OpenSSL headers/library, and PyTorch's raw
CUDA graph execution handle. Unavailable native support and strided inputs use
checked fallback paths. The [runtime guide](../../../docs/protocol/layer-commitments.md)
defines these contracts and the `preserves_inputs` callback promise. The result
does not establish the same percentage for CPU decode, larger state mappings,
or a different model or device.

## Reproduce

From the repository root after installing `.[test]`, with a C compiler and
OpenSSL development files available:

```sh
python -m experiments.benchmarks.proofs.layer_commitments \
  --device cuda --prepared --batch 1 --width 2048 --layers 16 \
  --warmups 101 --repeats 201 \
  --output experiments/results/proofs/layer_commitments_cuda_decode.json
```

Repeat with separate output paths to retain every trial. The reported
`native_execution`, `native_finalizer`, and `batched_hashes` fields must be true
to reproduce this configuration. `under_three_percent` checks the ratio of
medians; `paired_median_overhead_percent` also checks the result against changes
in baseline latency during a run.

## Earlier reference workloads

The following measurements predate the decode optimization and retain their
original source fingerprints and timing method. They are reference results,
not measurements of the current prepared implementation.

| Workload | Batch | Width | Overhead | Result |
| --- | ---: | ---: | ---: | --- |
| CPU decode | 1 | 256 | 39.017% | [JSON](layer_commitments.json) |
| CPU prefill | 128 | 256 | 12.782% | [JSON](layer_commitments_cpu_prefill.json) |
| Prepared CUDA prefill | 128 | 2048 | 5.723% | [JSON](layer_commitments_cuda_prefill.json) |
| Prepared CUDA prefill | 128 | 4096 | 2.224% | [JSON](layer_commitments_cuda_prefill_w4096.json) |

A [preceding width-4096 run](layer_commitments_cuda_prefill_w4096_prior_alias_fix.json)
measured 1.879%. These prefill results did not establish reliable overhead below
2%; the decode result above addresses the separate 3% target.

## Validation

All 276 relevant tests pass. They cover Merkle paths, malformed packets, context
and state tampering, CPU/CUDA digest agreement, native/Python equivalence, replay
drift, sparse checkpoints, in-place callbacks, overlapping mixed-dtype views,
nondefault streams, strided and empty inputs, completion readiness, retained
executions, and native failure cleanup. CPU references independently check the
native hashes and every authentication path.

```sh
python -m pytest tests/unit/proofs/test_merkle.py \
  tests/unit/proofs/test_state_digest.py tests/unit/proofs/test_layer_commitments.py \
  tests/unit/proofs/test_native_merkle.py tests/unit/gpu/test_gpu_hash.py -q
```
