# Run the large-prime preprocessing bridge

For the faster preprocessing replacement, use the [direct native-field
path](direct-field-preprocessing.md). This guide retains the q256-to-p31 bridge
for the existing p31 GPU runtime and the shared native build instructions.

The bridge generates authenticated `p = 2**31 - 1` matrix triples, square masks,
and owner-private input masks inside four-party MASCOT over `q = 2**256 - 189`.
Each worker samples its own contributions and imports only its private outputs.
There is no central correlation dealer. The [construction and security
assumptions](mascot-preprocessing.md) explain the native amplification-four
configuration, range checks, private export, and seven application MAC keys.

This is a bounded research bridge, not a fast preprocessing implementation.
Native preprocessing, model-dependent preparation, and private-input sharing
are charged separately from online inference. The local runner puts all roles
under one OS account; deployment requires separate security domains and
privately provisioned identities and role files.

## Build the native backend

Run from the repository root with the Python environment in the
[README](../README.md#run-locally). Supply an MP-SPDZ source matching
[`mascot256-source-lock.json`](../native/mascot256-source-lock.json) and a
dependency cache containing the locked OT archives and libsodium SDK:

```sh
python -m scripts.build_mascot_secure \
  --source /path/to/MP-SPDZ \
  --dependency-root /path/to/verified-native-dependency-cache \
  --build-dir work/mascot256 --jobs 2
python -m scripts.build_mascot_secure \
  --build-dir work/mascot256 --verify-only
```

The destination must be new. The helper leaves the supplied source unchanged,
checks pinned dependencies, applies amplification four and the fresh
post-authentication sacrifice-challenge patch, and builds
`mascot-party.x`. Its policy binds the binary, library, configuration, source,
and dependency hashes. This checks agreement with local build evidence; it is
not attestation against a malicious build host. Missing dependencies or a
source mismatch stop the build.

## Generate one query's material

The micro trained example is an 8 → 2 → square → 8 classifier for the successor
of one synthetic token modulo eight. Its public integer range proof covers all
eight training examples. The clear and private paths use the same calibrated
model. It does not claim held-out language-model accuracy.

```sh
python -m examples.spdz_micro_toy --batch 1 \
  --output results/spdz_micro_toy
python -m scripts.preprocess_mascot --native-root work/mascot256 \
  --manifest results/spdz_micro_toy/preprocessing.json \
  --output work/micro-query-1 --timeout 2700
```

The output directory must be new. Successful generation returns four zero exit
codes and writes a ready `public-manifest.json` beside private `party0` through
`party3` directories. Keep the private directories out of version control.
Each assigned worker creates its own private random input file; the coordinator
handles shapes, build evidence, process status, and public metadata.

The adapter enforces its export-size and integer-width limits before native
execution. It rejects malformed bit contributions before composing bounded
integers, then reduces output shares modulo the application field before
private export. The public manifest records compiler resource counts,
statistical parameters, source fingerprints, and separate compilation/native
timings. A successful run means the export completed; consult the accompanying
security-review status rather than treating a command-line security flag as a
proof.

## Consume the bundle once

```sh
python -m scripts.run_spdz_native --bundle work/micro-query-1 \
  --fixture micro --checkpoint results/spdz_micro_toy/model.pt \
  --fractional-bits 8 --batch 1 --device cuda \
  --backend device-deferred --online-mac-keys 7 --prepare-model \
  --replay-warmups 20 --replay-repeats 25 \
  --output results/micro-query-1.json
```

Each worker loads its own role, shares its designated inputs over mutual TLS,
prepares the model, and checks its final integer logits. The report records
exact-output checks, exit codes, source fingerprints, preprocessing and setup
costs, and ideal-network online timing estimates. CUDA requires the local
PyTorch and Triton installation. The [prepared-model contract](prepared-model.md)
explains why every query still consumes fresh correlations even when weights
remain shared.

The repeated timings replay one recorded public test fixture on restricted
benchmark transports. They do not reuse correlations for live queries. The
initial live inference consumes the entire bundle even if a later check fails.
Do not delete consumption markers: generate another bundle after a failure or
for another query. The large-prime provider exports seven application MAC
coordinates directly; it does not use the old ten-to-seven selection path.

The estimate assumes incoming messages already reside on each GPU and excludes
network latency and bandwidth. This runner does not implement physical
GPUDirect networking. Compare both eager and warmed CUDA-graph plaintext
baselines, and add preprocessing, per-query preparation, and input sharing for
a complete request-cost estimate.

## Retained small-field experiment

The earlier `preprocessing_mama.py` / `scripts.preprocess_mama` path has a
confirmed [mask-privacy flaw](native-mama-sacrifice.md#mask-privacy-limitation).
It can accept a correct correlation whose mask is known to malicious parties
with probability about `1/p`. Its successful correctness checks and timings
are functional evidence only; the independent-sacrifice patch does not repair
privacy. Retained MAMA artifacts must not be used as evidence for 128-bit-private
preprocessing. Use the separate large-prime bridge for the replacement design.
