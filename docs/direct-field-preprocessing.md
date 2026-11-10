# Direct native-field preprocessing

The direct path keeps authenticated values in `q = 2**256 - 189` from native
MASCOT through online evaluation. It removes the large-prime-to-p31 conversion
that dominated the earlier bridge. For the same one-query micro model, fresh
preprocessing fell from 1,340.8 seconds to 0.416 seconds. This is a different
field and runtime: these bundles cannot feed the p31 GPU implementation.

The current complete direct-field runtime uses Python big integers in four
TLS processes. Separate [exact q256 GPU matmul kernels](large-prime-kernels.md) pass bigint parity tests,
but they are not yet integrated into the complete authenticated online path.
The preprocessing-inclusive performance target remains unmet.

## Generate and consume a fresh bundle

Use the verified native build from the [native build guide](native-preprocessing.md#build-the-native-backend).
It must include amplification four and the fresh post-authentication sacrifice
challenge. From the repository root:

```sh
python -m scripts.preprocess_mascot_direct \
  --native-root work/mascot256-fresh-build \
  --manifest results/spdz_micro_toy/preprocessing.json \
  --output work/direct-query-1 --timeout 120
python -m scripts.run_spdz_prime256 --bundle work/direct-query-1 \
  --checkpoint results/spdz_micro_toy/model.pt --batch 1 \
  --output results/direct-query-1.json
```

The output directory must be new. Native preprocessing generates full-field
matrix triples, squares, and input masks directly; it does not implement those
objects using an integer-conversion circuit. The micro manifest needs 32
scalar matrix-product triples, two squares, and 42 random matrix-mask values.
It also needs 50 input-mask values: 42 for the model and eight for the query.
Each worker imports only its own native authenticated shares and MAC-key share.
Only an input mask's owner receives its clear value. The coordinator handles
public shapes, build evidence, process status, and public test-fixture outputs.

Native field words use Montgomery representation. The loader verifies the
native format and constant-one calibration before returning canonical q-field
integers. Owner-clear masks use four unsigned 64-bit limbs. The unsigned
conversion is necessary: centering a high-half q value before truncating its
low limb would shift its reconstructed mask by 189. Independent native tests
check matrix and square relations and 34 owner-mask reconstructions.

The CPU runtime uses one MAC coordinate over q, checked consistent broadcasts,
and salted commit/reveal full-vector MAC checks. It checks masked openings
before use and authenticates the final output. Correlation identifiers and
whole-bundle consumption markers prevent a second live use. After a failure,
retire the material and keys; do not delete a marker to retry.

## Prepare a fixed private model

The native fixed-model provider authenticates the model owner's weights once
per pool. For each future query it creates a fresh authenticated mask `A` and
product `F = A @ W`. Online evaluation checks the opening `D = X - A`, then
computes `F + D @ W`. Each row handle is consumed once; sharing immutable
weights across handles does not permit reusing their masks.

Run a fresh one-query fixture from the repository root:

```sh
python -m scripts.run_spdz_prime256_prepared \
  --native-root work/mascot256-fresh-build \
  --bundle work/fixed-model-query-1 \
  --output results/fixed-model-query-1.json
```

The runner uses the existing micro checkpoint, gives its plaintext weights
only to the assigned native model owner, and compares every opened integer
logit with the clear reference. The bundle directory must be new. This provider
supports only `2**256 - 189`; it rejects other fields before consuming handles
or communicating. Native generation still uses scalar MASCOT products, so this
change reduces online work without reducing the asymptotic preprocessing cost.

The [dependent-stream runner and measurements](../results/autoregressive_benchmark_design.md)
serve successive tokens using disjoint rows from a fresh pool. Its plaintext
comparators also execute dependent steps; batching future plaintext inputs
would measure a different workload. Compilation, process provisioning, and
the measured staged costs are identified separately in each report.

## Measured costs and limits

[Actual direct-field inference](../results/spdz_prime256_micro2.json) completed
at all four roles with exact integer output. Fresh native preprocessing took
416.276 ms and the checked CPU/TLS query took 17.361 ms. The
[stage accounting](../results/spdz_prime256_micro2_cost.json) adds role loading
and input/model sharing: its observed staged sum is 506.587 ms, with 314.588 ms
of static compilation separate. This sum is not a single end-to-end wall
measurement or an ideal-network GPU estimate.

The [preprocessing batch measurements](../results/mascot_direct_preprocessing_cost.json)
record fresh generation for 128 and 1,024 rows. Dividing generation by their
fresh row capacities gives 41.590 and 35.539 ms per row respectively. Both batches were subsequently served once with exact integer outputs. Their
[loader-inclusive observed staged costs](../results/spdz_prime256_served_capacity_costs.json)
were 42.788 and 36.059 ms per row. These are throughput figures, not request
latency. The report divides both MPC and the matched plaintext batch by the
same row count; replay does not create additional fresh capacity.

Eliminating conversion removes the main cost of the bridge, but native MASCOT
still generates scalar cryptographic products for matrix multiplication. A
matrix product with dimensions M×K×N still needs MKN scalar products. Reaching
the LLM target requires faster matrix preprocessing as well as an integrated
GPU runtime; these results do not establish that capability.

The [native security discussion](mascot-preprocessing.md) explains the
amplification/challenge patches, honest-role execution and code-approval
assumptions, and conditional status. The direct route removes field-conversion
masking from the computation but retains the selected native cryptographic
assumptions. The local shared-user fixture does not provide hostile-party OS
isolation or remote attestation.
