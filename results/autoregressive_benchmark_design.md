# Dependent-stream benchmark contract

This workload serves one fixed-length sequence through the trained 8 → 2 →
square → 8 micro model. It is a synthetic successor-token classifier, not a
transformer, attention benchmark, or held-out language-model evaluation.

The first one-hot token is privately supplied by owner 0. At every step the
four roles evaluate the complete authenticated MLP and open all eight checked
logits. The lowest-index argmax determines the next token. Because those
logits are public outputs, subsequent token inputs are public; they do not
need another private-input protocol. No future token input is known when its
correlation row is prepared.

## Fresh pools and timing boundary

The native owner supplies the model once per pool. Native preprocessing creates
T rows of independent A and F=A@W for each layer. Each role imports those rows
into T disjoint one-use handles sharing immutable W. The runner consumes one
handle from each layer and one distinct square mask at every step. It verifies
all T outputs against an independent Python-integer dependent trace, all
handles are used, and every role follows the same token trajectory. Replay
samples never increase the fresh token count.

The measured staged sequence cost includes fresh native pool generation,
maximum role setup/loading/import time, initial private-seed sharing, and the
complete dependent checked sequence. Native compilation and certificate/process
provisioning remain separate. The sum of stage maxima is an observed staged
aggregate, not one end-to-end stopwatch or an ideal-network projection.
Dividing that aggregate by T measures cost per actually served token. Online
step times are also retained; amortized pool cost is not first-token latency.

## Matched clear denominators

Every clear comparator runs T dependent MLP evaluations, deriving each next
input from its own preceding logits. The runner checks its whole trajectory
against the integer trace. It never substitutes a B-row independent matrix
batch for the dependent sequence.

CPU PyTorch and NumPy each execute the loop directly. Optional FP32 and FP16
GPU comparators retain the model and state on the device and include argmax
and next-input update at every step. They provide two delivery contracts:

- **Resident:** the next token remains on the GPU. Eager or per-step graph
  launches follow the same state dependency; there is no intermediate
  device-to-host transfer. This is the primary GPU comparator.
- **Host logits:** every step also copies all eight logits to the CPU. This
  separate comparator matches the CPU/TLS runner's public-output delivery.

A CUDA graph captures one complete stateful step, then is replayed in order;
it does not batch dependent steps. CUDA-event samples include any starvation
between CPU-submitted graph launches. Entire-sequence wall timings are retained
as well. GPU baselines require their own validated measurement window; CPU
protocol timings do not establish a GPU authenticated-runtime result.

The isolated implementation is
[`scripts/run_spdz_prime256_autoregressive.py`](../scripts/run_spdz_prime256_autoregressive.py),
with clear references in
[`benchmarks/spdz_autoregressive.py`](../benchmarks/spdz_autoregressive.py).
The first T=32 run passed all four TLS roles and all integer logits, using a
fresh pool once. Its [record](spdz_prime256_autoregressive32.json) and
[contention context](spdz_prime256_autoregressive32.context.json) retain the
CPU-only provisional measurements. No preprocessing-inclusive target is
claimed from that run.

## Provisional T=32 measurements

The four-role CPU/TLS stream checked every logit and consumed all 32 rows in
both layer pools plus all 32 square masks. Fresh pool generation took
5,016.791 ms; the checked dependent sequence took 659.453 ms. Including role
setup and initial seed sharing gives an observed staged sum of 5,883.410 ms,
or 183.857 ms per actually served token. Compilation is separate.

The [separate GPU baseline record](spdz_prime256_autoregressive32_gpu_baselines.json)
validates the complete same token trajectory for FP32 and FP16. It retains
fresh source hashes, timestamps, all samples, and its own contention context.

| Clear comparator | FP32, 32-step sequence | FP16, 32-step sequence |
| --- | ---: | ---: |
| Resident per-step graph, CUDA events | 0.278528 ms | 0.230400 ms |
| Resident per-step graph, wall | 0.284724 ms | 0.237396 ms |
| Per-step graph with host logits, wall | 0.761048 ms | 0.838192 ms |

These GPU measurements ran while a capped LowGear key-generation probe used
other CPU cores. The CPU/MPC and clear runs occurred in different contended
windows. Their staged ratios are provisional, and no preprocessing-inclusive
target is met. The faster resident comparator does not wait for the CPU
between tokens; the host-delivery variant is reported separately.
