# Rep4 GPU inference results

The GPU variant meets the **less than 100×** target for the tested decode and
prefill blocks against both FP32 and FP16 PyTorch under a zero-network timing
model. This uses bounded probabilistic Q14 rounding. Exact floor rounding meets
the decode target but remains about 120× FP16 for prefill. A checked
4096 × 4096 matrix-vector product also meets the preferred **less than 10×**
target. Complete inference has not reached 10× FP16. The malicious-majority
online experiments have improved, but the original small-field native
preprocessor has a [confirmed privacy limitation](docs/native-mama-sacrifice.md#mask-privacy-limitation).
Its correctness and timing results do not establish secure dealer-free inference.

These results come from one RTX 4060 Laptop GPU with 8 GB of memory. The
four-party runtime executes real masks, messages, integrity checks, and secret
arithmetic. The independent-GPU numbers below are **estimates from isolated
party replay**, not measurements on four networked GPUs.

## Inference latency

The network is a secret-weight 2048 → 8192 ReLU → 2048 feed-forward block.
Inputs, both weight matrices, and both biases are privately shared. Floor mode
uses exact signed Q14 rescaling; its outputs equal the integer reference.
Probabilistic mode rounds each rescaling to its floor or ceiling, with unbiased
rounding under uniform fresh masks. Its outputs satisfy propagated integer
bounds and agree at all parties. Decode has batch size 1; prefill has batch size 128.

| Runtime and workload | Zero-network estimate | Versus FP32 graph | Versus FP16 graph | Measured four roles on one GPU | Evidence |
| --- | ---: | ---: | ---: | ---: | --- |
| BLAKE2b, ring32 decode | 14.91 ms | 23.0× | 46.9× | 65.85 ms | [JSON](results/inference_decode2048_ring32_q14.json) |
| BLAKE2b, ring64 decode | 20.90 ms | 33.3× | 65.8× | 77.19 ms | [JSON](results/inference_decode2048_ring64_q14.json) |
| BLAKE2b, ring32 prefill | 579.64 ms | 503.6× | 1681.0× | 1451.02 ms | [JSON](results/inference_prefill2048_ring32_q14.json) |
| GPU BLAKE3, floor decode | 13.70 ms | 20.4× | 43.0× | 79.51 ms | [JSON](results/final_device_floor_decode.json) |
| GPU BLAKE3, floor prefill | 42.62 ms | 34.2× | 120.5× | 181.62 ms | [JSON](results/final_device_floor_prefill.json) |
| GPU BLAKE3, probabilistic decode | 10.11 ms | 14.9× | 31.8× | 54.51 ms | [JSON](results/final_device_prob_decode.json) |
| GPU BLAKE3, probabilistic prefill | 29.51 ms | 32.1× | 85.1× | 123.56 ms | [JSON](results/final_device_prob_prefill.json) |

![Inference slowdown against resident PyTorch](docs/assets/performance.png)

The BLAKE2b runtime retains MP-SPDZ's core message encodings and consistency
digests. Its benchmark uses AES-256-CTR correlated masks. The experimental GPU
runtime replaces mask generation with CUDA ChaCha20 and online digests with
BLAKE3, including its optional device-side output check in these final runs.
It preserves recipients, six joint inputs per multiplication, and
32-byte tag sizes; its cryptographic bytes are deliberately different.
See the [device protocol](docs/device-protocol.md).

The GPU BLAKE3 estimate assumes incoming online messages already reside on the
GPU. For the probabilistic variant, a replay that includes host staging instead
takes **12.16 ms for decode** and **51.58 ms for prefill**. That prefill result is
about **149× FP16**, so the GPU-incoming assumption matters to the target.
Physical GPUDirect networking is not implemented.
Both estimates exclude network latency and bandwidth. The measured four-role
column uses one shared GPU and includes contention between roles.

The probabilistic GPU decode is 3.31× the single-threaded NumPy baseline;
prefill is 0.323× NumPy. These CPU comparisons use the same
fixtures but are less demanding than FP16 CUDA inference. Floor mode's maximum absolute
log-probability error relative to FP32 is 0.00242 for decode and 0.00289 for
prefill. These are errors on generated fixtures, not language-model quality
measurements. Across the fresh probabilistic runs, the worst absolute
log-probability errors are 0.00252 for decode and 0.00291 for prefill.

The probabilistic prefill bound is 664,019,065 in encoded accumulator units,
strictly below its quarter-ring limit of 1,073,741,824. Propagated output
intervals span at most 40 Q14 integer units. The runtime does not privately
check this range for arbitrary models; the benchmark establishes it from its
public fixture before any sharing.

Exact prefill sends 352.72 MB of application payload globally in 25 rounds;
probabilistic prefill sends 239.60 MB in 20 rounds. Actual network bandwidth and
round latency remain additional deployment costs.

## Matrix multiplication

For a secret `(1, 4096) @ (4096, 4096)` product in ring32, the checked
zero-network estimate is **1.326 ms**, or **3.82× FP32 and 8.29× FP16**
CUDA-graph matrix multiplication. The measured four-role colocated time is
4.885 ms. It sends 98,496 application bytes globally in one communication
round. This timer includes masks, BLAKE2b checks, and resharing; output opening
is verified separately and excluded from this primitive's timer.
[Nine-sample evidence](results/primitives_decode_words32.json).

Prepared weights store each ring32 component in four bytes. A party holds
three components, so its resident weight storage is six times that of a
single FP16 plaintext weight matrix. The decode kernel computes the three
local multiplication contributions directly from those words. Larger matrix
products use exact balanced int8 limb products and tensor cores; they do not
cast secret ring values to approximate floating-point arithmetic.

## Correctness and communication evidence

The combined regression passes **1,438 tests and 8 subtests**, including the
trained GPU model and separate-process TLS configurations. The subsequent
direct-q256 addition passes 45 focused tests and native role-local checks.
[Validation summary](results/malicious_final_all_validation.json),
[complete regression log](logs/pytest_malicious_final_all.log),
[direct-field test log](logs/pytest_prime256_final.log).

- An independent Rep4 oracle compares recipients, protocol phases, payloads,
  digests, and local replicated components. Input ownership and checked opening
  are covered separately. Transport framing is custom, so this is application
  protocol correspondence rather than binary interoperability with MP-SPDZ.
- Actual four-party MP-SPDZ executables pass the retained ring32/ring64
  arithmetic and fixed-point fixtures. The installed snapshot's optimized
  ring32 binary composition fails a boundary fixture; its failure is retained,
  and the passing ring32 golden uses generic per-bit conversion.
  [Golden report](tests/golden/report.md).
- The trained synthetic 16 → 64 ReLU → 8 next-token MLP matches all **512
  integer logits across 64 contexts** on CPU, CUDA, and native ring32/ring64
  MP-SPDZ. Both floating-point and private fixed-point inference attain 100%
  accuracy on these training contexts; no held-out result is claimed.
  [Example](examples/README.md), [whole-network golden](tests/golden/toy_ffn_report.md).
  The fast CUDA-crypto path also passes every context in both ring sizes:
  floor mode matches all integer logits, and probabilistic mode stays within
  the proved intervals while retaining 100% classification accuracy.
- Separate processes complete private inference over mutually authenticated
  TLS with distributed correlated-seed setup and checked outputs.
  [Socket report and logs](results/socket_inference_report.md).
- The GPU crypto variant has independent CPU ChaCha20/BLAKE3 transcript
  comparisons and malformed-message tests. Hash tests cover partial blocks,
  odd trees, multi-megabyte inputs, streams, and deterministic graph replay.
  Live protocol graph replay must not freeze and reuse mask nonces.

Exact carry-based truncation and weighted binary-to-arithmetic conversion are
protocol optimizations. Their arithmetic matches the golden; their complete
activation traffic differs from MP-SPDZ's default circuits. ReLU is exact.
GELU and SiLU use documented polynomial approximations, and a polynomial
composition API supports other approximations. [Activation contracts](docs/activations.md).

## Timing boundaries and reproduction

Input sharing, model sharing, static weight preparation, and JIT compilation
are outside resident-model inference timing. Inference includes its checked
final opening. Each live execution uses fresh masks. Replay restores a fixed
public fixture solely to measure one role at a time, verifies outgoing bytes
during warmup, and retains actual cryptographic computation. The estimate sums
the maximum per-role median for each protocol phase. It excludes inter-party
network transfer time and does not demonstrate a functioning network pipeline.

PyTorch comparisons include eager and resident CUDA-graph measurements; the
table uses graph device time. FP32 disables TF32. NumPy BLAS and PyTorch CPU
threads are limited to one. Each JSON retains samples, phase timings, range
checks, package versions, source hashes, baseline errors, and separate setup
costs. Model sharing and static preparation can be amortized over reuse of the
shared model. Private-input sharing recurs for each fresh request and must be
added to an end-to-end latency estimate. A one-shot run must also include model
setup and compilation.

Eager inference can be faster than a captured graph for a particular shape.
Probabilistic prefill is 32.51× FP32 eager and 81.60× FP16 eager, so its target
does not depend on selecting the graph baseline. FP32 graph denominators vary
across runs; the JSON files retain each measured denominator rather than
replacing it with a common value.

After [installation](README.md#run-locally), run from the repository root:

```sh
python -m benchmarks.inference --device cuda --bits 32 --binary-bits 32 \
  --fused --prg aes --truncation compact --fractional-bits 14 \
  --prepared --weight-storage words32 --batch 1 --width 2048 --repeats 7 \
  --output results/inference-reproduction.json

python -m benchmarks.device_inference --bits 32 --binary-bits 32 \
  --fractional-bits 14 --prepared --weight-storage words32 \
  --optimized --gpu-open --rounding probabilistic \
  --batch 128 --width 2048 --repeats 5 \
  --output results/device-reproduction.json

python -m pytest -q
```

The public fixture bounds fit ring32 at Q14. This does not justify ring32 or
the same precision for arbitrary models: signed accumulators must fit before
rescaling. Ring64 is available when more headroom is needed.

## Malicious-majority target and measured costs

The malicious-majority target **includes amortized fresh cryptographic
preprocessing and is not met**. Removing work from the online timer does not
remove its throughput cost. A preprocessing batch can be divided only by the
number of distinct one-use query packs it creates; replay samples do not
increase that number. Static compilation and model sharing have separate
amortization rules.

The current GPU path uses seven independent application MAC coordinates,
canonical packed field shares, exact integer matrix kernels, committed joint
check randomness, and deferred internal checks that finish before output
shares are released. It supports a bounded integer square MLP, rather than a
complete fixed-point transformer. All parties matched the integer references
in the following synthetic workloads.

| Workload | Online ideal-network estimate | Versus FP32 graph | Versus FP16 graph | Fresh model preparation, four roles colocated | Evidence |
| --- | ---: | ---: | ---: | ---: | --- |
| 1024 → 4096 → 1024, one row | 5.315 ms | 114.7× | 80.9× | 117.43 ms | [JSON](results/spdz_prepared7_decode1024_quiet.json) |
| 512 → 2048 → 512, one row | 5.104 ms | 330.3× | 370.0× | 72.26 ms | [JSON](results/spdz_prepared7_decode_prefill512_quiet.json) |
| 512 → 2048 → 512, 128 rows | 7.592 ms | 95.6× | 227.2× | 295.70 ms | [JSON](results/spdz_prepared7_decode_prefill512_quiet.json) |
| 1024 → 4096 → 1024, 16 rows | 8.018 ms | 56.5× | 196.1× | 124.52 ms | [JSON](results/spdz_prepared7_batch16_32_width1024.json) |
| 1024 → 4096 → 1024, 32 rows | 8.597 ms | 69.8× | 211.6× | 119.48 ms | [JSON](results/spdz_prepared7_batch16_32_width1024.json) |

These are online timings with **trusted correlation fixtures**, not measured
large-model dealer-free throughput. Each row retains 20 discarded replay
warmups and 25 timed samples per role. The estimate sums per-phase maxima of
role medians and assumes GPU-resident incoming messages. The graph comparator
uses warmed CPU-submitted CUDA-graph repetitions. The 1024 one-row run has
remaining tail variation in two roles; its raw samples and fastest-group
comparators are retained. The 512 one-row and prefill tails were stable.
Physical GPUDirect networking is not implemented.

The preparation column is actual colocated wall time, while the query column
is an isolated-role projection. Their sum is an observed partial aggregate,
not a rigorous perfect-network lower bound. Secure preprocessing and new-input
sharing are also absent from that sum. The [stage accounting
report](results/spdz_pipeline_costs.json) keeps the missing secure cost unknown
and does not substitute trusted-dealer fixture generation for it.

The [prepared-model interface](docs/prepared-model.md) opens `E=W-B` and
computes `F=C+A@E` before a query. Every query needs fresh `A/F`. A new partition
operation divides one prepared matrix batch into disjoint one-use query
handles, sharing an owned immutable `W`. It permits the weight-sized opening
and preparation to be amortized across actual fresh rows without reusing query
masks. Its CPU and GPU tests cover complete partitions, mismatched partitions,
parent/child reuse, scope, and mutation checks.

A fused projection-accumulation epilogue reduces the measured three-segment
check kernel sequence from 0.218 ms to 0.127 ms in the isolated
[microbenchmark](artifacts/field_weighted_accumulate_bench.json). The table
predates that integration; it does not subtract the microbenchmark improvement
from whole-query results.

## Direct native-field replacement

Removing q256-to-p31 conversion reduced fresh preprocessing for the trained
micro model from **1,340.8 seconds to 0.416 seconds**, about 3,200×. The
[direct-field path](docs/direct-field-preprocessing.md) keeps native q256
shares throughout; it cannot feed the existing p31 GPU runtime.

The [actual direct native inference](results/spdz_prime256_micro2.json) passed
all four TLS roles and matched the independent integer oracle. Checked online
CPU latency was 17.361 ms. Including role loading and input/model sharing gives
an [observed staged sum](results/spdz_prime256_micro2_cost.json) of 506.587 ms;
static compilation is separate. This is still about 35,582× the measured CPU
FP32 eager baseline for this tiny model. It is not a GPU or perfect-network
measurement, and it does not meet the preprocessing-inclusive target.

The [exact q256 tensor-core kernels](docs/large-prime-kernels.md) now supply
value and MAC products inside the fixed-model runtime. A
[fresh native/GPU/TLS toy run](results/spdz_prime256_prepared_gpu_micro2.json)
passed all integer outputs. Its checked query took 11.609 ms, including
1.353 ms of GPU product calls with packing and transfers. GPU setup took
571.482 ms, including public-zero kernel warmup; fresh native preparation took
1,669.775 ms. This single-query fixture does not establish a speedup or an
amortized preprocessing target. Input sharing, activations, and MAC checks
remain on CPU, and the four CUDA contexts share one physical GPU.

Fresh direct-field batches of 128 and 1,024 rows took 5.324 and 36.392 seconds
of native generation. Their [per-capacity-row costs](results/mascot_direct_preprocessing_cost.json)
are 41.590 and 35.539 ms. Both batches were then served once, with all integer
outputs correct. Their [observed staged totals](results/spdz_prime256_served_capacity_costs.json),
including loading and sharing, were 42.788 and 36.059 ms per row. Matched-batch
FP32 comparisons remain about 303,885× and 937,788× respectively. Both sides
use the same batch size; these are throughput comparisons, not single-query
latency ratios. Scalar-triple generation remains the dominant cost.

### Dependent toy stream

The [32-step successor-token benchmark](results/autoregressive_benchmark_design.md)
uses the same small square MLP, preparing fresh correlations for future steps
before their token inputs are known. It checked every integer logit and
consumed every row once. Its observed staged sum, including fresh native pool
generation and online setup, was 5,883.410 ms, or 183.857 ms per token.
Native compilation and process/certificate provisioning are separate.

| Matched clear comparator | Time per token | Preprocessing-inclusive slowdown |
| --- | ---: | ---: |
| PyTorch CPU FP32 | 31.70 μs | 5,800× |
| NumPy CPU | 4.98 μs | 36,916× |
| PyTorch GPU FP32, resident per-step graph | 8.70 μs | 21,123× |
| PyTorch GPU FP16, resident per-step graph | 7.20 μs | 25,536× |

These are provisional shared-host measurements from separate contended windows.
Every clear comparator preserves token dependencies. The GPU comparator retains
the next token on device; host-logit delivery is measured separately. These
ratios are not perfect-network projections, and they do not establish a full
transformer result or meet the malicious-majority target.

### Goldilocks streams with persistent HE keys

The native LowGear route also evaluates the same dependent micro-model stream
over `p = 2**64 - 2**32 + 1`, using persistent HE keys and fresh correlation
pools. The configured settings are S40 and 128-bit computational security;
a whole-epoch statistical error bound has **not** been established. These are
conditional experimental CPU/TLS results, and the preprocessing-inclusive
performance target remains unmet.

The [128-token run](results/goldilocks_dependent128.json) served every dependent
step with exact integer logits and one-use rows. Its fresh pool cost, including
native compilation and provisioning, was 10.148 seconds. Including owner
fixture setup and the actual consumer-process stopwatch gives 14.394 seconds,
or 112.452 ms per served token. This is a warm-key epoch measurement, not a
cold-start result.

| Matched 128-step clear sequence | Clear sequence time | Warm-key process-inclusive slowdown |
| --- | ---: | ---: |
| PyTorch CPU FP32 | 2.851 ms | 5,048× |
| NumPy CPU | 0.522 ms | 27,564× |
| PyTorch GPU FP16, resident per-step graph, wall | 0.919 ms | 15,669× |

The FP16 CUDA-event comparator is 0.911 ms, giving 15,794×. Neither GPU
comparator waits for host logits between steps. The
[stage accounting](results/goldilocks_dependent128_costs.json) keeps the initial
cold cost separate: measured HE-key generation took about **471 seconds**, and
the complete first cold pool took 481.134 seconds. Those initial costs are
excluded from the warm-key figures above. Charging the whole first cold pool
to this stream gives a conservative 495.527-second staged case; it is not a
separately measured cold 128-token run, and its earlier independent rows do not
increase this stream's dependent-token denominator. Setup reuse never permits
correlation reuse or credit for unserved capacity.

The [fresh 2,400-token deferred-check run](results/goldilocks_dependent2400_deferred.json)
also served every dependent step with exact logits at all four roles. Internal
masked-opening checks finished before each public output. Its fresh pool,
including native compilation and provisioning, cost 10.934 seconds; the
checked sequence cost 16.056 seconds, or 6.690 ms per token. Including owner setup and the actual
consumer-process stopwatch gives **29.182 seconds**, or **12.159 ms per served
token**, with warm HE keys.

| Matched 2,400-step clear sequence | Clear sequence time | Warm-key process-inclusive slowdown |
| --- | ---: | ---: |
| PyTorch CPU FP32 | 51.941 ms | 562× |
| NumPy CPU | 9.372 ms | 3,114× |
| PyTorch GPU FP32, resident per-step graph, wall | 16.327 ms | 1,787× |
| PyTorch GPU FP16, resident per-step graph, wall | 17.119 ms | 1,705× |

The corresponding FP16 event measurement is 17.100 ms and gives 1,707×.
The larger pool amortizes fresh preprocessing over 2,400 **actually served**
dependent tokens; it does not batch future token inputs or count replay as new
capacity. Initial HE-key setup remains excluded from these warm figures.
Charging the complete first cold pool as a conservative setup case gives
510.315 seconds, or 212.631 ms per token, for this run. That sum is not a
separately measured cold 2,400-token sequence. The S40 whole-epoch bound remains
unestablished, and neither warm nor cold accounting meets the target.

## Dealer-free conversion-bridge evidence

The replacement [large-prime MASCOT bridge](docs/mascot-preprocessing.md) runs
four native roles over `q=2**256-189`, with amplification four and a fresh joint
sacrifice challenge after authentication. Each assigned worker samples its
private contributions. The secure computation produces canonical p31 value
shares, seven application MAC coordinates, input masks, and correlation
outputs privately for their designated roles. There is no central randomness
dealer. Honest roles still need approved code, private execution domains, and
authenticated confidential channels.

The [repaired native probe](results/mascot_postauth_preprocessing_validation.json)
completed with four successful native exits and four successful independent
role-local matrix/MAC checks. Two malformed-input runs, including a non-first
lane whose check has zero low 64 bits, caused all roles to abort before export.
These tests and the source argument support a conditional experimental
construction, not a formal implementation-security proof.

That one-use 1×1 matrix-triple probe required **47.02 seconds of fresh
preprocessing**, excluding 10.74 seconds of compilation. Native reports record
about **42.39 GB of logical protocol traffic**. The tiny probe includes native
batch/setup overhead and is not a large-batch throughput prediction. Its
[public cost report](results/mascot_fresh_probe_cost.json) distinguishes static
compilation, fresh generation, native-process times, and rounded traffic.
The [trained bridge inference](results/spdz_mascot_fresh_micro2.json) also
completed with exact output at all four roles. Its fresh preprocessing took
1,340.8 seconds and its ideal-network GPU query estimate was 3.884 ms. The
direct-field replacement above avoids that conversion cost.

The trained micro model is an 8 → 2 → square → 8 successor-token classifier.
Its floating-point and Q8 versions classify all eight training examples
correctly, with maximum logit error 0.06472. The public absolute-sum bound is
653,079,894, below the signed p31 limit. These are training examples, with no
held-out language-model claim. The [independent integer
validation](results/spdz_micro_toy/validation.json) checks every example; it is
separate from the fresh native private-inference test.

### Superseded preprocessing paths

The retained small-field MAMA path has a [known-mask privacy
flaw](docs/native-mama-sacrifice.md#mask-privacy-limitation). Its independent
sacrifice improves product checking but does not repair that flaw. The earlier
large-field build also used a sacrifice challenge predictable before
candidate authentication; it is superseded by the post-authentication joint
challenge build. Old functional/timing records retain their measured values
and explicit failed-security annotations. Neither old path establishes secure
dealer-free inference, and no stopped or superseded bundle is reused.

## Security scope and remaining work

Rep4 protects against one malicious party among four. The separate
prime-field protocol targets correctness and privacy with abort against three
malicious parties, conditional on its preprocessing, channel, randomness, and
honest-execution assumptions. The direct native-field path addresses the central-dealer requirement
experimentally and removes the conversion bridge, but its fresh scalar-triple
work remains too expensive for the preprocessing-inclusive target.

Efficient malicious matrix preprocessing is the principal missing capability.
The installed native malicious protocols generate scalar correlations; their
matrix interface does not provide the packed malicious matrix-OLE path needed
here. A persistent producer can amortize setup and hold a fixed private model,
but it must still charge every fresh correlation. Published constructions such
as [malicious matrix preprocessing](https://eprint.iacr.org/2020/451) are an
algorithmic direction, rather than an implemented performance claim.

The Rep4 implementation supports dense layers, exact ReLU, fixed-point
composition, and approximate GELU/SiLU. The malicious-majority prototype
currently supports field matmuls and polynomial operations without general
private rescaling. Full transformers, attention, authenticated inference
records, model binding, model-substitution detection, and physical multi-GPU
networking remain outside the implemented result.
