# Rep4 GPU inference results

The GPU variant meets the **less than 100×** target for the tested decode and
prefill blocks against both FP32 and FP16 PyTorch under a zero-network timing
model. This uses bounded probabilistic Q14 rounding. Exact floor rounding meets
the decode target but remains about 120× FP16 for prefill. A checked
4096 × 4096 matrix-vector product also meets the preferred **less than 10×**
target. Complete inference has not reached 10× FP16, and the malicious-majority
bonus is not established.

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

The full regression passes **763 tests and 8 subtests**, including the trained
GPU model and six separate-process TLS configurations.
[Validation summary](results/validation_summary.json), [complete test log](logs/pytest_final.log).

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

## Security scope and remaining work

Rep4 protects against one malicious party out of four; two colluding parties
can reconstruct secrets. It does not satisfy the design document's all-but-one
privacy requirement. The separate authenticated matrix-product experiment
uses a centralized trusted dealer and does not establish distributed
malicious-majority inference or the 500× bonus target.
[Experiment and limits](docs/malicious-majority.md).

The implementation supports dense layers, exact ReLU, fixed-point composition,
and approximate GELU/SiLU. It does not implement full transformer execution,
attention, model binding, authenticated inference records, or the document's
model-substitution detection claim. Real multi-GPU networking, larger-model
numerical validation, and independently reviewed malicious-majority
preprocessing remain separate work.
