# Private text classification

Train a small SMS spam classifier, share its inputs and weights with four Rep4
parties, and measure checked label inference against native floating-point
inference. This workload exercises a complete text classification path. Its
SMS labels do not measure whether a classifier can detect problematic behavior
in a model-control setup.

## Train and run

From the repository root, using the environment in the [setup guide](../../README.md#run-locally):

```sh
python -m experiments.examples.text_classifier
python -m experiments.benchmarks.inference.text_classifier \
  --checkpoint experiments/results/examples/text_classifier/model.npz \
  --device cpu --batch 128 \
  --output experiments/results/inference/text_classifier_cpu128.json
```

Training downloads the [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms)
to `.local/datasets/SMSSpamCollection`. It removes duplicate normalized messages
before a deterministic stratified train/test split. The checkpoint and
`metadata.json` are written to `experiments/results/examples/text_classifier/`.
The metadata records the dataset hash, classification metrics, quantization
agreement, and an accumulator bound.

For larger CUDA batches:

```sh
python -m experiments.benchmarks.inference.text_classifier \
  --checkpoint experiments/results/examples/text_classifier/model.npz \
  --device cuda --batch 8192 --repeats 5 --phase-graphs \
  --output experiments/results/inference/text_classifier_cuda8192_graph.json
```

The benchmark saves a JSON report and sibling `_logs` directory containing each
party's results and process log. A nonzero command exit indicates failed checks,
a failed worker, or a timeout. Temporary certificates, private keys, and owner
input files are removed when the launcher exits. Batches larger than the test
set cycle through the held-out features for timing; they add no evaluation data.

## Model and arithmetic

The input owner lowercases and tokenizes each message, then hashes its first
128 word tokens into 512 count buckets. A final constant-one column supplies
the bias. These features stay on the input owner's machine until sharing.
The public tokenizer has no learned parameters; the learned linear weights
and bias belong to the model owner.

Training fits a regularized logistic classifier. Inference uses rounded integer
weights and computes one matrix product. A nonnegative score means spam. There
is no sigmoid or fixed-point rescaling because neither changes the classification
threshold. The runtime computes the sign privately and opens only the binary
class to the four computing parties.

The training bound is `max_tokens * max(abs(weights_without_bias)) + abs(bias)`.
It covers every input produced by the tokenizer. The benchmark also validates
the loaded fixture's absolute-product bound before inference. Neither check is
a private proof that an adversarial input owner followed the tokenizer. An
application that needs that guarantee must bind the shared features to the
text and preprocessing it intends to classify.

The linear checkpoint above uses `Z/(2**32)` by default. The precision sweep
identifies weights whose bounds fit a smaller ring. The benchmark also supports
`--bits 16` on CPU and CUDA for checkpoints whose full arithmetic fits that ring:

```sh
python -m experiments.benchmarks.inference.text_classifier_precision --help
```

## Throughput interpretation

Weights are shared once before measurement. Each timed iteration uses fresh
input sharing, fresh protocol masks, private classification, and checked output
opening. The report separates input sharing from resident inference and also
reports their combined online time. Setup and graph capture are separate from
steady-state inference. Tokenization is outside both MPC and native timers.
Already-preprocessed features still need private input sharing; resident timing
assumes that sharing has completed. Use the combined online time when the
service starts with features held only by their owner.

The native baseline computes the same dense linear model and threshold with
floating-point weights. CUDA reports both eager execution and CUDA graphs in
FP32 and FP16; the fastest measured baseline defines the slowdown. A sparse
native implementation could be faster still. A successful correctness check
does not mean the less-than-20-times slowdown target passed; the report includes
a separate `target_met_resident` field.

All four processes run on one host, and CUDA processes share one GPU. TLS uses
loopback networking. These measurements include the actual transport and host
staging, but do not establish throughput on four independent GPUs or across a
WAN. Phase graphs reserve fresh nonces on each run and perform live checked
exchanges; they do not replay recorded messages.

The specialized CUDA kernel uses direct modular matrix-vector products for the
single output column. `--generic-kernel` selects the existing matrix-product
path for comparison. Both paths retain the same masking and consistency checks.

`--pinned-staging` batches outgoing GPU-to-host copies behind one stream
synchronization and reuses pinned host buffers for incoming messages. Identical
outgoing tensor slices are staged once, and the synchronous socket transport
borrows those buffers until its sends complete. It keeps the live TLS transport
and protocol checks. This can reduce staging overhead;
it does not provide GPUDirect networking.

### Keep classifier messages on the GPU

On a single Linux GPU, use `--device-transport cuda-ipc` to send input
corrections, masked products, and check tags between four local party
processes. The sender copies each outgoing frame into a dedicated CUDA
allocation. The receiver copies from that allocation into its own device
buffer. TLS carries the IPC handles and message sizes. This mode is for
trusted local processes because importing an IPC handle grants write access
to the exported buffer.

Run the trained 512-to-128 classifier from the repository root:

```sh
python -m experiments.benchmarks.inference.multiplex_text_classifier \
  --checkpoint experiments/results/examples/unscaled_text_classifier/model.npz \
  --sessions 2 --batch 4096 --repeats 50 --warmup 5 --device-transport cuda-ipc \
  --output .local/benchmarks/ipc-classifier.json
```

The report gives aggregate checked texts per second. The worker records in
the sibling `_sessions` directory distinguish logical MPC bytes, payload bytes
staged by the application, and socket bytes used to coordinate the parties.
The measured input-sharing path starts with features on the owner's device.
For inference with pre-shared inputs, add `--resident-inputs`. CUDA performs
the tensor operations and digest comparisons. The CPU establishes the seeds,
reserves mask counters, schedules exchanges, and enforces the GPU check
results before releasing dependent messages. The benchmark reads opened
labels back to the host to check predictions. The [device transport results](../../experiments/results/inference/classifier_device_transport.json) include throughput and an Nsight trace summary for the resident classifier.

For four distinct GPUs, use the [NCCL classifier launcher](nccl-classifier.md).
NCCL chooses a route for the available hardware. Confirm GPUDirect RDMA
on the cluster before interpreting inter-host measurements.

## Private neural encoder

The neural example learns both an encoder and a classifier head. The public
input-owner preprocessing remains token hashing; no pretrained encoder weights
are public. Its architecture is `counts/8 → Linear(512,128) → square → Linear(128,1)`.
Both layers and their biases are secret-shared from the model owner.

```sh
python -m experiments.examples.neural_text_classifier
python -m experiments.examples.unscaled_text_classifier
python -m experiments.benchmarks.inference.text_classifier \
  --checkpoint experiments/results/examples/unscaled_text_classifier/model.npz \
  --device cuda --batch 8192 --repeats 5 --phase-graphs --pinned-staging \
  --output experiments/results/inference/unscaled_text_classifier_cuda8192.json
```

Training uses separate training, validation, and test partitions. Validation
selects the epoch and threshold. The unscaled exporter selects coefficient
scales using arithmetic bounds, without using test metrics for selection.

The square activation makes the model a polynomial. Rather than rescaling after
each product, the exporter absorbs the positive scales into integer coefficients:

```text
H = raw_counts @ w1 + b1
score = (H * H) @ w2 + b2
label = (score >= 0)
```

Only the sign matters, so the final positive output scale needs no division.
The selected coefficients have a certified absolute output bound of
1,267,197,386 for every message within the 128-token budget. All intermediate
bounds also fit the signed 32-bit ring. The protocol therefore needs no
intermediate truncation or activation opening. The exported integer model is
an approximation of the learned float model; its metadata records the measured
prediction agreement and quality.

For comparison, the neural training command also produces `model_q6.npz` and
`model_q8.npz`. The same benchmark automatically detects those checkpoints and
uses exact floor rescaling after each product. It checks every measured label
against the corresponding integer model, whichever representation is selected.

### Pipeline the 512-to-128 encoder

The RTX 4060 encoder kernel can prefetch reduction tiles into shared memory
while tensor cores multiply the previous tile. Each party still completes the
checked exchange before it uses the encoder output. Separate sessions can
overlap independent batches on CUDA streams.

From the repository root, run two sessions on the same GPU:

```sh
python -m experiments.benchmarks.inference.multiplex_text_classifier \
  --checkpoint experiments/results/examples/unscaled_text_classifier/model.npz \
  --sessions 2 --batch 4096 --encoder-pipeline-stages 3 --prepare-encoder \
  --output .local/benchmarks/pipelined-classifier.json
```

The JSON reports aggregate checked texts per second including private input
sharing. Add `--resident-inputs` to share each fixed batch once before timing.
Each inference still generates fresh product masks. Omit both
`--encoder-pipeline-stages` and `--prepare-encoder` for the original kernel.

The experimental kernel supports one through four stages with compact ring32
weights. Preparing the encoder caches five weight matrices as signed byte
planes per party and permits a fifth stage. To compare depths independently of
the TLS transport, run
`python -m experiments.benchmarks.kernels.pipelined_encoder --prepared --stages 1 2 3 4 5 --output .local/benchmarks/encoder-kernels.json`.
The report includes CUDA event timings, shared memory per block and exact
comparisons with the current kernel and a CPU reference. The saved
[kernel timings](../../experiments/results/kernels/classifier_encoder_pipeline.json)
and [live throughput comparison](../../experiments/results/inference/classifier_encoder_pipeline.json)
cover the 512-to-128 encoder on the RTX 4060 Laptop GPU.

## Private encoder with a fixed square head

This variant learns an encoder and output bias while fixing the output signs
as architecture constants: half the squared activations contribute positively,
and half negatively. All learned weights and biases stay private. A fused
square-and-reduce operation shares one output column instead of every hidden
activation, reducing communication.

```sh
python -m experiments.examples.fixed_head_text_classifier
python -m experiments.benchmarks.inference.text_classifier \
  --checkpoint experiments/results/examples/fixed_head_text_classifier/model_h32.npz \
  --device cuda --batch 32768 --repeats 5 --warmup 2 \
  --phase-graphs --pinned-staging --device-input \
  --output experiments/results/inference/fixed_head_h32_cuda32768.json
```

Training writes a checkpoint for each candidate width and chooses `model.npz`
by validation score. The command above explicitly benchmarks the 32-wide
candidate. Its integer coefficients absorb scaling, so inference requires no
intermediate truncation. Checkpoint metrics record classification quality and
bounds; the [measurement report](../../experiments/results/inference/text_classifier_report.md)
compares model-only throughput with the fastest measured matched native model.

## A bounded 16-bit model

The ring16 example trains a sparse ternary encoder over binary hashed token
presence, then uses a square activation and an integer head. Validation selects
the architecture and decision threshold. Exact removal of zero head terms
reduces the deployed hidden width from 512 to 328 without changing scores.
Learned sparsity locations remain secret along with the weights and biases.

```sh
python -m experiments.examples.ring16_text_classifier
python -m experiments.benchmarks.inference.text_classifier \
  --checkpoint experiments/results/examples/ring16_text_classifier/model_pruned.npz \
  --bits 16 --device cuda --batch 8192 --repeats 5 \
  --phase-graphs --pinned-staging --device-input \
  --output experiments/results/inference/ring16_classifier_cuda8192.json
```

All allowed inputs have binary feature values and at most 128 nonzero features.
The certified maximum absolute accumulator is 2,184. Arithmetic messages use
two-byte ring elements; binary carry operations use 32-bit packed words. CUDA
ring16 supports exact floor rescaling but rejects probabilistic rescaling.

`--device-input` keeps the correction and its consistency hashing on the GPU.
It uses the same mask placement and three input rounds as the reference input
protocol, with BLAKE3 tags replacing BLAKE2b. All parties must select the same
variant. Both recipients check their corrections before returning the shared
input. This avoids large CPU hash work while retaining live TLS transfers.

For this sparse integer model, the benchmark validates and measures an optimized
sparse native implementation as well as dense PyTorch. The fastest measured
implementation defines the slowdown. Its learned indices are available to the
native baseline only; the MPC computation retains private dense shares.

## Compact ring16 encoder and packed GPU storage

The compact ring16 example trains a sparse ternary encoder with a fixed square
head. Validation selects four hidden units with 64 nonzero weights each. Its
learned coefficients, sparsity locations, and biases remain private. Binary
hashed features produce an absolute output bound of 15,927, so the complete
encoder and score fit the signed 16-bit ring without truncation.

```sh
python -m experiments.examples.tiny_ring16_fixed_head
python -m experiments.benchmarks.inference.text_classifier \
  --checkpoint experiments/results/examples/tiny_ring16_fixed_head/model.npz \
  --bits 16 --device cuda --batch 65536 --repeats 5 \
  --phase-graphs --pinned-staging --device-input --packed-arithmetic \
  --output experiments/results/inference/tiny_ring16_packed.json
```

`--packed-arithmetic` keeps CUDA arithmetic shares in `uint16` storage. It is
available only for ring16 CUDA execution; the default remains `int64` storage.
Binary carry words retain their existing representation. This reduces device
memory traffic while preserving the two-byte wire format, mask streams, and
protocol checks. Sign and exact truncation adapters widen values internally
when the existing conversion kernels require it.

The benchmark also measures an exact sparse native implementation. Its public
indices belong only to the native comparison; the private protocol evaluates
dense shares so it does not reveal the encoder's sparsity locations. Held-out
SMS quality and throughput are recorded separately in the
[measurement report](../../experiments/results/inference/text_classifier_report.md).

## Privacy assumptions

Rep4 tolerates at most one corrupt party among four; two colluding parties can
reconstruct the weights and inputs. The local launcher gives each worker only
its role's plaintext, but is a benchmark harness on one operating-system account,
not an independently administered deployment. Production parties need separate
trust domains and authenticated confidential channels.

Opening a class still releases information about the classifier. Private weights
do not prevent adaptive probing through repeated labels, and this example does
not enforce query limits, model identity, or a policy for who receives outputs.
