# Private Text CNN inference with byte-lane CRT polynomial MPC

[Documentation index](../README.md)

This experiment measures active-security 4-party checked inference for a
100%-dense 1D Text CNN trained on the UCI SMS Spam Collection, keeping **both
inputs and model weights secret-shared** throughout inference and opening only
the final binary classification labels after all consistency checks pass.

## Model and certified bounds

Trained and exported by
[`experiments/examples/text_cnn_classifier.py`](../../experiments/examples/text_cnn_classifier.py)
into [`experiments/results/examples/text_cnn_classifier/`](../../experiments/results/examples/text_cnn_classifier/):

- **Dataset**: UCI SMS Spam Collection (`sha256: 7d039a24...`), deduplicated
  stratified split (3,285 train / 820 val / 1,025 test).
- **Input encoding**: 48 tokens per message, each mapped to a 256-dim dense
  Rademacher vector in `{-1, +1}` (100.0% nonzero; shape `(B, 48, 256)`).
- **Architecture**: `Conv1d(256 -> 128, k=5)` -> `Square + GlobalSumPool(L_out=44)` ->
  `Linear(128 -> 1)` -> private `Sign(score >= 0)`.
- **CRT modulus**: 4 odd byte-lane primes `(251, 241, 239, 233)`, product
  $M = 3{,}368{,}562{,}317$, signed limit $\pm 1{,}684{,}281{,}158$.
- **Certified global worst-case integer bounds** (over all $2^{48 \times 256}$ inputs):
  - `|conv_out| <= 1,192`
  - `|pool_out| <= 62,518,016`
  - `|score| <= 680,607,688 < 1,684,281,158` (zero wraparound possible).
- **Held-out SMS Spam test quality (1,025 messages)**:
  - Float: `97.66%` accuracy, `93.16%` spam precision, `87.20%` spam recall (`8` FP, `16` FN).
  - Certified integer (exact MPC oracle): **`97.76%` accuracy, `91.80%` spam precision,
    `89.60%` spam recall (`10` FP, `13` FN), `99.32%` float/integer agreement**.

## Protocol and GPU kernels

- **Arithmetic engine** ([`src/rep4gpu/gpu/matmul/crt_conv.py`](../../src/rep4gpu/gpu/matmul/crt_conv.py),
  [`src/rep4gpu/protocol/crt_runtime.py`](../../src/rep4gpu/protocol/crt_runtime.py)):
  - 3-of-4 replicated shares over $\prod_{\ell=1}^{4} \mathbb{F}_{p_\ell}$ stored as
    three `uint8` CUDA tensors of shape `(4, *logical_shape)`.
  - **1-product quadratic polynomial gates** (`crt_conv1d_local`,
    `crt_square_sum_pool_local`, `crt_matmul_local`): each party evaluates its
    centered `int8` point $F(i), G(i)$ locally and runs one INT8 Tensor Core
    implicit-im2col 1D convolution (`BM=128, BN=128, BK=64`) or fused
    bias-add + square + global-sum-pool reduction (`44x` wire reduction),
    followed by a 2-collector checked broadcast (`7` bytes/output/prime globally,
    2 rounds).
  - **Active security**: collectors P1 and P2 verify the third difference
    $Y(3) - 3Y(2) + 3Y(1) - Y(0) \equiv 0 \pmod{p_\ell}$ *before* Round 2,
    exchange `BLAKE3(BLAKE3(Y(0)) || BLAKE3(Y(3)))` to detect sender/collector
    equivocation, and P3 verifies both collector digests and `BLAKE3(C) == D_C`.
  - **PRG**: official NVRTC BLAKE3-256 keyed XOF (`CHUNK_START|CHUNK_END|ROOT|KEYED_HASH`,
    8-byte monotonic nonce, 64-bit counter) with `mul.hi.u64` 4-digit base-$p$
    extraction (statistical distance $\le p^4 / 2^{64} \le 2.15 \times 10^{-10}$).
  - **Private sign extraction**: bit-sliced 32-bit Rep4 carry tree over the
    authenticated mutual-TLS mesh (`_FastCpuBinaryParty` with SHAKE-256 masks,
    BLAKE2b integrity tags, and single-round checked binary opening), keeping
    the raw score secret-shared and opening only the binary label.
- **Transport and graph execution**
  ([`src/rep4gpu/protocol/cuda_ipc_transport.py`](../../src/rep4gpu/protocol/cuda_ipc_transport.py),
  [`src/rep4gpu/protocol/transport.py`](../../src/rep4gpu/protocol/transport.py),
  [`src/rep4gpu/models/phase_graph.py`](../../src/rep4gpu/models/phase_graph.py)):
  - Captured `InferencePlan` CUDA graph segments with fresh atomic nonce uploads.
  - `CudaIpcTransport` uses interprocess CUDA events (`cudaEventInterprocess`)
    to overlap GPU execution with mutual-TLS descriptor exchange and dedicated
    `alloc_export_tensor` wire buffers to eliminate sender-side D2D copies.
  - `SocketTransport` coalesces frames `<= 32 KiB` without thread-pool scheduling.

## Measured performance (RTX 4060 Laptop GPU, 8 GB VRAM)

All live runs execute **four separate OS processes** with mutually authenticated
TLS, distributed `correlated_seeds`, fresh nonces per batch, and 100% verified
exact agreement with the integer oracle. Native slowdown is computed against the
fastest honest PyTorch baseline across FP32/FP16/BF16, `F.conv1d` and
`unfold+mm`, eager and CUDA graph (`cudnn.benchmark=True`).

| Configuration | Batch | Fastest native | Live 4-proc checked inference | Slowdown (resident) | Live 4-proc online (w/ `crt_input`) | Slowdown (online) | Throughput (resident) | Isolated-party GPU estimate (no network) | Evidence |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| CUDA IPC, resident inputs | 2560 | 2.833 ms (`bf16_conv1d_graph`) | **51.31 ms** | **18.11×** | 51.31 ms | — | 49,896 texts/s | 11.09 ms (**3.92×**) | [JSON](../../experiments/results/inference/text_cnn_b2560_resident_ipc.json) |
| CUDA IPC, fresh `crt_input` | 2560 | 2.829 ms (`bf16_conv1d_eager`) | **51.76 ms** | **18.30×** | 70.88 ms | 25.06× | 49,462 texts/s | — | [JSON](../../experiments/results/inference/text_cnn_b2560_ipc.json) |
| CUDA IPC, resident inputs | 2048 | 2.205 ms (`bf16_conv1d_eager`) | **42.14 ms** | **19.11×** | 42.14 ms | — | 48,605 texts/s | 9.02 ms (**4.09×**) | [JSON](../../experiments/results/inference/text_cnn_b2048_resident_ipc.json) |
| CUDA IPC, fresh `crt_input` | 2048 | 2.213 ms (`bf16_conv1d_eager`) | **42.42 ms** | **19.16×** | 57.51 ms | 25.98× | 48,283 texts/s | — | [JSON](../../experiments/results/inference/text_cnn_b2048_ipc.json) |
| CUDA IPC, resident inputs | 1024 | 0.979 ms (`fp16_conv1d_eager`) | 23.44 ms | 23.93× | 23.44 ms | — | 43,695 texts/s | 4.77 ms (**4.87×**) | [JSON](../../experiments/results/inference/text_cnn_b1024_resident_ipc.json) |
| Host-staged mutual TLS control | 2048 | 2.277 ms (`bf16_conv1d_eager`) | 565.92 ms | 248.51× | 565.92 ms | — | 3,619 texts/s | — | [JSON](../../experiments/results/inference/text_cnn_b2048_resident_host.json) |

Notes:
- Global checked inference traffic at `B=2048` is `330.36 MB` (`14` rounds for
  resident inputs; `16` rounds and `531.69 MB` including fresh dense input
  sharing). Only `160.9 KB` traverses mutual TLS in CUDA IPC mode (155.5 KB
  CPU sign/open payload + 5.5 KB IPC descriptors).
- The `isolated-party GPU estimate` replays the bottleneck party's exact
  captured CUDA graph segments (`party2`) plus input evaluation on one
  uncontended GPU; it excludes network latency, network bandwidth, and the
  final CPU sign tree (~1.8 ms).

## Reproduction

```bash
TRITON_CACHE_DIR=/tmp/triton_cache PYTHONPATH=. .venv/bin/python \
  -m experiments.benchmarks.inference.text_cnn_classifier \
  --batch 2048 --repeats 10 --warmup 3 --device-transport cuda-ipc \
  --resident-inputs --isolated-estimate \
  --output experiments/results/inference/text_cnn_b2048_resident_ipc.json
```
