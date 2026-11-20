# Examples and Benchmarks

Run all commands from the repository root after `pip install -e '.[test,crypto]'`.

## 1. End-to-end Sequential MLP (`examples/mlp_demo.py`)

Trains a small `16 -> 64 -> 8` ReLU MLP in PyTorch, shares weights once with
`share_sequential`, captures CUDA phase graphs, and checks exact private inference:

```sh
python -m examples.mlp_demo
python -m examples.mlp_demo --rounding probabilistic --bits 32
```

## 2. Four-process mTLS inference (`examples/mtls_inference.py`)

Spawns four independent OS processes with mutually authenticated TLS certificates
and distributed `correlated_seeds` setup:

```sh
python -m examples.mtls_inference --transport cuda-ipc
python -m examples.mtls_inference --transport pinned-tls
python -m examples.mtls_inference --transport cpu-tls
```

## 3. Ring Rep4 FFN benchmark (`examples/ring_ffn_benchmark.py`)

Benchmarks a resident `width -> 4*width ReLU -> width` FFN block with GPU
ChaCha20/BLAKE3, captured phase graphs, sparse/narrow truncation corrections,
and optional Strassen ring32 matmuls:

```sh
python -m examples.ring_ffn_benchmark --quick
python -m examples.ring_ffn_benchmark --batch 1 128 --width 2048 --storage strassen32
```

## 4. Odd-prime CRT 4-process benchmark (`examples/crt_benchmark.py`)

Benchmarks active-secure 4-process odd-prime CRT Rep4 over `CudaIpcTransport`
(with mTLS control) or host-staged mTLS:

```sh
python -m examples.crt_benchmark --workload mlp --quick
python -m examples.crt_benchmark --workload cnn --quick
python -m examples.crt_benchmark --workload cnn --batch 2048 --repeats 10
```
