# Test Suite

Run all tests from the repository root:

```sh
python -m pytest -q
```

Layout:
- `tests/unit/protocol/`: Rep4 ring runtimes (`Party`, `FusedParty`, `DeviceParty`, `DeviceInferenceParty`, `PhaseGraphParty`), odd-prime CRT runtime (`CrtPhaseGraphParty`, `crt_trunc_relu`), transports, PRG, nonce reservation, and PyTorch `Sequential` sharing (`rep4gpu.nn`).
- `tests/unit/arithmetic/`: Fixed-point operations (`FixedPoint`), exact carry-based truncation, probabilistic truncation, and compact bit conversions.
- `tests/unit/gpu/`: Exact ring8/16/32/64 Triton matmul kernels, SM89 modular-add & Strassen ring32 kernels, GPU ChaCha20 PRG, GPU BLAKE3-256, SWAR/prefix carry trees, sparse & narrow products.
- `tests/integration/`: Multi-process mTLS (`SocketTransport`), `CudaIpcTransport`, `NcclDeviceTransport` routing, commit/reveal `correlated_seeds` setup, transport limits, and end-to-end trained MLP inference.
