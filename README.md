# rep4gpu

GPU-accelerated four-party replicated secret sharing (Rep4) for private neural
network inference with active security against one malicious party (with abort).

Inputs, model weights, and intermediate activations remain secret-shared across
four parties until an explicit checked output opening.

## What is implemented

Two complementary GPU-native Rep4 execution engines share one unified runtime,
CUDA graph capture layer, and multi-process transport stack:

1. **Ring Rep4 ($\mathbb{Z}/2^k$, $k \in \{8, 16, 32, 64\}$)** —
   [`DeviceInferenceParty`](src/rep4gpu/protocol/ring_inference.py) and
   [`PhaseGraphParty`](src/rep4gpu/protocol/phase_graph.py):
   - Exact Triton ring matrix multiplication (`words32`, `strassen32`, `limbs`,
     SM89 modular-add tiles) — no floating-point approximation of ring values.
   - GPU-resident ChaCha20 correlated masks and official BLAKE3-256 integrity
     tags ([`gpu/crypto/`](src/rep4gpu/gpu/crypto/)).
   - Exact carry-based floor truncation and bounded small-gap probabilistic
     truncation with optional sparse (4-joint) and narrow (16-bit) correction
     products and bounded-sign ReLU.
   - Polynomial GELU/SiLU approximations and `torch.nn.Sequential` model sharing
     via [`share_sequential`](src/rep4gpu/nn.py).
   - Fixed-shape CUDA graph capture across checked communication boundaries with
     fresh atomic nonce uploads per inference call.

2. **Odd-prime byte-lane CRT Rep4 ($`\prod_\ell \mathbb{F}_{p_\ell}`$)** —
   [`CrtPhaseGraphParty`](src/rep4gpu/protocol/crt_runtime.py),
   [`crt_conv.py`](src/rep4gpu/gpu/matmul/crt_conv.py), and
   [`crt_relu_gpu.py`](src/rep4gpu/gpu/bits/crt_relu_gpu.py):
   - 3-of-4 replicated shares stored as `uint8` prime lanes (e.g. 3-lane
     `251, 241, 239` or 4-lane `251, 241, 239, 233`).
   - 1-product quadratic polynomial gates (`crt_matmul`, `crt_conv1d`,
     `crt_square_sum_pool`) using INT8 Tensor Cores and a 2-collector active-security
     third-difference + BLAKE3 broadcast.
   - Active-secure GPU CRT probabilistic truncation fused with exact ReLU
     (`crt_trunc_relu`) and private bit-sliced sign extraction.

3. **Transports** ([`src/rep4gpu/transports/`](src/rep4gpu/transports/)):
   - `SocketTransport`: mutually authenticated TLS mesh across processes/hosts.
   - `CudaIpcTransport`: direct CUDA IPC transfers between 4 local processes on
     one Linux GPU with mTLS control.
   - `NcclDeviceTransport`: direct GPU-to-GPU transfers across 4 NCCL GPUs.
   - `PinnedDeviceByteBridge` / `DeviceByteBridge`: pinned or standard host
     staging over any byte transport.
   - `IdealNetwork` / `DeviceIdealEndpoint`: in-process test harness.

## Repository layout

```text
src/rep4gpu/
  transports/          mTLS sockets, CUDA IPC, NCCL, pinned host bridge, ideal harness
  gpu/
    crypto/            CUDA ChaCha20 PRG (single, batched, graph) and BLAKE3-256
    matmul/            Exact ring8/16/32/64 Triton matmul, Strassen, and CRT kernels
    bits/              SWAR/prefix carry trees, probabilistic truncation, CRT ReLU
  arithmetic/          FixedPoint API, A2B/B2A conversions, exact & probabilistic truncation
  protocol/            Rep4 runtime stack: Party -> FusedParty -> DeviceParty
                       -> DeviceInferenceParty -> PhaseGraphParty -> CrtPhaseGraphParty
  nn.py                torch.nn.Sequential sharing and private forward execution
examples/              Self-contained demos and benchmarks (ring FFN, mTLS, CRT MLP/CNN)
docs/                  Architecture, ring protocol, and CRT protocol guides
tests/                 Unit and multi-process integration tests
```

See [docs/architecture.md](docs/architecture.md) for the layered design,
[docs/ring-protocol.md](docs/ring-protocol.md) for the $\mathbb{Z}/2^k$ protocol,
and [docs/crt-protocol.md](docs/crt-protocol.md) for the odd-prime CRT protocol.

## Quick start

Requires Linux with an NVIDIA CUDA GPU and CUDA-enabled PyTorch (CPU protocol
tests also run without a GPU):

```sh
python -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install -e '.[test,crypto]'
python -m pytest -q
```

### Run the examples

```sh
# 1. End-to-end private PyTorch Sequential MLP (CUDA graphs + floor/probabilistic)
python -m examples.mlp_demo

# 2. Four separate OS processes over mTLS + CUDA IPC (or pinned-tls / cpu-tls)
python -m examples.mtls_inference --transport cuda-ipc

# 3. Ring Rep4 FFN benchmark (width -> 4*width ReLU -> width)
python -m examples.ring_ffn_benchmark --quick

# 4. Active-secure 4-process CRT ReLU MLP and 1D CNN benchmarks
python -m examples.crt_benchmark --workload mlp --quick
python -m examples.crt_benchmark --workload cnn --quick
```

## Minimal API example

```python
from rep4gpu import (
    FixedPoint,
    PhaseGraphParty,
    share_sequential,
    sequential_spec,
)

# Each party holds only its 3 correlated seeds (derived via correlated_seeds on mTLS).
party = PhaseGraphParty(
    party_id, transport, local_seeds,
    bits=32, binary_bits=32,
    gpu_open=True, rounding='probabilistic',
    bounded_sign=True, sparse_products=True, narrow_products=True,
)

# Share a PyTorch Sequential model once (only model_owner passes the module).
shared_model = share_sequential(
    party, spec, model if party_id == 1 else None,
    owner=1, fractional_bits=12, weight_storage='words32',
)

# Share private inputs and capture a fixed-shape CUDA phase graph.
x = party.input(0, encoded_x if party_id == 0 else None, input_shape)
plan = party.prepare_inference(lambda inp: party.open(shared_model(inp)), x, check_at_send=True)
opened_logits = plan.run(x)
```

## Security scope and assumptions

- **Threat model**: Honest supermajority (3-of-4) with active security against
  at most one malicious party (security with abort). Two colluding parties can
  reconstruct secrets.
- **Channel & randomness requirements**: Live deployments require independent
  party isolation, mutually authenticated confidential channels (`SocketTransport`
  with mTLS, or a protected NCCL fabric for `NcclDeviceTransport`), and fresh
  session seeds from `correlated_seeds`. Keep party instances and their monotonic
  nonce counters alive for the session; never reuse seeds with reset counters.
- **Accumulator bounds**: Ring and CRT arithmetic wraps modulo $2^k$ or $M = \prod p_\ell$;
  the runtime does not privately range-check arbitrary models. Callers must verify
  accumulator headroom for their chosen ring width / CRT modulus and precision.
- **Research status**: Covered by unit, fault-injection, and multi-process
  integration tests, but not formally verified or independently audited.
