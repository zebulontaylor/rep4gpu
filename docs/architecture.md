# Architecture

`rep4gpu` is structured into five strict layers with no upward dependencies
between transports, kernels, and high-level model adapters:

```text
Layer 4: rep4gpu.nn          PyTorch Sequential sharing (share_sequential, SharedSequential)
Layer 3: rep4gpu.protocol    Party -> FusedParty -> DeviceParty -> DeviceInferenceParty
                             -> PhaseGraphParty -> CrtPhaseGraphParty
Layer 2: rep4gpu.arithmetic  FixedPoint, A2B/B2A conversion, exact & probabilistic truncation
Layer 1: rep4gpu.gpu         CUDA/Triton crypto (ChaCha20, BLAKE3), matmul/CRT, and bit kernels
Layer 0: rep4gpu.transports  IdealNetwork, SocketTransport (mTLS), CudaIpcTransport, NcclDeviceTransport
```

## Runtime inheritance spine

Every higher runtime specializes the layer beneath it while keeping the same
four-party replicated share representation (`Share(parts=(c0, c1, c2), binary=bool)`):

| Class | Module | Adds |
| --- | --- | --- |
| `Party` | `protocol/runtime.py` | Core Rep4 3-of-4 share layout, 6-joint multiplication schedule, private input, checked open, host PRG |
| `FusedParty` | `protocol/fused_runtime.py` | Fused Triton joint-input preparation and wire assembly |
| `DeviceParty` | `protocol/device_runtime.py` | GPU-resident ChaCha20 masks, BLAKE3-256 digests, `exchange_device`, 4-joint sparse products |
| `DeviceInferenceParty` | `protocol/ring_inference.py` | Exact floor & probabilistic rescaling, bounded-sign ReLU, `words32`/`strassen32` weights, packed ring8/16 |
| `PhaseGraphParty` | `protocol/phase_graph.py` | Fixed-shape CUDA graph capture across checked communication boundaries with atomic fresh nonces |
| `CrtPhaseGraphParty` | `protocol/crt_runtime.py` | Byte-lane odd-prime CRT 1-product polynomial gates (`crt_matmul`, `crt_conv1d`, `crt_square_sum_pool`, `crt_trunc_relu`) |

`DeviceInputMixin` (`protocol/device_input.py`) can be mixed into `DeviceParty`
subclasses to perform private input consistency checks directly on the GPU with
BLAKE3 instead of host-staged BLAKE2b.

## Transports

All transports expose either byte rounds (`exchange`) or CUDA tensor rounds
(`exchange_device` / `exchange_device_into`):

| Transport | Module | Use case |
| --- | --- | --- |
| `IdealNetwork` / `DeviceIdealEndpoint` | `transports/byte.py`, `transports/device.py` | Colocated in-process testing (no process isolation) |
| `SocketTransport` | `transports/byte.py` | Mutually authenticated TLS over TCP between 4 processes or hosts |
| `DeviceByteBridge` / `PinnedDeviceByteBridge` | `transports/device.py`, `transports/pinned.py` | Bridges CUDA byte tensors over `SocketTransport` (with reusable pinned host buffers) |
| `CudaIpcTransport` | `transports/cuda_ipc.py` | Direct GPU-to-GPU CUDA IPC copies across 4 local Linux processes with mTLS control |
| `NcclDeviceTransport` | `transports/nccl.py` | Direct GPU-to-GPU NCCL `batch_isend_irecv` across 4 GPUs (requires protected fabric + mTLS seed bootstrap) |
| `DeviceRecorder` / `DeviceReplay` | `transports/device.py` | Benchmark-only transcript recording and isolated-party replay (never for live secrets) |

## Correlated seed setup

`correlated_seeds(party, transport, context=...)` (`protocol/setup.py`) runs a
commit-and-reveal protocol over an authenticated transport so each party receives
only the three 256-bit seeds excluding its peers. `make_seeds()` is a trusted
fixture helper for local tests and benchmarks only.
