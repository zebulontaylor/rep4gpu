# Four-process checked TLS inference report

All eight CPU/CUDA cases completed with four zero exits and exact agreement with the integer two-layer network reference. Each case uses four fresh Python processes, four distinct mutually authenticated TLS certificate identities, distributed triple-seed commit/reveal, private input from party 0, private weights and biases from party 1, compact exact truncation, and a checked final opening. The network dimensions are 8 → 8 → 4 with Q12 fixed point. The public toy fixture's accumulator bounds were checked before selecting the 32-bit or 64-bit arithmetic ring.

| Execution | Ring bits | Batch | All-phase global payload | Inference global payload | Result |
|---|---:|---:|---:|---:|---|
| CPU | 32 | 1 | 18,840 | 13,792 | [report](socket_cpu32_b1.json) |
| CPU | 32 | 16 | 166,200 | 160,192 | [report](socket_cpu32_b16.json) |
| CPU | 64 | 1 | 24,536 | 18,560 | [report](socket_cpu64_b1.json) |
| CPU | 64 | 16 | 238,616 | 230,720 | [report](socket_cpu64_b16.json) |
| CUDA fused | 32 | 1 | 18,840 | 13,792 | [report](socket_cuda_fused32_b1.json) |
| CUDA fused | 32 | 16 | 166,200 | 160,192 | [report](socket_cuda_fused32_b16.json) |
| CUDA fused | 64 | 1 | 24,536 | 18,560 | [report](socket_cuda_fused64_b1.json) |
| CUDA fused | 64 | 16 | 238,616 | 230,720 | [report](socket_cuda_fused64_b16.json) |

Payload counts cover application messages, including consistency digests; TLS/TCP framing is excluded. All-phase totals include correlated setup and private input. Inference totals include the checked final reveal. CPU and fused CUDA payload counts agree for every matching case. Per-party timing, rounds, final outputs, and process exit codes are in the JSON reports; each sibling `_logs` directory contains four process logs and four per-party reports. Temporary certificates, keys, role input files, and correlated seeds are not exported.

CUDA runs place four isolated processes on one physical GPU. Their wall times include startup, CUDA context creation, and possible JIT compilation. These runs demonstrate real process isolation and checked network execution; they do not estimate four independent GPUs or production serving latency.

Coordinator tests cover CPU32/batch1, CPU64/batch16, successful exit-code/report preservation, ephemeral key cleanup, early rejection of fused CPU execution, environment-independent CLI help, and forced-timeout child cleanup.

## Experimental device protocol over TLS

The CUDA ChaCha20/BLAKE3 variant also passes complete inference in four separate
processes through `DeviceByteBridge` and the same mutual-TLS setup. These runs
use batch 2, binary words matching the arithmetic ring, and the same 8 → 8 → 4
Q12 network. Both reports contain four zero exit codes and exact integer outputs.
Host/device copies remain part of this real socket path; it does not use GPUDirect.

| Ring bits | All-phase global payload | Evidence |
| --- | ---: | --- |
| 32 | 15,256 bytes | [Report and sibling logs](socket_device32_b2.json) |
| 64 | 22,616 bytes | [Report and sibling logs](socket_device64_b2.json) |

The full regression also validates the optimized path and optional GPU output
opening. All six retained configurations have four zero exits, exact integer
outputs, authenticated TLS, and distributed seed setup. Each report has a
sibling `_logs` directory containing the four process logs and party reports.

| Ring | Optimized kernels | GPU output check | Retained report |
| --- | --- | --- | --- |
| 32 | No | No | [JSON](socket_final_device32_optimized0_gpuopen0.json) |
| 64 | No | No | [JSON](socket_final_device64_optimized0_gpuopen0.json) |
| 32 | Yes | No | [JSON](socket_final_device32_optimized1_gpuopen0.json) |
| 64 | Yes | No | [JSON](socket_final_device64_optimized1_gpuopen0.json) |
| 32 | Yes | Yes | [JSON](socket_final_device32_optimized1_gpuopen1.json) |
| 64 | Yes | Yes | [JSON](socket_final_device64_optimized1_gpuopen1.json) |

```sh
python -m benchmarks.socket_inference --device cuda --device-protocol --bits 32 \
  --batch 2 --output results/socket_device32_b2.json
```

To reproduce from the repository root:

```sh
python -m benchmarks.socket_inference --device cpu --bits 32 --batch 1 --output results/socket_cpu32_b1.json
python -m benchmarks.socket_inference --device cuda --fused --bits 64 --batch 16 --output results/socket_cuda_fused64_b16.json
python -m pytest tests/test_socket_inference.py -q
```
