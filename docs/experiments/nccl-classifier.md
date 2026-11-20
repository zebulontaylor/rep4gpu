# Run the classifier on four NCCL GPUs

The launcher assigns one Rep4 party to each torchrun rank and one CUDA device
to each local process. Seed setup uses the four parties' mutual TLS mesh.
During inference NCCL sends the masked tensors and verification tags directly
from CUDA buffers into the receiving party's CUDA buffers. The CPU schedules
rounds, checks integrity flags, and collects the opened classification labels.

Use a confidential, authenticated fabric with access restricted to the four
participating ranks. The TLS identity check applies to the control connection.
NCCL has a separate rendezvous and data channel. Across hosts NCCL selects the
available network route, which can use host staging. For physical GPUDirect RDMA,
verify the NIC, drivers and route on the target cluster using NVIDIA's
[GPU-to-NIC guide](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting/gpu_troubleshooting.html#gpu-to-nic-communication).

## Prepare the local files

Each party needs its own `party.pem` and `party.key` plus the common `ca.pem`.
The certificate DNS names are `party0` through `party3`. Only party 0 also
needs a `values.npz` containing the integer feature matrix under key `x`.
Party 1's `values.npz` contains its quantized model arrays under keys `w1`,
`b1`, `w2`, and `b2` for the unscaled square classifier. Use the same signed
ring and parameter shapes as the [classifier export](text-classifier.md#private-neural-encoder).

Give each rank the same public manifest. For the 512-to-128 model and a batch
of 4,096 examples, use this layout with the four hosts' reachable names and
free TLS listener ports in rank order:

```json
{
  "architecture": "square_unscaled",
  "fractional_bits": 0,
  "bits": 32,
  "input_shape": [4096, 512],
  "weight_shapes": {"w1": [512, 128], "b1": [128], "w2": [128, 1], "b2": [1]},
  "addresses": [["gpu-a", 18400], ["gpu-b", 18400], ["gpu-c", 18400], ["gpu-d", 18400]],
  "warmup": 3,
  "repeats": 10
}
```

## Launch and inspect the results

From the repository root with the SIV environment active, run this on each of
four hosts, setting `NODE_RANK` to that host's party number from 0 through 3.
Here `gpu-a` is rank 0's hostname. The job uses torchrun's separate rendezvous
port 29400. Each host reads only its local identity and owner inputs.

```sh
torchrun --nnodes=4 --nproc-per-node=1 --node-rank="$NODE_RANK" \
  --master-addr=gpu-a --master-port=29400 --module \
  experiments.benchmarks.inference.nccl_text_classifier \
  --manifest manifest.json --party-dir private --output-dir results
```

For one host with four distinct GPUs, use
`torchrun --standalone --nproc-per-node=4 --module` with the same module and
arguments, change `--party-dir` to `'private/party{rank}'`, and give each
loopback listener a different port.

Each rank writes `results/partyN.json` with its opened labels, per-iteration
times, protocol bytes, rounds, and check status. The default includes fresh
input sharing. Add `"resident_inputs": true` to share a fixed fixture before
timing. The `host_payload_bytes` counter measures application staging. For a cross-host run, set
`NCCL_DEBUG=INFO`, `NCCL_DEBUG_SUBSYS=INIT,NET,GRAPH`, and
`NCCL_DEBUG_FILE=nccl.%h.%p.log` on every host before launching. Inspect the
send and receive channel lines in each rank's log. NCCL marks network routes
using GPU Direct RDMA with `GDRDMA`. A socket route or an IB interface name
alone does not establish direct GPU transfers. See the
[NCCL network transport](https://github.com/NVIDIA/nccl/blob/master/src/transport/net.cc).

Use each rank's
elapsed durations when comparing hosts with separate monotonic clocks.

On a four-GPU machine, run
`python -m pytest -q tests/integration/test_nccl_classifier.py` to exercise
the full launcher and verify the checked labels against a known classifier.
