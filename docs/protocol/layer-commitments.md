# Layer commitments and prover replay

The [layer commitment runtime](../../src/rep4gpu/proofs/layer_commitments.py)
executes deterministic plaintext prover layers, commits every boundary to a
Merkle root, and reconstructs a requested boundary from a nearby checkpoint.
The verifier receives the reconstructed tensors and their authentication path.
Boundary `0` is the input; boundary `i + 1` is the output of layer `i`.

A state is a nonempty mapping of names to dense `torch.int32` or `torch.int64`
tensors. Include everything a later layer needs: hidden activations, residuals,
attention caches, and explicit random state. Device and strides do not affect
the commitment. Names, shapes, dtypes, and integer values do. The runtime does
not choose fixed-point scales or rounding rules; the layer callbacks define
them. Include those choices in the model identifier's specification.

## Commit, request, and receive a state

Run this example from the repository root after the editable installation in
the [README](../../README.md#run-locally):

```python
import json
import torch
from rep4gpu.proofs.layer_commitments import (
    Commitment, CommitmentRuntime, Layer, LayerOpening, verify_layer,
)

layers = [
    Layer('project', lambda state: {
        'hidden': state['x'] * 3,
        'residual': state['x'],
    }, preserves_inputs=True),
    Layer('residual_add', lambda state: {
        'x': state['hidden'] + state['residual'],
    }, preserves_inputs=True),
]
inputs = {'x': torch.tensor([1, -2, 3], dtype=torch.int64)}
with CommitmentRuntime(layers, checkpoint_interval=8) as runtime:
    execution = runtime.commit(inputs, model_id='example-v1', run_id='request-42')

# Send this message before the verifier chooses its boundary.
commitment_message = json.dumps(execution.commitment.to_dict())
accepted = Commitment.from_dict(json.loads(commitment_message))

# The verifier requests boundary 1. The prover reconstructs and sends it.
requested = 1
response_message = json.dumps(execution.open(requested).to_dict())
response = LayerOpening.from_dict(json.loads(response_message))
received = verify_layer(accepted, response, expected_boundary=requested)
assert received['hidden'].tolist() == [3, -6, 9]
assert received['residual'].tolist() == [1, -2, 3]
```

The JSON envelopes can travel through the application's authenticated channel;
this module does not start a network server. Set transport message-size limits
before parsing untrusted envelopes. The verifier must retain its original
accepted root and context, rather than accepting a replacement supplied with
an opening. Agree on a distinct `run_id` for each inference request. A
`model_id` can be a digest of the model, arithmetic configuration, and layer
schedule; the runtime binds the supplied identifier but does not compute or
validate a model-weight digest.

`verify_layer` returns new CPU tensors only after authentication succeeds.
It raises `ValueError` for a wrong boundary, altered state, bad authentication
path, or inconsistent context. To check a transition, request its input and
output boundaries and apply the relevant computation checks. A Merkle opening
proves membership in the committed transcript, not correct inference.

Openings reveal their plaintext state to the recipient. Unkeyed hashes also do
not hide low-entropy states against guessing. This API is separate from the
[Rep4 witness-input protocol](../experiments/witness-activations.md#supply-witnesses-and-recompute):
it does not bind Merkle openings to MPC shares or provide private openings.

## Checkpoints and replay

`checkpoint_interval=8` keeps boundary `0`, `8`, `16`, and so on. Requesting
boundary `13` copies checkpoint `8` and executes five layers. The runtime
checks each replayed boundary's fingerprint against the original transcript
and rejects drift before releasing a response. Checkpoint tensors and returned
states have independent storage; aliases and overlapping views within a state
are preserved during prover replay.

Callbacks must reproduce the same integer results and use unchanged parameters.
Random operations need explicit, reproducible state. Global RNG use, training
mode updates, and data-dependent external side effects do not meet this
contract. Checkpoint replay cannot reconstruct data omitted from the state.
Do not put verifier shares into these plaintext callbacks.

Only checkpoints and hashes survive in each `CommittedExecution`; full tensor
payloads are encoded when requested. More frequent checkpoints shorten replay
at the cost of retained tensor memory. Set the interval above the layer count
to retain only the input. A changed callback is detected when it changes a
replayed boundary; a request satisfied directly by a checkpoint does not rerun
that boundary's producer.

## CUDA execution and hashing

The eager runtime hashes CPU snapshots in reusable worker threads. On CUDA it
hashes tensors on the device, then transfers their 32-byte digests. Tensor
hashes run before later in-place mutations on the producer stream. Host work
can overlap subsequent layers; `max_pending` bounds queued eager work.
`close()` releases workers, and the runtime also supports `with`.

For repeated inference with fixed shapes, prepare a CUDA graph once:

```python
cuda_inputs = {name: value.cuda() for name, value in inputs.items()}
with CommitmentRuntime(layers, checkpoint_interval=8) as runtime:
    prepared = runtime.prepare(cuda_inputs)
    first = prepared.commit(cuda_inputs, model_id='example-v1', run_id='request-43')
    next_inputs = {'x': cuda_inputs['x'] + 1}
    second = prepared.commit(next_inputs, model_id='example-v1', run_id='request-44')
assert first.recompute(2)['x'].cpu().tolist() == [4, -8, 12]
```

Preparation warms and captures callbacks. They must support CUDA graph capture;
all state tensors must stay on one CUDA device. New inputs must match the
captured names, shapes, dtypes, device, and alias relationships. Python branches
and scalar callback configuration are fixed by capture. Reprepare when those
change. Previous execution checkpoints remain valid after another commit.

Set `Layer(..., preserves_inputs=True)` only when the callback does not mutate
its input mapping, any input tensor storage, or an earlier returned state. This
includes mutations through views and residual aliases. All layers in a prepared
model must make this promise to avoid intermediate snapshot copies. The default
remains safe for in-place callbacks.

When the complete boundary trace contains at most 1 MiB of tensor values, the
prepared path groups tensors by byte length and hashes each group in one batch.
Larger traces use a separate hash stream with periodic joins controlled by
`max_pending`. Checkpoints are packed into one buffer and copied once for each
returned execution. Digest transfer to pinned host memory is part of the graph.
CUDA graph buffers remain resident in the prepared runtime.

An optional native helper combines CUDA graph launch, completion waiting, and
OpenSSL SHA-256 finalization. It overlaps context encoding and response-object
allocation with inference. The helper compiles during preparation using `CC`
(or `cc`) and requires OpenSSL development headers and `libcrypto`. `CC` names
an executable, not a shell command. Compilation uses a private temporary
directory and leaves no binaries in the repository.

If native preparation is unavailable, the runtime uses Python graph launch and
hash finalization. `prepared.native_unavailable` records a compiler/library
failure; `prepared.native_cuda_unavailable` records a native CUDA setup failure.
The combined native launch also requires PyTorch's `raw_cuda_graph_exec` API
and contiguous, nonempty input buffers. Strided inputs retain the checked
PyTorch copy path. The fallback has the same commitment format and replay
semantics; the measured latency target uses the native path.

Both execution paths return only after the complete root and independent
checkpoint copy are available. Native tree hashes remain in an immutable byte
buffer; extracting a requested authentication path requires no further hashing.

See the [commitment performance report](../../experiments/results/proofs/layer_commitments_report.md)
for measured overhead and reproduction commands. The benchmark includes input
copies, checkpoints, hashing, transfers, and root completion. Prepared runs
compare against an inference-only CUDA graph, not eager Python execution.

## Commitment format

The format uses two levels of hashing:

1. Each tensor's contiguous, little-endian integer bytes receive a BLAKE3-256
   digest. A SHA-256 state fingerprint binds the sorted tensor names, dtypes,
   shapes, and ordered tensor digests. Its domain is
   `rep4gpu.state-digest.v1\0`, followed by the big-endian 64-bit metadata
   length, canonical ASCII JSON metadata, and the tensor digests.
2. A leaf contains a context digest followed by the state fingerprint. The
   context is SHA-256 of `rep4gpu.layers.v1\0` followed by canonical ASCII JSON
   `[layer_names, model_id, run_id]`. The
   [Merkle primitive](../../src/rep4gpu/proofs/merkle.py) uses separate SHA-256
   domains for indexed leaves, internal nodes, and the count-bound root.

Canonical JSON uses sorted keys, compact separators, and ASCII escaping.
The state metadata is a sorted list of `[name, dtype, shape]`, with dtypes
`<i4` and `<i8`. The Merkle leaf index is the boundary number. Odd tree levels
duplicate the last hash, and the verifier checks those duplicated siblings and
the exact path length. The root binds the boundary count, including the input.

Wire state payloads use `SIVSTATE` followed by version byte `1`, a big-endian
32-bit metadata length, the same canonical state metadata, and ordered raw
tensor bytes. JSON openings base64-encode that payload and hex-encode sibling
hashes. Decoding rejects duplicate or unsorted names, unsupported schemas,
truncation, and trailing data; it does not load pickle objects.
