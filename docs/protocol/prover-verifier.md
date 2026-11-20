# Prover and verifier over TLS

For verification without revealing intermediate states, use the
[private Rep4 audit](private-rep4-audit.md), which runs four verifier processes.
The two-process service described here uses plaintext openings.

The audit service runs a prover and verifier as separate processes over a real
TLS socket. Both peers load the same public toy integer model. The prover
commits its execution boundaries; the verifier authenticates openings and
recomputes transitions using its local model. This CPU reference demonstrates
an end-to-end audit protocol. It is not MPC, a zero-knowledge proof, a full
transformer implementation, or a production inference service.

Inputs, outputs, and opened intermediate states are plaintext to the peers.
TLS protects transport, while Merkle authentication binds openings to the
accepted transcript. Hashes do not hide low-entropy states against guessing.
See [layer commitments](layer-commitments.md) for state encoding, replay,
checkpointing, and the commitment format.

## Run locally

From the repository root, complete the editable installation in the
[README](../../README.md#run-locally), including the `crypto` extra. Then run:

```sh
python -m rep4gpu.proofs.audit_service demo --output-dir /tmp/siv-audit-demo
```

The output directory must not already exist. `demo` creates temporary setup
files, starts a prover process on an ephemeral loopback port, and starts a
verifier process that checks every transition. It removes the temporary setup
when the run ends. The retained output contains `prover.log`, `verifier.log`,
`verifier.json`, `prover/session-000001.json`, and `summary.json`. The summary
records both subprocess exit codes and any orchestration error. The command
returns zero only when both processes succeed; otherwise it returns one and
terminates any remaining prover process.

To exercise sampling instead, add `--samples 2`. Omit that option for full
verification. For this four-layer example, `--samples` accepts 1 through 4.
`--timeout` defaults to 60 seconds. The generated input is
`{"x": [[1, -2, 3, 4]]}` and the accepted output is
`{"x": [[91, 0, 0, 2]]}`.

## Run on two machines

Install the project and its `crypto` extra on both machines. On a trusted setup
machine, create a fresh directory:

```sh
python -m rep4gpu.proofs.audit_service init --directory /tmp/siv-audit-setup
```

`init` writes `model.json`, `input.json`, `ca.pem`, and a certificate and private
key for each role. Its test certificates expire after seven days. The CA
private key exists only during setup and is discarded, so this setup cannot
renew or issue certificates. For deployment work, operators must provision and
manage their own CA, role certificates, and private keys.

Distribute files through an authenticated channel with these role boundaries:

| File | Prover machine | Verifier machine |
| --- | --- | --- |
| `model.json`, `ca.pem` | Copy | Copy |
| `prover.pem`, `verifier.pem` | Copy both public certificates | Copy both public certificates |
| `prover.key` | Copy privately | Do not copy |
| `verifier.key` | Do not copy | Copy privately |
| `input.json` | Not needed as a setup file | Copy |

Keep each private key accessible only to its operator. `init` creates its
private keys with mode `0600`; preserve that protection during distribution.
The input reaches the prover in the protocol request. Both peers have the
public model weights and biases.

For example, place each machine's files in `/srv/siv-audit`. On the prover
machine, run from the repository root:

```sh
python -m rep4gpu.proofs.audit_service serve \
  --host 0.0.0.0 --port 9443 --sessions 1 \
  --model /srv/siv-audit/model.json --ca /srv/siv-audit/ca.pem \
  --cert /srv/siv-audit/prover.pem --key /srv/siv-audit/prover.key \
  --peer-cert /srv/siv-audit/verifier.pem \
  --output-dir /srv/siv-audit/reports
```

On the verifier machine, replace `prover.example.org` with the prover's
reachable address and run:

```sh
python -m rep4gpu.proofs.audit_service verify \
  --host prover.example.org --port 9443 --server-name prover.test \
  --model /srv/siv-audit/model.json --input /srv/siv-audit/input.json \
  --ca /srv/siv-audit/ca.pem \
  --cert /srv/siv-audit/verifier.pem --key /srv/siv-audit/verifier.key \
  --peer-cert /srv/siv-audit/prover.pem \
  --output /srv/siv-audit/verifier.json
```

These commands require TLS 1.3 or later, CA validation, and an exact peer
certificate match, compared through SHA-256 of the DER certificate. The
prover requires a client certificate. The verifier also checks the server's
DNS certificate identity: `--server-name prover.test` matches the generated
test certificate even when `--host` is a different network address. With
operator certificates, set `--server-name` to their intended server identity
and distribute the expected peer certificate to each role.

`verify` writes an accepted report and exits zero on success. On failure it
writes `accepted: false` with an error, prints the error to stderr, and exits
one. `serve` writes `session-000001.json` and subsequent numbered reports,
prints session results, and requires a fresh output directory so prior reports
are not overwritten. It exits one after its configured connection count
if any session failed. Its default `--sessions 0` listens until stopped.
Redirect stdout and stderr to retain logs when running the roles manually.

## What the verifier checks

The toy model accepts a CPU `int64` matrix named `x`, with batch size 1 through
64 and the model's width. Each layer computes `ReLU(x @ weight + bias)` using
signed 64-bit wrapping multiplication and addition. Model width is bounded at
128 and layer count at 64. All boundaries must retain the input tensor names,
dtypes, and exact shapes; in particular, their batch size must equal the
requested batch size. The verifier checks this schema on every opening.
The format identifier `siv-int64-relu-v1` defines these arithmetic and state semantics. A domain-separated SHA-256 digest of the
canonical model specification binds the format, weights, biases, and ordered
layer schedule. Both roles derive that identity locally; the verifier does not
accept a model identity supplied only by the prover.

For each request:

1. The verifier creates a fresh random 256-bit run identifier and sends its
   expected model identity and input.
2. The prover executes the model and sends a commitment covering every
   boundary, including input and output, with that model and run context.
3. After accepting the commitment, the verifier selects distinct transitions
   using system randomness and sends their sorted indices. The challenge is
   unpredictable when the prover commits.
4. The prover opens the input, output, and both boundaries of every selected
   transition. The verifier checks each authentication path against its
   original accepted commitment, validates the state schema, and compares the
   opened input byte-for-byte with the requested input.
5. The verifier independently recomputes each selected transition and compares
   its exact encoded result with the authenticated next boundary. It sends an
   accepted report only after all checks pass. The prover checks that report
   against the session before recording acceptance.

Reports include the root, model and run identifiers, selected transitions,
total transition count, verification mode, output, and plaintext semantics.
Full verification is the default and checks every transition. Sampled
verification checks only the selected transitions; endpoint authentication
alone does not establish that the output follows from the input.

For a fixed committed trace with `L` transitions, exactly `b` of which violate
the verifier's transition relation, a uniformly random sample of `k` distinct
transitions detects at least one violation with probability
`1 - C(L-b, k) / C(L, k)`, taking `C(n, k) = 0` when `k > n`. For one bad
transition this is `k/L`. This statement assumes the trace is fixed before the
challenge and the openings remain bound to it. It is not a claim that a
sampled acceptance proves correctness of unchecked transitions. `--samples`
must be between one and the model's layer count; zero is rejected.

## Wire exchange

Every message has `version: 1` and a `type` field. The remaining fields are
exactly those listed below; unexpected fields or message types are rejected.

| Type | Direction | Fields beyond `version` and `type` |
| --- | --- | --- |
| `request` | Verifier → prover | `model_id`, `run_id` (64 lowercase hexadecimal characters), `inputs` (base64-encoded `SIVSTATE` payload) |
| `commitment` | Prover → verifier | `commitment` (nested commitment envelope) |
| `challenge` | Verifier → prover | `indices` (sorted, distinct, zero-based transition indices) |
| `opening` | Prover → verifier | `opening` (nested state and authentication-path envelope), one message per requested boundary |
| `result` | Verifier → prover | `report` (nested accepted report) |

For transition `i`, the verifier requests boundaries `i` and `i + 1`. Openings
arrive in ascending boundary order, with duplicates removed and boundaries
`0` and `L` always included. The commitment and opening envelopes use the
[layer commitment format](layer-commitments.md#commitment-format). A protocol
error closes the connection without acceptance; there is no structured error
frame. CLI failure reports are local files, not wire messages.

## Resource bounds and integration

Protocol messages are JSON objects with a four-byte big-endian length prefix.
The receiver rejects frames above 16 MiB before reading their body; it also
rejects duplicate JSON keys and invalid message schemas. Model files are
limited to 32 MiB and verifier input files to 1 MiB. The configured socket
timeout applies to connection and socket operations, including TLS; it is not
a total execution-time budget for model computation. The server accepts and
handles connections serially, so one slow session delays later sessions.
`--sessions` bounds accepted connections, including failed handshakes.

Each request owns a commitment runtime and its checkpoint state. Context
managers close the runtime and sockets on success or failure. The default
checkpoint interval is eight; `serve --checkpoint-interval` changes the replay
and memory tradeoff described in [layer commitments](layer-commitments.md).
This small reference has no concurrent worker pool or service-level compute
quota.

To adapt the protocol to a real model, define its exact state boundaries and
arithmetic first, including quantization, rounding, overflow, normalization,
attention caches, and any explicit random state. Bind all weights, parameters,
layer ordering, and those semantics into a content-derived model identity
that the verifier independently derives from its trusted model. Supply a
verifier implementation of each checked transition and compare against
independent expected results; sharing callbacks in this example does not
protect against a bug common to both roles. Extend model and state validation,
transport limits, and resource controls to match the actual workload. Private
weights, private activations, and binding openings to MPC shares require
additional protocols; this service does not implement them.
