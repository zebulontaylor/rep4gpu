# Private Rep4 trace audit

The private audit runs an external prover and four Rep4 verifiers as five
separate processes. The prover executes the toy integer model and
commits secret-shared execution boundaries. The verifiers authenticate their
shares and check transitions together over a mutually authenticated TLS mesh.
They reveal one aggregate acceptance bit, then open only the final output if
the checks pass. Weights, biases, and intermediate boundary values stay shared
throughout verifier computation.

This CPU reference checks `ReLU(x @ weight + bias)` in the 64-bit ring, with
signed interpretation for ReLU and wrapping addition and multiplication. It
uses secret-shared integer weights and biases without fixed-point rescaling.
It is not a transformer implementation or a formal zero-knowledge security claim.
The [public prover/verifier protocol](prover-verifier.md) describes the toy
plaintext model semantics, state schema, and sampling guarantee. The private
model identity is defined below.

## Privacy and trust

The public information includes the model commitment, architecture, input,
tensor shapes, layer schedule, run metadata, sampled indices, acceptance, and
accepted output. All five roles load `input.json`: private input delivery is
not implemented. Weights, biases, and intermediate states are hidden from each
individual verifier. The prover knows the model and its execution trace.

A trusted model owner provisions four fresh additive components of every
weight and bias tensor. Each verifier receives only its three components and
their salts in `partyN-model.json`. `model-manifest.json` contains the dimensions
and four salted component digests. Its `model_id` hashes this manifest content,
so it binds the provisioned shares and arithmetic format. It is randomized on
each provisioning; it is not the plaintext model hash used by the public audit.
The owner must approve and authenticate the manifest independently of the
prover. The audit does not prove that these shares match a separately supplied
plaintext model hash. Changing the approved model requires provisioning new
shares and a new manifest.

`init` uses the repository's published toy parameters, so those example values
are already known. Role isolation demonstrates the private protocol; hiding a
real model requires provisioning parameters that are not publicly available.

Each verifier receives three of four random additive components of each
requested boundary and never receives the missing component or its salt.
Rep4's trust threshold is at most one malicious verifier; two colluding
verifiers can reconstruct the model and trace. A verifier colluding with the
prover can also learn them from the prover. Deploy the verifier roles under independent
operators if that separation is part of the privacy requirement. Running four
processes under one administrator does not establish that separation. Public
inputs and outputs can themselves disclose information about intermediate
states and model parameters; sharing does not hide what follows from public
information.

## Run the local example

From the repository root, follow the [installation instructions](../../README.md#run-locally)
including the `crypto` extra, then run:

```sh
python -m rep4gpu.proofs.private_audit_service demo \
  --output-dir /tmp/siv-private-demo
```

Use a new output directory. The command starts five real processes with TLS
connections on loopback, creates temporary role directories containing only
each role's own private key and model material, and removes setup files when
finished. This is a local protocol demonstration, not evidence of an independently operated
multi-server deployment.

The default checks every transition. Add `--samples 2` to check two of the four
example transitions, or `--timeout 120` to change the default 60-second timeout.
The example input is `{"x": [[1, -2, 3, 4]]}` and accepted output is
`{"x": [[91, 0, 0, 2]]}`. The output directory retains `prover.log`,
`party0.log` through `party3.log`, each role's `result.json` or `failure.json`,
and `summary.json` with process exit codes and any orchestration error. Each
role also writes `traffic.json`; a successful demo aggregates these counters
in its summary. The demo returns zero only when all five processes succeed; it terminates remaining
processes on failure or expiry of its execution deadline.

## Deploy on five servers

Install the project and `crypto` extra on each server. On a trusted setup
machine, create a fresh directory:

```sh
python -m rep4gpu.proofs.private_audit_service init \
  --directory /tmp/siv-private-setup
```

`init` generates the toy model, its private share bundles and public manifest,
input, deployment configuration, CA
certificate, and certificates and keys for `prover` and `party0` through
`party3`. These example certificates expire after seven days. The CA key is
discarded, so operators must provision their own CA and role certificates for
renewable deployments.

Edit `deployment.json` before distributing it. `prover` is a `[host, port]`
pair; `verifiers` contains four such pairs in party-number order. The defaults
use loopback ports 19443 through 19447. Replace these with addresses reachable
from the other servers that also bind to an interface on the respective
listening server. The same configuration must be present on every host;
roles compare it as part of their public context. The current format uses
each address for both binding and connecting, so a wildcard listening address
alone is not a deployment address. Allow the prover's listening port from all
four verifiers and the full verifier peer mesh through the relevant firewalls.

Set `samples` to `null` for all transitions or an integer from one through the
model's layer count. `timeout` is a finite positive number of seconds;
`checkpoint_interval` is a positive integer, defaulting to eight.

Distribute the following files through an authenticated channel, for example
into `/srv/siv-private` on every host:

| Files | Prover host | Each verifier host |
| --- | --- | --- |
| `deployment.json`, `model-manifest.json`, `input.json`, `ca.pem` | Copy | Copy |
| `model.json` | Copy privately | Do not copy |
| `partyN-model.json` | Do not copy | Copy privately only to party N |
| `prover.pem`, `party0.pem`, `party1.pem`, `party2.pem`, `party3.pem` | Copy all public certificates | Copy all public certificates |
| `prover.key` | Copy privately | Do not copy |
| `partyN.key` | Do not copy | Copy privately only to party N |

Preserve private-key and private-model-file mode `0600` and restrict each role directory to its
operator. Do not distribute the complete generated setup directory with all
five keys and all model shares to every server.

TLS requires version 1.3 or later, CA validation, and an exact role certificate
match using the SHA-256 digest of its DER encoding. Server identities are fixed
in this implementation: `prover.test` and `party0.test` through `party3.test`.
Operator certificates must contain those DNS names and the appropriate
server/client usages; changing network addresses does not change certificate
identity. Verifiers act as both TLS servers and clients. The prover requires
client certificates and pins each connection to its claimed verifier role.

Start these commands from the repository root on their respective hosts,
using fresh report directories. Redirect stdout and stderr to retain logs.

Prover host:

```sh
python -m rep4gpu.proofs.private_audit_service prover \
  --setup /srv/siv-private --output-dir /srv/siv-private/report-prover \
  > /srv/siv-private/prover.log 2>&1
```

Party 0 host:

```sh
python -m rep4gpu.proofs.private_audit_service verifier --party 0 \
  --setup /srv/siv-private --output-dir /srv/siv-private/report-party0 \
  > /srv/siv-private/party0.log 2>&1
```

Party 1 host:

```sh
python -m rep4gpu.proofs.private_audit_service verifier --party 1 \
  --setup /srv/siv-private --output-dir /srv/siv-private/report-party1 \
  > /srv/siv-private/party1.log 2>&1
```

Party 2 host:

```sh
python -m rep4gpu.proofs.private_audit_service verifier --party 2 \
  --setup /srv/siv-private --output-dir /srv/siv-private/report-party2 \
  > /srv/siv-private/party2.log 2>&1
```

Party 3 host:

```sh
python -m rep4gpu.proofs.private_audit_service verifier --party 3 \
  --setup /srv/siv-private --output-dir /srv/siv-private/report-party3 \
  > /srv/siv-private/party3.log 2>&1
```

Launch all roles within the configured timeout. These are single-session jobs,
not persistent services. Each exits zero after a matching accepted completion
and writes `result.json`. An exception exits one and writes `failure.json`
when its output directory was created; it does not write an accepted result.
Inspect all role logs when a peer aborts or times out. Socket timeouts cover
connection and communication operations, not a total computation budget;
the local demo adds its own execution deadline. A verifier opens the output
only after MPC acceptance, although a later communication failure can still
prevent final report completion.

## What the committee checks

1. Each verifier authenticates its three model components against the approved
   manifest and constructs shared weight and bias tensors. The four verifiers
   establish their TLS mesh and agree on the model commitment, public input,
   and configuration. Fresh nonce commitments and reveals produce a
   shared run identifier. They derive fresh correlated PRG seeds over the
   confidential mesh, with each excluded-party seed known only to its three
   holders; seeds are bound to this run rather than loaded from a shared file.
2. The prover checks that all four requests match. It executes the model and
   splits each boundary into four random additive ring components. For each
   component it constructs a Merkle tree over independently salted boundary
   encodings, binding model, run, schedule, shape, component, and boundary.
   Verifiers validate and agree on the four roots before selecting samples.
3. A second shared nonce commit/reveal, bound to that commitment, supplies the
   challenge seed. All verifiers derive the same sorted sample without
   replacement. A party can abort, but cannot choose its nonce after learning
   the honest reveals while still satisfying its commitment.
4. The prover delivers the input and output boundaries and both boundaries of
   each selected transition. Party N receives the components whose excluded
   indices are `(N+1)%4`, `(N+2)%4`, and `(N+3)%4`. Each component's salt,
   encoding, position, context, and authentication path must match its accepted
   root. The verifier checks schema and shape before constructing a `Share`.
   This binds delivered MPC operands to the agreed commitment without opening
   a plaintext boundary.
5. `verify_shared_trace` uses `WitnessSession` to check the input against the
   public input and recompute each sampled affine/ReLU transition using private
   Rep4 matrix multiplication and shared bias addition. Residuals require exact
   equality of all encoded 64-bit words, with
   zero tolerance. Sign-bit evaluation and multiplication implement signed
   ReLU without rescaling. Constraint results stay shared until one aggregate
   acceptance bit is opened. Rejection raises an error; acceptance releases
   the final output share for opening. The committee agrees on its report,
   and the prover requires four matching accepted reports before completion.

The salted component roots replace the plaintext trace root on this wire
protocol. They are not the same roots, and there is no proof connecting them
to a separately supplied plaintext root. Prover-local checkpoint replay uses
the existing commitment runtime, but only authenticated component commitments
bind the shares delivered to verifiers. The prover retains all encoded
components and salts for every boundary, so its component storage grows with
the full trace despite checkpointing.

The prover connections use the existing bounded four-byte length-prefixed
JSON framing, with wire `version: 3` for the shared-model path. Messages follow
`hello`, `request`, `commitment`, `challenge`, one
`share_opening` per requested boundary, `result`, and `complete`. A
`share_opening` contains exactly three salted component encodings and their
authentication paths, never a plaintext `LayerOpening`. The verifier mesh
uses `SocketTransport` framing with a 16 MiB frame limit. Protocol failures
close connections; local failure reports contain no intermediate tensors.

Full verification checks every transition of this toy relation. Sampling
checks only selected transitions; authenticated endpoints do not establish
unchecked transitions. The [sampling probability](prover-verifier.md#what-the-verifier-checks)
for a fixed committed trace applies to uniform samples here as well. Neither
sampling nor this implementation constitutes a general proof system for
arbitrary architectures or private inputs.

## Network traffic

The demo counts bytes passed through the TLS application sockets, including
four-byte JSON frame headers, eight-byte verifier-mesh frame headers (including
empty frames), and the mesh's one-byte role identities. The global total sums
outgoing bytes across all five roles, counting each transmission once. Aggregate
sent and received totals must match. These are application traffic measurements,
not packet-capture totals: TLS records and handshakes, TCP/IP overhead, and
certificate distribution are excluded.

`traffic.channels` separates prover/committee traffic from the verifier mesh.
`transition_checks_and_output_sent` is the subset of mesh traffic spent checking
the sampled transitions and input, opening acceptance, and opening the accepted
output. The remaining mesh traffic includes agreement and fresh randomness
setup. Each run's full protocol overhead is included in `traffic.total_sent`.

Model shares are provisioned before the online audit. `model_provisioning`
reports the serialized sizes of the four role bundles and four copies of the
manifest separately. These are file payload sizes, not measured network
transfers. The bundles can be reused for audits of the same approved model;
fresh trace shares and protocol randomness are generated for each run.

See the [shared-weight traffic report](../../experiments/results/proofs/private_rep4_shared_weights_report.md)
for full and sampled toy runs, with links to their summaries and role logs.
