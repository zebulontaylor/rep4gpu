# Persistent role-private OLE service

`ole_service` keeps one mutually authenticated TLS channel, endpoint entropy
root, and malicious OT extension context for an ordered scalar/vector role pair.
It accepts changing vector lengths and call counts without resetting those
contexts. Every call still runs a complete fresh
[Wolverine chosen-input OLE](DIRECT-OLE.md): new raw scalar coordinates, random
input vector, mask inputs, OT results, extension check, and post-check
extraction. It never reuses a correlation output.

This is an experimental backend component. It records
`security_reviewed=false`; retaining the transport and OT engines does not
change the construction or lower-layer security accounting in
[WOLVERINE.md](WOLVERINE.md). Four-party authentication, sacrifice, epoch
retirement, and input consistency remain the surrounding runtime's work.

## Build and check

From the repository root with the pinned dependencies built:

```sh
python native/matrix_vope/build_ole_service.py
python native/matrix_vope/check_ole_service.py
```

The executable, build manifest, logs, private request/output fixtures, and
public status report live under ignored `work/ole_service/`. The service uses
new files and leaves the one-request baseline executable intact.

```text
ole_service ole_role port self_role peer_role ca cert key peer_name epoch_session_hex [base_fault]
```

`ole_role=1` holds scalars and listens on localhost; `ole_role=2` holds vectors
and connects. The actual MPC roles are distinct integers 0 through 3. The
certificate and DNS SAN contract matches [the TLS endpoint](DIRECT-OLE.md#native-mutual-tls-endpoint).
The caller supplies a unique nonzero session identifier for the epoch,
ordered role pair, and public service configuration. The native TLS subchannel
also binds the protocol version, ordered scalar/vector roles, and prime field.

## Local control protocol

Each role controls only its own process through stdin and stdout. Secret values
remain in the role's canonical request and output files. Stdout contains only
bounded JSON status records with public counters, timing, role, and session
metadata. Stderr contains generic protocol errors. Do not attach stdout to a
coordinator that could issue private requests on another role's behalf.

[`ole_service_io.py`](ole_service_io.py) supplies the typed control helpers:

- `send_frame` writes a request or close frame to local stdin.
- `ReplyReader.read(timeout)` reads a bounded public status line under a total
  deadline and rejects fields outside the status schema.
- `check_ready` and `check_complete` verify the expected role, session,
  request identifier, shape, sequence, and stream position.
- `LocalServiceOutput.read` checks the role-private output against that same
  context and verifies every call advanced the ROT position by 508.

A binary control frame starts with magic `SIVSVC01`, then little-endian uint64
operation and sequence, a 32-byte request identifier, and little-endian uint64
`length`, `calls`, input-path byte length, and output-path byte length. The two
path strings follow. Operation 1 executes a request. Operation 2 closes; its
identifier, dimensions, and paths must all be empty/zero. Paths are bounded to
4096 bytes and cannot contain NUL. Input files use the existing `SIVOLE01`
role-private format.

Sequences start at zero and advance exactly once per completed request. Each
nonzero request identifier is single-use. Before processing any request, both
peers use a role-bound committed exchange to agree on the operation, sequence,
request identifier, dimensions, and starting OT stream position. This exchange
is bound to the persistent connection and its session. The private request's
shape must also match the agreed frame.

Each OLE call gets a domain containing the connection, request sequence,
identifier, and call index. After all calls finish and the private output is
written, a final role-bound acceptance exchange precedes `complete`. The caller
must receive and validate that status before loading output. A four-party
caller sends all active local frames before waiting for completions, then runs
its public phase barrier before releasing the batch.

To close, send the next canonical close frame to both endpoints before waiting
for either `closed` response. The peers commit to the same close position,
retire entropy, close TLS, and terminate. An unexpected EOF, replayed sequence
or identifier, mismatched peer metadata, failed base check, or I/O error
retires the process and exits with failure. A failed service cannot reset,
accept a retry, or restart inside its existing epoch. The surrounding runtime
must terminate its other contexts and retire that epoch too.

## Setup accounting

`ready` reports TLS and local context setup once. Lazy base-OT initialization is
included in the first actual request. Each `complete` reports request time,
application bytes, fresh ROT count, cumulative ROT position, and entropy
substream position. A setup-inclusive total adds the one `ready` cost and all
actual requests. It does not multiply a warm single-call rate by a hypothetical
number of requests. TLS framing and TCP overhead require a separate network
measurement.
