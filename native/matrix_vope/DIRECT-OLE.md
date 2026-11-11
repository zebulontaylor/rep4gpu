# Chosen-input OLE over the checked base

`direct_ole.hpp` turns a fresh [Wolverine base-sVOLE](WOLVERINE.md) into a
chosen-vector, chosen-scalar product. ALICE holds scalar `a`, BOB holds vector
`v`, and their output vectors satisfy `L + R = a*v`. Each native process reads
only its own private request. There is no dealer and no outer GGM or hash
comparison of private field values.

The default driver uses unauthenticated localhost `NetIO`. A separate TLS
build uses mutual certificate authentication and binds the actual party roles
and a 32-byte session identifier before running OLE. Both builds record
`security_reviewed=false` and `full_preprocessing=false`; the TLS build records
`transport_authenticated=true`. Review of the full lower-layer composition is
required before deployment. Matrix MAC
checks, sacrifice, and cross-call input consistency belong to the surrounding
protocol; arbitrary private input changes are not faults that OLE can detect.

## Adapter equations

Each call first generates fresh random `Gamma,u,K,M` with
`M = Gamma*u + K`, using the complete `r=1,rho=128` base construction.

1. BOB sends the canonical correction `d = v - u`.
2. ALICE receives the complete `d` before sending `c = a - Gamma`.
3. ALICE returns `L = -(K - Gamma*d)`. BOB returns `R = M + c*v`.

Thus `L + R = a*v`. Fresh raw keys, random inputs, OT results, checks, and
post-check extraction run on every call. Reusing a chosen scalar or vector
still consumes fresh base material. The base construction's single-coordinate
bound is about `2^-113`, before the separate OT, PRG, and repeated-call terms
listed in [WOLVERINE.md](WOLVERINE.md). The extraction parameter `rho=128` is
not a whole-implementation statistical security claim.

For a matrix cross-product, BOB holds `A[Q,K]` and ALICE holds `B[K,N]`.
For each `(k,j)`, run OLE with vector `A[:,k]` and scalar `B[k,j]`, then sum the
local output vectors into column `j`. This uses `K*N` calls of length `Q` and
returns two additive shares of `A@B`.

## Build and check

From the repository root, after building the pinned dependencies described in
[README.md](README.md):

```sh
python native/matrix_vope/build_direct_ole.py
python native/matrix_vope/check_direct_ole.py --throughput
```

The check runs two actual native processes per case. It verifies field parity,
the matrix accumulation, zero/one/minus-one inputs, fresh output shares, the
chosen-column provider sign convention, malformed private input rejection,
public shape mismatch, lower-layer check faults, and rejection of EMP test
mode. `--throughput` adds ten fresh calls each at lengths 128, 1024, and 8192.
The first call includes lazy OT initialization; setup-inclusive counters add
that call and context setup to all later calls. Shared-host timings are not a
plaintext comparison or an inference speed claim.

The executable, private requests and outputs, logs, build manifest, and public
summary live under ignored `work/direct_ole/`. Only the public summary contains
publishable statuses, dimensions, operation counts, byte counts, and timings.

## Role-private provider boundary

Run matching role-local requests in separate processes:

```sh
work/direct_ole/direct_ole 1 19432 work/direct_ole/role1.bin work/direct_ole/role1.json
work/direct_ole/direct_ole 2 19432 work/direct_ole/role2.bin work/direct_ole/role2.json
```

[`direct_ole_io.py`](direct_ole_io.py) defines immutable `ScalarOleRequest`,
`VectorOleRequest`, and `LocalOleBatch` types. A request has magic `SIVOLE01`,
then little-endian uint64 `length` and `calls`. ALICE's payload contains one
scalar per call. BOB's payload contains `calls*length` values in call order.
Each field value is a canonical 16-byte little-endian residue. Trailing data,
noncanonical residues, unsupported dimensions, and mismatched public request
shapes are rejected.

Each private output contains its own `sessions[call].share[length]`; it contains
neither the other party's inputs nor the raw base key. `LocalOleBatch.read`
checks exact dimensions, order, fresh-OT counts, canonical residues, and the
explicit experimental metadata before returning local shares.

The matrix reference backend uses `outgoing_u - incoming_w = delta*x`.
Map BOB's vector-side output to `outgoing_u = R` and ALICE's scalar-side output
to `incoming_w = -L`. The type's `outgoing_u()` and `incoming_w()` methods
apply those role checks and signs. Concatenating data columns and fresh check
masks into one vector per key is a caller operation; its exact layout must be
bound to the enclosing authentication batch and epoch. This executable does
not implement the four-party backend scheduler or authenticate those epochs.

## Native mutual TLS endpoint

```sh
python native/matrix_vope/build_direct_ole_tls.py
python native/matrix_vope/check_direct_ole_tls.py
```

The TLS executable is `work/direct_ole_tls/direct_ole_tls`. Its argument order
is:

```text
direct_ole_tls ole_role port request output self_role peer_role ca cert key peer_name session_hex [base_fault]
```

`ole_role=1` is the scalar holder and listens on localhost; `ole_role=2` is the
vector holder and connects. `self_role` and `peer_role` identify actual MPC
parties 0 through 3 and must differ. Certificates must have the matching
`party0` through `party3` DNS subject alternative names. `peer_name` names the
expected peer. `session_hex` contains a nonzero 32-byte public identifier encoded as 64 hex
digits. A four-party caller derives it from the epoch, operation purpose,
batch, ordered vector/scalar role pair, and public layout; each endpoint must
receive the same value.

The native endpoint uses [`TLSIO`](tls_io.h). It never falls back to `NetIO`.
A successful private output records the actual roles and session identifier.
`LocalOleBatch.read(..., authenticated=True, self_role=..., peer_role=...,
session=...)` requires all three values to match and returns an authenticated
batch. Its default unauthenticated mode rejects TLS output. This prevents a
caller from accepting a stale endpoint result or using the plaintext driver
through the TLS contract.

The TLS smoke checks execute OLE, lower-layer fault rejection, mismatched
sessions, and incorrect peer names. Transport tests in the TLS module cover
its independent channel contract. Native byte counters count application
payloads; TLS framing and TCP overhead require a separate network measurement.
