# SPAKE2 equality over the native TLS channel

`spake2_eq_tls.h` and `spake2_eq_tls.cpp` connect the
[success-only SPAKE2 check](SPAKE2-EQUALITY.md) to the native `TLSIO` transport.
`TLSAdapter` owns the channel, checks a concrete accepted-output barrier, and
closes the connection immediately after failure. It does not replace the
surrounding VOPE checks or supply cryptographic preprocessing.

## Ownership and public bindings

Construct `siv::spake2::TLSAdapter` with the listener/connect address, port,
`matrix_vope::TLSOptions`, and a required epoch-retirement function. TLSIO
verifies mutual TLS certificates and binds certificate roles, session, and
channel before application traffic. The adapter copies those options; it does
not adopt an unrelated preexisting channel.

`check(context, residual)` uses the smaller authenticated certificate role as
SPAKE2 A and the larger role as B. Context identities must be the exact ASCII
names `party<smaller-role>` and `party<larger-role>`. Context SID must equal the
32-byte TLS session. Configuration is nonempty and at most 65,000 bytes.

The adapter frames the caller's configuration together with its version domain,
TLS session, channel ID, an advancing sequence number, and the ordered role
numbers. The sequence advances before checking an attempt; a failed attempt
poisons the channel. `bind_tls_context` exposes this public context encoding for
independently implemented peers. It carries no residual-dependent data.

A fixed channel can serve successive checks, with fresh SPAKE2 ephemerals and a
new bound sequence each time. Both the environment variable `EMP_TEST_MODE`
and programmatic `emp::is_test_mode()` are rejected before connection or check.

## Concrete accepted-output barrier

After both RFC confirmations, the adapter commits and opens a fixed public
acceptance message. Each endpoint sends a fresh 32-byte salted SHA256 commitment,
receives the peer commitment, then opens its nonce and message. It verifies the
peer's commitment and the entire expected peer message.

The 124-byte message has this canonical layout:

| Offset | Length | Meaning |
| --- | ---: | --- |
| 0 | 8 | ASCII `SPEQACK1` |
| 8 | 8 | Sequence, little-endian |
| 16 | 4 | Sender certificate role, little-endian |
| 20 | 4 | Receiver certificate role, little-endian |
| 24 | 1 | SPAKE2 role: A=0, B=1 |
| 25 | 1 | Accepted status, exactly 1 |
| 26 | 2 | Reserved zeros |
| 28 | 32 | TLS session |
| 60 | 32 | TLS channel ID |
| 92 | 32 | Public SPAKE2 context binding |

The commitment preimage is three eight-byte-length-prefixed fields: ASCII
`SIV-SPAKE2-TLS-accepted-output-v1`, the 124-byte message, and the 32-byte nonce.
All length prefixes are little-endian. Sender/receiver roles are swapped in the
expected peer message; no reflection or alternate acceptance status is accepted.

The binding is the SPAKE2 wrapper's hash of **public context only**. It is
independent of the residual. Never replace it with `SHA256(TT)`: the RFC splits
that secret transcript hash into `Ke` and `Ka`. Neither a transcript hash nor a
residual hash belongs in this barrier.

## Failure and counted transport

Every failed check invokes `poison()`. Poison snapshots counters, marks the
adapter unusable, destroys its owned TLSIO without flushing or waiting for
`close_notify`, and invokes epoch retirement once. Destroying the socket wakes a
peer blocked on input; TLSIO's monotonic deadlines also bound a peer that simply
withholds data. The retirement function must retire local pending material and
must not perform channel I/O. It must complete its local retirement work even
when called during exception handling.

`transport()` borrows the channel as an `emp::IOChannel&` for the surrounding
native protocol. IOChannel already counts bytes and direction changes. A
forwarding counting view can borrow that reference, but neither the view nor
any saved reference remains valid after poison. The owner and transport have
one owner thread. If another protocol stage fails while using the borrowed
transport, call the adapter's `poison()` and stop using that reference.

`statistics()` remains available after failure and includes completed checks,
poison state, IOChannel counters, flushes, and TLS writes. On failed reads the
inherited receive counter includes requested bytes; it is not proof that all
those bytes arrived. One successful check on a fresh channel sends and receives
285 application bytes per endpoint: 97 SPAKE2 bytes plus 188 barrier bytes.
TLS handshakes, records, and TCP framing are separate.

## Reproduce the two-process tests

From the repository root:

```sh
taskset -c 6,7,14,15 .venv/bin/python native/matrix_vope/run_spake2_eq_tls_checks.py
```

The runner checks the pinned EMP sources, creates an ephemeral P-256 CA and two
distinct role certificates, and runs actual two-process mutual TLS cases. It
retains source/dependency hashes, build logs, status-only role reports, counters,
and exit codes under `work/matrix_vope/spake2_eq_tls/`. Temporary private keys
are removed. Test elapsed times include connection setup and are not protocol
performance estimates.

The cases cover equal inputs, two checks on one channel, unequal inputs,
correct and incorrect known-delta candidates, changed session/configuration/
identities, withheld confirmation, and independently constructed malicious
barrier bindings, roles, and rejected statuses. A failed comparison cannot reuse
the channel or borrow its transport, cannot add traffic, and retires once.
Programmatic EMP test mode rejects before transport.

The candidate tests demonstrate the intended online-guess boundary: a selected
input that actually equals the peer input succeeds, while an incorrect selected
input fails. They do not establish a new privacy proof or an end-to-end VOPE
security claim.
