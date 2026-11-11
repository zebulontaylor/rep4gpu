# Success-only private equality with SPAKE2

`spake2_eq.h` and `spake2_eq.cpp` provide an isolated equality check for two
canonical protocol residual byte strings. The production entry point is
`siv::spake2::check_private_equal(role, context, residual, callbacks)`. It returns
without a value after confirmation and the caller's accepted-output barrier.
Every failure invokes the poison callback and throws the same
`EqualityFailure("private equality failed")`.

The construction uses the existing PAKE and its success/abort result; it does
not define a VOPRF or transmit a residual digest. Under SPAKE2's stated group
and hash assumptions, plus the collision/preimage assumptions of the input
mapping, it retains PAKE's online-guess and authentication bounds. The outer
protocol still controls allowed comparisons and retires failed correlations.
This module does not establish security of the surrounding VOLE construction.

## Pinned profile

The core implements [RFC 9382](https://www.rfc-editor.org/rfc/rfc9382.html),
`SPAKE2-P256-SHA256-HKDF-HMAC`. M and N are the RFC's fixed P-256 points;
compressed encodings are used only to load those constants. Wire points and
transcript points use canonical 65-byte SEC1 uncompressed encoding.

The RFC transcript is six fields, each prefixed by its eight-byte little-endian
length: `A`, `B`, `pA`, `pB`, `K`, and `w`. The scalar `w` uses 32-byte
big-endian encoding. `SHA256(TT)` splits into 16-byte `Ke` and `Ka`.
HKDF-SHA256 uses `Ka`, empty salt, and `"ConfirmationKeys" || AAD` to produce
two 16-byte confirmation keys. Each confirmation is a full 32-byte HMAC-SHA256
over `TT`. The production interface exports none of these keys or transcripts.

The required order is `A:pA`, `B:pB`, `A:cA`, `B:cB`. B verifies cA before
sending cB; A verifies cB. Peer points must have prefix `04`, canonical affine
coordinates, and a byte-for-byte serialization round trip. Off-curve points,
infinity, reflected public messages, an unmasked infinity share, and an
infinity shared point all reject. Confirmation comparison uses
`CRYPTO_memcmp`.

## Bind the application inputs

`Context` contains nonempty common `sid`, `configuration`, `identity_a`, and
`identity_b` byte strings. Identities must differ. SID and each identity are at
most 1,024 bytes; configuration is at most 65,536 bytes. The residual is between
one byte and 1 MiB. The caller supplies a canonical encoding, with its field,
shape, basis, stage, and ordering bound in configuration. For example, an F127
extension residual uses the coordinate order defined in
[FP127EXT4.md](FP127EXT4.md).

Define `frame(x)` as eight-byte little-endian length followed by x. The common
context is the concatenation of framed profile domain, SID, configuration,
identity A, and identity B. The profile domain is
`SIV-matrix-FEQ/RFC9382/SPAKE2-P256-SHA256-HKDF-HMAC/v1`.

The wrapper makes an owned copy of the residual before any transport operation.
It computes

```text
input = frame("SIV-matrix-FEQ-password-SHA512-v1")
        || frame(common_context)
        || frame(SHA256(canonical_residual))
w = OS2IP(SHA512(input)) mod P256_order
```

Both roles use this same mapping. Their transcript identities prepend distinct
framed A/B role domains to the common context. AAD is `SHA256(common_context)`.
The residual, its digest, and w are never sent. The scalar reduction performs
512 fixed iterations of a four-word reduction, with one branchless 257-bit
subtraction per input bit. It does not use secret-dependent big-number division.

Production ephemerals come from `BN_priv_rand_range`, rejecting zero; they are
fresh nonzero scalars below the P-256 order. Secret BIGNUMs have
`BN_FLG_CONSTTIME`, and scalar multiplication uses OpenSSL's P-256 operations.
The implementation clears owned residual material, scalar BIGNUMs, shared-point
encodings, transcripts, and derived confirmation secrets after use.

## Transport and failure contract

The caller supplies four callbacks:

- `send(span)` consumes or copies the borrowed message before returning.
- `receive(size)` returns exactly that many bytes or throws. It must support
  interruption when either endpoint aborts.
- `accepted_output_barrier(binding)` authenticates the public 32-byte common
  binding and both endpoints' accepted statuses. It returns only when the
  existing role/session-bound barrier completes. No caller output may be
  released before this callback succeeds.
- `poison()` retires pending correlations and their epoch, and closes or
  interrupts transport so a blocked peer also aborts. Failed material must not
  be retried.

The equality exchange carries 65-byte points and 32-byte confirmations in each
direction, excluding transport framing and the final barrier. A peer that
withholds a message causes an abort, not a partial success.

## Build and test

From the repository root:

```sh
taskset -c 6,7,14,15 .venv/bin/python native/matrix_vope/run_spake2_eq_checks.py
```

The runner builds optimized and UBSan tests plus a production smoke executable,
retaining logs, compiler commands, exit codes, and source hashes under
`work/matrix_vope/spake2_eq/`. It checks all four RFC Appendix B vectors,
including exact points, shared point, transcript, key schedule, and tags.
Additional cases cover binary AAD, malformed points, scalar boundaries,
reflection, replay, wrong order, mismatched inputs/context, owned-input binding,
and the final acceptance/poison callbacks.

Deterministic scalars and secret trace access exist only when compiled with
`SIV_SPAKE2_EQ_TESTING`. They additionally require
`SIV_SPAKE2_EQ_TEST_MODE=1`; the test executable sets it only after an explicit
`--test-mode` argument. The production build has no such interface and refuses
any defined `EMP_TEST_MODE`, including `0` and an empty value, before transport.
