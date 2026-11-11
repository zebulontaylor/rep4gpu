# Native SiLU proofs

This helper proves the Q24 rational SiLU relation in
[`silu_rational.py`](../../rep4gpu/silu_rational.py). It uses native Pedersen
commitments to `X + 16*2^24` and `Y + 2^24`, and Bulletproofs R1CS constraints
over the Ristretto scalar field. Inputs cover inclusive `[-16,16]`; the final
division accepts either nearest neighbor at an exact tie. Intermediate power
and polynomial reductions round to nearest with ties upward.

The proof establishes a relation between committed plaintext values. The
calling protocol must bind these commitments to its existing shared inputs
and outputs before accepting them. This executable does not implement that MPC
binding or receive verifier data shares.

Bulletproofs R1CS is an experimental upstream API enabled by `yoloproofs`.
The library and this integration have not been independently audited for this
application. Its discrete-logarithm assumptions provide the conventional
Ristretto255 security level; this is not a post-quantum backend.

## Build and run

From this directory, with the pinned Rust toolchain available:

```sh
cargo build --release --locked
cargo test --release --locked
./target/release/siv-silu-proof metadata
```

`metadata` prints the exact compressed generators, scalar modulus, circuit
version, and multiplier count. `Cargo.lock` pins dependencies. The vendored
Bulletproofs 5.0.0 source has the compatibility change described in
[`vendor/README.md`](vendor/README.md).

Create an input JSON file with encoded signed 64-bit words:

```json
{"x":[-16777216,0,16777216]}
```

An optional `y` array supplies output claims; otherwise the helper computes
nearest-even output claims. To prove and verify:

```sh
./target/release/siv-silu-proof prove input.json artifacts/example 128
./target/release/siv-silu-proof verify artifacts/example/public.json
```

The output directory must not already contain `private-blindings.json`.
`public.json` contains compressed commitments and independent chunk proofs,
encoded as hexadecimal strings. `private-blindings.json` contains `output_y`
and the canonical little-endian scalar blindings. It is created with mode
`0600` on Unix and belongs only in the plaintext prover process. It is not a
public proof artifact. The prove command reports raw commitment and proof byte
counts, excluding JSON encoding and transport framing.

The circuit uses 524 multiplication constraints per coordinate. Generator
capacity comes from the actual constraint-system metrics. Chunking limits
prover memory; the final chunk can contain fewer coordinates. Independent chunks
use four worker threads by default; set `SIV_PROOF_JOBS` to a positive integer
to change that limit. Generator vectors are shared across chunks with matching
capacity. The transcript
binds the circuit version, total count, chunk size, offset, and actual chunk
count. Verification rejects missing, duplicated, empty, or reordered chunks.

## Project public commitments

After fixing the commitments, a calling protocol can derive 128 Bernoulli
projection commitments:

```sh
./target/release/siv-silu-proof project artifacts/example/public.json weights.bin projected.json
```

`weights.bin` contains exactly `ceil(128*N/8)` bytes for `N` coordinates. Bits
are contiguous in row-major order, least significant bit first within each
byte; rows have no padding. A set bit selects that coordinate. The output
contains 128 `commitments_x` and 128 `commitments_y` points, each the group sum
of selected original commitments. This is public local computation; the
calling protocol determines and agrees on the challenge seed and bit stream.
