# Bulletproofs source

`bulletproofs/` contains the library sources, package manifest, README, and MIT
license from the crates.io `bulletproofs` 5.0.0 package. The parent manifest
patches that exact version to this directory.

The only library changes are three scalar parsing expressions in
`src/r1cs/proof.rs`: each `Scalar::from_canonical_bytes(...)` result is converted
from `CtOption<Scalar>` to `Option<Scalar>` before calling `ok_or`. This restores
compilation of the experimental R1CS feature against curve25519-dalek 4.1.3
while retaining canonical scalar validation and its existing error behavior.

Upstream: <https://github.com/dalek-cryptography/bulletproofs>.
