# Checked random base sVOLE

`wolverine_base.hpp` implements the base-sVOLE construction in Figures 2.3 and
2.4 of [Chenkai Weng's thesis](https://www.mccormick.northwestern.edu/computer-science/documents/nu-cs-2024-11-chenkai-weng.pdf).
It generates a fresh scalar key at ALICE, a random vector at BOB, and respective
shares `K` and `M` satisfying `M[i] = K[i] + gamma * u[i]`. An optional final
correction programs BOB's vector from a private seed.

This is a standalone experimental transcription. Its output records
`security_reviewed=false` and `full_preprocessing=false`. It does not supply
multiparty authentication, matrix sacrifice, or a full neural inference
preprocessing provider. The older checked-COPE and seeded-VOLE experiments have
a confirmed scalar privacy failure; their separate prime-field check is not
used as the security upgrade here.

## Protocol and field contract

The prime field is `p = 2^127 - 1`. Raw correlations use `F_(p^4)` with basis
`[1, i, J, iJ]`, where `i^2 = -1` and `J^2 = 2 + i`. The header
[`fp127ext4.h`](fp127ext4.h) defines this basis and its strict 64-byte encoding.

1. ALICE samples four fresh prime-field coordinates of a raw extension-field
   key. BOB samples the random vector `u` and four independent mask inputs
   `a[0..3]` in the prime field.
2. Four raw COPE coordinate batches consume 127 fresh chosen ROT results each.
   All four batches receive the same vector and the same four mask inputs.
   Figure 2.3 uses prime-field PRF pads and corrections; the gadget sum
   assembles the extension-field result. There are no separate prime-field
   consistency checks or intermediate compressed outputs.
3. After all raw corrections are fixed, ALICE samples full extension-field
   challenges `chi[i]`. BOB sends
   `X = sum chi[i]*u[i] + sum a[h]*basis[h]` and
   `Z = sum chi[i]*W[i] + sum C[h]*basis[h]`.
   ALICE checks `Z = Y + raw_delta*X`, where
   `Y = sum chi[i]*V[i] + sum B[h]*basis[h]`.
4. After this one check succeeds, ALICE samples the fresh compression
   coefficients and sends them to BOB. Both sides compress the raw coordinate
   correlations, then discard the raw key batch. The OT engines persist;
   every later batch consumes fresh ROT indices and a fresh raw key.
5. In a programmed mode, BOB sends one correction vector `d = v - u`.
   ALICE changes each output coordinate's key share to `K - gamma*d`.
   BOB keeps its MAC share. This binds the coordinates within that call to
   one vector. It does not bind a malicious programmer to the same vector
   across separate calls.

| Mode | Output field | Extraction parameter | Compression |
| --- | --- | --- | --- |
| `random`, `programmed` | `F_p` | `rho=128`, `ell=4` | Four fresh `F_p` coefficients |
| `paired-random`, `paired-programmed` | `F_(p^2)` | `rho=120`, `ell=2` | Two fresh `F_(p^2)` coefficients |

Paired mode views the raw field as two blocks over `F_(p^2)`: `D0 + D1*J`.
It computes `Gamma = alpha0*D0 + alpha1*D1`, and exposes its real and imaginary
coordinates as two scalar keys against the same vector. It uses the same 508
fresh ROT choices and raw check as single mode.

Theorem 1 gives the construction's bound
`(r*log2(p))^2 / p^r + 2^-rho` in the stated COPEe hybrid. For single mode,
that is about `2^-113`, before lower-layer errors. Paired mode uses
`254^2 / p^2 + 2^-120`. These are separate from the computational security of
AES, SHA-256, and malicious OT. The pinned CSW base-OT implementation has a
40-bit statistical parameter; `rho=128` does not make the whole implementation
128-bit statistically secure. A security review must account for this OT
implementation, its composition with the prime-field PRF sampler, the full
protocol and all repeated calls before making a deployment claim.

## Build and check

From the repository root, with the pinned dependencies already built as
described in [README.md](README.md):

```sh
python native/matrix_vope/build_wolverine.py
python native/matrix_vope/check_wolverine.py
```

The build verifies dependency commits and selected source hashes from
`source-lock.json`. The executable, build log, source-hash manifest, private
fixtures, and public check summary live under ignored `work/wolverine_base/`.
The check summary contains statuses, counts, and timings, never keys or shares.
The role fixtures contain private values for offline parity checks and must not
be published as preprocessing data.

The checks cover random and programmed outputs, both paired coordinates,
multiple fresh sessions, exact vector lengths, full-width field values,
configuration mismatch, inconsistent coordinate inputs, corrupted extension
responses, raw high-bit correction faults, role-reflection rejection, entropy
retirement, and rejection of `EMP_TEST_MODE=1` before networking or entropy
construction. A raw-bit-zero fault can pass the intermediate check; this is the
selective-failure behavior that Figure 2.4's later fresh compression addresses.
That case verifies the accepted compressed correlation, not a privacy proof.

A manual two-role run uses matching public arguments:

```sh
work/wolverine_base/wolverine_base 1 19431 work/wolverine_base/role1.json 3 128 paired-programmed
work/wolverine_base/wolverine_base 2 19431 work/wolverine_base/role2.json 3 128 paired-programmed
```

Run those commands in separate terminals. Output includes setup and per-call
network counters. The first call includes lazy OT setup; later calls consume
fresh OT results from the persistent engines. Shared-host functional timings
are not an inference speed measurement.
