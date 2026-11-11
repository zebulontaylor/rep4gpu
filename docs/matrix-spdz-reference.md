# F127 matrix-MAC reference

`rep4gpu.matrix_spdz_reference` implements exact CPU arithmetic and protocol
checks for square matrices over `q = 2**127 - 1`. It follows Protocols 1–4 and
Appendix D of [Dishonest Majority Multiparty Computation over Matrix Rings](https://eprint.iacr.org/2023/1912).
It is a reference for backend integration, not a completed private preprocessing
implementation or a GPU performance result.

## Sharing and online interface

A party holds a matrix value share `X_i` of shape `(m,m)`, a key share `V_i` of
shape `(m,r)`, and a tag share `G_i` of shape `(m,r)`. Across parties,
`sum(G_i) = sum(X_i) @ sum(V_i)`. These are matrix-vector MACs, not the existing
per-scalar q256 or p31 MAC format.

`MatrixParty` uses the existing ordered byte broadcast with digest echoes.
Live transports require authenticated confidential channels. Unit tests use
`IdealNetwork` with the explicit test-transport flag. Public configuration,
including the prime, matrix width, statistical parameter, check budget and
repetition count, is bound into the wire session.

Local authenticated operations are addition, scalar multiplication, and
**public left multiplication**. A tag cannot be right-multiplied or locally
transposed to obtain the corresponding authenticated matrix.

`prepare_sextuple(candidate, backend)` returns an opaque party/epoch/key-bound
one-use handle after all preprocessing checks pass. `multiply(X,Y,handle)`:

1. Opens and checks `D = X-A` and `E = Y-B`.
2. Forms the authenticated value `F_transpose = E.T @ A.T - R.T`, then opens
   and checks it. Let `F = F_transpose.T = A@E-R`.
3. Computes value shares `C_i + D@B_i + A_i@E + public_share_i(D@E)` and tags
   `G_i(C) + D@G_i(B) + F@V_i + G_i(R) + (D@E)@V_i`.

`open` and `open_many` release values only after fresh post-opening random
linear checks and salted commit/reveal MAC residual checks. The implementation
checks intermediate openings before dependent local products, rather than
leaving those checks pending until output. Any protocol error poisons the
session; retire its keys and correlations. Handles, candidate IDs, authentication
batch IDs, and input masks are single-use.

## Candidate and authentication boundary

`SextupleCandidate` contains local additive shares of `A`, `A.T`, `B`, `C`,
`R`, and `R.T`, plus independent sacrificed `(A_prime,C_prime)` pairs. Every
sacrifice shares the same retained right factor `B`. Raw candidates are never
accepted directly as multiplication handles.

`ChosenColumnVoleBackend.authenticate_columns` receives only its local role's
matrices and local key share. For each ordered peer pair, matrix column `k`,
and key column `h`, it must provide fresh chosen-input VOLE outputs with
`u - w = V_receiver[k,h] * X_sender[:,k]`. Its returned `ColumnVoleBatch`
contains only this role's summed outgoing `u` and incoming `w`. The party
computes each tag locally as `X_i@V_i + outgoing_u - incoming_w`.

Before returning, the backend must finish its lower-layer malicious checks,
bind the ordered peers, MAC epoch, scalar stream, call counter and batch ID,
and ensure the required chosen-input privacy. `BackendEvidence` records these
integration obligations; its flags are not a cryptographic certificate.
Current candidate spVOPE and direct fixed-scalar COPE have unresolved
selective-abort privacy defects. They must remain `privacy_reviewed=False`
and are rejected as non-test providers. No private provider is registered in
this reference: setting a review flag alone cannot enable one. A future adapter
needs an explicitly integrated and reviewed implementation. Later authentication or sacrifice
does not repair biased or known preprocessing masks on an accepted branch.

Authentication adds an independent random matrix mask for each check
repetition. After pairwise outputs and local values are fixed, fresh public
coins select masked linear combinations. The opened combinations and committed
MAC residuals check aggregate consistency, including a corrupt party using
different chosen vectors or key scalars with different peers. Public coins use
salted commit/reveal contributions and domain-separated field sampling.

For triple sacrifice, each independent challenge checks
`D = chi*A-A_prime` and `chi*C-C_prime-D@B = 0`, with authenticated openings.
The implementation then checks the independently authenticated transpose pairs
with fresh masked linear combinations and independent mask pairs. A valid MAC
on a wrong `C` or wrong transpose is insufficient: relation checks must pass
before the handle is issued.

## Statistical configuration

`statistical_security_bits` defaults to 128; `statistical_check_budget` defaults
to `2**32`. The constructor requires enough independent key columns that

```text
q**r >= 2**statistical_security_bits * 2 * statistical_check_budget
```

It uses the same `r` for authentication masks, linear-check repetitions, and
shared-right-factor sacrifice pairs. It reserves each check's cost before its
first frame and preflights compound authentication/preparation/multiplication
costs under a per-party reentrant operation lock. Exhaustion aborts before any
new output frame. F127 uses two columns with the defaults. A clearly
labeled 40-bit statistical configuration can use one column for this budget.

A successfully constructed `MatrixParty` owns its role's entire MAC epoch. A
process-lifetime allocation registry rejects a second runtime using the same
role/session or the same role’s local key share, even if the original object is
collected. Do
not reset counters, reuse keys under a new session, or reconstruct a runtime
after abort. A future private provider must enforce durable allocation across
process restarts; the in-memory registry does not provide that persistence.

`statistical_accounting()` returns a conservative union-bound counter for the
checks in this single owned runtime allocation, not across restarted workers. It excludes commitment, entropy, channel, backend privacy,
and backend computational assumptions. A component offering approximately
122-bit computational security cannot establish a 128-bit overall claim just
because this statistical setting is 128. The reference always reports
`security_target_met=False`.

## Trusted fixtures and validation

`TrustedTestFixture` centrally constructs random candidate shares and input
masks. `TrustedTestColumnVole` centrally simulates fresh pairwise chosen-input
VOLE and sees all supplied local keys and values. Their names and evidence mark
them **test-only**; neither removes a trusted preprocessing dealer. They let a
future role-private provider be tested against the same types and checks.

Run the small CPU suite from the repository root:

```sh
.venv/bin/python -m pytest -q tests/test_matrix_spdz_reference.py
```

The tests cover private inputs, noncommuting matrix products, multiple widths,
40-bit and 128-bit statistical settings, whole-budget/wire binding, and one-use
state. Fault tests include MAC-consistent wrong products and transposes,
pairwise-valid but cross-peer-inconsistent VOLE inputs/keys, forged online
values or tags, and unverified or privacy-unreviewed backend evidence. These
algebra and fault tests do not validate a real VOPE/VOLE implementation.

The [experimental native adapter](matrix-native-ole.md) exercises this pipeline
with actual role-private OLE endpoints in an explicitly test-only mode. It does
not register a qualified production backend or change the privacy gate.
