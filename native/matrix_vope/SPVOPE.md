# Candidate vector-leaf single-point VOPE

Both backends inherit a confirmed scalar-bit privacy leak from bare COPE plus
its polynomial check. These matrix correlations must not supply private MPC
preprocessing. A passing relation check is compatible with the leak. A proven
malicious amplification/backend is required before this construction can be
considered for that use.

`spvope_candidate.hpp` produces a matrix-shaped correlation between two
separate roles. PA/ALICE receives a secret position `alpha`, a secret scalar
`beta`, and `U[a,b]`. PB/BOB receives a secret vector `v[b]` and `W[a,b]`.
The checked relation is `U-W = e_alpha(beta) v^T`. Outputs are released only
after the outer vector check passes.

This is an experimental composition, with `security_reviewed=false` and
`arbitrary_beta_malicious_privacy=false`. It demonstrates a selective-abort
limitation for low-entropy chosen beta. It does not implement the published
malicious VOPE functionality or provide a complete preprocessing protocol.

## Run the component checks

Use the pinned backend setup in [README.md](README.md), then run from the
repository root:

```sh
python native/matrix_vope/build_spvope.py
python native/matrix_vope/check_spvope.py --backend both --field fp127
python native/matrix_vope/check_spvope.py --backend both --field fp61
```

The executable, logs, summaries, and private role fixtures are written under
`work/spvope_candidate`. The `results_fp127.json` and `results_fp61.json`
summaries contain statuses and measurements, not private correlation values.
The role fixtures contain correlation secrets and are only for the offline
observer that checks parity after both processes exit.

Each backend exercises two fresh outer correlations of shape `16×128`, with
four distinct scalar masks in total. The observer checks every matrix entry,
vector length, session count, and canonical field value. Fault cases alter the
vector-tree correction, introduce opposite column errors, alter U, change the
outer check correction, and use inconsistent seeds between the two scalar-key
calls. Both roles must abort and retire their entropy context.

## Construction and typed boundary

1. Run a seeded-vector correlation with fresh scalar `x` held by PA. From
   `M=x v+K`, PA uses `z=-K` and PB uses `y=-M`, giving `z=x v+y`.
2. PA sends `beta-x`. PB expands a fresh GGM root into vector leaves in
   `Fq^b`. Punctured-tree keys use a separate malicious OT context in the
   reversed direction: PB sends and PA receives.
3. PB sends `d=y-(beta-x)v-sum(W)`. PA reconstructs all leaves except alpha,
   then sets `U_alpha=z-d-sum(other leaves)`.
4. After these tree messages and d are fixed, obtain a second seeded-vector
   correlation with fresh scalar `xstar`. Sample fresh outer coefficients r.
   PA sends `r_alpha beta-xstar`.
5. Compare the complete vectors `U^T r-zstar` and
   `W^T r+(r_alpha beta-xstar)v-ystar` using role-bound committed hash equality.
   There is no sum that permits residuals in different columns to cancel.

The output targets the second call's v. Honest roles use the same private right
seed for both calls. `backend=pcg` uses the Le Mans-style short-vector producer;
`backend=direct` uses checked chosen-vector COPE without an LPN expansion.
Both retain fresh scalar keys, base masks, GGM roots, OT indices, and check
coins. Field elements use the canonical codecs from the shared producer.

This implements the algebra of Appendix E.1 in
[the matrix-ring paper](https://eprint.iacr.org/2023/1912). The composition of
independently initialized seeded-vector calls is an additional experimental
choice. The paper's right-vector functionality requires persistent same-vector
binding; sharing a seed in honest application code does not enforce that
condition against a malicious endpoint.

## Selective-abort diagnostic

Let PB use vectors v1 and v2 in its first and second calls. The outer check is
`D^T r=0`, where `D=U-W-e_alpha(beta)v2^T` is fixed before r. A fixed nonzero
D passes with probability at most `1/q`, in addition to errors in the inner
protocols. This explains why an ordinary seed mismatch is usually rejected.

That correctness argument does not establish arbitrary-beta privacy. PB can
guess `beta0=1` and add `beta0(v1-v2)` to d. The residual then becomes
`r_alpha(beta-beta0)(v1-v2)`. The native diagnostic accepts when PA privately
uses beta=1 and rejects when PA privately uses beta=2. PB receives neither beta
value through the diagnostic's command line or protocol input. Thus acceptance
reveals information about a low-entropy chosen beta.

The harness also checks that beta=0 can accept seed equivocation: a zero product
has no right-vector binding requirement. A deterministic small-field algebra
check uses `r_alpha=1` to demonstrate these relations independently of random
challenge sampling.

The matrix application uses fresh random preprocessing scalars, but applying a
negligible-guess argument there still needs a complete composition and abort
analysis. Full VOPE also requires one fixed right vector across all of its
single-point calls. This component provides neither that global binding nor a
whole-execution security claim. The PCG parameters remain unreviewed, and the
pinned OT bootstrap has its own statistical extraction bound; increasing the
field width does not increase that bound.
