# Short-vector programmable VOLE experiment

The Le Mans COPE-backed producer described here has a confirmed malicious-privacy failure and
must not supply private preprocessing. A malicious programmer can perturb one
bit's correction and learn a scalar-key bit from acceptance versus abort, even
when accepted outputs satisfy the exact product relation. The component and
arithmetic checks below do not establish privacy. Outputs retain
`security_reviewed=false` and `malicious_privacy=false`.

The separate [Wolverine base-sVOLE candidate](WOLVERINE.md) implements the
published extension-field check and fresh key extraction. Its paired mode
generates two scalar correlations against one vector in a single call.
The [direct chosen-input OLE adapter](DIRECT-OLE.md) provides a separate
working scalar-times-vector path and a role-private matrix backend boundary.

This native two-party producer implements the Le Mans programmable VOLE layer
with chosen single-point VOLE and a public linear map. It produces a vector
`v = H e` from the programmer's private regular-noise seed, and two private
shares satisfying `M = Delta*v + K`. The programmer receives `(v,M)`; the scalar
holder receives `(Delta,K)`. Reusing the noise seed across independent scalar
keys reproduces `v` without sending it to the counterpart.

The default field is fp61, with prime `2^61-1`. An explicit `--field fp127`
runner option selects the independently tested field in `fp127.h`, with prime
`2^127-1`. Both use the checked base VOLE, GGM, and malicious chosen OT
components in `work/vole-labs`. It generates only the requested short vector;
it does not invoke that backend's 16K-output C-VOLE or its commitment matrices.

## Build and run

Run these commands from the repository root. The dependency checkout is pinned
by `source-lock.json`; the build helper checks all three commits, tracked-file
changes, and hashes of the protocol headers it uses. Building requires an x86-64
C++20 compiler with AES/RDSEED support, CMake, and OpenSSL development libraries.

```sh
git clone --no-checkout https://github.com/vole-labs/vole.git work/vole-labs
git -C work/vole-labs checkout --detach e266595c1811692a6b2df87a4b4d13d3a86ef80f
git -C work/vole-labs submodule update --init --recursive
cmake -S work/vole-labs -B work/vole-labs/build -DCMAKE_BUILD_TYPE=Release
cmake --build work/vole-labs/build --target emp-ot -j2
python native/matrix_vope/build.py
python native/matrix_vope/run_checks.py --sessions 10 --mode persistent
python native/matrix_vope/run_checks.py --honest-only --sessions 10 --mode fresh
python native/matrix_vope/run_checks.py --sessions 2 --field fp127
```

If the pinned dependency is already built, run only the Python commands.
The executable and compiler log are written to `work/matrix_vope`.

The runner starts separate scalar-holder and programmer processes, each with
its own private randomness. After both exit, it reads their saved role outputs
and verifies the correlations. This observer check does not add a debug opening
to the protocol. Role outputs contain test correlation secrets; they are test
fixtures, not consumable preprocessing files.

`work/matrix_vope/artifacts/summary_persistent_fp61.json` contains return codes,
timings, communication counts, and verification results. The honest case uses
the requested number of fresh scalar keys with one private programmer seed. Fault cases alter a GGM correction, a programmer MAC,
or a check correction; both roles must return failure. Each case retains its
role logs. The runner also reflects a commitment and its opening against the
honest endpoint for both coin and equality payloads; the honest endpoint must
reject the reflected sender role and retire its entropy context. `work/matrix_vope/build.log` retains compiler
output.

The native executable accepts:

```text
seeded_vole <1|2> <port> <output.json> <sessions> [cheat] [b] [t] [log_block] [persistent|fresh] [fp61|fp127]
```

Role 1 owns `Delta`; role 2 owns the noise seed. Supported fault names are
`none`, `tree` on role 1, and `mac` or `correction` on role 2. The defaults are
`b=128`, `t=8`, `log_block=6`, and three independent single-point checks.
The default OT mode is `persistent`, and the default field is `fp61`. The role
JSON includes the field name, prime bit length, modulus, and base-check degree.
All integer values use full decimal JSON numbers; the verifier reads them with
Python integers, including values above 64 bits.

FP127 messages use canonical 16-byte little-endian residues. The wide COPE path
validates its bulk correction arrays before arithmetic instead of relying on
the backend's untyped bulk IO. The same bit-decomposition relation and
polynomial check are used. Wire tests accept `p-1` and reject both `p` and
`2^127`. The separate `run_fp127_checks.py` checks the field arithmetic against
GMP, its strict codec, and template compatibility with the backend.

## Protocol layers

1. Generate fresh base VOLE correlations under the scalar holder's new key,
   including independent masks for every check. The backend executes its COPE
   consistency check.
2. Derive one hidden position and nonzero value per regular-noise block from
   the programmer's seed. Derandomize one base correlation to each chosen value.
3. Transfer punctured GGM keys using malicious chosen OT. Reconstruct the
   missing MAC using the corrected base correlation.
4. After all block corrections are fixed, execute three independent checks.
   Each check uses a joint committed coin to sample independent field
   coefficients for every point and leaf, and an independent base VOLE mask
   per point. Compare the entire vector of folded residuals using committed
   hash equality. No residuals are added together or opened.
5. Apply the fixed public dense `H` to the stacked values and MAC/key shares.

Every commitment binds its domain and immutable sender role, and verifies the
opposite role. A fresh role-bound connection coin supplies the per-connection
domain; each protocol stage also binds its session index and purpose. This
prevents a peer from reflecting an honest commitment and opening to force the
joint coin to zero. The public configuration is bound before each call. GGM
roots, base correlations, OT randomness, and check masks are fresh even when the
private noise seed is reused. This interface deliberately returns the ordinary
`M=Delta*v+K` convention. For matrix-ring `SeededRandomVOLE`, give the scalar
holder `x=Delta,z=-K`, and give the programmer `v,y=-M`; then `z=x*v+y`.

## Persistent OT and fresh correlations

Persistent mode retains two malicious SoftSpoken extension engines per peer,
one in each direction. Its first call includes their CSW bootstrap. Each new
scalar key consumes 61 or 127 fresh chosen random OTs to seed a new COPE instance;
the COPE consistency check runs on that instance's new correlations. Every
punctured-tree transfer also consumes fresh OT indices. Scalar keys, tree roots,
check masks, and correlations are never reused. Fresh mode creates new engines
and uses the backend's original CSW-based COPE initialization on every fp61
call. FP127 fresh mode creates independent malicious extension engines and uses
the same canonical wide COPE wrapper as its persistent mode.
Discard a peer context after any abort.

COPE retains its AES PRG storage across calls. The initial schedules use an
explicit placeholder key and never generate output; fresh ROT outputs replace
every key before each COPE call. This avoids drawing entropy for PRG objects
whose initial keys would immediately be overwritten.

The producer draws private coins, commitment nonces, GGM roots, and base values
from one entropy root per endpoint. Each purpose-separated substream consumes a
fresh root block and a strictly advancing index. The initial root seeding is
charged to context setup. The executable rejects EMP deterministic test mode
before constructing an entropy or OT context; the API constructors enforce the
same condition. Both the environment toggle and programmatic EMP flag are
checked. An exception retires that entropy source; discard the
peer OT context as well. The persistent EMP engines retain their own existing
randomness sources.

The role JSON records per-session wall time, process CPU time, sent bytes,
advancing entropy-substream counts, and transport direction runs. Phase timers separate configuration, base VOLE, tree OT, GGM/corrections,
checks, public map generation, and public map multiplication. Base-VOLE
subtimers separate ROT, PRG storage/reseed, COPE extension, and its check, with
both wall and process CPU time. The public map is
cached in both modes; its first generation is charged to the first call.

Context setup measures the connection-domain exchange and OT-engine
construction before the calls. Its communication counts show whether bootstrap
traffic occurred there. Include context time plus all calls in the amortized
total, and report the first call separately. For a bounded CPU diagnostic, the
runner accepts `--role-cpus 6,14 7,15` to pin the scalar holder and programmer
to separate CPU sets. Direction runs are a transport metric, not the
cryptographic round count. Role output files also contain correlation secrets;
share the summary and logs, not these fixtures.

## Security scope and next integration

Every output explicitly sets `security_reviewed=false`. The short-vector
parameters are for functional experiments. They have no assigned LPN security
level. The base consistency check is a polynomial check of degree
`d=weight*(1+check_repetitions)` over the selected field. Its component error
bound under uniform challenge sampling is `d/q`; repeating the later
single-point checks does not amplify this separate base check. For FP127 at
the default parameters, this is `32/(2^127-1)` per base check, before accounting
for other checks, pseudorandomness assumptions, and the full execution. This experiment therefore makes no 128-bit security
claim. Hash equality is binding under SHA-256 assumptions; a hash of a 61-bit
residual is not a 128-bit hiding commitment. Hashing the full residual vector avoids cancellation
and can increase entropy, but does not establish 128-bit conditional hiding:
an adversarial transcript may still depend on a single scalar key in the
selected field.
The equality and leakage argument needs review before a stronger claim. The
pinned CSW bootstrap documents a separate statistical extraction parameter
(sigma=40 in the reviewed setting); FP127 does not turn that into a 128-bit
statistical guarantee.

Parameter review must cover both the regular-noise code and all base/check
error bounds, including the recent regular syndrome decoding attacks. The
small dense map is a reference implementation; a faster structured map requires
its own supported parameter selection and must preserve the seed contract.

The [vector-leaf candidate](SPVOPE.md) uses two fresh scalar-key calls against
one honest right-vector seed and produces matrix-shaped correlations. Its
selective-abort diagnostic prevents treating it as a malicious primitive for
arbitrary chosen beta.
Matrix authentication, sacrifice, transpose checks,
and the fused matrix accumulation remain separate layers in the
[matrix-ring protocol](https://eprint.iacr.org/2023/1912).

The implementation follows Le Mans, Figures 13–15 in its full version
[ePrint 2021/1579](https://eprint.iacr.org/2021/1579), also reproduced as
Figures 4.13–4.15 in
[the author's thesis](https://cs.au.dk/fileadmin/ingen_mappe_valgt/phd_thesis/Sai_Rahul_Rachur.pdf).
The backend is [vole-labs/vole](https://github.com/vole-labs/vole), commit
`e266595c1811692a6b2df87a4b4d13d3a86ef80f`.
