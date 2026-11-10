# Authenticated prime-field online experiment

[`spdz_distributed.py`](../rep4gpu/spdz_distributed.py) implements per-party authenticated matrix and elementwise multiplication over `p = 2**31 - 1`. It accepts local trusted-dealer fixtures or role-private preprocessing bundles. The old small-field MAMA workflow is functional but has a confirmed source-derived mask-privacy attack; it does not meet the malicious-majority privacy target. The replacement large-prime MASCOT bridge is under validation, as described below. Each online protocol instance receives only its local additive value shares, MAC shares, and MAC-key shares. The protocol is designed for correctness and privacy with abort when up to all but one online party are malicious, assuming correct fresh preprocessing and authenticated confidential channels. It is an experimental implementation, not a security audit or an implementation of SPDZ2k.

This differs from [`spdz.py`](../rep4gpu/spdz.py), which retains the original centralized dealer/online simulator. The distributed module adds checked broadcasts, committed MAC residuals, owner-private masked inputs, and single-use correlation identifiers. The native adapters export distributed preprocessing; the online module itself does not prove correlations correct or private. This field experiment does not implement the fixed-point ReLU/GELU/SiLU inference path.

## Shares and multiplication

An authenticated matrix has additive value shares `x_i` and independent MAC shares `gamma[j,i]` satisfying

```text
sum_i gamma[j,i] = alpha[j] * x mod p
alpha[j] = sum_i alpha[j,i] mod p.
```

The trusted dealer defaults to five independent MAC keys. For a nonzero value forgery, guessing all five field authentication equations has term `p**-5`, approximately `2**-155`, under the commitment and channel assumptions. This is not a claim of 155-bit security for the complete implementation: the commitments and transcript digests also impose their computational assumptions. Distributed construction rejects fewer than five MAC keys. This five-key fixture bound is per check; it is not a 128-bit whole-session bound for the entire 64-bit broadcast-counter range.

A matrix triple authenticates `A`, `B`, and `C = A @ B`. Each role opens and checks `D = X-A` and `E = Y-B` together, then computes its share of

```text
Z = C + D @ B + A @ E + D @ E.
```

Only party 0 adds the public `D @ E` value term. Every role adds `alpha[j,i] * (D @ E)` to each local MAC share. The final result is checked when opened. Elementwise multiplication uses a distinct `ElementwiseTriple` with `C = A*B`; the API rejects exchanging the two correlation types.

`split_triple()` consumes the central dealer triple before creating local bundles. Online multiplication burns the local triple before validating its dimensions or attempting an opening. A required 32-byte identifier also records consumption across copied wrappers within that party instance. These checks identify trusted preprocessing objects; they do not prove the correlation correct or make renamed/repackaged old preprocessing fresh.

## Distributed opening and rushing prevention

`open_many()` checks every matrix element under every MAC key before returning an opened value. It performs three checked broadcasts:

1. Broadcast additive value shares and reconstruct the candidate public values.
2. Compute local residuals `sigma[j,i] = gamma[j,i] - alpha[j,i] * opened`. Broadcast a BLAKE2b-256 commitment to the residual bytes with a fresh 32-byte salt.
3. After the commitment broadcast and its consistency check finish, reveal the salt and residuals. Verify the commitments and require the sum of every residual element to be zero.

Every broadcast sends the same framed payload to all peers, followed by an ordered digest-vector echo of the received frames. Honest parties detecting different views abort. This provides consistent broadcast with abort, not reliable delivery or guaranteed output to every honest party. It requires no honest majority.

Frames bind the public session, monotonic broadcast round, sender, value/echo stage, operation label, and payload length. Opening labels additionally bind the field, little-endian uint32 encoding, MAC-key count, ordered matrix shapes, and caller label. Field payloads must have the exact expected length and encode values strictly below `p`; malformed, replayed, cross-session, or inconsistent frames abort.

The salted commitments prevent a rushing party from choosing its MAC residual after learning honest residuals. Because the implementation checks the entire residual vector, it needs no random linear-combination challenge or joint coin toss. An opening uses six transport exchanges: value plus echo, commitment plus echo, and reveal plus echo.

[`spdz_compression.py`](../rep4gpu/spdz_compression.py) supplies an optional compressed path with at least seven keys. The old MAMA generator uses ten coordinates; the replacement large-prime bridge generates seven application coordinates directly. A fresh committed joint coin follows the completed value broadcast. Each key receives an independent coefficient vector, generated with domain-separated ChaCha20 and exact rejection into the field. Parties commit all compressed residuals before revealing any of them. For a fixed nonzero value-error vector and independent uniform coefficients, the stated per-key bound is `2/p - 1/p**2`; ten independent keys give its tenth power, approximately `2**-300`, before an operation-count union bound. The implementation uses computational stream expansion, and the complete bound also depends on commitments, broadcast, key secrecy, and preprocessing. This calculation is not a proof of the entire MPC stack.

A shared coefficient vector would instead introduce one common zero-projection event with probability `1/p`, approximately `2**-31`, which bypasses all keys together. The implementation avoids that construction. The compressed opening performs five checked broadcasts: values, coin commitment, coin reveal, residual commitment, and residual reveal, for ten transport exchanges including echoes.

[`spdz_deferred.py`](../rep4gpu/spdz_deferred.py) queues only internal differences masked by fresh, consumed triples or square masks. It freezes local residual copies and combines their checks after tentative fixed-circuit evaluation. Public `open()` and `open_many()` flush every pending check **before sending any final output value share**, then check that output normally. A failed flush poisons and retires the queue and session. [Deferred-check contracts](deferred-mac-checks.md) explain why this boundary is necessary, the fixed-circuit restrictions, queue cap, and segmented coefficient domains. For final outputs of at most 4096 field values, the deferred backend defaults to the full-vector check under every selected key after the pending flush; larger outputs use compression. The public threshold and selected mode are bound into the operation context. Setting the threshold to zero forces compression. Three internal openings, one batch flush, and a small full-vector checked output use 20 exchanges, compared with 24 for a compressed final output and 40 with immediate compressed checks throughout. The device square identifier is bound into its masked-opening label rather than checked in a separate broadcast.

## Owner-private inputs and square activations

[`spdz_inputs.py`](../rep4gpu/spdz_inputs.py) consumes a fresh authenticated mask `r`. Only the designated input owner receives clear `r`; that owner broadcasts `epsilon = x-r`. All roles consistently check that correction and add its public value and corresponding MAC terms to their authenticated mask shares. Nonowners provide no plaintext or clear mask. The mask factory does not receive the private input.

In the trusted-dealer fixture, the dealer knows `r` and therefore must not observe the online correction: `r + epsilon` reveals the input. Preprocessing must reach each role privately, and the dealer must remain outside the online transcript. The demonstration coordinator also generates the public test fixture for reference checking; this does not make it a private-input deployment architecture.

[`spdz_polynomial.py`](../rep4gpu/spdz_polynomial.py) implements an authenticated elementwise square using valid preprocessed shares of `a` and `a*a`, supplied by a test dealer or a native adapter. It checks one masked opening `d=x-a`, then forms `a*a + 2*d*a + d*d`. Square masks have single-use identifiers and are consumed before the opening. They support a toy two-layer field polynomial; square is a different activation from ReLU, GELU, or SiLU.

## Failure and preprocessing contracts

A frame, authentication, validation, transport, or local-operation failure poisons the party instance and interrupts its transport. A poisoned instance rejects further operations. The controller must retire the entire session's MAC keys and all preprocessing after a failure. Constructing another Python object does not enforce persistent key retirement; a fresh session identifier alone does not make old keys or correlations safe to reuse.

This matters even after a previously successful opening: changing the reconstructed value on a later failed opening can expose MAC-key information through residual differences. No outputs or retries may use that failed key session. After a failure, restart with freshly generated MAC keys and correlations. Across clean restarts, continuing an uncompromised key session also requires durable protocol and preprocessing-consumption state; a new Python object does not provide it.

The trusted-dealer fixture relies on its dealer to generate correct authenticated triples, input masks, square correlations, and independent keys. A dealer that authenticates a wrong correlation can cause a wrong result while all MAC checks pass. The native workflows instead obtain these correlations from four protocol roles. The old MAMA path fails the mask-privacy requirement even when its exported correlations have correct products and MACs. The online checks do not implement sacrifice or prove preprocessing correct; they rely on the selected preprocessing contract and matching shape/identifier metadata.

Run one ordered protocol stream per party. All roles must agree on operations, public shapes, preprocessing identifiers, session, and key count. Production construction requires an authenticated confidential transport; the private insecure-transport override exists only for ideal-network tests.

## Native distributed preprocessing

### Old small-field MAMA path: functional, privacy broken

[`preprocessing_mama.py`](../rep4gpu/preprocessing_mama.py) runs four native MP-SPDZ MAMA roles with malicious SoftSpokenOT, ten prime-field MAC coordinates, and requested security parameter 128. The isolated native build enables `LONG_BIT_MAC_KEY` and applies the [independent-sacrifice patch](native-mama-sacrifice.md). The unpatched direct sacrifice branch's shared challenge is insufficient for the small-field target; this is a patched native protocol, not a claim about an unmodified upstream binary. The adapter requires the checked-in source lock, exact reviewed configuration and patch, and a matching build policy with binary/source/dependency hashes. These checks establish agreement with the supplied build evidence; they do not attest an executable against an adversarial build host.

The [mask-privacy attack](native-mama-sacrifice.md#mask-privacy-limitation) breaks the intended preprocessing privacy under all-but-one corruption. A malicious OT contribution can make acceptance identify a retained mask as zero with probability approximately `1/p`, even though the accepted triple has the correct product and valid MACs. A subsequent native Beaver difference can expose an exported query mask and then the private query. This is a source-derived attack, not an executed adversarial-native test. The independent-sacrifice patch, ten MAC coordinates, and requested `-S 128` do not repair it. Treat this path and its timing reports as functional experiments, not private malicious-majority inference.

Each native role writes its own authenticated values and MAC-key shares. Only that role's online worker imports the corresponding private directory; the coordinator handles public metadata and final outputs. After validating public metadata, the loader atomically creates and fsyncs a consumption marker before parsing secret correlations. It then checks the exact native field/Montgomery encoding and a constant-one calibration. A failure after that claim leaves the bundle retired. Fresh working directories and keys are generated for every run. These filesystem checks provide provenance and accidental-reuse protection, not remote attestation or isolation from a malicious same-user process. Removing the trusted preprocessing dealer means that no coordinator samples or reconstructs the roles' masks or MAC keys. Security still assumes at least one honest native/online role, independent role isolation, correctly provisioned TLS identities and confidential authenticated links, the malicious OT and cryptographic assumptions, correct local computation at honest roles, fresh single-use correlations, and session-wide retirement on abort. Those assumptions alone do not repair the old MAMA mask-privacy attack.

The [native preprocessing validation](../results/native_preprocessing_validation.json) records four successful native exits followed by four successful role-local Python TLS exits. Disposable material was opened to check the matrix and square relations and an owner-private input. That material was consumed for validation and cannot be used for inference. A [separate trained-toy run](../results/spdz_native_trained4.json) consumed a fresh native bundle and checked its CUDA field output. These are functional and provenance checks; they do not constitute an independent proof of the complete OT/preprocessing implementation. Native preprocessing time and online time are reported separately.

The trusted-dealer APIs remain useful test fixtures. Their existence does not mean the native workflow aggregates or regenerates the roles' MAC keys and masks in Python.

### Replacement large-prime MASCOT bridge

[`preprocessing_mascot.py`](../rep4gpu/preprocessing_mascot.py) constructs the application correlations inside native MASCOT over `q = 2**256 - 189`, with amplification `τ = 4` and the standard checked branch. The generated export circuit uses compiler statistical parameter 176 and bounds its integer intermediates to 78 bits. It creates seven application MAC coordinates over `p = 2**31 - 1`, validates the contributed bits over the full internal field, and privately exports canonical role-local values, MAC shares, key shares, and owner-only input masks. Each assigned role samples its own contributions; the coordinator does not generate or reconstruct them.

The [replacement preprocessing design](mascot-preprocessing.md) records the construction and challenge-order requirement. The first q256 build's probe and role-output checks succeeded, but its security approval was withdrawn: its sacrifice challenge was predictable before authentication, allowing a correlated pair of wrong products to pass. Those functional results are superseded, not evidence for secure preprocessing. A corrected build must draw a fresh joint challenge after authentication and pass fresh native, adversarial, role-output, and end-to-end checks. This replacement is not a formal proof of 128-bit security for the complete preprocessing, exporter, online runtime, and deployment.

The bridge is a bounded prototype: its exporter limits the requested share count and matrix accumulation width. Its seven application MAC coordinates are generated directly; they are not the old MAMA adapter's optional seven-of-ten projection described next. Build/source fingerprints and single-use role-bundle imports provide provenance and accidental-reuse checks, not protection against a malicious build host or same-user access to another role's private files.

### Old MAMA seven-key online selection

The old MAMA runner accepts explicit `--online-mac-keys 7`; omitting it keeps ten online keys. Native MAMA generation, its binary/build policy, and all nine independent sacrifice checks remain ten-key operations. Only the Python online representation changes. `load_role(..., online_mac_keys=7)` parses and checks canonical encoding and calibration for all ten native coordinates, then returns owned copies of the fixed coordinates `0..6` for every authenticated value and MAC-key share. It preserves owner-only clear masks and returns no second representation. Local parsing cannot prove authentication equations by itself; those equations rely on native preprocessing and online checks.

The derived online session binds the original native session, native key count, and ordered selected indices. Reports distinguish `native_mac_keys=10`, `online_mac_keys=7`, and the selected indices. The loader atomically consumes the whole native bundle regardless of the selected count. After a failed seven-key session, all ten native coordinates and remaining correlations stay retired: the unused three cannot be recovered through a ten-key reload or a new session identifier. Comparing modes requires separately generated fresh bundles.

For seven independent hidden keys, the combined full-vector/compressed online statistical term is bounded by `T*(2/p - 1/p**2)**7`. Each check consumes monotonic checked-broadcast rounds, whose live 64-bit guard prevents wraparound; one native key session therefore has `T < 2**64`. The resulting term is approximately `2**-146`, strictly below `2**-145`. Resetting a Python party does not create a new allowance under the same keys. [Online MAC-check parameters](online-mac-bounds.md) give the conditional first-forgery argument and distinguish this statistical term from hash, PRG, commitment, OT, and native-preprocessing assumptions. This is only a conditional online authentication calculation, not a 128-bit security claim for the old MAMA workflow. Selecting fewer online MAC coordinates does not repair the native mask-privacy limitation; authentication soundness and preprocessing privacy are distinct requirements.

Value-plus-MAC storage decreases from eleven to eight words per element. The round count and other fixed protocol costs remain unchanged. Seven-key mode must be selected consistently at every role before the online session starts.

## Arithmetic scope and running the examples

The field is not the ring used by Rep4 inference. Centered signed values have magnitude at most `(p-1)/2`. With Q12 encoding, even 64 products of encoded `1.0` reach `64 * 4096**2`, outside that signed range before truncation. Exact field products and toy polynomial tests therefore do not establish numerically equivalent fixed-point language-model inference.

`spdz.field_matmul` keeps a CPU fallback using 16-bit limbs and three exact integer matrix products, with an accumulation-bound check. CUDA dispatches to [`field_kernels.py`](../rep4gpu/field_kernels.py): it centers field elements, extracts four balanced signed-int8 limbs per operand, and evaluates all sixteen limb-pair products. Unlike ring32, high product diagonals remain relevant; reconstruction uses `2**31 = 1 mod p` and Mersenne reduction. Small-row CUDA products use an exact integer decode reduction. Reconstruction stays int64 so arbitrary field residues cannot fuse into saturating int32 MMA accumulators. [Raw field-kernel timings](../artifacts/field_kernel_bench.json) exclude authentication and transport; faster field arithmetic does not remove broadcast, commitments, or preprocessing costs.

From the repository root:

```sh
python -m pytest -q tests/test_spdz_distributed.py tests/test_spdz_inputs.py tests/test_spdz_polynomial.py
python -m examples.spdz_square_toy
python -m scripts.run_spdz_distributed --device cpu --m 8 --k 16 --n 8 \
  --output results/spdz_distributed.json
```

The square example runs four role threads over an ideal in-process transport and checks its bounded integer polynomial output. It is an algebra/protocol demonstration, not a network or LLM benchmark.

The TLS runner creates four isolated workers with distinct mutual-TLS identities. Each temporary role file contains only that role's preprocessing and its own input when it is an input owner. Reports preserve party exit codes and checked outputs, and separate preprocessing, owner-input protocol, and authenticated matrix-product timing. Matrix-product online time includes the checked masked openings and final opening. The demonstration workers share one OS account: process separation and 0600 fixture files do not prevent malicious same-user code from reading other role files or credentials. Deployment requires separate security domains with only each role's own preprocessing and credentials provisioned. Four CUDA workers share the selected local GPU; this is not a four-GPU measurement. The latency ratio compares one authenticated field matrix product with one FP32 matrix product, not complete inference. No complete malicious-majority inference latency target is established.

For the original centralized simulator, use `python -m pytest tests/test_spdz.py` and `python -m benchmarks.spdz_online`. Its logical contribution counts exclude actual broadcast, commitments, transport, and network latency; do not compare them directly with distributed wire counts.

## Relationship to MP-SPDZ

The local MP-SPDZ `Protocols/MAC_Check.hpp` commits to MAC residuals before revealing them. Its normal field check first derives a joint random seed and compresses openings into a random linear combination; its per-element diagnostic branch also uses commit-and-open. The base path retains the commitment requirement and checks each element directly; optional paths use independent per-key compressed checks and deferred masked openings as described above. The CUDA device paths use BLAKE3 commitments rather than the base path's BLAKE2b-256. None claims wire compatibility with MP-SPDZ.

MP-SPDZ's [SPDZ2k shares](https://github.com/data61/MP-SPDZ/blob/master/Protocols/Spdz2kShare.h) use authentication arithmetic in `Z/(2**(k+s))`, including a specialized ring check and random opening masks. At `k=64,s=64`, that requires 128-bit share arithmetic. Truncating it to the existing ring64 GPU kernel would discard part of its authentication contract. This prime-field experiment does not replace that protocol or its distributed preprocessing.
