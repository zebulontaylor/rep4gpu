# Large-prime MASCOT preprocessing for the p31 runtime

[`preprocessing_mascot.py`](../rep4gpu/preprocessing_mascot.py) generates the p31 runtime's authenticated matrices, square masks, input masks, and MAC-key shares inside a separate malicious-secure MPC computation. Four native roles compute over `q = 2**256 - 189`; each online role receives only its own outputs modulo `p = 2**31 - 1`. No coordinator generates or reconstructs the roles' randomness or keys. The previously tested q256 build has a predictable-sacrifice-challenge flaw. Its conditional approval is withdrawn. The current rebuilt experiment adds a fresh post-authentication challenge and has separate validation evidence below. It remains a conditional experiment, without a formal security audit or an end-to-end preprocessing-inclusive performance claim.

This construction replaces the small-field MAMA experiment. That experiment has a [known-mask selective-failure flaw](native-mama-sacrifice.md#mask-privacy-limitation): its checks can accept a correct triple whose mask is known to the coalition with probability about `1/p`. Correct matrix products and valid MACs do not repair that privacy failure. The bridge instead uses standard MASCOT's two-A/same-B sacrifice and amplification four over the large field.

## Sacrifice challenge order

The pre-repair native generator reused its public amplification PRNG to derive the later sacrifice challenge `t`. That stream was known before authenticating the candidate `C` values. A corrupt party could therefore change its own contributions by `C_0 += delta` and `C_1 += t*delta` before authentication. Both changed values receive valid MACs, while the sacrifice error `t*delta-t*delta` is identically zero. Neither a larger field nor amplification four fixes that ordering error.

The [fresh-challenge repair](../native/patches/mascot-fresh-sacrifice.patch) creates a fresh jointly committed seed **after candidate authentication finishes**, then samples sacrifice challenges from that fresh stream. The authenticated values are fixed before an honest seed contribution is revealed. The standard MASCOT correctness argument applies to this ordering; the parameter discussion below assumes it. The rebuilt policy pins this ordering change. A native old-accept/new-reject execution of the correlated-C attack has not been performed; the attack and repair have source and algebraic regression evidence. The previous [native validation evidence](../results/mascot_preprocessing_validation_superseded.json) remains preserved as functional evidence and is explicitly superseded.

## Construction and private outputs

Each native worker samples its own uniform p31 contributions with `secrets.randbelow(p)`. It supplies each contribution as 31 authenticated secret bits. Before using any range assumption, the circuit opens every `b*(1-b)`, completes a native MAC check, and rejects any nonzero result. The clear validation checks all four 64-bit limbs of every q-field result and reduces violations across every vector lane before the scalar abort instruction. It does not sum raw field errors, which could cancel.

After validation, the circuit canonicalizes a bit vector as

```text
r = sum_i 2**i * b_i - p * product_i b_i.
```

The all-ones encoding becomes zero. Honest workers reject that encoding when sampling, so their contributions remain exactly uniform modulo p; corrupt workers may choose zero or any other contribution. Summing the four contributions modulo p makes each global random value uniform and unknown to a coalition missing an honest contributor. The same construction creates seven independent application MAC keys. Each role receives only its own key-share contributions.

The circuit computes `C=A@B`, square correlations, and application MAC products modulo p using authenticated q-field arithmetic. For each exported value or MAC coordinate, roles 0 through 2 supply independent canonical masks `r_0,r_1,r_2`. Those roles receive their respective masks, while role 3 receives

```text
residual = (value + 3*p - r_0 - r_1 - r_2) mod p.
```

For a MAC coordinate, `value` is the secret product `alpha*x`. Reduction occurs **before** `reveal_to(3)`. Revealing the unreduced integer and reducing it at the recipient would expose carries and does not implement this construction. A designated input owner additionally receives its clear input-mask value through a private native output. Other roles do not receive that clear mask.

A checkpoint authenticates all internal computation before private outputs begin, and another checkpoint completes the output checks. A bundle becomes ready only when all four native processes exit successfully. Each online worker parses only its own output, validates the canonical field format and constant-one calibration, and durably consumes its entire bundle before parsing secret correlations. Failed or interrupted sessions retire their keys and material.

## Parameters and statistical terms

| Parameter | Setting |
| --- | --- |
| Native protocol | Standard MASCOT with malicious SoftSpokenOT |
| Internal prime | `q = 2**256 - 189` |
| Native MAC coordinates | One over q |
| Amplification | Four raw triples, two amplified A/C coordinates sharing B |
| Native runtime security option | Explicit `-S 128` |
| Compiler arithmetic masking | `program.security = 176` |
| Compiler integer bit length | 78, with explicit per-operation bounds |
| Application field and MAC coordinates | p31, seven independent keys |
| Exported scalar limit | 16,384, including calibration |
| Matrix inner-dimension limit | 65,536 |
| Compiled primitive budget | Sum of reported triples, bits, opens, and simple multiplications below `2**32` |

The [MASCOT paper](https://eprint.iacr.org/2016/505), Appendix C.2, analyzes the two-output privacy amplification. If an adversarial execution passes sacrifice with probability `beta=2**(-c)`, its weighted distinguishing term is bounded by `2**(-k*(tau-2)/2-c/2)`, where `k` is the field bit length. Amplification four therefore provides the approximately 256-bit field-sized term; amplification three would provide only approximately 128 bits per triple before a batch union bound. The implementation changes the pinned amplification constant to four and retains standard `check=true`, including the same-B sacrifice. It does not use MAMA's one-output, cross-B replacement. Authentication and other native check failure terms remain separate parts of the MASCOT analysis.

All integer reductions receive proven nonnegative inputs of at most 78 bits. In particular, for canonical operands and `K<=65536`, `K*(p-1)**2 < 2**78`. A MAC residual satisfies `3 <= alpha*x + 3*p - sum(r_i) < 2**62`. Mersenne reduction decomposes the bounded integer, folds bit positions modulo 31 using `2**31=1 mod p`, then conditionally subtracts p enough times to obtain a canonical result.

Each reduction uses one statistically masked bit decomposition and at most three comparisons. With `E` non-calibration exported scalars and `H=7` application keys, the generator enforces the conservative scalar-call bound `4*(9*E+H)`, below `2**20` at the export limit. The 176-bit masking parameter therefore leaves a union term below approximately `2**-156`; even the broader `2**32` call ceiling would leave approximately 144 bits. The largest masked opening remains below `2**255`, inside q. Native triple, authentication, and input-consistency terms are accounted for separately under the compiled-work cap and the large-field MASCOT assumptions. This is a parameter and source argument, not a complete independently verified concrete-security reduction of the implementation.

The online runtime uses the [seven-key check bound](online-mac-bounds.md). Its statistical term across fewer than `2**64` online check batches is approximately `2**-146`, distinct from the native preprocessing terms. Hashes, commitments, PRGs, OT, and authenticated transport retain their computational assumptions; the intended cryptographic target is 128 bits.

## Evidence and conditional security scope

The [post-authentication validation report](../results/mascot_postauth_preprocessing_validation.json) records four successful native exits using the rebuilt fresh-challenge binary, followed by four successful role-local checker exits. The checker opened disposable imported material under all seven application MAC keys and independently checked the matrix relation. Two negative native runs put `2` and a non-bit witness whose validation flag is exactly `2**64` in a non-first input-vector lane. Every role explicitly aborted at the bit guard, and those bundles stayed unready. All opened material is retired. Earlier [pre-repair evidence](../results/mascot_preprocessing_validation_superseded.json) remains superseded and supplies no security approval for the rebuilt binary.

The rebuilt source matches the stated fresh-challenge ordering and bounded-construction argument. Its recorded status is a conditional experiment, not a formal proof or an independent audit of MP-SPDZ, libOTe, the compiler, or the operating system. Security requires at least one honest native/online role, private role-local randomness and files, correctly provisioned identities, authenticated confidential channels, correct honest-role execution of the agreed circuit, the selected MASCOT/OT/cryptographic assumptions, fresh single-use correlations, and retirement after failure. The local same-user process fixture does not enforce hostile-party OS isolation; deployment needs separate security domains. Build policies and hashes record provenance, not remote attestation.

The circuit verifier regenerates the expected source and compares supplied artifact fingerprints. It does not independently compile inside every role, so the public manifest alone does not establish source-to-bytecode correspondence against a malicious distributor. The local harness assumes its reviewed coordinator compiles the approved public circuit. Independently deployed honest roles must compile that circuit themselves or approve its bytecode hashes through a trusted code-distribution process before supplying private bits. This software-distribution assumption is separate from generating or reconstructing private preprocessing randomness.

A small preprocessing probe does not establish complete trained-model inference or its latency target. Native generation, model-dependent preparation, private input sharing, and resident online evaluation must be reported separately. Repeated ideal-network replay is a benchmark fixture, not permission to reuse correlations for another live query. A target including fresh preprocessing must charge each fresh correlation generation once; replaying one consumed fixture does not supply an amortization denominator. Static compilation can be reused, but the current producer starts a new native key/OT session for each bundle and includes that fresh setup in its preprocessing time. See [prepared-model lifetime and costs](prepared-model.md).

For the `8 -> 2 square -> 8` micro model, a batch of `B` fresh query rows with one fixed model exports `E(B)=32B+74` authenticated p31 scalars and requires `Q(B)=190696B+439685` authenticated random bits shared over q in the current circuit. These are the compiler's `integer bits` requirement, not raw PRG or network bits. The constant term includes the model material shared once within that batch; each query row still has fresh masks and correlations. The formula matches the compiler's 630,381-bit requirement at `B=1` and approaches 190,696 bits per query, about a 3.31-fold reduction of this count component. The current scalar limit supports at most 509 rows. These are [analytical count projections](../results/mascot_micro_amortization_projection.json), not measured batched throughput, an overall speedup, or an ideal-network lower bound.

## Running the bounded producer

From the repository root, verify a pinned build and generate a fresh bundle from a public shape manifest:

```sh
python -m scripts.build_mascot_secure --build-dir work/mascot256-fresh-build --verify-only
python -m rep4gpu.preprocessing_mascot --native-root work/mascot256-fresh-build \
  --manifest /path/to/shapes.json --output work/mascot-query-1 --timeout 900
```

The output directory must be new. The current bridge returns seven application coordinates; the native online runner therefore requires `--online-mac-keys 7`. Use a fresh bundle for each live query or independent comparison. Keep role-private inputs, outputs, and keys out of version control.
