# MAMA preprocessing security findings

[Documentation index](../README.md)

This analysis records the sacrifice and mask-privacy properties of the removed
small-field MAMA preprocessing experiment. It does not describe a supported
preprocessing backend in this repository.

## Independent sacrifice challenges

For a triple `(a_i,b_i,c_i)`, write its multiplication error as `delta_i = c_i-a_i*b_i`. The direct MAMA sacrifice branch retains triple 0 and discards nine triples when using ten MAC keys. For each discarded triple it opens

```text
rho_i   = t_i*a_0-a_i
sigma_i = b_0-b_i
```

and checks the authenticated value

```text
t_i*c_0-c_i-b_i*rho_i-a_i*sigma_i-rho_i*sigma_i
    = t_i*delta_0-delta_i.
```

The original branch reuses one scalar `t` for every pair and every group. This has a field-size soundness floor: if `delta_0 != 0` and all discarded triples are correct, every equation passes at `t=0`, with probability `1/p`. Ten independent MAC keys do not repair this event because the incorrect triple values can themselves have valid authenticators. Public amplification before authentication does not exclude a malicious party changing its own multiplication-result contribution before that contribution is authenticated.

The experiment's [independent-sacrifice patch](../../src/native/patches/mama-independent-sacrifice.patch) drew a separate field challenge for every `(group,i)` from one fresh jointly seeded `GlobalPRNG`, after all candidate triples have been generated and authenticated. It stores those challenges and uses the same indexed challenge in the masked opening and its residual equation. It also constructs `TripleShuffleSacrifice` only when that branch is selected; its existing security guard remains unchanged.

For fixed errors with `delta_0 != 0`, each equation fixes exactly one independent challenge value. Nine checks therefore accept that bad retained triple with probability `p**-9` in the uniform-independent-challenge model. For `B` retained triples, a union bound gives `B*p**-9`, in addition to authentication, coin-generation, and cryptographic implementation failure terms. For `p=2**31-1`, nine checks supply approximately 279 bits before that union bound. The actual PRNG expands a jointly generated seed, so replacing independent uniform challenges by this stream adds its computational assumptions. This calculation is a sacrifice argument, not a formal proof of the complete native preprocessing stack.

The masking argument also requires the native protocol's fresh random operands and OT guarantees. Independent challenges do not substitute for secure OT or correct authentication. The build must retain all ten exported MAC coordinates and use the intended malicious OT implementation. MP-SPDZ labels its optional KOS route insecure; the intended build uses the default SoftSpokenOT route, without `INSECURE`.

## Mask privacy limitation

The independent-challenge calculation above proves a bound on accepting a **nonzero multiplication error**. It does not establish that an accepted triple's masks remain private. The removed native MAMA path has a separate source-derived selective-failure counterexample under all-but-one corruption.

In a targeted raw OT group, a corrupt party can supply correlation `b_i+1` while retaining `b_i` for its stored/authenticated input. This adds the honest party's raw `a_h` to the multiplication-result error. The public amplification in `OT/Triple.hpp` is linear in both `a` and `c`; adding the coalition's known amplified `a` shares to its local `c` before authentication makes the retained triple's total error `delta_0=A_0`. Other sacrifice triples can remain correct. `SimpleMascotTripleGenerator` with `check=false` authenticates only one amplified `a,c` coordinate from that raw group.

Every direct sacrifice check then has residual `t_i*A_0`. They all pass when `A_0=0`, an event with probability approximately `1/p`, even if the challenges are independent. If at least one challenge is nonzero, acceptance tells the coalition that the retained mask is zero. That accepted triple has the correct product relation and valid MACs, so subsequent correctness checks do not detect its lost privacy. Increasing the number of cross-group sacrifice checks does not remove this event.

The exporter uses native triples when computing its matrix products. A known native mask reveals the corresponding operand through its native Beaver difference, and that operand can become an exported query mask. Functional matrix/square/output parity therefore does not establish that the exported masks are private at the requested 128-bit level. This counterexample follows from the source dataflow and algebra; an adversarial native executable has not been run to reproduce it.

The primary [MASCOT paper](https://eprint.iacr.org/2016/505) treats privacy amplification separately from correctness sacrifice. The removed small-field MAMA adapter would require an additional justified mask-privacy construction before it supports a complete 128-bit malicious-majority preprocessing claim. Its `-S 128` setting, ten MAC coordinates, successful execution, and the independent-challenge patch alone do not establish that claim.

## Squares and owner masks

MAMA inherits `BitPrep::buffer_squares`: it obtains a checked triple `(a,b,c)`, opens and checks `a+b`, and returns `(a, a*(a+b)-c)`. Correct authenticated triples give `a**2` directly. This path introduces no additional small-field sacrifice challenge.

`NPartyTripleGenerator::generateInputs` follows a separate OT authentication path. For an honest owner it samples the clear mask locally, derives the owner's additive share from that mask and pairwise PRGs, and includes the owner's own MAC-key contribution locally. Authenticated OT supplies the remaining MAC-key contributions. A corrupt role can change its own value/MAC state, but producing a validly authenticated aggregate mask different from the honest owner's clear mask requires defeating the unknown honest MAC-key authentication. For a corrupt owner, choosing its effective input is already permitted by the online functionality.

The native input preprocessing check uses a common scalar linear combination across MAC coordinates. That check alone is not ten independent field checks for every individual mask: some inconsistent authenticators can survive a projection cancellation. This is distinct from an authenticated triple with an incorrect product relation. Inconsistent mask authenticators remain subject to the online per-element or independent-per-key checks before any unmasked output. Local corruption of one's own shares is therefore not an end-to-end counterexample to the owner-mask protocol. This argument relies on the native OT/authenticator privacy and consistency guarantees and on fresh keys being retired after failure; it does not replace an independent proof of that stack. Every exported MAC coordinate, native key share, and owner-only clear mask must be preserved.
