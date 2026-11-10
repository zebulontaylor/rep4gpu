# Online MAC-check parameters

The online compressed-check backends require at least seven independent MAC-key coordinates over `p = 2**31 - 1`. Native generation, its sacrifice checks, and the default online configuration retain ten coordinates. The optional native adapter mode selects the fixed first seven coordinates only after validating the complete ten-coordinate bundle. It returns owned copies and binds the ordered selection into the session. The selection cannot change after a challenge or a failed check, and loading a subset still consumes the complete bundle once.

## Statistical bound

For a fixed nonzero error vector in consistently broadcast opened values, let `r_k` be the fresh coefficient vector for MAC coordinate `k`. Its error projection is zero with probability `1/p` under independent uniform field coefficients. If the projection is nonzero, producing an accepting committed residual requires guessing an independent hidden honest MAC-key share, with probability at most `1/p`. Thus a compressed check with `K` independent coordinates has the bound

```text
q = 2/p - 1/p**2
Pr[an incorrect opening passes one check] <= q**K
```

Each coordinate has its own challenge row. Reusing one challenge row for every coordinate would leave a common projection-cancellation event with probability `1/p`.

The argument applies to the first accepted incorrect opening. Before that event, accepted correct checks do not reveal hidden MAC-key coordinates through a nonzero error projection. A union bound over `T` online authentication checks using the same keys gives `T*q**K`. With `K = 7` and `T < 2**64`, this is approximately `2**-146`, and is strictly below `2**-145`. Six coordinates do not meet a `2**-128` target for that entire counter range. A full-vector check has the stronger per-check bound `p**-K` and is covered by the same conservative union bound.

These are online statistical terms under correct authenticated preprocessing and hidden independent keys. SHAKE256 or ChaCha20 expansion from a public coin supplies computational pseudorandom coefficients, rather than literal information-theoretic entropy for the entire coefficient matrix. Commitment, hash, generator, transport, and preprocessing assumptions remain separate; the calculation is not a proof of the whole implementation. The seven-coordinate statistical term remains below the nominal 128-bit cryptographic target.

## Required ordering and state

Opened value broadcasts and their consistency echoes finish before the fresh joint coin is revealed. Every party commits its coin contribution before any contribution is revealed. Coefficients are sampled independently per key, with rejection of the non-field 31-bit word. Every compressed residual is committed before an honest residual is revealed.

Deferred checks bind the ordered opening schemas and freeze their authentication snapshots before generating the batch coin. All pending checks complete before any unmasked output share is sent. See [deferred MAC checks](deferred-mac-checks.md) for that boundary.

One ordered protocol stream uses each native key set. The `T < 2**64` bound uses the monotonic 64-bit broadcast-round guard: each authentication check consumes at least one broadcast, and live execution never resets that counter. Creating a new party object does not grant another allowance under the same keys. A failed or interrupted check poisons its session and permanently retires all ten native coordinates, including any three not selected online, and the remaining correlations. Retrying under the same keys or reloading the bundle in another online mode is forbidden. The online subset is fixed before the session begins. The optional seven-coordinate mode does not reduce native generation, verification, or sacrifice to seven coordinates; see [native preprocessing](native-preprocessing.md) and the [native sacrifice correction](native-mama-sacrifice.md).
