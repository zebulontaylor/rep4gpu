# Batched product witnesses

[Documentation index](../README.md)

`ProductWitnessBatch` checks prover-supplied ring64 products before a
`WitnessSession` releases outputs. It replaces elementwise product resharing
with checked witness input and a batch of 128 scalar checks. The default
activation programs retain their original product protocol; this verifier is
an explicit alternative.

For every claimed `c = a*b`, the verifier freezes all three share tensors.
After registration closes, each Rep4 role privately inputs a fresh 256-bit
coin. The verifier shares all four contributions before opening their sum.
An honest contribution keeps that sum unpredictable while the claims become
fixed. SHAKE256 expands the resulting public seed into 128 coefficient vectors.

Each row checks a weighted sum of `a*b-c`. Rep4 computes the weighted local
bilinear terms and reduces them before resharing, so the product-check payload
has 128 entries regardless of the number of coordinates. Residuals remain
private. The session combines exact zero tests with its other constraints and
opens one acceptance bit. The public challenge seed is also opened; input,
product, and residual values are not.

## Soundness over ring64

Fix any nonzero residual vector over `Z/(2**64)`. At least one coordinate is
`2**v` times an odd word, with `v < 64`. Conditioned on all other coefficients,
its uniform coefficient makes the weighted sum uniform over a subgroup with
at least two elements. One independent row therefore accepts an invalid vector
with probability at most `1/2`. All 128 independent rows accept with probability
at most `2**-128`, including when the only error is `2**63`.

That bound is for ideal independent coefficient rows. The implementation uses
SHAKE256 to expand a 256-bit seed and additionally relies on its pseudorandomness.
It does not treat the ring as a field or claim that one row supplies 128-bit
soundness. The bound applies per batch; separate batches use fresh coins.

All parties must register the same public shapes and use the same chunk size.
Claims cannot be added after verification starts. Product identities establish
modular multiplication, not integer range bounds. Activation recipes must still
prove the bounds needed for truncation and real-valued identities. For a chain
of claimed products, those bounds can follow by induction from the checked
input: each valid product establishes the next reduction's range. An invalid
earlier claim prevents release except under the stated false-acceptance bound.

## Exact SiLU with product witnesses

[`silu_product_witness.py`](../../src/rep4gpu/proofs/silu_product_witness.py) preserves the
existing direct-output SiLU integer formula and inclusive `[-16,16]` domain.
Its plaintext builder returns the output and ten raw product claims. The
verifier retains exact range checks and exact floors, and batches the product
identities.

```python
from rep4gpu.proofs.silu_product_witness import (
    silu_product_witness_program, silu_product_witnesses,
)

program = silu_product_witness_program()
# Run the builder in the prover process on encoded plaintext inputs.
claims = silu_product_witnesses({'x': encoded_x})
```

Supply these claims through `input_witnesses`, then use
`WitnessSession.evaluate` and `finish` as in the
[activation interface](witness-activations.md#supply-witnesses-and-recompute).
No claimed intermediate may be opened or consumed outside that interface
before acceptance.

## Rational SiLU

[`silu_rational.py`](../../src/rep4gpu/proofs/silu_rational.py) provides a different Q24
approximation on the same inclusive `[-16,16]` domain. Use
`rational_silu_program()` and `rational_silu_witnesses({'x': encoded_x})` with
the same session interface. It supplies six raw product claims and computes
six exact reductions.

With `S=2**24`, the verifier first computes `z=nearest(X*X/2**26)`. It forms
`z²`, `z³`, and `z⁴` at Q30, rounding each product to nearest with ties upward.
The public coefficient arrays in the module produce positive Q21 polynomials
`A` and `B`. The output satisfies the nearest-integer relation for
`(X*B+2*S*A)/(2*B)`. The plaintext builder chooses nearest-even at exact ties;
the verifier accepts either nearest neighbor, as the exponential recipe does.

Every power lies in `[0,2**30]`. Positive coefficients bound the two raw
polynomial accumulators, including their rounding offsets, by
`3,490,456,837,516,427,264` and `436,307,242,598,268,928`, both below `2**62`.
The denominator has `2**21 <= B <= 406,342,784`. The input and output range
checks prevent wrap in the final rational identity. Product bounds follow
inductively when the batch identities hold; invalid earlier claims retain the
batch's rejection guarantee.

The [exhaustive production-oracle report](../../experiments/results/proofs/silu_rational34_exhaustive_quality.json)
covers all 536,870,913 Q24 encodings in the domain. Its maximum absolute error
against float64 SiLU is `9.28304e-7`. Allowing the other exact tie neighbor adds
at most one Q24 unit, giving the conservative bound `9.87909e-7` for either
accepted neighbor. The report includes hashes of the production formula and
quality benchmark.

## Measured communication

The [comparison](../../experiments/results/proofs/silu_product_witness.json) uses 14,336 values on
`[-4,4]`, four live CUDA Rep4 roles, and Q24. Global bytes include every witness
input, public coin, check, and acceptance opening. Already-shared input and the
validation output opening are excluded.

| Program | Global bytes | Maximum fixture error |
| --- | ---: | ---: |
| Default exact-product SiLU | 37,300,080 | 4.65314e-7 |
| Same formula with product witnesses | 32,761,840 | 4.65314e-7 |
| Rational formula with product witnesses | 24,357,248 | 7.71902e-7 |

All measured outputs match their program's integer oracle. These variants
remain above 10 MB. These are payload measurements, not proving-time or WAN
latency measurements.

The separate experimental [committed-input proof backend](proof-activations.md)
uses the same rational formula and reduces measured global traffic to
6,109,416 bytes by replacing per-coordinate MPC checks with native proofs.

Run from the repository root with CUDA available:

```sh
python -m experiments.benchmarks.proofs.silu_product_witness
python -m experiments.benchmarks.proofs.silu_rational34_quality --exhaustive \
  --output experiments/results/proofs/silu_rational34_exhaustive_quality.json
```
