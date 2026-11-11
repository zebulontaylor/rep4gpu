# Fixed-point activations

[Documentation index](../README.md)

`rep4gpu.arithmetic.activations.FixedPoint` composes Rep4 arithmetic, binary conversion,
and multiplication protocols. Construct it with a party runtime and the number
of fractional bits:

```python
from rep4gpu.arithmetic.activations import FixedPoint

fp = FixedPoint(party, fractional_bits=16)
# x is an arithmetic Share containing integers scaled by 2**16.
y = fp.gelu(x)
```

`encode` converts plaintext to signed `torch.int64` with nearest-even rounding.
Pass that tensor to the runtime's secret input operation. `decode` converts an
explicitly opened integer tensor to floating point. Neither method performs
secret sharing or opens a share.

`rep4gpu.protocol.runtime.Party` uses the general arithmetic-to-binary conversion for
truncation. `rep4gpu.models.inference.InferenceParty` uses carry-based exact truncation
by default (`rounding='floor'`), with the packed carry tree on CUDA and compact
weighted binary-to-arithmetic conversion. Both defaults produce the same
signed floor-rounded result. `InferenceParty(rounding='probabilistic', ...)`
selects the bounded-input rounding contract described below; it also changes
the rounding used by `FixedPoint` multiplications and matrix products.

## Arithmetic and rounding

`add` and `sub` preserve scale. With default floor rounding, `mul` multiplies
the encoded integers and performs an exact signed arithmetic right shift by
the fractional bit count. Negative results round toward negative infinity.
`matmul` accumulates integer products and truncates the sum once. This can differ from truncating each product before
summing. `matmul_relu(x, y, bias=None)` adds an optional fixed-point shared
bias to the product accumulator at product scale, then uses one fused exact
truncation and ReLU conversion. Its result equals ReLU of the floor-rounded
biased matrix product when the accumulator fits the ring.

`Party`'s general exact truncation converts the arithmetic word to binary
shares, shifts with sign extension, and converts all retained bits back to
arithmetic shares. `InferenceParty` instead combines shared upper parts with
three converted carry and sign bits, as described below. Both circuits account
for carries between additive components and operate without opening the secret.

All arithmetic uses the runtime's ring. Signed products and accumulated sums
must fit before truncation. For a 64-bit ring and 16 fractional bits, an encoded
product must remain within `[-2**63, 2**63-1]`. Increasing fractional precision
reduces this headroom. The layer cannot detect secret overflow without another
secure circuit.

`polynomial(x, coefficients)` takes coefficients in ascending degree order.
For example, `[1, 2, 3]` evaluates `1 + 2*x + 3*x*x` with Horner's method.
Coefficients use the same nearest-even encoding as inputs, and every Horner
multiplication truncates. The first multiplication uses a public coefficient
and needs no secret multiplication protocol. The result follows fixed-point
operations, not a single final rounding of a real-valued polynomial.

## ReLU, clipping, GELU, and SiLU

`relu` extracts the signed ring sign bit and multiplies by its complement.
It computes exact `max(x, 0)`, including the most negative ring value.
`clamp(x, lower, upper)` compares against public encoded bounds and selects
secretly. Signed comparisons account for subtraction overflow.

`gelu` and `silu` use public degree-ten polynomial approximations on `[-4,4]`.
They evaluate in the normalized variable `t=x/4`; this keeps high-degree
coefficients representable at ordinary fixed-point precisions. They exploit
the even structure as `x/2 + t*t*P(t*t)`, with fixed-point floor rounding at
each multiplication and division. Inputs are clipped before evaluation. The
two comparison results are reused for clipping and tail selection. Outside
the interval, the output is zero on the left and the original input on the right. These tail selections use
secret comparison bits. Shapes and circuit schedules depend on public inputs,
not secret values.

The polynomials were fitted by least squares on 20,001 evenly spaced samples
of the interval, using the even part of each activation and adding the exact
linear term. Odd coefficients above degree one and the constant term are zero.
The coefficient tuples in `activations.py` are the definition of the
approximations; no fitting occurs during a protocol run.

| Operation | Polynomial error on `[-4,4]` before quantization | Tail behavior |
| --- | --- | --- |
| GELU | About 0.0058 maximum absolute error on a dense sample grid | Maximum tail error about 0.00013 |
| SiLU | About 0.00094 maximum absolute error on a dense sample grid | Maximum tail error about 0.072 at the boundary |

These are sampled approximation errors, not certified bounds. Fixed-point
coefficient quantization, division by four, and intermediate truncations add
error. Tests cover fractional precisions 12, 16, and 20 on a dense input grid.
The piecewise approximations can have a small discontinuity at each boundary;
SiLU's identity and zero tails account for its larger overall error. Use
`polynomial` with application-specific coefficients when a different interval
or error target is required.

For numerical evaluation of the original SiLU and GELU formulas, use
[`NumericalFixedPoint`](numerical-primitives.md). It provides private exponential,
reciprocal, and erf primitives plus SwiGLU, with explicit ring64 precision and
input-range contracts.

## Bounded quadratic alternatives

`bounded_quadratic` substitutes a degree-two polynomial for an activation using
one secret square. It performs no comparisons, clipping, or tail selection.
The existing `relu`, `gelu`, and `silu` methods retain their behavior.

```python
fp = FixedPoint(party, fractional_bits=14)
plan = fp.quadratic_plan('gelu', bound=4., profile='local')
# The caller has established the public range contract for x.
y = fp.bounded_quadratic(x, kind='gelu', bound=4., profile='local')
# For a matrix product, combine input rescaling with the accumulator truncation.
h = fp.matmul_quadratic(x, weight, bias, kind='gelu', bound=4., profile='local')
```

Both methods default to `kind='relu'`, `bound=4.`, and `profile='uniform'`.
The local profile supports GELU and SiLU. Coefficients are public constants:

| Kind | Uniform profile | Local profile |
| --- | --- | --- |
| ReLU | `x/2 + x*x/8 + 1/4` | Unavailable |
| GELU | `x/2 + x*x/8 + 951/4096` | `x/2 + 3*x*x/8` |
| SiLU | `x/2 + x*x/8 + 427/4096` | `x/2 + x*x/4` |

Uniform profiles have nonzero output at zero and can produce negative outputs.
Local profiles preserve zero and approximate typical inputs near zero more
closely, but have much larger errors near the ends of `[-4,4]`. Neither profile
supplies the exact activation or the degree-ten method's tail behavior.

### Integer formula and range contract

`fp.quadratic_plan(kind, bound, profile=...)` returns a `QuadraticPlan`. The
standalone `rep4gpu.arithmetic.activations.quadratic_plan` also accepts `fractional_bits`
and `bits`. Plans depend only on these public parameters. Supported bounds
satisfy `0 < bound <= 4`; the planner rejects precision and range combinations
that cannot fit the selected 32-bit or 64-bit ring.

Let `f` be the fractional bit count, `S=2**f`, and `X` the signed encoded input.
The caller must establish `abs(X) <= plan.input_bound`, where
`plan.input_bound=floor(bound*S)`. This is a public assumption about private
inputs, not a check performed by the secure computation. Inputs outside the
bound invalidate the overflow and approximation assumptions.

For quadratic coefficient `m/2**q`, offset `c`, and prescale `t`, evaluation is:

```text
t = plan.prescale
q = plan.denominator_power2
m = plan.quadratic_coeff
d = plan.shift = f + q - 2*t
u = trunc(X, t)
N = m*u*u + 2**(t+d-1)*u + round(c*S)*2**d
Y = trunc(N, d)
```

`Y` retains scale `S`; `round(c*S)` uses nearest-even rounding. `trunc` follows the runtime's floor or probabilistic
rounding policy. A zero prescale leaves `X` unchanged. Prescaling reduces input
precision before the square; the integer result can differ from encoding a
single evaluation of the real polynomial.

The planner chooses the smallest `t` whose conservative numerator bound is
strictly below `2**(bits-2)`. It includes either allowed rounding outcome of
input rescaling and also prevents square overflow. The plan exposes
`normalized_bound`, `linear_coeff`, `constant_numerator`, and
`numerator_abs_bound`. `plan.encoded_numerator(u)` evaluates the public
plaintext integer formula and checks `u` against `normalized_bound`; it accepts
no shares and is intended for fixture or application range analysis.

`matmul_quadratic` forms `A=x@weight + bias*S` at product scale, then computes
`u=trunc(A, f+t)` directly. It avoids a separate prescale truncation after the
usual matrix-product rescaling. The caller must additionally establish that
`A` fits the signed ring and `abs(A) <= plan.input_bound*S`. Probabilistic
rounding also requires `abs(A) < 2**(bits-2)`. Modular accumulator overflow
cannot be repaired by the later polynomial. Bias has the same scale as the
encoded inputs.

At ring32 Q14 with `bound=4`, uniform profiles use `t=2,d=13`; local GELU uses
`t=3,d=11`, and local SiLU uses `t=2,d=12`. The fused path uses one input
truncation, one secret square, and one final truncation. Public coefficient
multiplications require no secret multiplication protocol. No intermediate
activation is opened.

### Approximation tradeoffs

The [CPU quality report](../../experiments/results/activations/local_quadratic_cpu_quality.json) compares
these fixed polynomials with their corresponding floating-point activations.
The Gaussian sample contains one million independent inputs with standard
deviation 0.5. The synthetic fixture has batch 128, width 2048, hidden width
8192, and 2048 outputs. Its top-1 agreement compares polynomial and reference
outputs from the same synthetic weights.

| Kind and profile | Maximum error on dense `[-4,4]` grid | Gaussian activation RMSE | Fixture logit RMSE | Fixture top-1 agreement |
| --- | ---: | ---: | ---: | ---: |
| GELU uniform | 0.23252 | 0.18795 | 0.18849 | 35.94% |
| GELU local | 4.00013 | 0.02352 | 0.02328 | 89.84% |
| SiLU uniform | 0.17619 | 0.08419 | 0.08443 | 56.25% |
| SiLU local | 2.07194 | 0.01113 | 0.01101 | 92.19% |

These measurements evaluate floating-point polynomial substitution; fixed-point
rounding adds error. Dense-grid maxima are sampled errors, not certified
bounds. Synthetic output agreement does not establish trained language-model
quality. Choose a profile using the target model's activation distribution and
end-to-end quality measurements, including its tails.

The [higher-degree approximation evaluation](../experiments/activation-approximation-evaluation.md)
compares degree-four through degree-ten fits with these quadratics, including
fixed-point FFN error, synthetic SwiGLU error, and private GPU activation costs.

## Protocol assumptions

The activation layer inherits the runtime's adversary model, randomness,
transport, and consistency checks. Its arithmetic and Boolean circuit
composition does not by itself supply MP-SPDZ's malicious-security checks.
A process that owns all four parties' shares can reconstruct secrets; co-located
simulation does not isolate those parties. Run each party under the deployment's
intended isolation and trust model.

The conversion approach follows the structure exposed by MP-SPDZ's
`Protocols/Rep4.hpp` (`split`) and `Protocols/Rep4Share2k.h`: represent arithmetic
components as binary shares, then add them using a binary circuit. Rep4's
optimized two-summand split uses joint inputs, whereas a four-component split
can reuse each existing replicated component. Binary-to-arithmetic conversion
and sign extension must preserve the full ring width for exact signed shifts.

## Optional probabilistic truncation

`rep4gpu.arithmetic.probabilistic.trunc_pr(party, x, shift)` implements the MP-SPDZ Rep4
small-gap masked truncation for 32-bit and 64-bit rings. It rounds the encoded
integer divided by `2**shift` to its floor or ceiling. For a uniform fresh mask,
the probability of the ceiling is `(x mod 2**shift) / 2**shift`, including for
negative inputs. Integer multiples of `2**shift` are exact. This differs from
the default fixed-point methods, which always round toward negative infinity.

The caller must ensure `abs(signed(x)) < 2**(party.bits-2)` and
`0 <= shift < party.bits-1`. The first condition is a public range contract;
the protocol does not reveal or secretly validate the input's range. Violating
it invalidates the rounding guarantee. Check product and accumulated-product
ranges before choosing this operation. A zero shift returns the input without
communication.

The mask uses the two replicated components known jointly to parties 0 and 1.
Parties 0 and 1 send complementary masked-opening messages to parties 2 and 3,
with crosswise hashes. The protocol checks those hashes before resharing the
masked value's most significant bit and upper bits. Two batched joint inputs
and one secret multiplication compute the carry correction. Only masked values
are opened to the receiving pair; the secret operand is not reconstructed.
The same randomness provisioning and transport assumptions as the runtime
apply. This implementation supplies the small-gap correction and does not
include an alternative mode with a statistical large-gap failure probability.

## Exact truncation from carry bits

`rep4gpu.arithmetic.exact_truncation.trunc(party, x, shift)` is the exact full-ring
signed truncation circuit used by `InferenceParty` with default rounding. It converts three binary bits per input
word back to arithmetic, rather than converting every retained output bit.
`trunc_relu` additionally gates the arithmetic result by the complement of the
converted sign bit. Both functions preserve the default floor-rounding rule.

The default circuit follows MP-SPDZ's two-summand split. Parties 0 and 1
jointly know the modular sum of components 2 and 3; parties 2 and 3 know the
sum of components 0 and 1. The runtime reshares both pair words in the binary
domain, and their unsigned upper parts in the arithmetic domain, using two
joint-input rounds. On CPU, a binary prefix addition computes the modular sum
and its carries. On CUDA, `carry_tree_gpu.truncation_bits` computes only the
carries into `shift`, the ring boundary, and the sign position. The sign is
the XOR of the two top operand bits and the carry into the sign position.

The GPU tree packs generate and propagate nodes from multiple input words into
binary sharing words. It combines adjacent nodes at each level and repacks the
surviving nodes into smaller buffers. Carries requested at shorter positions
finish earlier; the longest tree determines the number of rounds. No full
binary output word is needed. Packing follows the runtime's `binary_bits`
setting, which defaults to the arithmetic ring width in `InferenceParty`.

Only the low carry, ring carry, and sign enter compact weighted conversion.
Their weights are `1`, `-2**(w-shift)`, and `-2**(w-shift)` respectively.
Adding this converted correction to the two shared upper parts gives the
signed quotient. Fused ReLU requests the sign as a second weighted output in
the same conversion, then gates the quotient with one multiplication.

For nonempty inputs and a nonzero shift, both the CPU prefix path and GPU tree
path use eleven communication rounds for a 64-bit ring and ten for a 32-bit
ring: two joint-input rounds, seven or six carry-computation rounds, and two
compact conversion rounds. Fused ReLU adds one round. A zero shift returns
the original share; fused ReLU still securely computes its sign and selection.

`trunc_four_component` retains a reference circuit that starts directly from
all four replicated components. For ring width `w`, write these unsigned
additive components as `a0, a1, a2, a3`. Two binary additions compute the component pairs in parallel,
and a third adds the pair results. Each prefix adder supplies its carry into
position `shift` and its carry out of position `w`. Let `L` be the sum of the
three carries into `shift`, `H` the sum of the three carries out of the ring,
and `s` the final signed word's sign bit. Then the exact signed quotient is

```text
sum_i floor(ai / 2**shift) + L - 2**(w-shift) * (H + s).
```

This follows by applying the two-input unsigned addition identity at each
node: a low-bit carry adds one to the quotient, while a ring overflow subtracts
`2**(w-shift)`. The two's-complement sign correction subtracts that same amount
once more for a negative final word. The local component quotients retain their
original replicated ownership. Only the six carry bits and final sign bit need
binary-to-arithmetic conversion, performed together in two multiplication
rounds. No intermediate word or carry is opened.

For nonzero shifts, the reference circuit in a 64-bit ring uses two levels of
seven-round binary prefix addition followed by two binary-to-arithmetic
rounds. A 32-bit ring uses two levels of six-round addition. This trades
additional rounds for smaller conversion payloads than the general full-word
conversion; the relative cost depends on tensor size and network latency.

## Compact binary-to-arithmetic conversion

`rep4gpu.arithmetic.compact_conversion.b2a` converts a binary-shared word with the same
selected-bit and signedness arguments as the general conversion. It computes
the parity of the four binary components in two stages. The product of
components 0 and 1 is already known jointly by parties 2 and 3; the product of
components 2 and 3 is known by parties 0 and 1. Two joint inputs therefore
replace the first stage's general secret multiplications, while preserving
the runtime's masked messages and hash checks.

The final XOR combines the two arithmetic parities. Its weighted bit products
are summed locally before multiplication contributions are reshared. This
keeps the final multiplication payload proportional to output words rather
than bit planes. The operation uses two communication rounds and opens no
secret value.

`weighted_b2a(party, planes, weights)` exposes the same conversion for selected
bit planes of shape `(P, *S)`. Integer weights of shape `(P,)` return one shared
weighted sum of shape `S`. Weights of shape `(O, P)` return `O` weighted sums
with shape `(O, *S)` in the same two rounds. Each input plane represents a
binary secret bit; the method selects the low bit of each local component.
Weights and arithmetic outputs use the runtime's ring, so signed weights
follow ordinary modular arithmetic. For example, the carry correction and
sign output in exact truncation can share one conversion using weights
`[[1, -2**d, -2**d], [0, 0, 1]]`, where `d` is ring width minus shift.
