# Activations with untrusted shared witnesses

Witness-assisted evaluation replaces selected nonlinear iterations with shared
intermediate values from a prover. Rep4 recomputes the remaining expression and
checks the witnesses before returning output shares. The prover's values are
untrusted; the activation recipe and its public bounds are trusted code.

The implementation is in [`witness_activations.py`](../rep4gpu/witness_activations.py).
It uses ring64 fixed point with 20 through 24 fractional bits. Existing
[polynomial activations](activations.md) and
[numerical primitives](numerical-primitives.md) remain available.

## Verification contract

A `WitnessProgram` names its witness tensors, declares their bounds, and supplies
a recomputation callback. `WitnessSession.evaluate` queues an output;
`WitnessSession.finish` checks the accumulated constraints and returns the queued
shared outputs only when they pass. An invalid witness raises `InvalidWitness`.
Do not open or consume callback intermediates outside this interface.

Witnesses must arrive through the runtime's checked input protocol or another
input protocol with equivalent consistency guarantees. A plaintext prover is
not an additional Rep4 role: an application must deliver its values to an input
owner or integrate an external sharing protocol. This module does not provide
an external-prover network service.

Range checks retain full-ring semantics for untrusted words. Clamp-based
recipes replace invalid values privately and retain the violation. Other
recipes register a sticky range failure and establish subsequent bounds
conditional on acceptance. Checking a product only after it has wrapped would
not establish a real-number identity.

The acceptance check reveals whether the batch passed. It does not open witness
values or residuals. Checks that only affect acceptance retain their rejection
bits in binary sharing. A packed OR tree combines these bits before one scalar
conversion joins them to the bounded arithmetic violation count. Checks needed
for private clamping retain arithmetic bits. The default implementation does not use a random linear combination
over `Z/(2**64)` to claim field-style soundness. Nor does it assume that the
existing arithmetic-to-binary conversion takes one round. Actual communication
includes the runtime's comparisons, truncations, consistency hashes, and final
acceptance check.

The opt-in [batched product verifier](product-witnesses.md) uses 128 ring
challenges under an explicit pseudorandomness assumption. It opens a fresh
public challenge seed in addition to the acceptance bit.

The separate [committed-input proof backend](proof-activations.md) verifies
the Q24 rational SiLU relation with native Bulletproofs R1CS and binds its
commitments to live Rep4 shares. It is an opt-in experimental backend with
an explicit external-client traffic benchmark.

Custom callbacks must constrain every untrusted quantity that affects their
result, establish product bounds before multiplication, and register the
identities defining the desired function. A witness schema alone cannot prove
that a callback computes the advertised activation.

## Supply witnesses and recompute

Each party runs the same public schedule. Here `x` is an existing Q24 arithmetic
share, and the prover sends encoded SiLU output values to party 0. Only party 0
sets `prover_values`; the other parties pass `None`.

```python
from rep4gpu import WitnessSession, input_witnesses, numerical_program

program = numerical_program('silu')
witnesses = input_witnesses(
    party, program, owner=0,
    shapes={'output': tuple(x.shape)},
    values=prover_values,  # {'output': encoded_tensor} on party 0
)
session = WitnessSession(party, fractional_bits=24)
session.evaluate(program, {'x': x}, witnesses)
output, = session.finish()
# output remains shared and can feed the next layer.
```

SiLU uses an `output` witness for its nearest encoded result. SwiGLU uses the
same SiLU gate witness and additionally takes a shared `value` input; the
verifier multiplies the checked gate by `value`. GELU retains its `reciprocal`
witness for the denominator in its erf approximation. The verifier still
recomputes the exponential privately; an output claim alone does not establish
the activation identity.

`numerical_program('silu', direct_output=False)` and the corresponding SwiGLU
option retain the reciprocal recipe on `[-4,4]`. Pass the same option to
`numerical_witnesses` when constructing its claims. The default direct-output
recipe accepts `x` in `[-16,16]` and requires exact-floor arithmetic.
Multiple calls to `evaluate` can share one session and acceptance check.
`finish` returns their results in call order. A rejected batch releases none
of them, and a finished or failed session cannot be reused.

## Add a recipe

Witness names and shapes need not match across different recipes. Bounds apply
to decoded values, while `residual` tolerances are integer units of the supplied
residual expression. This example checks two shared intermediate tensors before
adding them:

```python
from rep4gpu import WitnessProgram


def recompute(ctx, inputs, witnesses):
    if set(inputs) != {'x'} or any(
        w.shape != inputs['x'].shape for w in witnesses.values()
    ):
        raise ValueError('double requires matching input and witness shapes')
    x = ctx.bounded(inputs['x'], 0, 2)
    ctx.residual(ctx.runtime.sub(witnesses['left'], x), tolerance_lsb=0)
    ctx.residual(ctx.runtime.sub(witnesses['right'], x), tolerance_lsb=0)
    return ctx.runtime.add(witnesses['left'], witnesses['right'])


program = WitnessProgram(
    'double', {'left': (0, 2), 'right': (0, 2)}, recompute,
)
```

The callback must validate its public shapes and input names. Built-in recipes
require exact shapes rather than relying on implicit broadcasting of witnesses.
The session rejects a callback that registers neither a residual constraint
nor a `nonnegative` constraint. The latter requires an encoded expression to
be nonnegative and, like residual checks, requires a non-wrapping expression.

## Built-in domains and identities

All encoded inputs and witnesses use the session's scale `S = 2**f`.

| Recipe | Inputs | Witnesses | Checked relation |
| --- | --- | --- | --- |
| `numerical_program('reciprocal')` | `x` in `[1,2]` | Elementwise `reciprocal` in `[0,1]` | `-X <= 2*(X*W-S*S) <= X` |
| `numerical_program('silu')` | `x` in `[-16,16]` | Elementwise `output` in `[-16,16]` | `-D <= 2*(Y*D-P) <= D`, defined below |
| `numerical_program('swiglu')` | `x` in `[-16,16]`, `value` in `[-4,4]` | Elementwise SiLU gate `output` in `[-16,16]` | Checked SiLU gate multiplied by `value` |
| `numerical_program('gelu')` | `x` in `[-4,4]` | Elementwise `reciprocal` in `[0,1]` | Reciprocal identity for the recomputed denominator |
| `rmsnorm_program(epsilon=1e-5)` | Rows of `x` in `[-4,4]` | Row `inverse_sqrt` in `[1/8,512]` | `s*(V*s)` approximates one, with positive bounded `s` |
| `softmax_program()` | Rows of `x` in `[-4,4]`, width at most 4096 | Row `shift` in `[-4,4]`, `reciprocal` in `[0,1]` | Exact encoded maximum and reciprocal of the recomputed exponential sum |

### Direct-output SiLU

For encoded input `X`, the verifier extracts its sign and forms its absolute
value, then checks the inclusive bound `abs(X) <= 16*S`. An invalid input
registers a violation and its magnitude is clamped before exponential
arithmetic. The minimum signed ring word is also rejected by this check.
The verifier computes `T = floor(-abs(X)/16)` and evaluates a degree-six
Remez approximation to `exp(T/S)` on `[-1,0]`. The Q24 coefficients are:

```text
[16777216, 16777188, 8388203, 2793717, 691720, 128514, 14276]
```

The quadratic evaluation uses four floors and three secret products:

```text
U  = floor(T*T/S)
Q2 = C4 + floor((C5*T + C6*U)/S)
Q1 = C2 + floor((C3*T + U*Q2)/S)
H  = C0 + floor((C1*T + U*Q1)/S)
```

Each subsequent square uses `floor((H*H + S/2)/S)` four times, rounding to
nearest with ties upward. The Q24 linear coefficient compensates for this
schedule's floor bias. The rounding schedule and coefficients define the
verified function and the required witness. The implementation and scaling
for Q20 through Q24 are in [`silu_exp.py`](../rep4gpu/silu_exp.py).

For `-S <= T <= 0`, `0 <= U <= -T <= S`. The coefficients establish
`0 <= Q2 <= C4+C6` and `0 <= Q1 <= C2+C4+C6 <= C1`. Therefore
`-C1*S <= C1*T+U*Q1 <= 0`, so `H` lies in `[0,S]`. Nearest squaring
preserves this interval. The final exponential `E` lies in `[0,S]`, and
`D=S+E` lies in `[S,2*S]`, without an additional denominator range check.

The verifier selects `M=-E` for negative `X` and `M=S` otherwise, then forms
raw products `P=abs(X)*M` and `Y*D`. It checks the nearest-integer relation:

```text
-D <= R = 2*(Y*D-P) <= D
```

Either neighbor is accepted at an exact tie; the prover uses integer
nearest-even rounding. The output witness is not clamped. An exact unsigned
range check on `Y+S` establishes `-S <= Y < 31*S`, which keeps `R` within
`(-2**56,2**56)` at Q24. The lower relation check requires `R+D` in
`[0,8*S)`. Once this holds, `D-R = 2*D-(R+D)` lies in `(-8*S,4*S]`, so
its sign needs only `f+4` bits. Invalid earlier bounds remain sticky even
if later arithmetic on rejected inputs wraps. SiLU returns the checked `Y`;
SwiGLU multiplies it by its bounded `value` input and rescales once.

For an unsigned range check at bit `k`, let `U,V` be the upper parts of the
two private pair sums and `c` the carry out of their low `k` bits. The upper
part is zero exactly when `V = -U-c` modulo `2**(64-k)`. Two binary equality
trees test `V=-U` and `V=-U-1`; the carry selects the correct result. One
Boolean-to-arithmetic conversion produces a violation bit for the exact
acceptance count. Raw ring residuals must not be summed into this count:
nonzero residuals can cancel modulo `2**64`.

The [exhaustive Q24 report](../results/silu_quadratic_horner_quality.json)
evaluates all 536,870,913 input encodings in `[-16,16]` against float64
references. Maximum absolute error is `5.61785240355e-7`; the conservative
bound allowing either accepted tie neighbor is `6.21389885130e-7`, below
`6.41e-7`. These numerical checks apply to Q24 SiLU, not to lower fractional
precisions or the additional SwiGLU multiplication.

### Other built-in recipes

Row witnesses have shape `x.shape[:-1] + (1,)`. RMSNorm excludes its optional
learned weight. Its epsilon lies in `[1e-5,1]`; power-of-two row widths may be at
most `2**20`, and other widths at most 4096. The implementation reduces raw
squares locally in tiles of at most 512 coordinates, reshares and truncates
each tile sum, then averages and adds encoded epsilon. A tile's reconstructed
raw sum is at most `8192*S*S`, below `2**62` at Q24. This avoids per-coordinate
square truncations without allowing a full 4096-coordinate raw sum to wrap.
The prover uses the same tile schedule.

RMSNorm bounds the intermediate `V*s` to `[0,5]` before its final product.
This permits the usual `1e-5` epsilon without forming a potentially overflowing
`s*s` intermediate. Its raw residual tolerance is `518*S`, or `518/S` in the
normalized identity. Reciprocal checks use the secret encoded denominator `D`
itself: `-D <= 2*(D*W-S*S) <= D`. This requires the nearest encoded reciprocal,
accepting either neighbor at an exact tie. It avoids a row-width-dependent
slack that could admit inaccurate reciprocals when the actual sum is small.
The RMSNorm tolerance admits deviations larger than nearest-root rounding;
measured honest accuracy is not a worst-case error guarantee for all accepted
root witnesses.

Softmax checks that every shifted logit is nonpositive and that each row has a
zero. It recomputes the existing exponential approximation on `[-8,0]`.
Approximate exponentials are not exactly shift-invariant, so accepting an
arbitrary upper-bound shift would change the function being verified.

The plaintext helpers in [`witness_prover.py`](../rep4gpu/witness_prover.py) build
claims for exact-floor verification:

```python
from rep4gpu import numerical_witnesses, rmsnorm_witnesses, softmax_witnesses

silu_claims = numerical_witnesses('silu', {'x': encoded_x}, fractional_bits=24)
rms_claims = rmsnorm_witnesses(encoded_rows, fractional_bits=24, epsilon=1e-5)
softmax_claims = softmax_witnesses(encoded_rows, fractional_bits=24)
```

These helpers run on plaintext in the prover process. They must not receive
verifier shares. With probabilistic truncation, the verifier's intermediate
values can differ from the exact-floor denominator used by these builders;
this integration is validated with the exact-floor runtime.
Direct CUDA execution is supported by the tested device runtime. `finish`
reads the opened acceptance bit on the host and must run outside CUDA graph
capture; this interface is not a drop-in captured inference graph.

## Bounded exact-floor contract

[`trunc_bounded`](../rep4gpu/exact_truncation.py) computes the same signed floor
as the full-ring truncation circuit when the reconstructed input satisfies
`abs(Z) < 2**62` in ring64. The caller must establish this strict bound from
public arithmetic bounds or prior private checks. The routine does not check
it; use ordinary `trunc` for arbitrary ring words.

Adding `2**62` places the value in `[0,2**63)`. For the two unsigned pair sums
`u` and `v`, the wrap bit is then `msb(u) OR msb(v)`. The protocol absorbs the
individual MSB contributions into the arithmetic upper shares and converts
only the low fractional carry and the MSB product back to arithmetic shares.
The CUDA implementation shares the required low bits and top bits and uses a
pruned carry tree. The CPU implementation is an integer-protocol reference.

SiLU's Horner products have magnitude at most `S*S`; its rounded-square
numerators are at most `S*S+S/2`, both below `2**62` at Q24. Its bounded inputs
also justify the other explicitly selected bounded operations. This does not
establish the bound for arbitrary FFN projections: each dot-product
accumulator needs bounds on its operands and reduction width before selecting
this protocol. Q24 by itself provides no accumulator guarantee. Untrusted
input and witness checks retain full-ring semantics. The conditional output
comparison uses the narrower signed range established above.

## Current SiLU communication

The current CUDA measurement uses 14,336 inputs on `[-4,4]`, four optimized
roles, and ring64/Q24. Global bytes include witness ingestion, verification,
and acceptance, excluding initial activation sharing and the validation output
opening. The production path sends **37,300,080 bytes**, or **2,601.85 bytes
per coordinate**.

The [current accounting](../results/silu_binary_acceptance_accounting.json)
includes a semantic expansion that matches a separate run of the production
program in both exact outputs and total bytes. The measured maximum absolute error remains
`4.65314e-7` on this fixture. Traffic remains above 10 MB.

A bare secret multiplication costs 688,320 bytes for this batch. One bounded
Q24 floor costs 1,900,928 bytes, compared with 4,338,496 for the full-ring
floor. The exponential uses seven secret products, eight Q24 floors, and one
shift-by-four floor. The direct output relation adds two raw products without
rescaling. Witness upper-zero checking costs 2,443,296 bytes; input sign,
absolute value, and clamp cost 7,300,608 bytes. The two output checks together
cost 3,942,624 bytes. Final rejection aggregation and acceptance cost 38,192
bytes. The upper-zero check derives its second high-bit equality target from
an existing binary share, saving one checked joint input per value.

The [FFN decomposition](../results/ffn_communication_breakdown.json) reports
both bounded projection costs and a full-ring projection fallback. Its bounded
projection totals require the accumulator contract above; the benchmark does
not prove that contract for a trained model. These measurements describe
communication payloads, not WAN latency or end-to-end model accuracy.

Reproduce the current communication and numerical reports from the repository root
(the accounting benchmark requires CUDA):

```sh
python -m benchmarks.silu_accounting --output results/silu_binary_acceptance_accounting.json
python -m benchmarks.silu_minimax_quality --exhaustive \
  --output results/silu_quadratic_horner_quality.json
```


## Reference communication and accuracy comparison

The figures in this section describe the initial witness implementation,
before the check, polynomial, and RMS reduction improvements below. Run the
comparison on the current implementation from the repository root:

```sh
python -m benchmarks.witness_activations --elements 17 257 \
  --output results/witness_activations.json
python -m pytest tests/test_witness_activations.py tests/test_witness_activations_gpu.py -q
```

The [report](../results/witness_activations.json) measures four live CPU `Party`
roles over `IdealNetwork`, with exact-floor truncation. Bytes sum outgoing
payloads across all roles, including consistency hashes, witness input, domain
checks, residual checks, and acceptance opening. Existing activation inputs
are already shared. Final output opening is excluded for every method. These
are reference-runtime communication measurements, not optimized GPU transport
costs or distributed-network latency.

The numerical baselines assume caller-established domains; witness recipes
privately check them. Mixed polynomial baselines use ring32/Q14 and the
[previously selected fits](activation-approximation-evaluation.md), whereas
numerical and witness paths use ring64/Q24. These differences are included in
the comparison rather than presented as an isolated algorithm speedup.

For 257 elements, global payloads in decimal MB are:

| Function | Numerical baseline | With witnesses | Reduction | Selected mixed polynomial |
| --- | ---: | ---: | ---: | ---: |
| Reciprocal | 20.456 | 5.741 | 71.9% | — |
| SiLU | 53.813 | 39.098 | 27.3% | 3.555 |
| GELU | 69.956 | 55.242 | 21.0% | 4.192 |
| SwiGLU | 55.518 | 42.239 | 23.9% | — |

Witness ingestion contributes 4,240 bytes to each 257-element witness row.
Verification and recomputation dominate its total. The selected mixed
polynomials remain substantially cheaper than witness-assisted numerical
activation evaluation.

A separate 65,537-point exact-floor grid gives the following maximum absolute
errors against float64 PyTorch on the encoded inputs:

| Function | Numerical baseline | Honest witnesses |
| --- | ---: | ---: |
| Reciprocal | `5.960e-8` | `2.980e-8` |
| SiLU | `6.408e-7` | `6.408e-7` |
| GELU | `5.321e-7` | `5.203e-7` |
| SwiGLU | `2.423e-6` | `2.423e-6` |

On the 257-element live fixture, selected mixed SiLU and GELU have maximum
absolute errors of `0.00887` and `0.01429`, respectively. These are activation
errors on synthetic inputs, not trained-model accuracy. The dense witness
accuracy grid runs plaintext formulas; the smaller live cases additionally
verify agreement across all four parties.

The repository had no prior RMSNorm or softmax activation implementation to
measure as a before case. Their new witness recipes have these absolute costs
for two rows, with default RMSNorm epsilon `1e-5`:

| Function | Row width | Global payload (MB) | Honest maximum absolute error |
| --- | ---: | ---: | ---: |
| RMSNorm | 16 | 0.719 | `5.628e-8` |
| RMSNorm | 256 | 9.749 | `1.280e-7` |
| Softmax | 16 | 4.496 | `3.064e-7` |
| Softmax | 256 | 69.722 | `5.993e-8` |

The CPU tests exercise malformed ring words, wrong reciprocal values, negative
roots, false maxima, cancellation attempts, arbitrary witness shapes, and
session lifecycle. Direct CUDA tests compare SiLU, RMSNorm, and softmax with
the prover's exact-floor arithmetic across all four parties. Benchmark source
hashes and complete payload/error rows are retained in the JSON report.

## Auditing the full-model traffic estimate

The supplied 64-layer estimate counts multiplication output coordinates rather
than scalar multiply-accumulate instructions. That is a useful communication
model for fused dot products. With its dimensions, the counts per layer are
`77,824 + 64*S` multiplication coordinates and `14,336 + 32*S` unary evaluations.
At context 1024, the stated 143,360 and 47,104 counts are correct. At context
256, the multiplication count is 94,208, rather than 91,136.

Its per-coordinate prices describe hypothetical protocols, not the current
implementation:

- General Rep4 multiplication reshares six joint contributions: approximately
  48 bytes per coordinate in ring64 or 24 in ring32, plus consistency tags.
  The four-joint sparse operation applies to specially structured operands;
  it cannot replace every general activation multiplication.
- The existing optimized ring64/Q24 probabilistic multiply-plus-truncate path
  costs approximately 128 bytes per coordinate plus tags. The ring32 path
  with narrow corrections costs approximately 56, rather than 16.
- There is no private FSS lookup implementation in this repository. Exponential
  uses a polynomial followed by four squarings. The selected mixed SiLU and
  GELU approximations also require several multiplication/rescaling stages.
  Pricing each whole unary function as one 16- or 32-byte operation omits them.
- The estimate includes 66 normalization witness values per layer: two inverse
  roots, 32 maxima, and 32 reciprocals. Its stated 34 counts only normalization
  identities. Activation witnesses are additional: the current SiLU recipe
  uses one output witness per coordinate. Input range and maximum checks also
  touch vector coordinates.

Using the estimate's own hypothetical prices and witness overhead, its
context-128 totals would be approximately 893.94, 213.98, and 106.99 MB global
for tiers 1, 2, and 3, rather than 835, 197.1, and 98.6 MB. Correcting that
arithmetic does not establish those prices for this implementation. RMS
variance evaluation and model-dependent operations such as rotary encoding
also need explicit accounting before claiming a complete transformer budget.

A 64-token commitment chunk does not shorten attention over an existing prefix.
Sliding windows or prover-selected top-k support change dense attention unless
the model specifies that support or the protocol verifies a separate error
contract. They cannot be counted as transparent implementation savings.

The sampling guarantee also needs a defined cheating event. Independent samples
with detection probability `0.001` require 4,603 samples for 99% detection.
If each cheated example has one bad layer and the verifier checks one random
layer out of 64, detection per sample becomes `0.001/64`. Then 4,603 samples
provide about 6.94% detection, and 294,729 are needed for 99%. The claimed
64-fold layer-slicing reduction does not preserve that assurance under this
attack model.

## Private LUTs and recurring preprocessing

The static public activation table is reusable and costs no MPC communication
to construct. The secret mask and correlated lookup material are different:
they must be fresh for each scalar lookup. Reusing a mask reveals input
differences through repeated masked openings. Batching amortizes message tags,
not the fresh per-coordinate payload.

[`private_lut.py`](../rep4gpu/private_lut.py) implements a dealer-free dense
reference under the existing Rep4 protocols. Six checked vector corrections
prepare a randomly shifted table for each input. Online evaluation opens only
the masked index modulo the table size and selects shared output components.
Material is runtime-bound and single-use, including after an opening failure.
The caller remains responsible for input quantization and domain checks.

For 12-bit tables, recurring preprocessing is `6*N*4096*(ring_bits/8)+192`
bytes per batch of `N` scalar lookups. Online lookup is
`4*N*(ring_bits/8)+128` bytes. Thus the asymptotic total is **196,640 bytes per
ring64 lookup** or **98,320 per ring32 lookup**, before quantization and range
checks. The 32- or 16-byte online price is real for this construction; treating
its preprocessing as one-time is not. This backend is an explicit reference,
not the default activation path, because it increases total traffic.

The [LUT report](../results/private_lut.json) measures 32 fresh lookups per
batch on CPU and CUDA. Including tags, totals are 196,650 bytes per ring64
lookup and 98,330 per ring32 lookup. These are lower bounds on activation
traffic because input quantization and private range checks are excluded.
The dense 65,537-point accuracy check gives maximum errors of about `0.00205`
for the 12-bit SiLU table on `[-4,4]` and `0.00183` for exponential on `[-8,0]`.
These table errors are separate from the numerical-path error figures above.

```sh
python -m benchmarks.private_lut --elements 32 --output results/private_lut.json
python -m pytest tests/test_private_lut.py -q
```

Compact FSS keys could reduce preprocessing size, but the claimed zero-WAN
helper-pair generation is not specified by ordinary Rep4 PRG seeds. Helpers
can regenerate their own masks and keys; evaluators still need the corresponding
fresh key/correction material. Giving evaluators the full generation seed would
also reveal the mask. A usable replacement needs a concrete correlation
construction and malicious-evaluator checks, with both costs counted.

The [FSS preprocessing construction](https://ntt-research.com/wp-content/uploads/2022/07/Secure-computation-with-preprocessing-via-function-secret-sharing.pdf)
distinguishes this offline material from online evaluation.
[Dealerless FSS](https://dev.ndss-symposium.org/wp-content/uploads/2025-2233-paper.pdf)
provides a semi-honest construction with recurring offline work, while
[SHARK](https://eprint.iacr.org/2025/716.pdf) uses authenticated outputs to address
active cheating. An evaluator's ordinary checked Rep4 input proves consistency
of its submitted share, not correct evaluation of an FSS key. Importing a
semi-honest LUT would therefore change the verifier's security contract.

A 12-bit table also defines a quantized approximation, not exact pretrained
floating-point execution. Similarly, right-shifting individual arithmetic
shares is not secure truncation: it loses carries between share components.
Using a narrower comparison window requires a proved range and a correct
rescaling protocol; typical observed activation magnitudes are insufficient.

## Boolean packing and exact rescaling

In [`carry_tree_swar.py`](../rep4gpu/carry_tree_swar.py), each round allocates
`ceil(gate_count/binary_bits)` words and evaluates the gates with binary Rep4
multiplication. Each 64-bit word carries 64 AND gates, including gates from
different coordinates as trees shrink. Arithmetic conversion of a sign is a
separate operation with arithmetic shares.

Exact truncation computes the carry into the sign bit once, deriving the ring
carry with two final packed AND gates per coordinate. After bounds are
established, internal absolute-value signs use `f+4` bits, and the softmax zero
check uses `f+5` bits. At Q24 these are 28 and 29 bits. Original untrusted input
and witness checks remain full-width, so arbitrary high ring bits cannot alias
into an accepted narrow-domain value.

Narrowing uses the modular carry protocol. Locally shifting arithmetic share
components does not implement the same operation. High-bit alias tests check
the rejection path.
