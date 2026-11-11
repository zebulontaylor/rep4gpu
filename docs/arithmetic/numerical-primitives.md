# Private numerical primitives

[Documentation index](../README.md)

`NumericalFixedPoint` evaluates exponential, reciprocal, and error-function
algorithms on Rep4 shares, then composes the original SiLU, GELU, and SwiGLU
formulas. It uses ring64 with 20 through 24 fractional bits. Inputs must already
have that ring and scale; it does not convert existing ring32/Q14 shares.

```python
from rep4gpu import NumericalFixedPoint

# party is a ring64 runtime; x and value are arithmetic shares at Q24.
fp = NumericalFixedPoint(party, fractional_bits=24)
silu_output = fp.silu(x)
gelu_output = fp.gelu(x)
swiglu_output = fp.swiglu(x, value)
```

Every result remains shared. Only the caller's explicit `party.open(...)`
reconstructs an output. Iteration counts and communication schedules depend on
public configuration, not private inputs. The runtime's checked multiplication,
comparison, and truncation protocols supply the underlying Rep4 operations.

## Domains and algorithms

| Method | Decoded input domain | Computation |
| --- | --- | --- |
| `exp(x)` | `[-9,9]` | Degree-12 Taylor evaluation at `x/16`, then four squarings |
| `reciprocal(x)` | `[1,2]` | Six Newton steps `y = y*(2-x*y)`, starting at `1/2` |
| `erf(x)` | `[-3,3]` | A&S 7.1.26 using private absolute value, reciprocal, and exponential |
| `silu(x)` | `[-4,4]` | `x/(1+exp(-x))`, evaluated using `exp(-abs(x))` for stability |
| `gelu(x)` | `[-4,4]` | `x*(1+erf(x/sqrt(2)))/2` |
| `swiglu(gate, value)` | Both inputs in `[-4,4]` | `silu(gate)*value` |

The caller must establish these ranges. The functions neither clip nor privately
check their inputs. The constructor rejects other rings and fractional
precisions, and the numerical methods reject binary shares. Inputs outside the
stated domains invalidate the numerical and intermediate-range guarantees.

For SiLU, `exp(-abs(x))` stays in `[0,1]`, so its reciprocal denominator stays in
`[1,2]`. A private sign selection chooses the appropriate sigmoid expression.
For GELU, `x/sqrt(2)` stays inside the erf domain. The erf denominator
`1 + 0.3275911*abs(x)` also stays in `[1,2]`.

These are finite-precision numerical evaluations, not correctly rounded or
bit-identical replacements for PyTorch. Exponential uses a local series after
range reduction. Error-function evaluation uses the rational/exponential
approximation in [Abramowitz and Stegun, equation 7.1.26](https://digital.library.unt.edu/ark:/67531/metadc40302/m1/315/ocr/).
Its real-arithmetic error bound is `1.5e-7`; coefficient quantization and protocol
rounding add error. The [NIST definition of erf](https://dlmf.nist.gov/7.2.E1)
identifies the function used in the GELU formula.

The existing `FixedPoint.gelu` and `FixedPoint.silu` methods retain their
polynomial activation approximations. Select `NumericalFixedPoint` explicitly
to use these numerical algorithms.

## Precision and intermediate range

With `S = 2**fractional_bits`, the largest raw products occur in exponential's
last squaring near `x=9`. The implementation bounds them by `8105*S*S`, below
`2**62` for Q20 through Q24. The other primitives have much smaller products.
This satisfies the strict quarter-ring requirement of probabilistic truncation,
as well as signed ring64 storage. Q25 is rejected because this range argument
no longer holds.

Each rescaling follows the runtime's rounding policy: floor or probabilistic
floor/ceiling. A larger fractional precision reduces quantization error but
does not change the mathematical input domain. Error near a function's zero
is best interpreted as absolute error; relative error can be large there.

## Validation and measurements

The CPU quality report samples 65,537 encoded inputs across each declared
interval. At Q24 with floor rounding, maximum absolute errors are:

| Function | Maximum absolute error |
| --- | ---: |
| Exponential | `9.013e-3` |
| Reciprocal | `5.960e-8` |
| Error function | `2.59e-6` |
| SiLU | `6.408e-7` |
| GELU | `5.322e-7` |
| SwiGLU | `2.424e-6` |

Exponential's largest absolute error occurs near the large positive end of its
interval. The report includes relative error, Q20 results, and conservative
interval bounds for probabilistic rounding. The SwiGLU fixture pairs a gate
sweep from -4 to 4 with a value sweep from 4 to -4; it is not an exhaustive
check of every possible gate/value pair.

Integer-oracle tests check products before int64 wraparound, including floor,
ceiling, and randomized rounding. GPU checks compare all parties with the
integer oracle, then require recorded replay to reproduce both output and
outgoing messages. GPU benchmarks include fresh masks and checked output
opening, with inputs already shared. These are primitive and activation costs,
not complete FFN or transformer inference.

## GPU timing

The RTX 4060 Laptop GPU run uses Q24, probabilistic rounding, and three timing repetitions. These are isolated-party phase estimates that exclude network latency and bandwidth. Actual execution places all four roles on one GPU. Neither column represents distributed-network latency.

| Function | 8,192-element estimate | 65,536-element estimate | 8,192-element four-role time | Communication rounds |
| --- | ---: | ---: | ---: | ---: |
| `exp` | 6.134 ms | 8.001 ms | 32.140 ms | 68 |
| `reciprocal` | 4.265 ms | 5.772 ms | 23.691 ms | 50 |
| `erf` | 15.273 ms | 17.467 ms | 67.533 ms | 158 |
| `silu` | 12.352 ms | 15.973 ms | 58.941 ms | 136 |
| `gelu` | 15.647 ms | 18.185 ms | 73.254 ms | 168 |
| `swiglu` | 13.148 ms | 15.029 ms | 61.979 ms | 140 |

The 8,192-element probabilistic runs have maximum observed absolute errors of `6.132e-7` for SiLU, `4.157e-7` for GELU, and `1.703e-6` for SwiGLU. Each result also satisfies propagated integer bounds covering all permitted truncation outcomes. These observed errors are sample measurements, not certified global error bounds.

The 1,048,576-element batch-128 primitive case has not been measured.
[Complete numerical FFN forwards](numerical-forward.md) have been measured at
batches 1 and 8, including private projections and the SwiGLU gate product.
Ring64/Q24 numerical costs are not directly interchangeable with the earlier
ring32/Q14 polynomial results. The reported CUDA peak fields describe process
allocation during each case; they are not an isolated deployment-memory requirement.

Validation passed 14 numerical CPU tests, six private GPU protocol tests, and 50 existing activation and approximation checks.

## Reproduce

From the repository root with the documented Python environment activated:

```sh
# CPU accuracy and conservative rounding intervals.
python -m experiments.benchmarks.activations.numerical_quality

# CUDA: exact floor outputs against the integer oracle.
python -m experiments.benchmarks.activations.numerical_primitives --elements 9 --rounding floor --repeats 1 \
  --output experiments/results/activations/numerical_primitives_floor.json

# CUDA: probabilistic timing at two activation sizes.
python -m experiments.benchmarks.activations.numerical_primitives --elements 8192 65536 --repeats 3 \
  --output experiments/results/activations/numerical_primitives_decode_scaling.json

python -m pytest tests/unit/arithmetic/test_numerical.py tests/unit/arithmetic/test_numerical_protocol.py tests/unit/arithmetic/test_activations.py -q
```

The protocol tests require CUDA and are skipped when it is unavailable. The CPU
oracle tests still run. Benchmark reports retain source hashes, individual live
timing samples, payload sizes, round counts, and accuracy measurements.

- [Implementation](../../src/rep4gpu/arithmetic/numerical.py)
- [CPU quality results](../../experiments/results/activations/numerical_primitives_cpu_quality.json)
- [GPU scaling results](../../experiments/results/activations/numerical_primitives_decode_scaling.json)
- [Exact-floor GPU results](../../experiments/results/activations/numerical_primitives_floor.json)
- [GPU test log](../../experiments/results/activations/logs/numerical_primitives_pytest.log)
