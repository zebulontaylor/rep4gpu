# Authenticated matrix product experiment

`rep4gpu/spdz.py` simulates authenticated Beaver matrix multiplication over the
prime field with modulus `2**31 - 1`. It keeps four parties and a trusted dealer
in one Python process. It supports exact CPU and CUDA matrix arithmetic and
checks authentication before using an opened value. This experiment does not
implement distributed malicious-majority inference and does not establish the
bonus goal of less than 500 times clear inference latency.

## Arithmetic and authentication

An authenticated matrix has additive value shares `x_i` and five independent
sets of MAC shares `gamma[j, i]`. The dealer generates additive MAC-key shares
`alpha[j, i]` satisfying

```
sum_i gamma[j, i] = (sum_i alpha[j, i]) * x  (mod p)
p = 2147483647
```

A fresh matrix triple contains authenticated `A`, `B`, and `C = A @ B`. The
online evaluator opens and checks `D = X - A` and `E = Y - B`, then computes

```
Z = C + D @ B + A @ E + D @ E
```

Only the first party adds the public `D @ E` value term. Every party adds its
share of each MAC key times that public term to its MAC shares. The final output
is checked on opening. Each triple is consumed before the first opening, even
when authentication fails, so a failed attempt cannot reuse its masks.

The simulator checks every element and every key. It sums all parties' MAC
residuals atomically in this process. Five keys provide approximately 155 bits
of key entropy, but this is not a security claim for the simulator: it exposes
all parties' state and implements no broadcast, commitments, coin tossing, or
network adversary. In particular, a real distributed check must prevent a party
from choosing its residual after seeing honest parties' contributions.

CUDA field multiplication splits canonical field elements into 16-bit limbs.
Three exact integer matrix products recover the low, cross, and high terms.
Reduction between shifts prevents signed int64 overflow. The code checks an
inner-dimension bound before accumulating those products. Signed integer inputs
are reduced modulo the prime; fixed-point truncation, comparisons, activation
functions, and model inference are outside this experiment. A centered signed
interpretation has magnitude at most `(p - 1) / 2`. With 12 fractional bits, even
64 products of encoded `1.0` reach `64 * 4096**2`, beyond that signed range before
truncation. This field is therefore unsuitable as a drop-in replacement for the
64-bit fixed-point inference path; the measured matrix shapes do not establish
numerically correct LLM inference.

## What preprocessing includes

The trusted dealer samples field elements from OS randomness by rejection
sampling. It knows every input, triple, MAC key, and share. Its work includes
MAC-key generation, input authentication and sharing, fresh random matrix masks,
`C = A @ B`, and authenticated sharing of all three triple matrices. The benchmark
reports key setup separately and measures input sharing plus triple production
as `preprocessing_ms` for every invocation. It includes host sampling, tensor
allocation, and host-to-device copies when CUDA is selected.

The dealer does not prove its triples correct. A dealer that authenticates an
incorrect `C` can make the output wrong while all opening checks pass. Replacing
this dealer requires a proven distributed preprocessing protocol, consistency
checks or sacrifice, private input protocols, and a distributed MAC check. The
current simulator supplies none of those services.

## Run the experiment

From the repository root, run:

```sh
python -m pytest -q tests/test_spdz.py
python benchmarks/spdz_online.py --device cuda --shape 1 256 256 --repeats 5 \
  --output results/spdz_online.json
```

The JSON reports separate preprocessing and authenticated online times. Online
time includes checking the two masked openings and the final output. The clear
reference is one FP32 matrix product on the same device. Times use synchronized
host wall time, so both paths include Python dispatch and synchronization. The ratio compares
matrix-product latency only; it does not compare complete inference pipelines.
The benchmark checks the result against an exact field product.

`logical_field_contribution_bytes` counts one four-byte field element per
party's value contribution and MAC residual contribution, including the output
opening. It is an algebraic contribution count, not measured traffic: framing,
broadcast topology, commitments, transport, and communication latency are absent.
`network_ms` is `null`, and `full_malicious_majority_protocol` is `false`.

## Relationship to MP-SPDZ

The available MP-SPDZ tree implements the malicious dishonest-majority
[SPDZ2k protocol](https://eprint.iacr.org/2018/482). Its
[`Spdz2kShare`](https://github.com/data61/MP-SPDZ/blob/master/Protocols/Spdz2kShare.h)
uses values and MACs in `Z/(2**(k+s))`, rather than ordinary `k`-bit ring shares,
and selects a specialized ring MAC check. At `k=64, s=64`, this requires 128-bit
share arithmetic. Its ring check also uses random opening masks. Truncating this
state to the existing GPU ring64 kernel would discard part of the authentication
contract. This experiment therefore uses a separate prime-field design and does
not import or claim compatibility with SPDZ2k preprocessing.

MP-SPDZ's
[protocol table and preprocessing documentation](https://github.com/data61/MP-SPDZ)
distinguish its malicious protocols from semi-honest variants and explicitly
label generated fake offline data. A meaningful end-to-end comparison must run
real distributed preprocessing and online execution, or identify ideal/trusted
preprocessing and report its cost separately. The current implementation follows
the latter accounting convention while retaining the centralized simulation
limitation.
