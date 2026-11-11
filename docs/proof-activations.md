# SiLU with a committed-input proof

The opt-in proof backend verifies the Q24 rational SiLU formula on inclusive
`[-16,16]`. It replaces per-coordinate MPC range and rescaling checks with a
zero-knowledge circuit and private commitment-binding checks. The four Rep4
roles keep the activation input and output shared. The external plaintext
prover receives no verifier shares or private projections.

This backend uses the experimental Bulletproofs R1CS implementation behind
the upstream `yoloproofs` feature. It is a research backend, not an audited
production replacement for the existing activation protocols. The native
[build and CLI guide](../native/silu_proof/README.md) pins the SDK, compiler,
and compatibility patch.

## Statement and numerical bounds

Each coordinate has two Pedersen commitments, to `X+16*S` and `Y+S`, with
`S=2**24`. The native circuit range-checks the shifted input, including both
endpoints, and the shifted output. It constrains the six nearest reductions
and final nearest-output relation of
[`silu_rational.py`](../rep4gpu/silu_rational.py). Quotients and remainders are
both bounded, so prime-field equations cannot substitute modular aliases for
the intended integer operations.

The [exhaustive numerical report](../results/silu_rational34_exhaustive_quality.json)
checks all 536,870,913 encoded inputs. The oracle's maximum absolute error is
`9.28304e-7`. The circuit permits either neighbor at an exact final tie; adding
one Q24 unit gives a conservative accepted-output bound of `9.87909e-7`.

Circuit chunks bind their version, total length, offset, length, and chunk
size in the proof transcript. Verification requires a complete, contiguous
partition of the committed vectors. Every party verifies the native proofs
against the same agreed commitment bundle.

## Bind the proof to live Rep4 shares

A correct proof about some plaintext vector does not establish that it is the
current shared input. The protocol therefore performs these additional steps:

1. The client sends the commitments and proofs to all four roles and supplies
   the encoded output to its checked input owner. The roles agree on the
   public bundle digest and fix the output shares before drawing challenges.
2. The roles privately share fresh coin contributions and open their sum.
   SHAKE256 expands the agreed seed into 128 rows of Bernoulli weights. Each
   role computes weighted sums of its existing input and output components
   modulo `2**64`, without exchanging the full vectors.
3. The public commitments are projected with the same weights. The client
   sends replicated scalar-field shares of the corresponding summed
   blindings. The holders check component consistency; no MPC role receives
   an unshared projection blinding.
4. The verifier lifts only the 256 projected ring words to canonical unsigned
   scalars in the Ristretto group field. For each projection it forms the
   secret group difference `G*t + H*r - C`. The generators are pinned to the
   native SDK's Pedersen generators, not selected by the client.
5. After these differences are fixed, the roles draw fresh secret field
   weights, compute one checked scalar/group dot product, and open its single
   aggregate point. Identity accepts. Otherwise the point is distributed
   independently of the individual errors under the masking assumptions.
   No individual projection or difference is opened.

The external client receives the public challenge seed only. The native
verifier and commitment-projection subprocesses receive public artifacts only;
the field/group binding operates inside each Rep4 role.

## Binding and privacy assumptions

For a fixed nonzero vector difference in ring64, condition on every Bernoulli
weight except one at a nonzero coordinate. At most one of that coordinate's
two weights makes the weighted sum zero. Thus 128 independent rows miss an
incorrect vector with probability at most `2**-128`. SHAKE expansion adds its
pseudorandomness assumption. Candidate commitments must be fixed before the
seed is drawn for this argument to apply.

The supported count is `N < 2**35`. Each committed shifted value is at most
`2**29`, so the candidate projection sums fit below `2**64`;
the live ring sums are lifted as unsigned words. Pedersen binding prevents a
different shared blinding from compensating for a different projected value
without breaking the discrete-log assumption. If any group difference is
nonidentity, the final secret-weighted aggregate is uniform under ideal field
weights and equals identity with probability `1/q`, where `q` is the Ristretto
group order. The implementation additionally relies on its domain-separated
PRSS expansion; wide scalar reduction has negligible bias.

Scalar and group products use the existing Rep4 ownership schedule: each
partial product has two holders, and every reshared payload is checked against
its backup's digest. Merely checking replicated input consistency would not
be sufficient for these multiplications. The lift checks the original full
64-bit words, including high-bit aliases; shortened binary wires are rejected.

## Communication accounting

The byte-mailbox harness counts every external-client frame, including its
length prefix, and the four-party transport counts all MPC payloads and tags.
This includes four copies of commitments and proofs, plaintext output delivery
to its input owner, checked output sharing, challenge traffic, shared projected
blindings, ring lifting, and the aggregate commitment check. Initial input
sharing and the final output opening used only for validation are excluded,
as in the existing activation benchmark.

The external client is a witness provider, not a fifth Rep4 party or a trusted
dealer. Its commitments, output, proof, and blinding shares are all checked.
The harness is an in-process transport and does not establish distributed
latency, proving throughput, or TLS deployment behavior.

For 14,336 Q24 values, the [CUDA report](../results/silu_proof.json) measures
**6,109,416 global bytes**, below decimal 10 MB:

| Traffic | Bytes |
| --- | ---: |
| External client, including all four proof copies and framing | 4,568,488 |
| Four-party MPC, including output sharing and private binding | 1,540,928 |
| Total | 6,109,416 |

All four outputs match the production rational oracle. This reduces traffic
by 83.6% from the 37,300,080-byte default exact-product SiLU benchmark.
The tradeoff is computation: [proof generation](../results/silu_proof_generation.json)
took 363.0 seconds with four workers, and colocated verification and binding
took 72.0 seconds. The private binding portion took about 1.2 seconds per role.
These timings describe this run, not a distributed deployment.

## Run the benchmark

Build the [native helper](../native/silu_proof/README.md), then run these
commands from the repository root. The Python bridge also requires libsodium
with Ristretto255 support. Use a fresh artifact directory when generating a
new proof; the helper refuses to overwrite private blindings.

```sh
mkdir -p work/silu_proof_full
python - <<'PY'
import json
from pathlib import Path
import torch
x = torch.round(torch.linspace(-4, 4, 14336, dtype=torch.float64) * 2**24)
Path('work/silu_proof_full/input.json').write_text(json.dumps({'x': x.long().tolist()}))
PY
native/silu_proof/target/release/siv-silu-proof prove \
  work/silu_proof_full/input.json work/silu_proof_full 128
python -m benchmarks.silu_proof --artifact-dir work/silu_proof_full \
  --backend cuda --output results/silu_proof.json
```

Use `--backend cpu` to run the bridge without CUDA. The report separates
client and MPC bytes and records source and public-artifact hashes. Its
execution times exclude proof generation. The traffic fixture uses `[-4,4]`;
the numerical report separately covers every Q24 input in `[-16,16]`.
