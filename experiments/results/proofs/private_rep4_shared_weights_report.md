# Private-weight Rep4 audit traffic

The CPU toy audit uses four width-4 affine/ReLU layers, one input row, and
ring64 arithmetic without fixed-point rescaling. All weights and biases are
secret-shared among the four verifiers. The prover retains the plaintext model;
the input remains public. Each verifier receives only its own three model
components, authenticated against the trusted model owner's manifest.

Both runs used five separate processes over mutually authenticated loopback TLS.
All five processes exited zero and agreed on output `[[91, 0, 0, 2]]`.
The full run checked transitions `[0, 1, 2, 3]`; the sampled run checked
`[0, 1]`. A sampled acceptance establishes only the selected transitions.

| Traffic category | Full audit (bytes) | Two-transition audit (bytes) |
| --- | ---: | ---: |
| Prover ↔ committee | 40,724 | 35,508 |
| Verifier mesh, including role identities | 98,366 | 67,726 |
| **Total online traffic** | 139,090 | 103,234 |
| Transition checks, acceptance, and output (subset of mesh) | 75,488 | 44,848 |
| One-time model provisioning file payload | 17,204 | 17,204 |

Online totals sum outgoing application bytes across all five roles, counting
each transmission once. They include JSON messages, share encodings, commitment
paths, agreement, randomness setup, MPC exchanges, result messages, and framing.
They exclude TLS handshakes and record overhead, TCP/IP overhead, certificate
provisioning, and model-file delivery. Global sent and received counts match.
These are toy protocol measurements, not transformer traffic estimates.

One-time model provisioning consists of four 3,817-byte private role bundles
and four 484-byte manifest copies: 17,204 bytes in total. This is the serialized
file payload size, not a measured transfer. Certificates, public input, deployment
configuration, and the prover's plaintext model delivery are outside that figure.
The model bundles can be reused for the same approved model. Fresh trace shares,
challenges, and MPC seeds are generated per audit. Sampled prover-link traffic
can vary with the number of distinct requested boundaries.

From the repository root, reproduce with fresh output directories:

```sh
.venv/bin/python -m rep4gpu.proofs.private_audit_service demo \
  --output-dir /tmp/siv-shared-weights-full
.venv/bin/python -m rep4gpu.proofs.private_audit_service demo --samples 2 \
  --output-dir /tmp/siv-shared-weights-sampled
```

Each directory contains `summary.json`, five role logs, and per-role
`result.json` and `traffic.json` files. A failed process produces `failure.json`
and the launcher exits nonzero.

- [Full run summary](private_rep4_shared_weights_full/summary.json)
- [Sampled run summary](private_rep4_shared_weights_sampled/summary.json)
- [Protocol and model provisioning](../../../docs/protocol/private-rep4-audit.md)

Validation covered secret-share reconstruction, component tampering and bundle
substitution, full-ring private arithmetic, full and sampled TLS audits, role
file isolation, corrupted traces, substituted prover weights, tampered Merkle
paths, and traffic conservation across the five processes.
