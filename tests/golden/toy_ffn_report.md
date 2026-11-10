# Trained toy MLP: whole-network MP-SPDZ parity

Actual four-party MP-SPDZ execution matches every Q12 integer logit for all 64 contexts in the trained synthetic next-token fixture. Both native arithmetic rings pass all 512 logits and return zero from all four parties. The 32-bit and 64-bit runs use identical checkpoint and quantized fixture hashes and produce identical logits.

| Ring | Contexts | Checked logits | Exit codes | Global payload reported by MP-SPDZ | Evidence |
|---|---:|---:|---|---:|---|
| 32 bits | 64 | 512 | `0,0,0,0` | 3.75149 MB | [summary](toy_ffn32/summary.json) |
| 64 bits | 64 | 512 | `0,0,0,0` | 4.63574 MB | [summary](toy_ffn64/summary.json) |

The model is the saved `results/toy_next_token/model.pt` state dict for a 16 → 64 ReLU → 8 MLP. Party 0 privately inputs the one-hot contexts; party 1 privately inputs both trained weight matrices and bias vectors. The MP-SPDZ program performs a secret matrix product, adds the hidden bias scaled by 4096, discards the low twelve bits with signed floor semantics, and applies hidden ReLU. It then performs the second secret matrix product, floors by 4096, adds the output bias, and checks the final opening. There is no half-up offset.

The reference comes directly from `examples.toy_next_token.quantized_fixture`, including its exact integer accumulation and signed-ring range checks. Its maximum absolute first accumulator is 44,900,352; its maximum absolute second accumulator is 238,702,729. All encoded values, accumulators, hidden outputs, and logits fit signed ring32, so ring32 and ring64 agree without wraparound on this public trained fixture. This range proof is specific to these contexts and model weights.

The native 32-bit run uses generic per-bit Boolean-to-arithmetic conversion to avoid the snapshot's separately recorded optimized masked-compose failure. The native 64-bit run uses optimized masked composition. Both compile with four-component arithmetic-to-binary splits. The programs follow the MP-SPDZ circuits rather than the runtime's optimized carry/weighted-conversion schedule; exact output parity does not imply equal whole-network traffic.

Each evidence directory preserves the generated program, the canonical quantized fixture and context list, the exact per-owner input files, expected and observed logits, compiler and party logs, source/executable/checkpoint fingerprints, and all exit codes. TLS certificates are copied into an isolated temporary execution directory; the original MP-SPDZ tree and trained model are unchanged. The fixture is synthetic and all 64 contexts were also used for training; this result makes no held-out language-model accuracy claim.

Reproduce from the repository root:

```sh
python3 scripts/golden_toy_ffn.py
```

Use `--contexts 4` for a smaller check on contexts `(0,0)`, `(0,7)`, `(7,0)`, and `(7,7)`. The default checks every context.
