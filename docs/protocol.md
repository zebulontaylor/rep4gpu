# Rep4 protocol mapping

This implementation follows the four-party replicated protocol in the local MP-SPDZ `Protocols/Rep4.hpp`, `Rep4Input.hpp`, `Rep4MC.hpp`, and `GC/Rep4Secret.h`. The executable reference and source fingerprints are recorded in [`tests/golden/mpspdz/summary.json`](../tests/golden/mpspdz/summary.json). The reference executable is `/home/zeb/Desktop/mp-spdz-toy/MP-SPDZ/rep4-ring-party.x`.

The threat model is at most one malicious party among four, with abort on inconsistent messages. It is an honest-majority protocol. Hash comparisons authenticate consistency between two holders of the same value; they are not arithmetic MACs and do not provide dishonest-majority security. Security also depends on private authenticated channels, correctly shared fresh randomness, and completing all checks before releasing results. Matching arithmetic or message counts alone does not establish these properties.

## Share layout and arithmetic

Write a secret as `x = x_0 + x_1 + x_2 + x_3 (mod 2^64)`, where component `x_e` is held by all parties except party `e`. Party `p` stores three components in this order:

| Local slot | Excluded party |
|---|---|
| `0` | `(p + 1) mod 4` |
| `1` | `(p + 2) mod 4` |
| `2` | `(p + 3) mod 4` |

A public constant is placed in component `x_0`; party 0 stores zero in every slot. All arithmetic wraps modulo `2^64`. Signed interpretation occurs at interfaces and in the binary sign circuit. C++ arithmetic kernels must use unsigned values to preserve this wrapping behavior.

For multiplication, each party calculates three local contributions and stores them in a five-slot array. For `i = 0, 1`, it stores `(x[i] + x[i+1]) * y[i] + x[i] * y[i+1]` in `a[(p+i-1) mod 4]`. It stores `x[0]*y[2] + x[2]*y[0]` in `a[4]`. Each value used in the schedule below has two holders that can independently calculate it.

A dot product accumulates these three local contributions across its reduction dimension before communication. Thus a matrix product communicates per output element, rather than per scalar multiplication. The six joint inputs reconstruct the sum of all sixteen cross-products without opening the input or output.

## Joint inputs and multiplication messages

`Rep4::exchange()` executes these joint inputs in order. The last two rows both use the local `a[4]`, at different parties.

| Local value | Sender | Backup | Receiver | Outsider |
|---|---:|---:|---:|---:|
| `a[0]` | 0 | 1 | 3 | 2 |
| `a[1]` | 1 | 2 | 0 | 3 |
| `a[2]` | 2 | 3 | 1 | 0 |
| `a[3]` | 3 | 0 | 2 | 1 |
| `a[4]` | 0 | 2 | 3 | 1 |
| `a[4]` | 1 | 3 | 2 | 0 |

For a joint input value `v`, the three parties other than the receiver draw the same fresh mask `r` from their PRNG shared by that triple. They add `r` to the component excluded from the receiver. The sender and backup calculate `v-r`, add it to the component excluded from the outsider, and the sender transmits it to the receiver. The receiver adds that value to its component excluded from the outsider. The backup transmits a digest of the same packed masked values to the receiver. A mismatch aborts.

Digests use libsodium `crypto_generichash` with an unkeyed 32-byte output, equivalent to BLAKE2b-256. They cover packed value bytes in vector order. `Hash::append_final()` appends raw digest bytes and resets the hash state. `append_hashes()` orders digests by receiver's positive offset from the local party, then by sender number. The zero-length `octetStream::append(0)` calls do not append a delimiter. They finalize a partially packed bit byte.

For a batch of `n` 64-bit ring output values, the complete malicious multiplication exchange carries `48*n + 192` application payload bytes across all parties: six masked vectors of `8*n` bytes and six 32-byte digests. There are ten active directed channels. Parties 0 and 1 each send `16*n + 32` bytes; parties 2 and 3 each send `8*n + 64` bytes. These counts exclude transport lengths, TLS records, connection setup, PRNG setup, and other protocol phases. Semi-honest mode omits the six digests.

## Private inputs and opening

For input owned by party `p`, the owner draws its local slots 0 and 2 from triple-shared PRNGs, sets slot 1 to `input-slot0-slot2`, and sends slot 1 to both adjacent parties. The opposite party obtains slots 0 and 2 from its correlated PRNGs. The remaining component, excluded from the owner, is zero. The two neighbors hash what they receive and exchange digests with their opposite party. The cross-comparison detects an owner sending different residual vectors to its two neighbors. For a nonempty input batch from one owner, its two neighbors each finalize and send both receive hashes, including the hash of the empty stream. This sends two residual vectors and four 32-byte comparison digests globally. When all four owners supply nonempty vectors, the input phase sends eight residual vectors and eight digests.

For opening, each party sends its slot 0 vector to the next party, receives the previous party's slot 0 vector, and adds it to its three local components. The received component is precisely the missing component. Each party also hashes its slot 2 vector. `Rep4MC::Check()` sends this hash to the previous party and compares the hash received from the next party against the opened incoming vector. A checked nonempty opening batch of `n` ring values therefore sends `32*n + 128` application payload bytes globally. MP-SPDZ can defer this check across openings; it must complete before results are trusted.

## Binary and fixed-point semantics

`GC::Rep4Secret` uses the same layout and joint-input schedule with XOR as addition/subtraction and bitwise AND as multiplication. This is a packed Boolean domain, not multiplication in `GF(2^64)`. The valid lane mask must be applied to outputs and complements. Full words pack as eight bytes on the local little-endian platform. For `n_bits >= 8`, MP-SPDZ packs `ceil(n_bits/8)` bytes; for fewer than eight bits it packs consecutive values into bit fields, then aligns at the joint-input boundary.

The two-summand arithmetic-to-binary split (`-Z 2`) forms local arithmetic sums of adjacent components: slots 1+2 for parties 0 and 2, slots 0+1 for parties 1 and 3. It transposes these words into bit planes and jointly inputs the two partial sums using tuples `(0,1,3,2)` and `(2,3,1,0)`. A Boolean carry circuit adds them modulo `2^64`. This is necessary because XORing the two words would lose carries.

The small Q12 executable golden fixture computes `biased = accumulator + 2048 + (bias << 12)` modulo `2^64`, decomposes all 64 bits, retains bits 12 through 63, and interprets the retained 52-bit word as signed. Its explicit 2048 offset tests half-up rounding. The runtime's default `FixedPoint` operations and trained-model golden instead use signed floor rounding without that offset, as defined in the [activation guide](activations.md). ReLU clears every retained bit when bit 63 is set. Binary-to-arithmetic conversion must preserve the privacy of every bit; reconstructing binary shares to one party changes the privacy contract.

The older 63-bit decomposition has different wraparound behavior: it drops the original sign bit and interprets bit 62 as sign. It must not be substituted for the 64-bit circuit on inputs near the ring boundary.

## Executable references

Run these commands from the repository root:

```sh
python3 scripts/golden_rep4.py --output tests/golden/rep4_transcripts.json
python3 scripts/golden_mpspdz.py
python3 scripts/golden_mpspdz.py --split 4 --output tests/golden/mpspdz_z4 --port 18644
python3 -m unittest discover -s tests/golden -p 'test_*.py'
```

The first command independently exercises the local products, all six joint inputs, replicated output consistency, BLAKE2b digests, overflow, and packed Boolean full-word operations. Its masks are deterministic test masks, not MP-SPDZ's random stream; the resulting full transcript is a reproducible fixture rather than a byte-for-byte capture of an MP-SPDZ run.

The second command compiles [`rep4_reference.mpc`](../tests/golden/rep4_reference.mpc) in an isolated temporary directory with the installed MP-SPDZ compiler (`-R 64 -Z 2`), then runs four actual Rep4 parties using TLS. It copies the installed local test certificates into that directory and leaves the original MP-SPDZ tree unchanged. It checks a 2-by-3 matrix product against a 2-by-3 transposed matrix, including ring overflow, and eight binary Q12/sign/ReLU cases. Compiler and party logs, expected and observed output, source fingerprints, and all four exit codes are saved under `tests/golden/mpspdz/`. The process returns a failure code when compilation, a party, or output comparison fails.

## Runtime audit and circuit differences

[`test_runtime_golden.py`](../tests/golden/test_runtime_golden.py) injects the fixture masks into each runtime party and compares every serialized multiplication edge byte-for-byte, including digest bytes and payload order. It also compares all three local output slots at all parties. Private-input tests verify correction routes, both empty/nonempty hash streams in their two separate rounds, and opening's data round followed by its hash round. The runtime completes opening checks eagerly; MP-SPDZ's default can defer them until `Check()`. Socket framing and cryptographic PRNG streams are implementation-specific, so the protocols do not interoperate on the same sockets.

The runtime's activation circuits intentionally use a different schedule from the `-Z 2` reference. `conversion.a2b()` embeds all four arithmetic components as binary shares in their existing ownership slots, corresponding to MP-SPDZ's supported four-component split (`-Z 4`). Two carry-save stages and a seven-round parallel-prefix adder form the binary sum. This uses nine multiplication exchanges without opening a word. `conversion.b2a()` bit-decomposes each local component, embeds those bits as arithmetic shares, and computes their parity using `a XOR b = a+b-2ab`. Two batched arithmetic multiplication exchanges form the four-component parity, then a local weighted sum recomposes the signed or unsigned result. This replaces MP-SPDZ's daBit conversion; its preprocessing and communication counts differ.

The executable harness supports `--split 4` against the same twelve output checks. Its output belongs in a separate directory, such as [`tests/golden/mpspdz_z4/`](../tests/golden/mpspdz_z4/summary.json). Whole-program traffic from either reference measures that MP-SPDZ program, rather than the runtime's optimized circuit schedule. See the [validation report](../tests/golden/report.md) for recorded results.

## Optional 32-bit arithmetic ring

`Party(..., bits=32)` selects arithmetic modulo `2^32`; the default remains 64 bits. Arithmetic shares use canonical unsigned low-32-bit values in `torch.int64` tensors. Each arithmetic input, joint-input, or opening component packs as four little-endian bytes, and hashes cover those four-byte values. A batch of `n` multiplication outputs therefore sends `24*n + 192` application payload bytes globally. Opening sign-extends the reconstructed bit 31 into the public `int64` result.

The lower-level `Party` defaults to 64-bit packed binary words and eight-byte serialization in either arithmetic ring. With arithmetic ring32, `binary_bits=32` instead selects four-byte binary words. `InferenceParty` and `DeviceInferenceParty` default their binary width to the arithmetic width. The 32-bit conversion circuit retains low bits 0 through 31, uses bit 31 for sign, and stops the carry-prefix network at distance 16. Binary-to-arithmetic recomposition returns a value modulo `2^32`. This full-word binary transport deliberately differs from MP-SPDZ's compact lane packing for Boolean batches shorter than 64 lanes.

Weight preparation and local matrix-product kernels receive the selected ring width. Prepared weights cannot be reused with a different arithmetic ring. Reducing the ring width changes wraparound and the representable fixed-point range: a 32-bit Q12 accumulator must remain within signed 32-bit bounds to agree with a nonwrapping 64-bit calculation. The range contract must be established for the selected model and input bounds; selecting the smaller ring alone does not preserve inference semantics.

The runtime supports mutually authenticated TLS transport and distributed triple-seed setup through `rep4gpu.setup.correlated_seeds()`. The setup uses commit/reveal contributions and consistency checks among each triple's holders. `make_seeds()` remains a trusted-dealer helper for tests. Raw TCP transport provides neither confidentiality nor authentication; use the TLS path when applying the protocol's network security assumptions. These implementation choices have functional and transcript tests, rather than a full security audit.

For the installed snapshot's native 32-bit golden, use the generic per-bit conversion path: its optimized masked `sint.bit_compose(binary_bits)` path fails the recorded 32-bit Q12 cases. `scripts/golden_mpspdz.py --ring 32` selects the generic path and records the generated source. The [validation report](../tests/golden/report.md) preserves both the failing optimized reference and successful generic references.
