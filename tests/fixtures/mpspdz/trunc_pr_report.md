# Native Rep4 probabilistic truncation validation

Both actual MP-SPDZ executables passed the small-gap probabilistic truncation contract used by `rep4gpu.arithmetic.probabilistic.trunc_pr`: arithmetic ring width `w`, source length `k=w-1`, and shift `m=12`. The generated program invokes the native `TRUNC_PR` opcode directly, so the compiler does not substitute a different truncation circuit. Code-location logs confirm execution of `Rep4<T>::trunc_pr` and its correction multiplication.

| Ring | Checked outputs | Exact multiples | Party exit codes | Complete-program global payload | Evidence |
|---|---:|---:|---|---:|---|
| 32 bits | 304 | 112 | `0,0,0,0` | 22,432 bytes | [summary](trunc_pr32/summary.json) |
| 64 bits | 304 | 112 | `0,0,0,0` | 44,320 bytes | [summary](trunc_pr64/summary.json) |

Each ring tests nineteen signed values sixteen times under fresh masks. Cases include `-2^(w-2)+1`, `2^(w-2)-1`, exact multiples near those strict range boundaries, negative and positive nonmultiples around 4096 and 8192, and zero. Every output equals either signed floor division by 4096 or its ceiling. All exact-multiple outputs equal the exact quotient. This check validates the output support, not the statistical frequency of rounding up.

The source, compiler output, all four party logs, observed outputs with individual floor/ceiling checks, executable fingerprint, and protocol source fingerprints are preserved in each evidence directory. The 32-bit run uses the isolated native build recorded in [mpspdz_ring32/build.json](mpspdz_ring32/build.json). The original MP-SPDZ tree remains unchanged.

## Phase topology

The source audit checks the `Rep4.hpp` clauses implementing the following three phases, then compares the real Python runtime's outgoing edge byte counts against that mapping for the same 304-value batch. MP-SPDZ's native opcode execution is confirmed by logs; this is not a network packet capture.

1. Parties 0 and 1 send masked arithmetic partial sums to parties 2 and 3 respectively, and send their 32-byte hashes crosswise to parties 3 and 2. Global payload is `2*n*(w/8)+64` bytes.
2. The mask's MSB/upper values use joint-input tuple `(0,1,3,2)`; the opened masked values use `(2,3,1,0)`. Each tuple carries `2*n` arithmetic values and one digest. Global payload is `4*n*(w/8)+64` bytes.
3. One standard six-joint-input multiplication computes the product of the two MSB shares. Global payload is `6*n*(w/8)+192` bytes. The XOR correction and subtraction of the signed offset are local arithmetic.

For this batch the runtime sends 14,912 bytes across these phases in ring32 and 29,504 bytes in ring64. The complete-program MP-SPDZ totals above additionally include input, opening, checks, and other setup accounting.

The tuples, routes, hashes, and arithmetic agree with the source mapping. MP-SPDZ interleaves `(msb, upper)` for each value, while the runtime stacks the MSB and upper planes. This changes batch serialization order without changing values, roles, byte counts, or the correction formula. The runtime also batches the checked partial-opening payloads and hashes into one transport exchange. Its transport framing and PRG streams remain implementation-specific.

Run from the repository root:

```sh
python3 tools/validation/golden_trunc_pr.py
```

The command compiles both programs in temporary directories, runs four TLS parties per ring, checks their outputs, saves failure exit codes, and returns nonzero if any compilation, party, semantic check, or topology check fails. Override `--mpspdz32` or `--mpspdz64` to select another compatible native build.
