# Rep4 golden validation report

The installed MP-SPDZ executable passed the matrix and fixed-point reference with both `-R 64 -Z 2` and `-R 64 -Z 4`. All four parties exited zero in both runs, and all twelve observed outputs matched independently calculated expectations. The executable and source hashes, exact compiler arguments, expected/observed values, and return codes are recorded in the linked summaries.

| MP-SPDZ split | Global application payload | Party 0 payload | Summary |
|---|---:|---:|---|
| Two summands (`-Z 2`) | 20,742 bytes | 5,878 bytes | [summary](mpspdz/summary.json) |
| Four components (`-Z 4`) | 23,778 bytes | 6,870 bytes | [summary](mpspdz_z4/summary.json) |

These totals include the complete program's input, preprocessing, computation, opening, and checks. They do not measure TCP/TLS overhead. The reference executable ran four parties with TLS using copies of its installed local test certificates; neither run changed the original MP-SPDZ tree.

The matrix case multiplies a 2-by-3 matrix by a transposed 2-by-3 matrix. Operands include signed 64-bit limits and products that wrap modulo `2^64`. Eight fixed-point cases test the Q12 rounding boundary, negative values, both 64-bit signed limits, binary sign extraction, and ReLU.

The independent deterministic oracle passed 200 randomized ring and Boolean multiplication cases, replicated output consistency, and saved fixture reproduction. The runtime then matched every serialized per-edge payload and digest byte against all three saved fixtures under injected masks. Every local output component matched as well. Input and opening trace tests checked all four input owners, correction routing, empty/nonempty digest order, two separate input hash rounds, and opening's separate data and hash rounds.

The golden suite passes with:

```sh
python -m pytest tests/integration/test_rep4_oracle.py tests/integration/test_runtime_golden.py -q
```

The audit found that the full-word multiplication schedule and core input/open payloads match the MP-SPDZ mapping. Conversion preserves ring semantics while deliberately using a different circuit schedule: four-component carry-save A2B and parity-circuit B2A. Exact transport interoperability, MP-SPDZ PRNG stream compatibility, and whole-program traffic equality are outside this validation. The runtime's raw TCP transport does not provide authenticated confidential channels. Its optional mutual-TLS path and distributed correlated-seed setup address those deployment requirements; the one-malicious-party protocol claim still depends on using those mechanisms correctly.

## Native 32-bit executable

An isolated source copy was built with `-DRING_SIZE=32`; [build.json](mpspdz_ring32/build.json) records the configuration, compiler, libsodium header archive hash, and shared library. All protocol source fingerprints match the original snapshot. The original executable and source tree were unchanged.

The native 32-bit matrix outputs matched immediately. The snapshot's optimized masked `sint.bit_compose(binary_bits)` path returned incorrect Q12 outputs while all four parties exited zero. The original failed outputs and compiler logs are preserved in [optimized_compose_failure/summary.json](mpspdz_ring32/optimized_compose_failure/summary.json). This is a reference-path limitation; successful process exit alone is insufficient to validate the binary conversion.

The 32-bit reference therefore converts each Boolean bit with the generic `sint.conv` path and composes the resulting arithmetic bits. Both two-summand and four-component splits then pass all twelve matrix/Q12 outputs with four zero exits. Each summary identifies the conversion mode, hashes the generated program, and stores its exact source beside the logs.

| Native 32-bit split | Global application payload | Party 0 payload | Summary |
|---|---:|---:|---|
| Two summands (`-Z 2`) | 16,204 bytes | 5,236 bytes | [summary](mpspdz_ring32_z2/summary.json) |
| Four components (`-Z 4`) | 16,844 bytes | 5,396 bytes | [summary](mpspdz_ring32/summary.json) |

The runtime's additional 32-bit deterministic multiplication fixture matches every four-byte arithmetic value, digest, directed edge, and local output component. Ring32 tests also cover full-ring products, matrix overflow, signed comparison, sign extension, truncation at shifts 0/1/12/31, and retained full-word binary transport. Reducing the arithmetic ring is valid only when the application accepts modulo-32-bit semantics or separately establishes its nonwrapping accumulator bounds.
