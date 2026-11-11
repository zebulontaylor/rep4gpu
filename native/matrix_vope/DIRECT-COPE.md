# Direct checked seeded-vector COPE

This experiment has an accepted-branch privacy defect and must not supply private
model scalars or preprocessing material. A malicious programmer can change one
bit-indexed COPE correction and learn the corresponding scalar bit from whether
the check accepts. Fresh masking correlations and whole-context retirement do
not remove this leakage. The implementation remains a functional cost reference
while a malicious-input construction is selected.

`direct_seeded_cope.cpp` provides a short-vector reference without a regular-noise
assumption. The programmer expands one private seed into a vector `v` over
`F_(2^127-1)`. Each call uses a fresh scalar `Delta`, fresh chosen ROTs, and a
fresh masking value for its polynomial consistency check. The scalar holder
receives `K`; the programmer receives `(v,M)`, with `M=K+Delta*v`.

Build and exercise disposable fixtures from the repository root:

```sh
python native/matrix_vope/build_direct_cope.py
python native/matrix_vope/check_direct_cope.py --fault cope-high --fault base-response0 --fault base-response1 --fault base-mac
```

The build requires the pinned EMP libraries described in [README.md](README.md).
Compiler logs, source hashes, private fixtures, and the public test summary go to
`work/direct_seeded_cope`. The verifier reads the private outputs only after both
processes finish. The protocol itself sends no debug opening of those values.
Private output JSON is a test fixture, not a consumable preprocessing format.

The native interface is:

```text
direct_seeded_cope <1|2> <port> <private-output.json> <sessions> [length] [fault]
```

Role 1 holds the scalar; role 2 holds the vector seed. Length is public and
limited to 1–4096; each context accepts 1–32 calls. Both roles bind the field,
length, session count, and call index before COPE. A failed check retires the
entropy source and terminates the context. `EMP_TEST_MODE` is rejected before
entropy or OT construction. Canonical field arrays use 16-byte little-endian
residues; residues at least `p` are rejected. The `cope-high` fault forces scalar
bit 64 to one and corrupts that correction bit; it aborts. The diagnostic
`cope-high-zero` forces the same bit to zero and injects the same corruption; it
accepts, exposing the selective-abort channel.

Output retains `security_reviewed=false`. The component field-check bound is
`length/(2^127-1)` per call. The following calculation covers check-error components only; it does not
address the privacy defect above. A conditional statistical-40 budget allows at most
`2^20` independent contexts with 32 calls each if the 128-base-OT bootstrap has
at most two extraction events bounded by `2^-63` each. Their aggregate is at most
`2^-42`; field checks add less than `2^-90` at the maximum length. The CSW
extraction theorem, actual bootstrap event count, and malicious OT/equality
composition still need review. AES-128 and the underlying curve/OT assumptions
are nominal computational-128 components, not a proof for this whole program.

This reference generates one checked correlation per vector entry. It removes
the uncertain short-vector LPN parameters while giving up the expansion needed
for the matrix paper's asymptotic saving. Compare setup plus all fresh calls,
not just a warmed call, when assessing its cost.
