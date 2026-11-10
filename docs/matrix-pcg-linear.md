# Fusing matrix-correlation expansion

`rep4gpu.matrix_pcg_linear` implements the local linear algebra for a proposed
matrix-correlation producer. It does not implement cryptographic seed
generation, oblivious transfer, authentication, or consistency checks. Its
synthetic benchmarks are component timings, not preprocessing or inference
latency.

The construction follows the VOPE reduction in Appendix E of
[Dishonest Majority Multiparty Computation over Matrix Rings](https://eprint.iacr.org/2023/1912).
For a fixed public linear map `H`, an outer-product correlation has local
outputs `H @ Y` and `-H @ Z`. Matrix multiplication adds these correlations
over the inner dimension and peer pairs. Linearity lets each party sum its raw
terms before applying `H`:

```text
S_i = E_i @ B_i + sum(Y_i) - sum(Z_i)
C_i = H @ S_i
A_i = H @ E_i
```

Here each column of `E_i` has one nonzero entry in each regular block. The
identity `A_i @ B_i = H @ (E_i @ B_i)` also moves the local product into the
raw accumulator. Computing `E_i @ B_i` as sparse outer products avoids forming
a dense `E_i` for that step. The separate `A_i` expansion remains necessary for
authentication.

All combined terms must use the same public map, field, and raw dimensions.
This rearrangement does not remove any VOPE consistency check, authentication,
or sacrifice. A protocol integration must finish those checks before treating
the resulting matrix as consumable preprocessing. The linear map and regular
noise parameters also need an appropriate LPN security analysis; arbitrary
test matrices and sparsity values do not supply one.

## Local API

`PublicLinearMap` owns a canonical snapshot of `H` and binds its field, shape,
and contents to a SHA-256 identity. `RegularSparseColumns` owns the positions
and nonzero coefficients of `E`. Construction validates that every position
belongs to its designated block.

`LinearExpansionAccumulator` starts with zero raw shares. `add_local` adds the
sparse local product. `add` adds or subtracts one raw matrix; `add_batch` handles
up to 64 matrices while retaining every field addition. `finish` seals the
accumulator and applies `H` once. Every call supplies the map identity.

The current arithmetic backend supports only `p = 2**31 - 1`. Sparse products
are reduced before atomic addition. One position per block and fewer than
`2**31` columns bound each raw row sum below `2**62`. Batch additions reduce
each input before summing at most 64 terms in int64. A larger-field backend
needs different multiplication and accumulation bounds.

## Reproduce the component experiment

From the repository root, with the CUDA device available:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m pytest -q \
  tests/test_matrix_pcg_linear.py
.venv/bin/python -m benchmarks.matrix_pcg_linear_bench \
  --output artifacts/matrix_pcg_linear_bench.json
```

The tests compare local fused and unfused shares and reconstruct a four-party
matrix product from synthetic valid correlations. The benchmark includes all
`6K` signed raw additions for one party, its local product, and the required
final expansions. It cycles eight resident random buffers; it excludes
cryptographic generation, checks, communication, and common map/seed setup.
The benchmark records both streamed and batched accumulation because batching
can be slower when its working set exceeds the device cache.
