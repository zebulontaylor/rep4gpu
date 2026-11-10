# Preparing a fixed model for private queries

[`spdz_prepared_right.py`](../rep4gpu/spdz_prepared_right.py) moves work that depends only on the authenticated model into a separate preparation step. Every query still consumes fresh correlations. This optimization applies to a fixed model whose weights are available before the query arrives.

For an authenticated model matrix `W` and a fresh authenticated matrix triple `(A, B, C=A@B)`, preparation opens and authenticates

```text
E = W - B
F = C + A @ E = A @ W.
```

The prepared handle owns authenticated shares of `A`, `F`, and a snapshot of `W`. Because `E` is public, each role computes its value and MAC shares of `F` with local field matrix products. Preparing `E` completes its MAC check even when the runtime defers checks of internal query operations.

For a query `X`, online evaluation opens the fresh masked difference `D=X-A` and returns

```text
Z       = F + D @ W
gamma_Z = gamma_F + D @ gamma_W.
```

This removes the large weight-mask opening from query latency and reduces the local matrix work. A runtime with deferred checks keeps `D` inside the fixed arithmetic circuit; it checks every pending opening before sending any unmasked output shares and checks the output before accepting it.

## Use and lifetime

All roles call the same operations with matching public shapes and identifiers:

```python
from rep4gpu.spdz_prepared_right import prepare_right_matmul, matmul_prepared_right

# Before the query arrives; weights are already authenticated shares.
prepared = prepare_right_matmul(party, weights, fresh_triple, model_id=b"layer-1")

# Once, after the query arrives.
result = matmul_prepared_right(party, query, prepared, model_id=b"layer-1")
output = party.open(result)
```

Flush any pending deferred openings before preparation; it requires a checked boundary. Preparation burns the original triple before communication. Query evaluation consumes the prepared handle before opening `D`; a failed attempt cannot restore it. Each handle belongs to its original party, session, and MAC-key object. Its owned tensor snapshots prevent later mutations of caller-supplied weights or triples from changing the prepared operation. `model_id` identifies the application snapshot; it does not prove that an authorized model supplied the weights.

Generate a new triple and prepared handle for every query or batch. Reusing `A` would expose differences between private queries. After an abort, retire the session's keys and all remaining correlations as required by the [malicious-majority protocol](malicious-majority.md#failure-and-preprocessing-contracts).

## Prepare several independent queries together

A matrix triple with `A` shaped `(Q,K)`, `B` shaped `(K,N)`, and `C` shaped
`(Q,N)` supplies Q fresh row masks for one fixed model. Open `E=W-B` once for
that batch, then partition the prepared A/F rows into independent query handles:

```python
from rep4gpu.spdz_prepared_right import partition_prepared_right

parent = prepare_right_matmul(party, weights, fresh_batch, model_id=b"layer-1")
queries = partition_prepared_right(party, parent, (1,) * Q, model_id=b"layer-1")
for query_share, prepared in zip(query_shares, queries):
    result = matmul_prepared_right(party, query_share, prepared, model_id=b"layer-1")
    output = party.open(result)
```

Every role supplies the same partition. Positive row counts must cover the
whole parent exactly; they can also group several rows into a query. The
partition consumes its parent, and each child consumes its own rows once.
Children own their A/F tensors and share the immutable owned model snapshot.
Opening labels bind the entire partition and row range, so disagreement aborts.
Mutation of shared prepared weights invalidates the children. A failed
partition attempt consumes the parent and poisons the session.

This amortizes the weight-sized opening and preparation over the queries that
actually use the batch. It does not allow a second query under an already-used
row mask. Each query also needs fresh activation correlations and private-input
masks. Charge unused or discarded material to the served queries when
reporting achieved throughput, rather than dividing by an unused capacity.

## Cost and preprocessing assumptions

Report these costs separately:

1. Native distributed OT preprocessing, including correlation checks.
2. Private model sharing and model-dependent `E`/`F` preparation.
3. Query input sharing.
4. Resident query evaluation through the final authenticated output.

The second stage requires the actual model, so it is not model-independent preprocessing. It can run before a future query, but its work and traffic recur for that query's fresh correlations. Excluding it from online latency does not remove its throughput or total-cost contribution.

The large-prime [MASCOT bridge](mascot-preprocessing.md) supplies fresh authenticated triples without a central preprocessing dealer. Preparing `E` and `F` adds no dealer. The retained small-field MAMA path has a known privacy flaw and is functional test material only. The local timing fixtures assume ideal network delivery; they do not measure four physical GPUs or GPUDirect networking.

The performance target includes amortized fresh cryptographic preprocessing.
Divide a preprocessing batch's cost by the number of distinct one-use query
packs it actually produces. Static compilation, connection setup, and model
sharing can be amortized across a healthy persistent session. Fresh query
correlations, model-dependent preparation, and input sharing still contribute
to throughput cost even when performed before the query arrives. Overlapping
these stages can hide latency but cannot remove their work or capacity needs.

Prepared handles also support the native adapter's optional [seven-key online selection](malicious-majority.md#seven-key-online-selection). The selection happens before the worker receives its material and is bound into the session. Preparation does not select or discard authentication coordinates itself.

### Why a final matrix check cannot replace private candidate generation

A semi-honest OT product generator followed by Freivalds verification is insufficient for a private fixed model. Malicious OT protects the transfer itself; it does not bind a receiver's choice bits to that receiver's already authenticated share of `A`.

For a scalar example with one honest model share `w_h`, a corrupt receiver can use `a_i+1` in the cross-product OT while retaining authenticated `a_i`. This changes the candidate product by `w_h`. By adding its known corrupt model shares minus a public guess to its candidate share, the coalition makes the total product error `W-guess`. Authenticating that candidate does not prove its multiplication relation. A later repeated Freivalds check accepts when `W=guess` and rejects with high probability otherwise. Its accept/abort result therefore reveals a predicate of the private model, even if the failed random mask is retired before any query arrives.

[MASCOT](https://eprint.iacr.org/2016/505) distinguishes product correctness from privacy: inconsistent OT inputs can cause selective failures, and its triple generator adds consistency and privacy-amplification machinery. The prepared-model interface consequently requires fresh triples satisfying both authenticated correctness and mask privacy under the selected preprocessing protocol. A late matrix-identity check alone cannot establish that contract for an input-inconsistent candidate generator.
