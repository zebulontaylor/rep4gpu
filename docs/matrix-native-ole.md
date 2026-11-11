# Experimental native OLE matrix integration

This runner connects the [F127 matrix-MAC reference](matrix-spdz-reference.md)
to actual native chosen-input OLE over mutually authenticated TLS. Four Python
role processes generate their own keys and random matrix shares. Each role
launches only its local native endpoints and reads only their private output
files. It never calls the trusted test VOLE oracle or central candidate dealer.
The backend remains experimental, test-only, and privacy-unreviewed; the
production provider gate is unchanged and `security_target_met` remains false.

## Protocol path

For a candidate, role `i` samples its own stacked `Ahat_i`, `B_i`, and `R_i`.
Native OLE supplies additive shares of every cross-party outer product needed
for `sum(Ahat_i) @ sum(B_i)`. Each role accumulates its local product, outgoing
vector-side shares, and incoming scalar-side shares, then splits the stacked
result into the retained and sacrificed row blocks.

Authentication uses one vector per fixed key scalar containing the columns of
**all** matrices and all fresh authentication masks. The adapter normalizes
native `L + R = delta*x` as outgoing `u=R`, incoming `w=-L`; the reference forms
`X_i@V_i + outgoing_u - incoming_w`. It then performs masked consistency,
shared-right-factor sacrifice, and separately authenticated transpose checks.

Private input masks follow Protocol 10 of the matrix-ring paper. Only the input
owner samples the complete random mask and distributes private additive shares.
The OLE tags authenticate that owner's complete masks, not an unchecked sum
reported by recipients. Masked consistency verifies distribution before the
mask is used on a real input. Online multiplication checks `D`, `E`, the masked
transpose opening, and the final output.

The baseline creates separate native endpoint processes for each phase. TLS
binds roles, peer names, epoch, ordered direction, public dimensions, and OLE
instance identifiers. Every call consumes fresh raw correlations and fresh ROT
indices. A failed run is not retried with its old keys or private work files.

## Run and inspect

Use the pinned native dependencies described in
[the direct-OLE guide](../native/matrix_vope/DIRECT-OLE.md) and
[TLS transport guide](../native/matrix_vope/TLS-IO.md). From the repository root:

```sh
.venv/bin/python native/matrix_vope/build_direct_ole_tls.py
.venv/bin/python -m scripts.run_matrix_native_ole \
  --width 2 --statistical-security-bits 40 \
  --work work/matrix_native_ole/fresh_m2 \
  --output results/matrix_native_ole_fresh_m2.json
```

The work directory must be new. Its role-private requests and outputs contain
secret shares; public JSON reports include only final outputs, public sizes,
return codes, source/binary hashes, and timing/counter summaries. The known test
matrices are supplied only to input owners 0 and 1. The shared OS account does
not isolate mutually hostile processes; deployments need independent private
execution domains and credential provisioning.

## Functional evidence

Both [m=2](../results/matrix_native_ole_m2_retry.json) and
[m=4](../results/matrix_native_ole_m4.json) completed with four successful role
exits, exact independent matrix-product outputs, unchanged source hashes, and
all correlations consumed. The runs used a labeled 40-bit statistical setting
with one F127 key column and the configured whole-epoch check budget.

| Measured stage, maximum role wall time | m=2 | m=4 |
| --- | ---: | ---: |
| Candidate generation, including native setup | 161.5 ms | 224.8 ms |
| Authentication, sacrifice, transpose checks | 228.7 ms | 239.7 ms |
| Private input-mask preparation | 167.9 ms | 173.4 ms |
| Checked multiplication and output | 22.1 ms | 28.0 ms |
| Actual worker-process wall time | 2,796.0 ms | 2,835.4 ms |

Each run started 60 native endpoint processes: 84 two-party OLE calls for m=2
and 264 for m=4. Public native summaries include context setup, fresh ROT
counts, bytes, and call phases; local output vectors stay private. The report
also retains static provisioning, input sharing, and same-operation CPU clear
comparators. These are cold shared-host functional runs, not ideal-network GPU
measurements or token-throughput claims. Persistent peer contexts are a
separate optimization; these baseline artifacts and their binary stay intact.
