# Odd-Prime Byte-Lane CRT Protocol ($\prod_\ell \mathbb{F}_{p_\ell}$)

[`CrtPhaseGraphParty`](../src/rep4gpu/protocol/crt_runtime.py) implements an
active-secure 4-party Rep4 engine over a product of distinct odd primes
$p_\ell < 256$ (for example 3-lane `(251, 241, 239)` with $M = 14{,}457{,}349$,
or 4-lane `(251, 241, 239, 233)` with $M = 3{,}368{,}562{,}317$).

## Share representation

Each arithmetic secret $x \bmod M$ is represented by its residues
$(x \bmod p_0, \dots, x \bmod p_{L-1})$. Each party holds three `uint8` CUDA
tensors of shape `(L, *logical_shape)`. Signed integers live in
$[-\lfloor M/2 \rfloor, +\lfloor M/2 \rfloor]$.

## 1-product quadratic polynomial gates (`gpu/matmul/crt_conv.py`)

Linear-followed-by-quadratic operations (`crt_matmul`, `crt_conv1d`,
`crt_square_sum_pool`) use a degree-2 replicated polynomial evaluation:

1. **Local evaluation (1 INT8 Tensor Core product per lane)**:
   Each party $i \in \{0, 1, 2, 3\}$ locally maps its three replicated `uint8`
   shares to centered `int8` evaluations $F(i), G(i) \in [-(p-1)/2, (p-1)/2]$ and
   computes one local INT8 Tensor Core matmul / implicit-im2col 1D convolution /
   fused bias-add + square + global-sum-pool reduction.
2. **PRG masking**:
   Masked shares use an NVRTC BLAKE3-256 keyed XOF (`CHUNK_START|CHUNK_END|ROOT|KEYED_HASH`)
   with 8-byte monotonic nonces and `mul.hi.u64` 4-digit base-$p$ extraction
   (statistical distance $\le p^4 / 2^{64} \le 2.15 \times 10^{-10}$).
3. **2-collector active-security broadcast (2 rounds, 7 bytes/output/prime globally)**:
   - **Round 1**: $P_0, P_3$ send $Y(0), Y(3)$ to collectors $P_1, P_2$;
     $P_1 \leftrightarrow P_2$ exchange $Y(1), Y(2)$.
   - **Pre-Round-2 check**: Both collectors verify the degree-2 third difference
     $Y(3) - 3Y(2) + 3Y(1) - Y(0) \equiv 0 \pmod{p_\ell}$ *before* sending Round 2,
     compute leading coefficient $C = (Y(2) - 2Y(1) + Y(0)) \cdot 2^{-1} \bmod p_\ell$,
     $D_{\text{all}} = \text{BLAKE3}(\text{BLAKE3}(Y(0)) \parallel \text{BLAKE3}(Y(3)))$,
     and $D_C = \text{BLAKE3}(C)$.
   - **Round 2**: $P_1$ sends $D_{\text{all}}$ to $P_0, P_2$ and
     $C \parallel D_{\text{all}}$ to $P_3$; $P_2$ sends $D_{\text{all}}$ to
     $P_0, P_1$ and $D_C \parallel D_{\text{all}}$ to $P_3$.
   - All parties verify matching $D_{\text{all}}$ (and $P_3$ verifies
     $\text{BLAKE3}(C) == D_C$) before any dependent send or output release.

## Active-secure CRT truncation + exact ReLU (`gpu/bits/crt_relu_gpu.py`)

`crt_trunc_relu(party, x, shift)` evaluates $\text{ReLU}(\text{trunc\_pr}(x, s))$
directly from CRT shares back into CRT shares for inputs satisfying
$|x| < \lfloor M/4 \rfloor$:

1. **1-round CRT masking**: Mask $x_{23}$ with correlated CRT mask $r$ (checked
   via BLAKE3 backup tag), giving exact wrap indicator
   $w = \mathbb{1}[r > M/2] \cdot \mathbb{1}[c \le M/2]$.
2. **Bit-sliced GPU binary Rep4 carry tree**: Computes the sign bit of the
   truncated value in compact `uint32` bit-planes via CSA + carry prefix tree
   using complementary 4-joint products where supports are disjoint.
3. **CRT selection**: Converts the sign bit back to CRT with a checked 2-sender
   joint input and applies a fused register-level $y \cdot (1 - s)$ CRT gate.

## Private sign opening

`CrtPhaseGraphParty` provides both GPU (`crt_nonnegative_label`) and low-latency
bit-sliced CPU (`crt_nonnegative_label_open_cpu`, using SHAKE-256 + BLAKE2b over
the mTLS control mesh) sign extraction so final classification labels can be
opened without revealing raw logits.

## CUDA graph capture and direct-export IPC

`CrtPhaseGraphParty` inherits `PhaseGraphParty.prepare_inference`. When used with
`CudaIpcTransport`, `InferencePlan` automatically detects `alloc_export_tensor`
and allocates persistent interprocess wire buffers before capture, avoiding
sender-side D2D copies during graph replay.
