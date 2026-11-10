# Exact F127 GPU matrix products

`rep4gpu.field_fp127` multiplies matrices over `p = 2**127 - 1` using integer
tensor-core products. It provides the arithmetic needed for a fixed public map
`H` applied to changing raw matrices, and for matrices with fixed right weights.
It does not select protocol security parameters or generate authenticated
preprocessing.

## Tensor contract

A matrix with shape `(M,K)` uses a CUDA tensor with shape `(M,K,8)`. Its eight
little-endian radix-65536 limbs use `torch.uint16` or `torch.int32`. Limbs 0–6
are in `[0,65536)`; limb 7 is in `[0,32768)`. The encoding of `p` itself is
excluded. The caller establishes these canonical ranges; the matrix functions
check shape, dtype, device, and the inner-dimension bound without scanning every
input value. Outputs are canonical `torch.int32` tensors with shape `(M,N,8)`.

This is the same byte order as `native/matrix_vope/fp127.h`: each native element
has 16 little-endian bytes. Decode wire bytes as eight little-endian unsigned
16-bit limbs, rather than interpreting the entire element as a signed integer.
Native field validation and protocol framing remain the caller's responsibility.

The supported inner dimension is `K <= 8192`, including `K=0`. The empty inner
product returns zero. Both inputs must use the same CUDA device.

## Prepared operands

- `field_matmul_fp127(a, b)` packs both raw operands on each call.
- `prepare_fp127_left(h)` prepares a fixed left operand such as public `H`.
  Pass that snapshot as the first argument to `field_matmul_fp127` for each
  changing raw right operand.
- `prepare_fp127_weight(w)` prepares a fixed right operand. Pass that snapshot
  as the second argument; the left operand is packed on each call.
- Both operands can be prepared when measuring or computing a fixed product.

Prepared snapshots own centered-byte tensors and their integer row or column
sums. They retain no original input tensor, so modifying or releasing the input
does not change the snapshot. Do not mutate a prepared snapshot: tensor-version
checks reject ordinary in-place PyTorch edits before multiplication.

A left snapshot uses `16*M*K + 64*M` bytes. A right snapshot uses
`16*K*N + 64*N` bytes. A product temporarily allocates `256*M*N` int32 cross
products, plus `8*M*N` int64 carry coefficients and the output. This expanded
cross-product representation is useful for the tested small batches and decode
shapes; its temporary storage grows with the output matrix.

## Exact arithmetic and bounds

Each field element is split into 16 unsigned bytes. A byte `u` is represented
as signed int8 `s = u-128`. One matrix product over all byte-row and byte-column
pairs computes signed dot products. The unsigned dot is recovered exactly as

```text
sum_k u_k*v_k
  = sum_k s_k*t_k + 128*(sum_k s_k + sum_k t_k) + 16384*K.
```

There are no floating-point secret values. With `K<=8192`, every signed MMA
accumulator has absolute value at most `K*16384 <= 2**27`. Row and column sums
have absolute value at most `K*128 <= 2**20`. The correction and subsequent
convolution use int64. Arbitrary field residues never enter an MMA accumulator.

A product's byte diagonal `d` has weight `2**(8*d)`. For `d>=16`, the identity
`2**127 = 1 mod p` changes that weight to `2**(8*d-127)`. Each radix-65536
coefficient therefore combines four diagonals with shifts 0, 8, 1, and 9.
A conservative coefficient bound is

```text
16 * K * 255**2 * (1 + 256 + 2 + 512) < 2**43.
```

Individual products can occupy 254 bits; summing over K can exceed that width.
The bounded convolution coefficients preserve the full sum without fitting it
into one machine integer. The carry pass uses seven radix-65536 limbs and one
radix-32768 top limb. It propagates carries from low to high, then folds the top
carry into limb zero. The first pass leaves a numeric value below
`2**127 + 2**29`; the second leaves a value at most `p`. A third pass normalizes
any low-limb carry left by the wrap. Finally, the all-ones encoding of `p` maps
to zero. Every coefficient and carry fits signed int64.

## Reproduce correctness checks and component timings

From the repository root:

```sh
taskset -c 6,7,14,15 .venv/bin/python -m pytest -q tests/test_field_fp127.py
taskset -c 6,7,14,15 .venv/bin/python -m benchmarks.field_fp127_bench \
  --output artifacts/field_fp127_bench.json
```

The tests compare against independent Python big-integer products, including
full-field random values, byte boundaries, maximum inner-dimension carries,
canonical zero, native byte layout, and prepared snapshot ownership. The
benchmark records left/right preparation separately from raw and prepared
multiplication. Its inputs are uniform over the complete canonical field by
127-bit rejection. Large cases include independent big-integer samples and
whole-output equality across all preparation modes.

The artifact contains CUDA-event graph replay and eager host timings. These
are single field-product measurements. Native byte conversion, input transfer,
preprocessing, authentication, and communication are outside those timings.
