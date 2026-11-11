# F127 degree-four extension arithmetic

`fp127ext4.h` defines `FP127Ext4`, also named `FP1274`, over the base field
`FP127` with `p = 2**127 - 1`. Its common coordinate and wire basis is

```text
[1, i, v, i*v],     i² = -1,     v² = 2+i.
```

An element `(a0,a1,a2,a3)` means `a0 + a1*i + a2*v + a3*i*v`. Use this same
basis when decomposing a key into four base-field coordinates and constructing
basis masks. Extension multiplication uses these field relations; it is not
coordinatewise multiplication.

## Interface

- The default constructor returns zero. `FP127Ext4(FP127)` embeds a canonical
  base-field element in coordinate zero.
- The four-coordinate constructor and `Coordinates` array constructor require
  canonical `FP127` coordinates. `coordinates()`, `coefficients()`, and
  `operator[]` expose const coordinate access.
- `basis_unit(h)` returns the basis element at index `h`, for `0 <= h < 4`.
- Addition, subtraction, negation, multiplication, division, equality, and
  compound operators use exact field arithmetic. Multiplication by a base-field
  scalar is supported in both operand orders.
- `inv()` rejects zero with `std::domain_error`. This differs from the base
  header's zero-inverse convention.
- `pow(exponent)` accepts an unsigned 128-bit nonnegative exponent.

For example, `FP127Ext4(FP127(3),FP127(4),FP127(5),FP127(6))` constructs
`3+4*i+5*v+6*i*v`. `FP127Ext4::basis_unit(2)` constructs `v`.

## Why this is a field

The base prime satisfies `p = 3 mod 4`, so `-1` is a nonsquare and adjoining
`i` produces a degree-two field. The norm of `2+i` down to the base field is

```text
(2+i)*(2-i) = 5.
```

Because `p = 2 mod 5`, quadratic reciprocity gives `(5/p) = (2/5) = -1`.
For any nonzero element `z` of the quadratic field,

```text
z**((p²-1)/2) = Norm(z)**((p-1)/2).
```

Thus `2+i` is a nonsquare in that quadratic field. Adjoining `v` with
`v²=2+i` produces a second quadratic extension, hence degree four over the
base field. Eliminating `i=v²-2` gives the irreducible polynomial
`v⁴-4*v²+5`.

The implementation treats an element as `A+B*v`, where `A,B` are quadratic
pairs over the base field. Karatsuba products use nine base-field
multiplications. Inversion uses

```text
(A+B*v)**-1 = (A-B*v) / (A²-(2+i)*B²).
```

The denominator lies in the quadratic field; its inverse uses the base-field
inverse of its norm.

## Canonical codec and sampling

`encode()` returns exactly 64 bytes: four consecutive canonical 16-byte
little-endian `FP127` encodings, in basis order `[1,i,v,i*v]`.
`decode(pointer,length)` rejects every length except 64, a null pointer, and
any coordinate at least `p`. An array overload accepts
`std::array<unsigned char,64>`. The codec never reduces a noncanonical wire
value into the field.

`send(io)` writes one 64-byte element. `recv(io)` reads one such element and
updates the destination only after all four coordinates decode successfully.
The IO object supplies `send_data` and `recv_data`; protocol framing and the
ordering of elements remain the caller's responsibility.

`uniform(prg)` and `rand(prg)` draw four independent base-field coordinates
through `FP127::rand`. The PRG must supply fresh uniform bytes, or its selected
cryptographic pseudorandom analogue, through `random_data_unaligned`. Each draw
masks a fresh 128-bit word to 127 bits and rejects the single value `p`.
Every accepted coordinate is therefore exactly uniform over the base field
under uniform source bytes; the four-coordinate result is uniform over the
extension. Sampling does not reduce a 128-bit integer modulo `p` and does not
reuse a sampled coordinate.

## Reproduce arithmetic checks

From the repository root, using the existing pinned `work/vole-labs` backend:

```sh
taskset -c 6,7,14,15 .venv/bin/python native/matrix_vope/run_fp127ext4_checks.py
```

The script builds an optimized C++ executable and a clang executable with
`-fsanitize=undefined -fno-sanitize-recover=all`. Build logs, run logs, source
hashes, and exact compiler commands are written under
`work/matrix_vope/fp127ext4/`.

The independent GMP oracle works in polynomial basis `[1,v,v²,v³]`, using
`v⁴=4*v²-5`; it does not duplicate the tower multiplication. The conversion
from wire coordinates to oracle coordinates is

```text
(c0,c1,c2,c3) = (a0-2*a1, a2-2*a3, a1, a3).
```

The checks cover polynomial products and inverses, base embeddings, basis
relations, Frobenius, canonical lengths and coordinate encodings, rejection
sampling, and receive failure behavior. These are arithmetic checks; they do
not implement or establish security of a VOLE protocol.
