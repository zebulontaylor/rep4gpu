"""A carry prefix consisting of a power-of-two block and one final bit."""
import math
import torch
import triton
import triton.language as tl
from .runtime import Share
from .carry_tree_swar import _layout, _pack_gates, _next_tree, carry_bits


@triton.jit
def _pack_prefix(A0, A1, A2, OUT, N: tl.constexpr, POS: tl.constexpr,
                 WIDTH: tl.constexpr, WORDS: tl.constexpr, BLOCK: tl.constexpr):
    component = tl.program_id(1)
    word = tl.program_id(0)*BLOCK+tl.arange(0, BLOCK)
    if component == 0:
        ap = A0
    elif component == 1:
        ap = A1
    else:
        ap = A2
    DT: tl.constexpr = tl.uint32 if WIDTH == 32 else tl.uint64
    packed = tl.full((BLOCK,), 0, DT)
    for slot in tl.static_range((WIDTH+POS-1)//POS+1):
        item = word*WIDTH//POS+slot
        offset = item*POS-word*WIDTH
        value = tl.load(ap+item, (item < N) & (offset < WIDTH), 0).to(DT)
        value &= (1 << POS)-1
        packed |= tl.where(offset >= 0, value << tl.maximum(offset, 0),
                           value >> tl.maximum(-offset, 0))
    if WIDTH == 32:
        packed &= 0xffffffff
    tl.store(OUT+component*WORDS+word, packed.to(tl.int64), word < WORDS)


@triton.jit
def _prefix_word(P, index, WORDS: tl.constexpr, WIDTH: tl.constexpr, valid):
    word, offset = index//WIDTH, index%WIDTH
    DT: tl.constexpr = tl.uint32 if WIDTH == 32 else tl.uint64
    first = tl.load(P+word, valid & (word < WORDS), 0).to(DT)
    second = tl.load(P+word+1, valid & (word+1 < WORDS), 0).to(DT)
    return (first >> offset) | tl.where(offset != 0, second << (WIDTH-offset), 0)


@triton.jit
def _initialize_prefix(G0, G1, G2, P0, P1, P2, OUT,
                       N: tl.constexpr, POS: tl.constexpr, PREFIX: tl.constexpr,
                       WIDTH: tl.constexpr, WORDS: tl.constexpr,
                       TOTAL: tl.constexpr, BLOCK: tl.constexpr):
    component = tl.program_id(1)
    word = tl.program_id(0)*BLOCK+tl.arange(0, BLOCK)
    if component == 0:
        gp, pp = G0, P0
    elif component == 1:
        gp, pp = G1, P1
    else:
        gp, pp = G2, P2
    DT: tl.constexpr = tl.uint32 if WIDTH == 32 else tl.uint64
    g = tl.full((BLOCK,), 0, DT)
    p = tl.full((BLOCK,), 0, DT)
    for slot in tl.static_range(WIDTH//PREFIX):
        item = word*(WIDTH//PREFIX)+slot
        gv = _prefix_word(gp, item*POS, WORDS, WIDTH, item < N)
        pv = _prefix_word(pp, item*POS, WORDS, WIDTH, item < N)
        g |= (gv & ((1 << PREFIX)-1)) << (slot*PREFIX)
        p |= (pv & ((1 << PREFIX)-1)) << (slot*PREFIX)
    tl.store(OUT+component*TOTAL+word, g.to(tl.int64), word < TOTAL)
    tl.store(OUT+3*TOTAL+component*TOTAL+word, p.to(tl.int64), word < TOTAL)


@triton.jit
def _tail_gates(P0, P1, P2, STATE, OUT, N: tl.constexpr,
                POS: tl.constexpr, WIDTH: tl.constexpr, WORDS: tl.constexpr,
                PACKED_WORDS: tl.constexpr, STRIDE: tl.constexpr, BLOCK: tl.constexpr):
    component = tl.program_id(1)
    word = tl.program_id(0)*BLOCK+tl.arange(0, BLOCK)
    bit = tl.arange(0, WIDTH)
    item = word[:, None]*WIDTH+bit[None, :]
    if component == 0:
        pp = P0
    elif component == 1:
        pp = P1
    else:
        pp = P2
    # Adjacent lanes load adjacent inputs; each row reduces to one protocol word.
    p = _prefix_word(pp, item*STRIDE+POS, PACKED_WORDS, WIDTH, item < N) & 1
    packed = tl.sum(p << bit[None, :], 1)
    carry = tl.load(STATE+component*WORDS+word, word < WORDS, 0)
    tl.store(OUT+component*WORDS+word, packed.to(tl.int64), word < WORDS)
    tl.store(OUT+3*WORDS+component*WORDS+word, carry.to(tl.int64), word < WORDS)


@triton.jit
def _tail_result(G0, G1, G2, Q0, Q1, Q2, OUT, N: tl.constexpr,
                 POS: tl.constexpr, WIDTH: tl.constexpr, PACKED_WORDS: tl.constexpr, STRIDE: tl.constexpr, BLOCK: tl.constexpr):
    component = tl.program_id(1)
    item = tl.program_id(0)*BLOCK+tl.arange(0, BLOCK)
    if component == 0:
        gp, qp = G0, Q0
    elif component == 1:
        gp, qp = G1, Q1
    else:
        gp, qp = G2, Q2
    g = _prefix_word(gp, item*STRIDE+POS, PACKED_WORDS, WIDTH, item < N) & 1
    q = tl.load(qp+item//WIDTH, item < N, 0).to(tl.uint64)
    result = g ^ ((q >> (item % WIDTH)) & 1)
    tl.store(OUT+component*N+item, result.to(tl.int64), item < N)


def carry_power_plus_one(rt,a,b,position):
    """Return carry into position when position-1 is a power of two.

    Every AND uses the runtime's checked multiplication. The final gate combines
    the packed lower block carry with the last input bit without padding the
    prefix to twice the block length.
    """
    prefix=position-1
    if prefix<1 or prefix&(prefix-1) or position>rt.bits:
        raise ValueError('position-1 must be a positive power of two below ring width')
    if not a.binary or not b.binary or a.device.type!='cuda' or a.shape!=b.shape or a.device!=b.device:
        raise ValueError('carry operands must be matching CUDA binary shares')
    if not a.parts[0].numel():
        return rt.public(0,like=a,binary=True)
    count, width = math.prod(a.shape), rt.binary_bits
    if prefix > width:
        return carry_bits(rt, a, b, [position])[0]
    # Omit unused high bits before the first AND. Packing is local and linear
    # over XOR shares, so the checked multiplication still evaluates every leaf.
    packed_words = (count*position+width-1)//width
    packed = []
    for operand in (a, b):
        parts = tuple(part.contiguous().reshape(-1) for part in operand.parts)
        buffer = torch.empty((3, packed_words), dtype=torch.int64, device=a.device)
        _pack_prefix[(triton.cdiv(packed_words, 128), 3)](*parts, buffer,
            count, position, width, packed_words, 128)
        packed.append(Share(tuple(buffer.unbind(0)), True))
    return carry_packed_prefix(rt, *packed, position, a.shape, position)


def carry_packed_prefix(rt, a, b, position, shape, stride, *, complementary=False):
    """Evaluate the prefix from contiguous packed XOR shares with public stride."""
    prefix = position-1
    count, width = math.prod(shape), rt.binary_bits
    packed_words = a.parts[0].numel()
    sizes = (prefix,)
    words, offsets, total = _layout(count, sizes, width)
    state_dtype = torch.uint32 if width == 32 else torch.int64
    state = torch.empty((2, 3, total), dtype=state_dtype, device=a.device)
    generate = (rt._mul_sparse_pairs(a, b)
                if complementary and getattr(rt, 'sparse_products', False) else rt.mul(a, b))
    propagate = rt.add(a, b)
    gp = tuple(t.contiguous().reshape(-1) for t in generate.parts)
    pp = tuple(t.contiguous().reshape(-1) for t in propagate.parts)
    _initialize_prefix[(triton.cdiv(total, 128), 3)](*gp, *pp, state,
        count, stride, prefix, width, packed_words, total, 128)
    while any(size > 1 for size in sizes):
        gate_offsets, gate_count = [], 0
        for size in sizes:
            gate_offsets.append(gate_count)
            if size > 1:
                gate_count += count*size
        gate_words = (gate_count+width-1)//width
        gates = torch.empty((2, 3, gate_words), dtype=torch.int64, device=a.device)
        _pack_gates[(triton.cdiv(gate_words, 128), 3)](state, gates, count, total, width,
                                                    gate_words, gate_count, sizes, offsets,
                                                    tuple(gate_offsets), 128)
        product = rt.mul(Share(tuple(gates[0].unbind(0)), True), Share(tuple(gates[1].unbind(0)), True))
        next_sizes = tuple(size//2 if size > 1 else 1 for size in sizes)
        next_words, next_offsets, next_total = _layout(count, next_sizes, width)
        next_state = torch.empty((2, 3, next_total), dtype=state_dtype, device=a.device)
        for old_size, size, old_offset, offset, word_count, gate_offset in zip(sizes, next_sizes, offsets, next_offsets, next_words, gate_offsets):
            _next_tree[(triton.cdiv(word_count, 128), 3)](state, *product.parts, next_state, count,
                                                       old_size, size, total, old_offset, next_total,
                                                       offset, word_count, width, gate_offset,
                                                       gate_words, 128)
        state, sizes, words, offsets, total = next_state, next_sizes, next_words, next_offsets, next_total
    # G_final and P_final*C_lower are disjoint, so XOR is the carry-out.
    # Only that AND is needed; the final propagate is never consumed.
    final_words = (count+width-1)//width
    gates = torch.empty((2, 3, final_words), dtype=torch.int64, device=a.device)
    _tail_gates[(triton.cdiv(final_words, 4), 3)](*pp, state, gates, count,
        prefix, width, final_words, packed_words, stride, 4)
    product = rt.mul(Share(tuple(gates[0].unbind(0)), True),
                     Share(tuple(gates[1].unbind(0)), True))
    result = torch.empty((3, *shape), dtype=torch.int64, device=a.device)
    _tail_result[(triton.cdiv(count, 256), 3)](*gp, *product.parts, result,
        count, prefix, width, packed_words, stride, 256)
    return Share(tuple(result.unbind(0)), True)
