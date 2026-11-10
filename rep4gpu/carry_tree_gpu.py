"""GPU carry reductions with bit-packed state and active AND gates.

Each tree stores its nodes as binary words throughout the reduction. The
implementation gathers leaf bits from the input words and repacks each level
without allocating per-node int64 bit planes.
"""
import math
import torch
import triton
import triton.language as tl
from .runtime import Share


@triton.jit
def _read_bit(STATE, index, component, kind: tl.constexpr,
              TOTAL: tl.constexpr, OFFSET: tl.constexpr, WIDTH: tl.constexpr, valid):
    word = tl.load(STATE + kind*3*TOTAL + component*TOTAL + OFFSET + index//WIDTH, valid, 0)
    return (word.to(tl.uint64) >> (index % WIDTH)) & 1


@triton.jit
def _initialize(G0, G1, G2, P0, P1, P2, OUT,
                COUNT: tl.constexpr, LENGTH: tl.constexpr, SIZE: tl.constexpr,
                TOTAL: tl.constexpr, OFFSET: tl.constexpr, WORDS: tl.constexpr,
                WIDTH: tl.constexpr, PARTY: tl.constexpr, BLOCK: tl.constexpr):
    component = tl.program_id(1)
    word = tl.program_id(0)*BLOCK + tl.arange(0, BLOCK)
    lane = tl.arange(0, WIDTH)
    index = word[:, None]*WIDTH + lane[None, :]
    source = index//SIZE
    bit = index % SIZE
    valid = index < COUNT*SIZE
    if component == 0:
        gp, pp = G0, P0
    elif component == 1:
        gp, pp = G1, P1
    else:
        gp, pp = G2, P2
    g = tl.load(gp + source, valid & (bit < LENGTH), 0).to(tl.uint64)
    p = tl.load(pp + source, valid & (bit < LENGTH), 0).to(tl.uint64)
    gb = (g >> bit) & 1
    pb = tl.where(bit < LENGTH, (p >> bit) & 1, ((PARTY+component+1) % 4) == 0)
    pb = tl.where(valid, pb, 0).to(tl.uint64)
    gw = tl.sum(gb << lane[None, :], 1)
    pw = tl.sum(pb << lane[None, :], 1)
    tl.store(OUT + component*TOTAL + OFFSET + word, gw.to(tl.int64), word < WORDS)
    tl.store(OUT + 3*TOTAL + component*TOTAL + OFFSET + word, pw.to(tl.int64), word < WORDS)


@triton.jit
def _pack_gates(STATE, OUT, COUNT: tl.constexpr, TOTAL: tl.constexpr,
                WIDTH: tl.constexpr, GATE_WORDS: tl.constexpr, GATE_COUNT: tl.constexpr,
                SIZES: tl.constexpr, OFFSETS: tl.constexpr, GATE_OFFSETS: tl.constexpr,
                BLOCK: tl.constexpr):
    component = tl.program_id(1)
    word = tl.program_id(0)*BLOCK + tl.arange(0, BLOCK)
    lane = tl.arange(0, WIDTH)
    gate = word[:, None]*WIDTH + lane[None, :]
    left = tl.full((BLOCK, WIDTH), 0, tl.uint64)
    right = tl.full((BLOCK, WIDTH), 0, tl.uint64)
    for tree in tl.static_range(len(SIZES)):
        if SIZES[tree] > 1:
            groups = COUNT*(SIZES[tree]//2)
            relative = gate-GATE_OFFSETS[tree]
            valid = (relative >= 0) & (relative < 2*groups) & (gate < GATE_COUNT)
            node = relative % groups
            even = (node//(SIZES[tree]//2))*SIZES[tree] + (node % (SIZES[tree]//2))*2
            odd_p = _read_bit(STATE, even+1, component, 1, TOTAL, OFFSETS[tree], WIDTH, valid)
            even_g = _read_bit(STATE, even, component, 0, TOTAL, OFFSETS[tree], WIDTH, valid)
            even_p = _read_bit(STATE, even, component, 1, TOTAL, OFFSETS[tree], WIDTH, valid)
            left = tl.where(valid, odd_p, left)
            right = tl.where(valid, tl.where(relative < groups, even_g, even_p), right)
    lw = tl.sum(left << lane[None, :], 1)
    rw = tl.sum(right << lane[None, :], 1)
    tl.store(OUT + component*GATE_WORDS + word, lw.to(tl.int64), word < GATE_WORDS)
    tl.store(OUT + 3*GATE_WORDS + component*GATE_WORDS + word, rw.to(tl.int64), word < GATE_WORDS)


@triton.jit
def _next_tree(STATE, Q0, Q1, Q2, OUT,
               COUNT: tl.constexpr, OLD_SIZE: tl.constexpr, SIZE: tl.constexpr,
               OLD_TOTAL: tl.constexpr, OLD_OFFSET: tl.constexpr,
               TOTAL: tl.constexpr, OFFSET: tl.constexpr, WORDS: tl.constexpr,
               WIDTH: tl.constexpr, GATE_OFFSET: tl.constexpr, BLOCK: tl.constexpr):
    component = tl.program_id(1)
    word = tl.program_id(0)*BLOCK + tl.arange(0, BLOCK)
    lane = tl.arange(0, WIDTH)
    index = word[:, None]*WIDTH + lane[None, :]
    valid = index < COUNT*SIZE
    if OLD_SIZE > 1:
        odd = (index//SIZE)*OLD_SIZE + (index % SIZE)*2 + 1
        old_g = _read_bit(STATE, odd, component, 0, OLD_TOTAL, OLD_OFFSET, WIDTH, valid)
        if component == 0:
            qp = Q0
        elif component == 1:
            qp = Q1
        else:
            qp = Q2
        g_index = GATE_OFFSET+index
        p_index = GATE_OFFSET+COUNT*SIZE+index
        qg = tl.load(qp+g_index//WIDTH, valid, 0).to(tl.uint64)
        qg = (qg >> (g_index % WIDTH)) & 1
        pg = tl.load(qp+p_index//WIDTH, valid, 0).to(tl.uint64)
        pb = (pg >> (p_index % WIDTH)) & 1
        gb = old_g ^ qg
    else:
        gb = _read_bit(STATE, index, component, 0, OLD_TOTAL, OLD_OFFSET, WIDTH, valid)
        pb = _read_bit(STATE, index, component, 1, OLD_TOTAL, OLD_OFFSET, WIDTH, valid)
    gw = tl.sum(gb << lane[None, :], 1)
    pw = tl.sum(pb << lane[None, :], 1)
    tl.store(OUT + component*TOTAL + OFFSET + word, gw.to(tl.int64), word < WORDS)
    tl.store(OUT + 3*TOTAL + component*TOTAL + OFFSET + word, pw.to(tl.int64), word < WORDS)


@triton.jit
def _extract(STATE, OUT, COUNT: tl.constexpr, TOTAL: tl.constexpr,
              OFFSET: tl.constexpr, WIDTH: tl.constexpr, BLOCK: tl.constexpr):
    component = tl.program_id(1)
    index = tl.program_id(0)*BLOCK + tl.arange(0, BLOCK)
    value = _read_bit(STATE, index, component, 0, TOTAL, OFFSET, WIDTH, index < COUNT)
    tl.store(OUT+component*COUNT+index, value.to(tl.int64), index < COUNT)


@triton.jit
def _sign(A0, A1, A2, B0, B1, B2, C0, C1, C2, OUT,
           COUNT: tl.constexpr, BITS: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0)*BLOCK + tl.arange(0, BLOCK)
    component = tl.program_id(1)
    if component == 0:
        a, b, c = A0, B0, C0
    elif component == 1:
        a, b, c = A1, B1, C1
    else:
        a, b, c = A2, B2, C2
    x = tl.load(a+index, index < COUNT, 0)
    y = tl.load(b+index, index < COUNT, 0)
    z = tl.load(c+index, index < COUNT, 0)
    value = ((x >> (BITS-1)) & 1) ^ ((y >> (BITS-1)) & 1) ^ z
    tl.store(OUT+component*COUNT+index, value, index < COUNT)


def _layout(count, sizes, width):
    words = tuple((count*size+width-1)//width for size in sizes)
    offsets, total = [], 0
    for word_count in words:
        offsets.append(total)
        total += word_count
    return words, tuple(offsets), total


def carry_bits(rt, a, b, positions):
    """Return selected carry bits using packed GPU tree states."""
    if not a.binary or not b.binary:
        raise ValueError('carry trees require binary shares')
    if a.device.type != 'cuda':
        raise ValueError('GPU carry trees require CUDA operands')
    if a.shape != b.shape or a.device != b.device:
        raise ValueError('carry-tree operands must have matching shape and device')
    positions = tuple(positions)
    if any(not isinstance(p, int) or not 0 <= p <= rt.bits for p in positions):
        raise ValueError(f'carry positions must be integers in [0, {rt.bits}]')
    requested = sorted(set(positions)-{0})
    zero = rt.public(0, like=a, binary=True)
    if not requested or a.parts[0].numel() == 0:
        return [zero for _ in positions]
    count, width = math.prod(a.shape), rt.binary_bits
    sizes = tuple(1 << (p-1).bit_length() for p in requested)
    words, offsets, total = _layout(count, sizes, width)
    state = torch.empty((2, 3, total), dtype=torch.int64, device=a.device)
    generate, propagate = rt.mul(a, b), rt.add(a, b)
    gp = tuple(t.contiguous().reshape(-1) for t in generate.parts)
    pp = tuple(t.contiguous().reshape(-1) for t in propagate.parts)
    for length, size, word_count, offset in zip(requested, sizes, words, offsets):
        _initialize[(triton.cdiv(word_count, 32), 3)](*gp, *pp, state, count, length, size,
                                                    total, offset, word_count, width, rt.party, 32)
    while any(size > 1 for size in sizes):
        gate_offsets, gate_count = [], 0
        for size in sizes:
            gate_offsets.append(gate_count)
            if size > 1:
                gate_count += count*size
        gate_words = (gate_count+width-1)//width
        gates = torch.empty((2, 3, gate_words), dtype=torch.int64, device=a.device)
        _pack_gates[(triton.cdiv(gate_words, 32), 3)](state, gates, count, total, width, gate_words,
                                                    gate_count, sizes, offsets, tuple(gate_offsets), 32)
        product = rt.mul(Share(tuple(gates[0].unbind(0)), True), Share(tuple(gates[1].unbind(0)), True))
        next_sizes = tuple(size//2 if size > 1 else 1 for size in sizes)
        next_words, next_offsets, next_total = _layout(count, next_sizes, width)
        next_state = torch.empty((2, 3, next_total), dtype=torch.int64, device=a.device)
        for old_size, size, old_offset, offset, word_count, gate_offset in zip(sizes, next_sizes, offsets, next_offsets, next_words, gate_offsets):
            _next_tree[(triton.cdiv(word_count, 32), 3)](state, *product.parts, next_state, count,
                                                       old_size, size, total, old_offset, next_total,
                                                       offset, word_count, width, gate_offset, 32)
        state, sizes, words, offsets, total = next_state, next_sizes, next_words, next_offsets, next_total
    results = {0: zero}
    for length, offset in zip(requested, offsets):
        result = torch.empty((3, *a.shape), dtype=torch.int64, device=a.device)
        _extract[(triton.cdiv(count, 256), 3)](state, result, count, total, offset, width, 256)
        results[length] = Share(tuple(result.unbind(0)), True)
    return [results[p] for p in positions]


def truncation_bits(rt, a, b, shift):
    """Return carry into shift, ring carry, and modular-sum sign."""
    if not isinstance(shift, int) or not 0 <= shift < rt.bits:
        raise ValueError(f'shift must be an integer in [0, {rt.bits-1}]')
    low, high, before_sign = carry_bits(rt, a, b, (shift, rt.bits, rt.bits-1))
    count = math.prod(a.shape)
    result = torch.empty((3, *a.shape), dtype=torch.int64, device=a.device)
    if count:
        ap = tuple(t.contiguous().reshape(-1) for t in a.parts)
        bp = tuple(t.contiguous().reshape(-1) for t in b.parts)
        _sign[(triton.cdiv(count, 256), 3)](*ap, *bp, *before_sign.parts, result, count, rt.bits, 256)
    return low, high, Share(tuple(result.unbind(0)), True)
