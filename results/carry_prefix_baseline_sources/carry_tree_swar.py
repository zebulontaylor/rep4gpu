"""Carry-tree packing with whole-word bit compaction.

The node and gate ordering matches carry_tree_gpu. The compaction sequence
handles even or odd bits of a packed word without expanding individual nodes.
Internal 32-bit tree words use uint32 storage; protocol shares remain int64.
"""
import math

import torch
import triton
import triton.language as tl

from .runtime import Share
from .carry_tree_gpu import _extract, _layout, _sign


@triton.jit
def _low_mask(n, WIDTH: tl.constexpr):
    if WIDTH == 32:
        ones = tl.full((), 0xffffffff, tl.uint32)
    else:
        ones = tl.full((), 0xffffffffffffffff, tl.uint64)
    return tl.where(n <= 0, 0, ones >> ((WIDTH-tl.minimum(n, WIDTH)) % WIDTH))


@triton.jit
def _compact(value, ODD: tl.constexpr, WIDTH: tl.constexpr):
    if WIDTH == 32:
        value = value.to(tl.uint32)
        if ODD:
            value >>= 1
        value &= 0x55555555
        value = (value | (value >> 1)) & 0x33333333
        value = (value | (value >> 2)) & 0x0f0f0f0f
        value = (value | (value >> 4)) & 0x00ff00ff
        return (value | (value >> 8)) & 0x0000ffff
    else:
        value = value.to(tl.uint64)
        if ODD:
            value >>= 1
        value &= 0x5555555555555555
        value = (value | (value >> 1)) & 0x3333333333333333
        value = (value | (value >> 2)) & 0x0f0f0f0f0f0f0f0f
        value = (value | (value >> 4)) & 0x00ff00ff00ff00ff
        value = (value | (value >> 8)) & 0x0000ffff0000ffff
        return (value | (value >> 16)) & 0x00000000ffffffff


@triton.jit
def _state_word(STATE, index, component, KIND: tl.constexpr,
                TOTAL: tl.constexpr, OFFSET: tl.constexpr, WORDS: tl.constexpr, valid):
    return tl.load(STATE+KIND*3*TOTAL+component*TOTAL+OFFSET+index,
                   valid & (index >= 0) & (index < WORDS), 0)


@triton.jit
def _compacted_at(STATE, word, component, KIND: tl.constexpr, ODD: tl.constexpr,
                  TOTAL: tl.constexpr, OFFSET: tl.constexpr, WORDS: tl.constexpr,
                  START: tl.constexpr, WIDTH: tl.constexpr, valid):
    # Output bit zero corresponds to global gate bit word*WIDTH. START binds
    # that stream to a tree's compressed node stream, including partial words.
    HALF: tl.constexpr = WIDTH//2
    REM: tl.constexpr = (-START) % HALF
    index = (word*WIDTH-START-REM)//HALF
    first = _compact(_state_word(STATE, index, component, KIND, TOTAL, OFFSET, WORDS, valid), ODD, WIDTH)
    second = _compact(_state_word(STATE, index+1, component, KIND, TOTAL, OFFSET, WORDS, valid), ODD, WIDTH)
    value = (first >> REM) | (second << (HALF-REM))
    if REM:
        third = _compact(_state_word(STATE, index+2, component, KIND, TOTAL, OFFSET, WORDS, valid), ODD, WIDTH)
        value |= third << (WIDTH-REM)
    return value


@triton.jit
def _initialize(G0, G1, G2, P0, P1, P2, OUT,
                COUNT: tl.constexpr, LENGTH: tl.constexpr, SIZE: tl.constexpr,
                TOTAL: tl.constexpr, OFFSET: tl.constexpr, WORDS: tl.constexpr,
                WIDTH: tl.constexpr, PARTY: tl.constexpr, BLOCK: tl.constexpr):
    component = tl.program_id(1)
    word = tl.program_id(0)*BLOCK+tl.arange(0, BLOCK)
    if component == 0:
        gp, pp = G0, P0
    elif component == 1:
        gp, pp = G1, P1
    else:
        gp, pp = G2, P2
    if WIDTH == 32:
        gw = tl.full((BLOCK,), 0, tl.uint32)
        pw = tl.full((BLOCK,), 0, tl.uint32)
    else:
        gw = tl.full((BLOCK,), 0, tl.uint64)
        pw = tl.full((BLOCK,), 0, tl.uint64)
    if SIZE <= WIDTH:
        for slot in tl.static_range(WIDTH//SIZE):
            source = word*(WIDTH//SIZE)+slot
            valid = source < COUNT
            g = tl.load(gp+source, valid, 0).to(tl.uint64)
            p = tl.load(pp+source, valid, 0).to(tl.uint64)
            leaf_mask = _low_mask(LENGTH, WIDTH)
            gb = g & leaf_mask
            pb = p & leaf_mask
            padding = _low_mask(SIZE, WIDTH) ^ leaf_mask
            pb |= tl.where(((PARTY+component+1)%4)==0, padding, 0)
            pb = tl.where(valid, pb, 0)
            gw |= (gb << (slot*SIZE)).to(gw.dtype)
            pw |= (pb << (slot*SIZE)).to(pw.dtype)
    else:
        source = word//(SIZE//WIDTH)
        start = (word % (SIZE//WIDTH))*WIDTH
        g = tl.load(gp+source, source < COUNT, 0).to(tl.uint64)
        p = tl.load(pp+source, source < COUNT, 0).to(tl.uint64)
        leaf_mask = _low_mask(LENGTH-start, WIDTH)
        gw = ((g >> start) & leaf_mask).to(gw.dtype)
        pw = ((p >> start) & leaf_mask).to(pw.dtype)
        pw |= tl.where(((PARTY+component+1)%4)==0, _low_mask(WIDTH, WIDTH) ^ leaf_mask, 0).to(pw.dtype)
        pw = tl.where(source < COUNT, pw, 0)
    tl.store(OUT+component*TOTAL+OFFSET+word, gw.to(tl.int64), word < WORDS)
    tl.store(OUT+3*TOTAL+component*TOTAL+OFFSET+word, pw.to(tl.int64), word < WORDS)


@triton.jit
def _segment(STATE, word, component, TOTAL: tl.constexpr, OFFSET: tl.constexpr,
              WORDS: tl.constexpr, START: tl.constexpr, GROUPS: tl.constexpr,
              KIND: tl.constexpr, WIDTH: tl.constexpr, GATE_WORDS: tl.constexpr):
    relative = word*WIDTH-START
    valid = (relative < GROUPS) & (relative+WIDTH > 0) & (word < GATE_WORDS)
    lo = tl.maximum(-relative, 0)
    hi = tl.minimum(GROUPS-relative, WIDTH)
    mask = _low_mask(hi, WIDTH) ^ _low_mask(lo, WIDTH)
    mask = tl.where(valid, mask, 0)
    lp = _compacted_at(STATE, word, component, 1, True, TOTAL,
                        OFFSET, WORDS, START, WIDTH, valid)
    rp = _compacted_at(STATE, word, component, KIND, False, TOTAL,
                        OFFSET, WORDS, START, WIDTH, valid)
    return lp & mask, rp & mask


@triton.jit
def _pack_gates(STATE, OUT, COUNT: tl.constexpr, TOTAL: tl.constexpr,
                WIDTH: tl.constexpr, GATE_WORDS: tl.constexpr, GATE_COUNT: tl.constexpr,
                SIZES: tl.constexpr, OFFSETS: tl.constexpr, GATE_OFFSETS: tl.constexpr,
                BLOCK: tl.constexpr):
    component = tl.program_id(1)
    word = tl.program_id(0)*BLOCK+tl.arange(0, BLOCK)
    if WIDTH == 32:
        left = tl.full((BLOCK,), 0, tl.uint32)
        right = tl.full((BLOCK,), 0, tl.uint32)
    else:
        left = tl.full((BLOCK,), 0, tl.uint64)
        right = tl.full((BLOCK,), 0, tl.uint64)
    for tree in tl.static_range(len(SIZES)):
        if SIZES[tree] > 1:
            lg, rg = _segment(STATE, word, component, TOTAL, OFFSETS[tree],
                              (COUNT*SIZES[tree]+WIDTH-1)//WIDTH,
                              GATE_OFFSETS[tree], COUNT*(SIZES[tree]//2),
                              0, WIDTH, GATE_WORDS)
            lp, rp = _segment(STATE, word, component, TOTAL, OFFSETS[tree],
                              (COUNT*SIZES[tree]+WIDTH-1)//WIDTH,
                              GATE_OFFSETS[tree]+COUNT*(SIZES[tree]//2),
                              COUNT*(SIZES[tree]//2), 1, WIDTH, GATE_WORDS)
            left |= lg | lp
            right |= rg | rp
    tl.store(OUT+component*GATE_WORDS+word, left.to(tl.int64), word < GATE_WORDS)
    tl.store(OUT+3*GATE_WORDS+component*GATE_WORDS+word, right.to(tl.int64), word < GATE_WORDS)


@triton.jit
def _read_packed(Q, word, START: tl.constexpr, WIDTH: tl.constexpr,
                 WORDS: tl.constexpr, valid):
    REM: tl.constexpr = START % WIDTH
    index = word+START//WIDTH
    if WIDTH == 32:
        a = tl.load(Q+index, valid & (index < WORDS), 0).to(tl.uint32)
    else:
        a = tl.load(Q+index, valid & (index < WORDS), 0).to(tl.uint64)
    value = a >> REM
    if REM:
        if WIDTH == 32:
            b = tl.load(Q+index+1, valid & (index+1 < WORDS), 0).to(tl.uint32)
        else:
            b = tl.load(Q+index+1, valid & (index+1 < WORDS), 0).to(tl.uint64)
        value |= b << (WIDTH-REM)
    return value


@triton.jit
def _next_tree(STATE, Q0, Q1, Q2, OUT,
               COUNT: tl.constexpr, OLD_SIZE: tl.constexpr, SIZE: tl.constexpr,
               OLD_TOTAL: tl.constexpr, OLD_OFFSET: tl.constexpr,
               TOTAL: tl.constexpr, OFFSET: tl.constexpr, WORDS: tl.constexpr,
               WIDTH: tl.constexpr, GATE_OFFSET: tl.constexpr,
               GATE_WORDS: tl.constexpr, BLOCK: tl.constexpr):
    component = tl.program_id(1)
    word = tl.program_id(0)*BLOCK+tl.arange(0, BLOCK)
    valid = word < WORDS
    OLD_WORDS: tl.constexpr = (COUNT*OLD_SIZE+WIDTH-1)//WIDTH
    if OLD_SIZE > 1:
        old_g = _compacted_at(STATE, word, component, 0, True, OLD_TOTAL,
                               OLD_OFFSET, OLD_WORDS, 0, WIDTH, valid)
        if component == 0:
            qp = Q0
        elif component == 1:
            qp = Q1
        else:
            qp = Q2
        qg = _read_packed(qp, word, GATE_OFFSET, WIDTH, GATE_WORDS, valid)
        pg = _read_packed(qp, word, GATE_OFFSET+COUNT*SIZE, WIDTH, GATE_WORDS, valid)
        gb, pb = old_g ^ qg, pg
    else:
        gb = _state_word(STATE, word, component, 0, OLD_TOTAL, OLD_OFFSET, OLD_WORDS, valid)
        pb = _state_word(STATE, word, component, 1, OLD_TOTAL, OLD_OFFSET, OLD_WORDS, valid)
    tail = _low_mask(COUNT*SIZE-word*WIDTH, WIDTH)
    gb, pb = gb & tail, pb & tail
    tl.store(OUT+component*TOTAL+OFFSET+word, gb.to(tl.int64), valid)
    tl.store(OUT+3*TOTAL+component*TOTAL+OFFSET+word, pb.to(tl.int64), valid)


def carry_bits(rt, a, b, positions):
    """Match the packed carry-tree transcript using whole-word compaction."""
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
    state_dtype = torch.uint32 if width == 32 else torch.int64
    state = torch.empty((2, 3, total), dtype=state_dtype, device=a.device)
    generate, propagate = rt.mul(a, b), rt.add(a, b)
    gp = tuple(t.contiguous().reshape(-1) for t in generate.parts)
    pp = tuple(t.contiguous().reshape(-1) for t in propagate.parts)
    for length, size, word_count, offset in zip(requested, sizes, words, offsets):
        _initialize[(triton.cdiv(word_count, 128), 3)](*gp, *pp, state, count, length, size,
                                                     total, offset, word_count, width, rt.party, 128)
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
