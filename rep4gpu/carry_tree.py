"""Packed Boolean carry trees for selected public carry positions.

Each reduction packs active gates across words and trees into packed
Rep4 binary words. It keeps the carry-tree depth while shrinking communication
at every level. No arithmetic or Boolean share is opened.

This CPU reference materializes local bit planes and rejects CUDA inputs. A
GPU implementation must gather and repack active bits without those planes.
"""
import math

import torch

from .runtime import Share


def _pack_parts(parts, word_bits=64):
    """Pack local binary-share bits in row-major order, low lane first."""
    count = parts[0].numel()
    words = (count + word_bits-1) // word_bits
    result = []
    for part in parts:
        flat = part.reshape(-1)
        if words == 0:
            result.append(torch.empty(0, dtype=torch.int64, device=part.device))
            continue
        padded = torch.zeros(words*word_bits, dtype=torch.int64, device=part.device)
        padded[:count] = flat
        weights = torch.ones(word_bits, dtype=torch.int64, device=part.device) << torch.arange(word_bits, device=part.device)
        result.append((padded.reshape(words, word_bits)*weights).sum(dim=1))
    return Share(tuple(result), True)


def _unpack_parts(share, count, word_bits=64):
    positions = torch.arange(word_bits, dtype=torch.int64, device=share.device)
    return tuple(((part[:, None] >> positions) & 1).reshape(-1)[:count] for part in share.parts)


def _initial_tree(rt, generate, propagate, length):
    size = 1 << (length-1).bit_length()
    count = math.prod(generate.shape)
    positions = torch.arange(length, dtype=torch.int64, device=generate.device)
    gs, ps = [], []
    # Public padding uses the same component-zero embedding as Party.public.
    padding = rt.public(torch.ones((count, size-length), dtype=torch.int64, device=generate.device), binary=True)
    for component in range(3):
        g = (generate.parts[component].reshape(-1, 1) >> positions) & 1
        p = (propagate.parts[component].reshape(-1, 1) >> positions) & 1
        gs.append(torch.cat((g, torch.zeros_like(padding.parts[component])), dim=1))
        ps.append(torch.cat((p, padding.parts[component]), dim=1))
    return [tuple(gs), tuple(ps)]


def carry_bits(rt, a, b, positions):
    """Return binary shares of carries into the requested bit positions.

    Position 0 has public carry zero. Position ``rt.bits`` is carry out of the
    ring. Other positions lie between them. Inputs are packed binary shares
    with matching shapes; positions and shapes are public. Duplicate positions
    reuse a tree, and the returned list follows the caller's order.
    """
    if not a.binary or not b.binary:
        raise ValueError('carry trees require binary shares')
    if a.device.type != 'cpu':
        raise ValueError('the dense carry-tree reference supports CPU only')
    if a.shape != b.shape or a.device != b.device:
        raise ValueError('carry-tree operands must have matching shape and device')
    positions = tuple(positions)
    if any(not isinstance(position, int) or not 0 <= position <= rt.bits for position in positions):
        raise ValueError(f'carry positions must be integers in [0, {rt.bits}]')
    if not positions:
        return []
    requested = sorted(set(positions)-{0})
    zero = rt.public(torch.zeros(a.shape, dtype=torch.int64, device=a.device), binary=True)
    if not requested or a.parts[0].numel() == 0:
        return [zero for _ in positions]
    generate = rt.mul(a, b)
    propagate = rt.add(a, b)
    trees = {length: _initial_tree(rt, generate, propagate, length) for length in requested}
    while True:
        active = [(length, tree) for length, tree in trees.items() if tree[0][0].shape[1] > 1]
        if not active:
            break
        left, right = [[] for _ in range(3)], [[] for _ in range(3)]
        # G and P are mutually exclusive for each group. Thus the two
        # terms in G_odd XOR (P_odd AND G_even) cannot both be one.
        for _, (gs, ps) in active:
            for component in range(3):
                odd_p = ps[component][:, 1::2].reshape(-1)
                left[component].extend((odd_p, odd_p))
                right[component].extend((gs[component][:, ::2].reshape(-1), ps[component][:, ::2].reshape(-1)))
        left = tuple(torch.cat(parts) for parts in left)
        right = tuple(torch.cat(parts) for parts in right)
        count = left[0].numel()
        products = _unpack_parts(rt.mul(_pack_parts(left, rt.binary_bits), _pack_parts(right, rt.binary_bits)),
                                 count, rt.binary_bits)
        offset = 0
        for _, tree in active:
            gs, ps = tree
            shape = gs[0][:, 1::2].shape
            groups = math.prod(shape)
            next_g, next_p = [], []
            for component in range(3):
                product_g = products[component][offset:offset+groups].reshape(shape)
                product_p = products[component][offset+groups:offset+2*groups].reshape(shape)
                next_g.append(gs[component][:, 1::2] ^ product_g)
                next_p.append(product_p)
            tree[:] = [tuple(next_g), tuple(next_p)]
            offset += 2*groups
        assert offset == count
    results = {length: Share(tuple(part.reshape(a.shape) for part in tree[0]), True)
               for length, tree in trees.items()}
    results[0] = zero
    return [results[position] for position in positions]


def truncation_bits(rt, a, b, shift):
    """Return carry into shift, ring carry, and sign of the modular sum.

    These three bits are sufficient to correct unsigned upper-word sums into
    exact signed truncation. This function returns Boolean shares and leaves
    their arithmetic conversion to the caller.
    """
    if not isinstance(shift, int) or not 0 <= shift < rt.bits:
        raise ValueError(f'shift must be an integer in [0, {rt.bits-1}]')
    low, high, before_sign = carry_bits(rt, a, b, (shift, rt.bits, rt.bits-1))
    sign = Share(tuple(((x >> (rt.bits-1)) & 1) ^ ((y >> (rt.bits-1)) & 1) ^ c
                       for x, y, c in zip(a.parts, b.parts, before_sign.parts)), True)
    return low, high, sign


def reduction_word_counts(count, bits, positions, word_bits=64):
    """Return packed AND word counts, including the initial generate round.

    Each word contributes ``6 * (word_bits // 8)`` global value bytes,
    with another 192 digest bytes per nonempty multiplication round.
    """
    if count < 0 or bits not in (32, 64) or word_bits not in (32, 64):
        raise ValueError('count must be nonnegative and ring/word bits must be 32 or 64')
    if word_bits == 32 and bits != 32:
        raise ValueError('32-bit binary words require a 32-bit arithmetic ring')
    positions = set(positions)-{0}
    if any(not isinstance(position, int) or not 0 < position <= bits for position in positions):
        raise ValueError('invalid carry position')
    if not positions or count == 0:
        return []
    sizes = [1 << (position-1).bit_length() for position in positions]
    words = [count]
    while any(size > 1 for size in sizes):
        gates = count*sum(size for size in sizes if size > 1)
        words.append((gates+word_bits-1)//word_bits)
        sizes = [size//2 if size > 1 else size for size in sizes]
    return words
