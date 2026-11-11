"""Range-aware sign extraction after quarter-ring probabilistic truncation."""
import math

import torch
import triton
import triton.language as tl

from .runtime import Share
from .carry_tree_swar import carry_bits
from .carry_prefix_gpu import carry_packed_prefix, _pack_prefix, _prefix_word
from .compact_conversion_gpu import sign_b2a_gpu


@triton.jit
def _unpack_prefix(P0, P1, P2, OUT, N: tl.constexpr, POS: tl.constexpr,
                   WIDTH: tl.constexpr, WORDS: tl.constexpr, BLOCK: tl.constexpr):
    component = tl.program_id(1)
    item = tl.program_id(0)*BLOCK+tl.arange(0, BLOCK)
    if component == 0:
        pp = P0
    elif component == 1:
        pp = P1
    else:
        pp = P2
    value = _prefix_word(pp, item*POS, WORDS, WIDTH, item < N)
    tl.store(OUT+component*N+item, (value & ((1 << POS)-1)).to(tl.int64), item < N)


def _bounded_joint_inputs(rt, pair, shape, width, *, packed_output=False):
    count, word_width = math.prod(shape), rt.binary_bits
    words = triton.cdiv(count*width, word_width)
    packed = torch.empty(words, dtype=torch.int64, device=pair.device)
    if words:
        _pack_prefix[(triton.cdiv(words, 128), 1)](pair, pair, pair, packed,
            count, width, word_width, words, 128)
    shares = rt.joint_input_batch(
        [(0,1,3,2,packed if rt.party<2 else None),
         (2,3,1,0,packed if rt.party>=2 else None)],
        (words,), True, pair.device)
    if packed_output:
        return shares
    result = []
    for share in shares:
        unpacked = torch.empty((3, *shape), dtype=torch.int64, device=pair.device)
        if count:
            _unpack_prefix[(triton.cdiv(count, 256), 3)](*share.parts, unpacked,
                count, width, word_width, words, 256)
        result.append(Share(tuple(unpacked.unbind(0)), True))
    return result


def posttrunc_sign(rt,x,shift):
    """Extract sign after trunc_pr under its strict quarter-ring contract.

    For shift > 0, the rounded result lies in the inclusive interval
    [-2**(bits-shift-2), 2**(bits-shift-2)]. Consequently its sign equals
    bit bits-shift-1, including both endpoints. The sign is evaluated modulo
    that shorter power-of-two ring; arithmetic shares remain in the full ring.
    """
    if x.binary or not isinstance(shift,int) or not 1<=shift<rt.bits-1:
        raise ValueError('posttrunc_sign requires arithmetic shares and shift in [1,bits-2]')
    return signed_width_sign(rt, x, rt.bits-shift)


def signed_width_sign(rt, x, width, *, binary=False):
    """Extract sign when x lies in [-2**(width-1), 2**(width-1)).

    Arithmetic shares stay in the full ring. Only the low width bits of the
    pair sums enter the binary addition, whose top bit is the signed result.
    The caller must establish the range before relying on this comparison.
    ``binary`` retains the sign in XOR sharing for rejection aggregation.
    """
    if x.binary or type(width) is not int or not 2 <= width < rt.bits:
        raise ValueError('signed-width sign requires arithmetic shares and width in [2,bits-1]')
    pair=rt._normalize(x.parts[1]+x.parts[2 if rt.party%2==0 else 0])
    # Only this shorter ring enters the sign circuit. Pack before sharing so
    # masks and checked joint-input wires cover exactly its private bits.
    position=width-1
    prefix=position-1
    if not binary and math.prod(x.shape) and prefix>0 and not prefix&(prefix-1):
        a,b=_bounded_joint_inputs(rt,pair.contiguous(),x.shape,width,packed_output=True)
        carry=carry_packed_prefix(rt,a,b,position,x.shape,width,complementary=True)
        return sign_b2a_gpu(rt,a,b,carry,width,shape=x.shape,packed_width=rt.binary_bits)
    a,b=_bounded_joint_inputs(rt,pair.contiguous(),x.shape,width)
    carry=carry_bits(rt,a,b,[position])[0]
    if binary:
        return Share(tuple(((left >> position)&1)^((right >> position)&1)^bit
                           for left,right,bit in zip(a.parts,b.parts,carry.parts)),True)
    return sign_b2a_gpu(rt,a,b,carry,width)
