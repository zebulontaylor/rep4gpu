"""Label-only inference for integer linear classifiers with private weights."""
from ..protocol.runtime import Share
import torch


def nonnegative_label(rt, score):
    """Return a binary-shared bit for signed ``score >= 0`` without opening it.

    The caller must establish that the intended integer score fits the signed
    ring. This compares the ring value exactly, including its minimum value.
    Keeping the result in binary sharing avoids an arithmetic conversion when
    the only released output is a classification label.
    """
    rt._validate_share_storage(score)
    if score.parts[0].dtype == torch.uint16:
        score = Share(tuple(part.to(torch.int64) for part in score.parts))
    if score.binary:
        sign = Share(tuple((part >> (rt.bits - 1)) & 1 for part in score.parts), True)
    else:
        pair = rt._normalize(score.parts[1] + score.parts[2 if rt.party % 2 == 0 else 0])
        a, b = rt.joint_input_batch(
            [(0, 1, 3, 2, pair if rt.party < 2 else None),
             (2, 3, 1, 0, pair if rt.party >= 2 else None)],
            score.shape, True, score.device)
        if score.device.type == 'cuda':
            from ..gpu.bits.carry_tree_swar import carry_bits
        else:
            from ..gpu.bits.carry_tree import carry_bits
        carry = carry_bits(rt, a, b, [rt.bits - 1])[0]
        sign = Share(tuple(((left >> (rt.bits - 1)) & 1)
                           ^ ((right >> (rt.bits - 1)) & 1) ^ bit
                           for left, right, bit in zip(a.parts, b.parts, carry.parts)), True)
    return rt.add(sign, rt.public(1, like=sign, binary=True))


def linear_labels(rt, features, weights):
    """Return shared labels; features include a final constant-one bias column.

    Both arguments remain shared, and this function releases no output. Call
    ``rt.open`` explicitly to release labels to the computing parties.
    """
    return nonnegative_label(rt, rt.matmul(features, weights))
