"""Checked signed ring8-to-ring16 lifting of replicated arithmetic shares."""
import torch

from ..protocol.runtime import Share


def signed_lift(rt, x, *, source_bits=8):
    """Return the signed byte value in the active ring16, without opening it.

    Two jointly known source-ring pair sums u,v satisfy x=(u+v) mod 256.
    Their unsigned sum lifts as u+v-256*(carry8+sign7). Both correction bits
    are computed and converted under the existing checked Rep4 protocol.
    All conversion work happens in this call; no preprocessed bits are used.
    """
    if (source_bits != 8 or rt.bits != 16 or not rt.packed_arithmetic
            or x.binary or x.device != rt.device
            or any(part.dtype != torch.uint8 for part in x.parts)):
        raise ValueError('signed lift requires packed ring8 input and an active packed ring16 runtime')
    from ..gpu.bits.carry_tree_swar import carry_bits
    from ..gpu.bits.compact_conversion_gpu import weighted_b2a_gpu

    if x.parts[0].numel() == 0:
        return rt.public(torch.empty(x.shape, dtype=torch.int64, device=x.device))
    pair = (x.parts[1].to(torch.int64)
            + x.parts[2 if rt.party % 2 == 0 else 0].to(torch.int64)) & 255
    specs = [(0, 1, 3, 2, pair if rt.party < 2 else None),
             (2, 3, 1, 0, pair if rt.party >= 2 else None)]
    a, b = rt.joint_input_batch(specs, x.shape, True, x.device)
    upper_a, upper_b = rt.joint_input_batch(specs, x.shape, False, x.device)
    carry7, carry8 = carry_bits(rt, a, b, (7, 8))
    sign = Share(tuple(((u >> 7) & 1) ^ ((v >> 7) & 1) ^ c
                       for u, v, c in zip(a.parts, b.parts, carry7.parts)), True)
    planes = Share(tuple(torch.stack((c, s)) for c, s in zip(carry8.parts, sign.parts)), True)
    correction = weighted_b2a_gpu(rt, planes, [-256, -256])
    return rt.add(rt.add(upper_a, upper_b), correction)
