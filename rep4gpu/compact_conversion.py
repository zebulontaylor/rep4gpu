"""Binary-to-arithmetic conversion with joint parity inputs and weighted dots."""

import torch

from .conversion import _components
from .runtime import Share


def _weighted_sum(value, weights):
    if weights.ndim == 1:
        return (value * weights.reshape((-1,) + (1,)*(value.ndim-1))).sum(dim=0)
    layout = weights.T.reshape((*weights.T.shape,) + (1,)*(value.ndim-1))
    return (value.unsqueeze(1) * layout).sum(dim=0)


def _weighted_local_products(xs, ys, weights):
    x0, x1, x2 = xs
    y0, y1, y2 = ys
    return tuple(_weighted_sum(value, weights) for value in (
        (x0+x1)*y0 + x0*y1,
        (x1+x2)*y1 + x1*y2,
        x0*y2 + x2*y0))


def weighted_b2a(rt, planes, weights):
    """Convert and combine shared bits without opening any bit.

    ``planes`` has shape (P, *S), with each secret plane holding 0 or 1.
    Integer weights of shape (P,) produce shape S; weights (O,P) produce
    O separate weighted outputs with shape (O,*S).
    """
    if not planes.binary or len(planes.shape) == 0:
        raise ValueError('weighted_b2a requires binary bit planes with a leading plane axis')
    weights = torch.as_tensor(weights, dtype=torch.int64, device=planes.device)
    if weights.ndim not in (1, 2) or weights.shape[-1] != planes.shape[0]:
        raise ValueError('weights must have shape (planes,) or (outputs,planes)')
    planes = Share(tuple(part & 1 for part in planes.parts), True)
    a, b, c, d = _components(rt, planes, False)
    known = {(rt.party+j+1)%4: value for j, value in enumerate(planes.parts)}
    product01 = known[0]*known[1] if rt.party in (2, 3) else None
    product23 = known[2]*known[3] if rt.party in (0, 1) else None
    p01, p23 = rt.joint_input_batch(
        [(2, 3, 1, 0, product01), (0, 1, 3, 2, product23)],
        planes.shape, False, planes.device)
    left = rt.sub(rt.add(a, b), rt.mul_public(p01, 2))
    right = rt.sub(rt.add(c, d), rt.mul_public(p23, 2))
    if planes.device.type == 'cuda':
        from .weighted_kernels import weighted_local_products
        local = weighted_local_products(left.parts, right.parts, weights, bits=rt.bits)
    else:
        local = _weighted_local_products(left.parts, right.parts, weights)
    contributions = [None]*5
    contributions[(rt.party-1)%4], contributions[rt.party], contributions[4] = local
    product = rt._reshare_product(contributions, False, planes.device)
    linear = rt._share(tuple(_weighted_sum(l+r, weights) for l, r in zip(left.parts, right.parts)))
    return rt.sub(linear, rt.mul_public(product, 2))


def b2a(rt, x, *, bits=None, signed=True):
    """Convert selected low binary word bits with two communication rounds."""
    if not x.binary:
        raise ValueError('b2a requires binary shares')
    bits = rt.bits if bits is None else bits
    if not isinstance(bits, int) or not 1 <= bits <= rt.bits:
        raise ValueError(f'bits must be in [1, {rt.bits}]')
    positions = torch.arange(bits, dtype=torch.int64, device=x.device)
    positions = positions.reshape((bits,) + (1,)*len(x.shape))
    planes = Share(tuple((part.unsqueeze(0) >> positions) & 1 for part in x.parts), True)
    weights = torch.ones(bits, dtype=torch.int64, device=x.device) << torch.arange(bits, device=x.device)
    if signed and bits < 64:
        weights[-1] = -(1 << (bits-1))
    return weighted_b2a(rt, planes, weights)
