"""Authenticated square activations with single-use trusted preprocessing."""
from dataclasses import dataclass
import os

import torch

from .spdz import PRIME, _random
from .spdz_distributed import AuthShare, split_share


@dataclass
class SquareMask:
    a: AuthShare
    a_squared: AuthShare
    identifier: bytes
    used: bool = False


def make_square_masks(dealer, shape):
    """Share a random ``a`` and ``a*a`` for one elementwise square."""
    if len(shape) != 2 or any(not isinstance(n, int) or n < 1 for n in shape):
        raise ValueError('square shape must contain two positive dimensions')
    a = _random(tuple(shape), dealer.device)
    shares = split_share(dealer.share(a))
    squares = split_share(dealer.share((a * a).remainder(PRIME)))
    identifier = os.urandom(32)
    return [SquareMask(x, y, identifier) for x, y in zip(shares, squares)]


def square(party, x, mask):
    """Compute ``x*x`` using one fresh masked opening.

    The correlated masks are valid only for squaring the same secret. Reusing
    them for two different inputs would expose the difference between inputs.
    The runtime checks the opening immediately or defers it within a fixed
    circuit whose pending checks finish before an unmasked output is sent.
    """
    device_impl=getattr(party,'_square_device',None)
    if device_impl is not None:
        return device_impl(x,mask)
    try:
        if party.poisoned:
            raise RuntimeError('party session is poisoned')
        spent = getattr(party, '_spent_square_masks', None)
        if spent is None:
            spent = party._spent_square_masks = set()
        if mask.used or mask.identifier in spent:
            raise ValueError('square masks are single-use')
        mask.used = True
        spent.add(mask.identifier)
        if len(mask.identifier) != 32:
            raise ValueError('invalid square mask identifier')
        for share in (x, mask.a, mask.a_squared):
            party._validate(share)
        if x.values.shape != mask.a.values.shape or x.values.shape != mask.a_squared.values.shape:
            raise ValueError('square mask shape does not match input')
        # Agree on the preprocessing identity before exposing its masked input.
        headers = party.broadcast(mask.identifier, label=b'square-mask')
        if any(header != mask.identifier for header in headers.values()):
            raise ValueError('square preprocessing identifiers differ')
        d = party.open(party.subtract(x, mask.a))
        values = (mask.a_squared.values + 2 * d * mask.a.values).remainder(PRIME)
        macs = (mask.a_squared.macs + 2 * d.unsqueeze(0) * mask.a.macs).remainder(PRIME)
        result = AuthShare(values, macs)
        return party.add_public(result, (d * d).remainder(PRIME))
    except BaseException as error:
        party.abort(error)
        raise
