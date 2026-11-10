"""Owner-private inputs using single-use authenticated preprocessing masks."""
from dataclasses import dataclass
import os
import struct

import torch

from .spdz import PRIME, _random
from .spdz_distributed import AuthShare, split_share


@dataclass
class InputMask:
    share: AuthShare
    clear: torch.Tensor | None
    owner: int
    identifier: bytes
    used: bool = False


def make_input_masks(dealer, owner, shape):
    """Return one mask per role; only the input owner receives the clear mask.

    The trusted dealer learns the random mask, but this function never receives
    the private input. The dealer must not observe the subsequent online input
    transcript: the clear mask and public correction together reveal the input.
    """
    if owner not in range(dealer.parties):
        raise ValueError('input owner is outside the party set')
    if len(shape) != 2 or any(not isinstance(n, int) or n < 1 for n in shape):
        raise ValueError('input shape must contain two positive dimensions')
    clear = _random(tuple(shape), dealer.device)
    shares = split_share(dealer.share(clear))
    identifier = os.urandom(32)
    return [InputMask(share, clear.clone() if p == owner else None, owner, identifier)
            for p, share in enumerate(shares)]


def private_input(party, mask, value=None):
    """Consume a mask and consistently broadcast the owner's masked input.

    All roles call in the same order. Only the owner supplies ``value``. A failed
    attempt retires the party session, and its input mask cannot be retried.
    """
    try:
        if party.poisoned:
            raise RuntimeError('party session is poisoned')
        spent = getattr(party, '_spent_input_masks', None)
        if spent is None:
            spent = party._spent_input_masks = set()
        if mask.used or mask.identifier in spent:
            raise ValueError('input masks are single-use')
        mask.used = True
        spent.add(mask.identifier)
        if mask.owner not in range(party.parties) or len(mask.identifier) != 32:
            raise ValueError('invalid input mask metadata')
        party._validate(mask.share)
        shape = tuple(mask.share.values.shape)
        if len(shape) != 2 or min(shape) < 1:
            raise ValueError('input mask must be a nonempty matrix')
        if party.party == mask.owner:
            if value is None or mask.clear is None:
                raise ValueError('owner requires an input and clear preprocessing mask')
            if mask.clear.dtype != torch.int64 or tuple(mask.clear.shape) != shape:
                raise ValueError('clear preprocessing mask has the wrong type or shape')
            if torch.any((mask.clear < 0) | (mask.clear >= PRIME)).item():
                raise ValueError('clear preprocessing mask must be canonical')
            if value.dtype != torch.int64 or tuple(value.shape) != shape:
                raise ValueError('input must be an int64 matrix matching the mask shape')
            epsilon = (value.to(party.device).remainder(PRIME) - mask.clear.to(party.device)).remainder(PRIME)
            payload = epsilon.cpu().numpy().astype('<u4').tobytes()
        else:
            if value is not None or mask.clear is not None:
                raise ValueError('only the input owner receives an input or clear mask')
            payload = b''
        label = b'private-input/' + mask.identifier + struct.pack('<I', mask.owner) + str(shape).encode('ascii')
        received = party.broadcast(payload, label=label)
        if any(data for p, data in received.items() if p != mask.owner):
            raise ValueError('non-owner supplied an input correction')
        data = received[mask.owner]
        if len(data) != 4 * shape[0] * shape[1]:
            raise ValueError('input correction has the wrong length')
        import numpy as np
        correction = torch.from_numpy(np.frombuffer(data, dtype='<u4').astype(np.int64)).reshape(shape).to(party.device)
        if torch.any(correction >= PRIME).item():
            raise ValueError('input correction is not a canonical field encoding')
        return party.add_public(mask.share, correction)
    except BaseException as error:
        party.abort(error)
        raise
