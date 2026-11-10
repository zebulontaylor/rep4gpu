"""Authenticated polynomial toy inference with bounded integer activations."""
from concurrent.futures import ThreadPoolExecutor
import os

import pytest
import torch

from rep4gpu.spdz import PRIME, TrustedDealer
from rep4gpu.spdz_distributed import DistributedParty, split_triple
from rep4gpu.spdz_inputs import make_input_masks, private_input
from rep4gpu.spdz_polynomial import make_square_masks, square
from rep4gpu.transport import IdealNetwork


def test_owner_private_two_layer_square_network():
    torch.manual_seed(123)
    x = torch.randint(-2, 3, (4, 8), dtype=torch.int64)
    w1 = torch.randint(-2, 3, (8, 16), dtype=torch.int64)
    w2 = torch.randint(-2, 3, (16, 4), dtype=torch.int64)
    expected = ((x @ w1).square() @ w2).remainder(PRIME)
    dealer = TrustedDealer()
    masks = [make_input_masks(dealer, owner, tuple(value.shape))
             for owner, value in zip([0, 1, 1], [x, w1, w2])]
    first = split_triple(dealer.triple(4, 8, 16))
    second = split_triple(dealer.triple(4, 16, 4))
    squares = make_square_masks(dealer, (4, 16))
    net = IdealNetwork(timeout=10)
    session = os.urandom(32)
    parties = [DistributedParty(p, net.endpoint(p), dealer.alpha_shares[:, p].clone(),
                                session=session, _allow_insecure_test_transport=True) for p in range(4)]
    def run(p):
        party = parties[p]
        shares = [private_input(party, mask[p], value if p == owner else None)
                  for mask, value, owner in zip(masks, [x, w1, w2], [0, 1, 1])]
        hidden = party.matmul(shares[0], shares[1], first[p])
        activated = square(party, hidden, squares[p])
        return party.open(party.matmul(activated, shares[2], second[p]))
    with ThreadPoolExecutor(max_workers=4) as pool:
        outputs = list(pool.map(run, range(4)))
    assert all(torch.equal(output, expected) for output in outputs)
    assert all(mask.used for mask in squares)
    with pytest.raises(ValueError, match='single-use'):
        square(parties[0], squares[0].a, squares[0])
    assert parties[0].poisoned
