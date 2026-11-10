"""Private input ownership and preprocessing-mask consumption."""
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

from rep4gpu.spdz import PRIME, TrustedDealer
from rep4gpu.spdz_distributed import DistributedParty
from rep4gpu.spdz_inputs import make_input_masks, private_input
from rep4gpu.transport import IdealNetwork


def setup():
    dealer = TrustedDealer()
    network = IdealNetwork(timeout=5)
    parties = [DistributedParty(p, network.endpoint(p), dealer.alpha_shares[:, p].clone(),
                                session=b'private-input-test-session',
                                _allow_insecure_test_transport=True) for p in range(4)]
    return dealer, parties


def together(fn):
    with ThreadPoolExecutor(max_workers=4) as pool:
        return list(pool.map(fn, range(4)))


@pytest.mark.parametrize('owner', range(4))
def test_owner_only_input_and_open(owner):
    dealer, parties = setup()
    value = torch.tensor([[-3, 0, PRIME - 1], [4, 5, 6]], dtype=torch.int64)
    masks = make_input_masks(dealer, owner, value.shape)
    assert [m.clear is not None for m in masks] == [p == owner for p in range(4)]
    shares = together(lambda p: private_input(parties[p], masks[p], value if p == owner else None))
    outputs = together(lambda p: parties[p].open(shares[p]))
    assert all(torch.equal(output, value.remainder(PRIME)) for output in outputs)
    assert all(mask.used for mask in masks)
    with pytest.raises(ValueError, match='single-use'):
        private_input(parties[owner], masks[owner], value)
    assert parties[owner].poisoned


def test_nonowner_input_aborts_session():
    dealer, parties = setup()
    masks = make_input_masks(dealer, 0, (1, 1))
    with pytest.raises(ValueError, match='only the input owner'):
        private_input(parties[1], masks[1], torch.ones((1, 1), dtype=torch.int64))
    assert masks[1].used and parties[1].poisoned


def test_mask_copy_identifier_cannot_be_reused():
    import copy
    dealer, parties = setup()
    masks = make_input_masks(dealer, 0, (1, 1))
    duplicate = copy.copy(masks[0])
    together(lambda p: private_input(parties[p], masks[p], torch.ones((1, 1), dtype=torch.int64) if p == 0 else None))
    with pytest.raises(ValueError, match='single-use'):
        private_input(parties[0], duplicate, torch.ones((1, 1), dtype=torch.int64))
