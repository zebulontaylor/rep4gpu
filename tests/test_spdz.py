"""Algebra and fault detection for the trusted-dealer online simulator."""
import pytest
import torch

from rep4gpu.spdz import (PRIME, AuthenticationError, AuthenticatedOnline,
                         TrustedDealer, field_matmul)


def test_field_product_matches_python_integers():
    a = torch.tensor([[PRIME - 1, PRIME - 2, 65537], [0, 1, -1]], dtype=torch.int64)
    b = torch.tensor([[PRIME - 3, 5], [7, PRIME - 1], [PRIME - 1, 99999]], dtype=torch.int64)
    expected = torch.tensor([[sum(int(a[i, k]) * int(b[k, j]) for k in range(3)) % PRIME
                              for j in range(2)] for i in range(2)])
    assert torch.equal(field_matmul(a, b), expected)


@pytest.mark.parametrize("parties", [2, 4])
def test_authenticated_product(parties):
    dealer = TrustedDealer(parties=parties)
    online = AuthenticatedOnline(dealer)
    a = torch.tensor([[1, -2, 3], [PRIME - 1, 4, 5]], dtype=torch.int64)
    b = torch.tensor([[7, 8], [9, 10], [11, 12]], dtype=torch.int64)
    result = online.matmul(dealer.share(a), dealer.share(b), dealer.triple(2, 3, 2))
    assert torch.equal(online.open(result), field_matmul(a, b))


@pytest.mark.parametrize("where", ["value", "mac"])
def test_tampered_opening_aborts(where):
    dealer = TrustedDealer()
    online = AuthenticatedOnline(dealer)
    dealer.alpha_shares[0].zero_()
    dealer.alpha_shares[0, 0] = 1
    dealer.alpha[0] = 1
    value = dealer.share(torch.ones((2, 2), dtype=torch.int64))
    if where == "value":
        value.values[0, 0, 0] += 1
    else:
        value.macs[0, 0, 0, 0] += 1
    with pytest.raises(AuthenticationError):
        online.open(value)


def test_triples_cannot_be_reused_after_success_or_abort():
    dealer = TrustedDealer()
    online = AuthenticatedOnline(dealer)
    a = dealer.share(torch.ones((1, 1), dtype=torch.int64))
    triple = dealer.triple(1, 1, 1)
    online.matmul(a, a, triple)
    with pytest.raises(ValueError, match="single-use"):
        online.matmul(a, a, triple)
    bad = dealer.triple(1, 1, 1)
    a.macs[0, 0, 0, 0] += 1
    with pytest.raises(AuthenticationError):
        online.matmul(a, a, bad)
    assert bad.used


def test_authenticated_output_tampering_aborts():
    dealer = TrustedDealer()
    online = AuthenticatedOnline(dealer)
    a = dealer.share(torch.tensor([[123, -456]], dtype=torch.int64))
    b = dealer.share(torch.tensor([[789], [1000]], dtype=torch.int64))
    result = online.matmul(a, b, dealer.triple(1, 2, 1))
    result.macs[2, 3, 0, 0] += 1
    with pytest.raises(AuthenticationError):
        online.open(result)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_field_product_matches_cpu_and_authentication():
    a = torch.randint(PRIME, (3, 67), dtype=torch.int64)
    b = torch.randint(PRIME, (67, 17), dtype=torch.int64)
    expected = field_matmul(a, b)
    assert torch.equal(field_matmul(a.cuda(), b.cuda()).cpu(), expected)
    dealer = TrustedDealer(device="cuda")
    online = AuthenticatedOnline(dealer)
    result = online.matmul(dealer.share(a), dealer.share(b), dealer.triple(3, 67, 17))
    assert torch.equal(online.open(result).cpu(), expected)


def test_trusted_dealer_can_authenticate_an_incorrect_triple():
    """Authentication alone does not establish the triple relation."""
    dealer = TrustedDealer()
    online = AuthenticatedOnline(dealer)
    one = torch.ones((1, 1), dtype=torch.int64)
    x, y = dealer.share(one), dealer.share(one)
    triple = dealer.triple(1, 1, 1)
    correct_c = triple.c.values.sum(0).remainder(PRIME)
    triple.c = dealer.share(correct_c + 1)
    result = online.open(online.matmul(x, y, triple))
    assert result.item() == 2
