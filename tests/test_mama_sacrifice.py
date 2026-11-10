"""Exhaustive algebra checks for the isolated MAMA sacrifice patch."""
from itertools import product
import pytest


def accepts(errors, challenges, p):
    return all((t * errors[0] - e) % p == 0
               for t, e in zip(challenges, errors[1:]))


@pytest.mark.parametrize('p', [2, 3, 5, 7])
def test_common_challenge_has_field_size_floor(p):
    errors = [1, 0, 0, 0]
    hits = sum(accepts(errors, [t] * 3, p) for t in range(p))
    assert hits == 1  # All checks pass at the same zero challenge.


@pytest.mark.parametrize('p,checks', [(2, 4), (3, 3), (5, 2)])
def test_independent_challenges_unique_for_every_bad_retained_error(p, checks):
    for errors in product(range(p), repeat=checks + 1):
        if errors[0] == 0:
            continue
        hits = sum(accepts(errors, ts, p)
                   for ts in product(range(p), repeat=checks))
        assert hits == 1  # Exact probability p**(-checks), including zero.


def test_sacrifice_residual_matches_error_equation():
    p = 5
    for a0, b0, ai, bi, d0, di, t in product(range(p), repeat=7):
        c0, ci = (a0*b0+d0) % p, (ai*bi+di) % p
        rho, sigma = (t*a0-ai) % p, (b0-bi) % p
        residual = (t*c0-ci-bi*rho-ai*sigma-sigma*rho) % p
        assert residual == (t*d0-di) % p
