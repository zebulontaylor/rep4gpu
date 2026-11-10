"""Algebraic evidence for the masked-evaluation/checked-output boundary."""
from itertools import product


def beaver(a, b, c, d, e, p):
    return (c+d*b+a*e+d*e) % p


def test_unmasked_faulty_output_leaks_before_late_abort():
    p = 7
    for a, b, y in product(range(p), repeat=3):
        d, e = -a % p, (y-b) % p
        assert beaver(a, b, a*b % p, d, e, p) == 0
        assert beaver(a, b, a*b % p, (d+1) % p, e, p) == y


def test_fresh_mask_hides_arbitrary_tentative_secret():
    p = 7
    for tentative in range(p):
        candidates = [(tentative-mask) % p for mask in range(p)]
        assert sorted(candidates) == list(range(p))


def test_nonzero_opening_error_leaves_nonzero_authentication_residual():
    p = 7
    for alpha in range(1, p):
        for value, error in product(range(p), range(1, p)):
            mac = alpha*value % p
            residual = (mac-alpha*((value+error) % p)) % p
            assert residual != 0
